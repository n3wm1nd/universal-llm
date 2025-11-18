{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module OpenAIComposableSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.Providers.OpenAI

-- Helper to build request for GLM45 (GLM45 with full composition)
buildRequest :: GLM45
             -> [ModelConfig OpenAI GLM45]
             -> [Message GLM45 OpenAI]
             -> OpenAIRequest
buildRequest _model = buildRequestGeneric TestModels.openAIGLM45 GLM45 ((), ((), ((), ())))

-- Generic helper to build request with explicit composable provider
buildRequestGeneric :: forall model s. ComposableProvider OpenAI model s
                    -> model
                    -> s
                    -> [ModelConfig OpenAI model]
                    -> [Message model OpenAI]
                    -> OpenAIRequest
buildRequestGeneric composableProvider model s configs = snd . toProviderRequest composableProvider OpenAI model configs s

-- Helper to parse OpenAI response for GLM45
parseOpenAIResponse :: GLM45
                    -> [ModelConfig OpenAI GLM45]
                    -> [Message GLM45 OpenAI]
                    -> OpenAIResponse
                    -> Either LLMError [Message GLM45 OpenAI]
parseOpenAIResponse _model configs _history (OpenAIError (OpenAIErrorResponse errDetail)) =
  Left $ ProviderError (code errDetail) $ errorMessage errDetail <> " (" <> errorType errDetail <> ")"
parseOpenAIResponse _model configs _history resp =
  let msgs = parseOpenAIResponseGeneric TestModels.openAIGLM45 GLM45 configs ((), ((), ((), ()))) resp
  in if null msgs
     then Left $ ParseError "No messages parsed from response"
     else Right msgs

-- Generic helper to parse response with explicit composable provider
parseOpenAIResponseGeneric :: forall model s. ComposableProvider OpenAI model s
                           -> model
                           -> [ModelConfig OpenAI model]
                           -> s
                           -> OpenAIResponse
                           -> [Message model OpenAI]
parseOpenAIResponseGeneric composableProvider model configs s resp =
  snd $ fromProviderResponse composableProvider OpenAI model configs s resp

spec :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
spec getResponse = do
  describe "OpenAI Composable Provider - Basic Text" $ do

    it "sends message, receives response, and maintains conversation history" $ do
      let model = GLM45
          configs = [MaxTokens 500]  -- Increased for reasoning models

          -- First exchange
          msgs1 = [UserText "What is 2+2?" :: Message GLM45 OpenAI]
          req1 = buildRequest GLM45 configs msgs1

      resp1 <- getResponse req1

      case parseOpenAIResponse GLM45 configs msgs1 resp1 of
        Right [AssistantText txt] -> do
          T.isInfixOf "4" txt `shouldBe` True

          -- Second exchange - append to history
          let msgs2 = msgs1 <> [AssistantText txt, UserText "What about 3+3?"]
              req2 = buildRequest GLM45 configs msgs2

          resp2 <- getResponse req2

          case parseOpenAIResponse GLM45 configs msgs2 resp2 of
            Right [AssistantText txt2] -> do
              T.isInfixOf "6" txt2 `shouldBe` True

              -- Verify request has full conversation history
              length (messages req2) `shouldBe` 3

        Right other -> expectationFailure $ "Expected [AssistantText], got: " ++ show other
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "merges consecutive user messages" $ do
      let model = GLM45
          configs = [MaxTokens 50]
          msgs = [ UserText "First part" :: Message GLM45 OpenAI
                 , UserText "Second part"
                 ]
          req = buildRequest GLM45 configs msgs

      -- Should merge into single user message
      length (messages req) `shouldBe` 1
      case head (messages req) of
        OpenAIMessage "user" (Just content) _ _ _ ->
          content `shouldBe` "First part\nSecond part"
        _ -> expectationFailure "Expected merged user message"

  describe "OpenAI Composable Provider - Tool Calling" $ do

    it "completes full tool calling conversation flow" $ do
      let model = GLM45
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the current weather in a given location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("The city name" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [Tools [toolDef], MaxTokens 800]  -- Increased for reasoning models with tools

          -- Step 1: Initial request with tools
          msgs1 = [UserText "What's the weather in Paris?" :: Message GLM45 OpenAI]
          req1 = buildRequest GLM45 configs msgs1

      -- Verify tools are in request
      case tools req1 of
        Just [tool] -> do
          tool_type tool `shouldBe` "function"
          name (function tool) `shouldBe` "get_weather"
        _ -> expectationFailure "Expected exactly one tool in request"

      resp1 <- getResponse req1

      case parseOpenAIResponse GLM45 configs msgs1 resp1 of
        Right [AssistantTool toolCall] -> do
          getToolCallName toolCall `shouldBe` "get_weather"

          -- Step 2: Execute tool (simulated) and send result
          let toolResult = ToolResult toolCall
                             (Right $ object ["temperature" .= ("22°C" :: Text)])
              msgs2 = msgs1 <> [AssistantTool toolCall, ToolResultMsg toolResult]
              req2 = buildRequest GLM45 configs msgs2

          -- Verify history has all messages
          length (messages req2) `shouldBe` 3

          resp2 <- getResponse req2

          case parseOpenAIResponse GLM45 configs msgs2 resp2 of
            Right [AssistantText finalTxt] -> do
              -- Should incorporate the tool result
              T.isInfixOf "22" finalTxt `shouldBe` True
            Right other -> expectationFailure $ "Expected [AssistantText], got: " ++ show other
            Left err -> expectationFailure $ "parseResponse failed: " ++ show err

        Right other -> expectationFailure $ "Expected [AssistantTool], got: " ++ show other
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

  describe "OpenAI Composable Provider - JSON Mode" $ do

    it "requests and receives JSON response" $ do
      let model = GLM45
          schema = object
            [ "type" .= ("object" :: Text)
            , "properties" .= object
                [ "colors" .= object
                    [ "type" .= ("array" :: Text)
                    , "items" .= object ["type" .= ("string" :: Text)]
                    ]
                ]
            , "required" .= (["colors"] :: [Text])
            ]
          configs = [MaxTokens 500]  -- Increased for reasoning models
          msgs = [UserRequestJSON "List 3 primary colors" schema :: Message GLM45 OpenAI]
          req = buildRequest GLM45 configs msgs

      -- Verify response_format is set with correct schema
      case response_format req of
        Nothing -> expectationFailure "Expected response_format to be set"
        Just format -> do
          responseType format `shouldBe` "json_schema"
          case json_schema format of
            Just (Aeson.Object obj) ->
              case KM.lookup "properties" obj of
                Just (Aeson.Object props) ->
                  case KM.lookup "colors" props of
                    Just _ -> return ()  -- Schema has 'colors' field
                    Nothing -> expectationFailure "Schema missing 'colors' property"
                _ -> expectationFailure "Schema 'properties' is not an object"
            _ -> expectationFailure "json_schema is not an object"

      resp <- getResponse req

      case parseOpenAIResponse GLM45 configs msgs resp of
        Right [AssistantJSON jsonVal] -> do
          case jsonVal of
            Aeson.Object obj ->
              case KM.lookup "colors" obj of
                Just (Aeson.Array arr) -> length arr `shouldSatisfy` (>= 3)
                _ -> expectationFailure "JSON missing 'colors' array"
            _ -> expectationFailure "Response not a JSON object"
        Right other -> expectationFailure $ "Expected [AssistantJSON], got: " ++ show other
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "latest UserRequestJSON sets the schema for the request" $ do
      let model = GLM45
          schema1 = object ["type" .= ("string" :: Text)]
          schema2 = object ["type" .= ("number" :: Text)]
          configs = [MaxTokens 50]

          -- First JSON request
          msgs1 = [UserRequestJSON "Give me a string" schema1 :: Message GLM45 OpenAI]
          req1 = buildRequest GLM45 configs msgs1

          -- Second request after response - add new JSON request with different schema
          msgs2 = msgs1 <> [AssistantJSON (Aeson.String "hello"), UserRequestJSON "Give me a number" schema2]
          req2 = buildRequest GLM45 configs msgs2

      -- req1 should have schema1
      case response_format req1 of
        Just format -> case json_schema format of
          Just (Aeson.Object obj) ->
            case KM.lookup "type" obj of
              Just (Aeson.String "string") -> return ()
              _ -> expectationFailure "Expected schema1 (type: string)"
          _ -> expectationFailure "Expected json_schema object"
        Nothing -> expectationFailure "Expected response_format to be set"

      -- req2 should have schema2 (latest UserRequestJSON wins)
      case response_format req2 of
        Just format -> case json_schema format of
          Just (Aeson.Object obj) ->
            case KM.lookup "type" obj of
              Just (Aeson.String "number") -> return ()
              _ -> expectationFailure "Expected schema2 (type: number)"
          _ -> expectationFailure "Expected json_schema object"
        Nothing -> expectationFailure "Expected response_format to be set"

  describe "Compile-Time Safety Demonstrations" $ do

    it "allows tool use with tool-capable model" $ do
      let model = GLM45  -- HasTools
          toolDef = ToolDefinition "test_tool" "Test" (object [])
          configs = [Tools [toolDef], MaxTokens 50]
          msgs = [UserText "test" :: Message GLM45 OpenAI]
          req = buildRequest GLM45 configs msgs

      case tools req of
        Just [_] -> return ()
        _ -> expectationFailure "Expected tools in request"

    -- These tests demonstrate compile-time safety by being commented out
    -- Uncommenting them will cause compilation to fail

    {-
    it "SHOULD NOT COMPILE: tools with BasicTextModel (no HasTools instance)" $ do
      let model = BasicTextModel  -- NO HasTools instance!
          toolDef = ToolDefinition "test_tool" "Test" (object [])
          -- This line will fail to compile:
          -- No instance for (HasTools BasicTextModel)
          configs = [Tools [toolDef], MaxTokens 50]
          msgs = [UserText "test" :: Message BasicTextModel OpenAI]
          -- Cannot use fullComposableProvider without HasTools and HasJSON
          req = buildRequest GLM45 configs msgs
      return ()
    -}

    {-
    it "SHOULD NOT COMPILE: JSON mode with BasicTextModel (no HasJSON instance)" $ do
      let model = BasicTextModel  -- NO HasJSON instance!
          schema = object ["type" .= ("string" :: Text)]
          -- This line will fail to compile:
          -- No instance for (HasJSON BasicTextModel)
          msgs = [UserRequestJSON "test" schema :: Message BasicTextModel OpenAI]
          -- Cannot use fullComposableProvider without HasJSON
          req = buildRequest model [] msgs
      return ()
    -}

    {-
    it "SHOULD NOT COMPILE: AssistantTool without HasTools" $ do
      let model = BasicTextModel  -- NO HasTools!
          toolCall = ToolCall "id" "name" (object [])
          -- This line will fail to compile:
          -- No instance for (HasTools BasicTextModel)
          msgs = [AssistantTool toolCall :: Message BasicTextModel OpenAI]
          req = buildRequest model [] msgs
      return ()
    -}

  describe "OpenAI Provider - Message Structure" $ do
    it "correctly structures OpenAI messages with all optional fields" $ do
      -- Test that we can create messages with various combinations of fields
      let msg1 = OpenAIMessage
            { role = "assistant"
            , content = Just "Hello"
            , reasoning_content = Nothing
            , tool_calls = Nothing
            , tool_call_id = Nothing
            }
      role msg1 `shouldBe` "assistant"
      content msg1 `shouldBe` Just "Hello"

      -- Test with reasoning content
      let msg2 = OpenAIMessage
            { role = "assistant"
            , content = Just "The answer is 42"
            , reasoning_content = Just "Let me think..."
            , tool_calls = Nothing
            , tool_call_id = Nothing
            }
      reasoning_content msg2 `shouldBe` Just "Let me think..."
      content msg2 `shouldBe` Just "The answer is 42"

      -- Test with tool calls
      let toolFunc = OpenAIToolFunction "get_weather" "{\"location\": \"Paris\"}"
          toolCall = OpenAIToolCall "call1" "function" toolFunc
          msg3 = OpenAIMessage
            { role = "assistant"
            , content = Nothing
            , reasoning_content = Nothing
            , tool_calls = Just [toolCall]
            , tool_call_id = Nothing
            }
      tool_calls msg3 `shouldBe` Just [toolCall]

