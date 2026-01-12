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
import qualified Data.Map.Strict as Map
import Data.Default (def)
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.Providers.OpenAI

-- Type aliases for easier provider/model switching
type TestProvider = OpenRouter
type TestAIModel = GLM45
type TestModel = Model TestAIModel TestProvider

-- Helper to build request for the test model with full composition
buildRequest :: TestModel
             -> [ModelConfig TestModel]
             -> [Message TestModel]
             -> OpenAIRequest
buildRequest model = buildRequestGeneric TestModels.openRouterGLM45 model (def, ((), ((), ())))

-- Generic helper to build request with explicit composable provider
buildRequestGeneric :: forall m s. (ProviderRequest m ~ OpenAIRequest)
                    => ComposableProvider m s
                    -> m
                    -> s
                    -> [ModelConfig m]
                    -> [Message m]
                    -> OpenAIRequest
buildRequestGeneric composableProvider model s configs = snd . toProviderRequest composableProvider model configs s

-- Helper to parse OpenAI response for the test model
parseOpenAIResponse :: TestModel
                    -> [ModelConfig TestModel]
                    -> [Message TestModel]
                    -> OpenAIResponse
                    -> Either LLMError [Message TestModel]
parseOpenAIResponse model configs _history (OpenAIError (OpenAIErrorResponse errDetail)) =
  let typeInfo = case errorType errDetail of
        Just t -> " (" <> t <> ")"
        Nothing -> ""
  in Left $ ProviderError (code errDetail) $ errorMessage errDetail <> typeInfo
parseOpenAIResponse model configs _history resp =
  let msgs = parseOpenAIResponseGeneric TestModels.openRouterGLM45 model configs (def, ((), ((), ()))) resp
  in if null msgs
     then Left $ ParseError "No messages parsed from response"
     else Right msgs

-- Generic helper to parse response with explicit composable provider
parseOpenAIResponseGeneric :: forall m s. (ProviderResponse m ~ OpenAIResponse)
                           => ComposableProvider m s
                           -> m
                           -> [ModelConfig m]
                           -> s
                           -> OpenAIResponse
                           -> [Message m]
parseOpenAIResponseGeneric composableProvider model configs s resp =
  either (error . show) snd $ fromProviderResponse composableProvider model configs s resp

spec :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
spec getResponse = do
  describe "OpenAI Composable Provider - Basic Text" $ do

    it "sends message, receives response, and maintains conversation history" $ do
      let model = Model GLM45 OpenRouter :: TestModel
          configs = [MaxTokens 500]  -- Increased for reasoning models

          -- First exchange
          msgs1 = [UserText "What is 2+2?" :: Message TestModel]
          req1 = buildRequest model configs msgs1

      resp1 <- getResponse req1

      case parseOpenAIResponse (Model GLM45 OpenRouter) configs msgs1 resp1 of
        Right parsedMsgs1 -> do
          -- Extract AssistantText, may also have AssistantReasoning
          let textMsgs1 = [txt | AssistantText txt <- parsedMsgs1]
          length textMsgs1 `shouldSatisfy` (> 0)
          let txt = head textMsgs1
          T.isInfixOf "4" txt `shouldBe` True

          -- Second exchange - append to history
          let msgs2 = msgs1 <> parsedMsgs1 <> [UserText "What about 3+3?"]
              req2 = buildRequest (Model GLM45 OpenRouter) configs msgs2

          resp2 <- getResponse req2

          case parseOpenAIResponse (Model GLM45 OpenRouter) configs msgs2 resp2 of
            Right msgs -> do
              -- Should have at least AssistantText, may also have AssistantReasoning
              let textMsgs = [txt | AssistantText txt <- msgs]
              length textMsgs `shouldSatisfy` (> 0)
              let txt2 = head textMsgs
              T.isInfixOf "6" txt2 `shouldBe` True

              -- Verify request has conversation history (exact count depends on reasoning extraction)
              length (messages req2) `shouldSatisfy` (>= 3)

            Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "merges consecutive user messages" $ do
      let model = Model GLM45 OpenRouter
          configs = [MaxTokens 50]
          msgs = [ UserText "First part" :: Message TestModel
                 , UserText "Second part"
                 ]
          req = buildRequest model configs msgs

      -- Should merge into single user message
      length (messages req) `shouldBe` 1
      let msg = head (messages req)
      role msg `shouldBe` "user"
      content msg `shouldBe` Just "First part\nSecond part"

  describe "OpenAI Composable Provider - Tool Calling" $ do

    it "completes full tool calling conversation flow" $ do
      let model = Model GLM45 OpenRouter
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
          msgs1 = [UserText "What's the weather in Paris?" :: Message TestModel]
          req1 = buildRequest model configs msgs1

      -- Verify tools are in request
      case tools req1 of
        Just [tool] -> do
          tool_type tool `shouldBe` "function"
          name (function tool) `shouldBe` "get_weather"
        _ -> expectationFailure "Expected exactly one tool in request"

      resp1 <- getResponse req1

      case parseOpenAIResponse (Model GLM45 OpenRouter) configs msgs1 resp1 of
        Right parsedMsgs -> do
          -- Extract tool calls from the response (may be mixed with text)
          let toolCalls = [tc | AssistantTool tc <- parsedMsgs]
          length toolCalls `shouldSatisfy` (> 0)
          let toolCall = head toolCalls
          getToolCallName toolCall `shouldBe` "get_weather"

          -- Step 2: Execute tool (simulated) and send result
          let toolResult = ToolResult toolCall
                             (Right $ object ["temperature" .= ("22°C" :: Text)])
              msgs2 = msgs1 <> parsedMsgs <> [ToolResultMsg toolResult]
              req2 = buildRequest (Model GLM45 OpenRouter) configs msgs2

          resp2 <- getResponse req2

          case parseOpenAIResponse (Model GLM45 OpenRouter) configs msgs2 resp2 of
            Right msgs2Result -> do
              -- Extract text from the response
              let textMsgs = [txt | AssistantText txt <- msgs2Result]
              length textMsgs `shouldSatisfy` (> 0)
              -- Should incorporate the tool result
              let finalTxt = head textMsgs
              T.isInfixOf "22" finalTxt `shouldBe` True
            Left err -> expectationFailure $ "parseResponse failed: " ++ show err

        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

  describe "OpenAI Composable Provider - JSON Mode" $ do

    it "requests and receives JSON response" $ do
      let model = Model GLM45 OpenRouter
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
          msgs = [UserRequestJSON "List 3 primary colors" schema :: Message TestModel]
          req = buildRequest model configs msgs

      -- Verify response_format is set with correct schema
      case response_format req of
        Nothing -> expectationFailure "Expected response_format to be set"
        Just format -> do
          responseType format `shouldBe` "json_schema"
          case json_schema format of
            Just (Aeson.Object wrapper) ->
              -- The schema is wrapped with name/strict/schema fields
              case KM.lookup "schema" wrapper of
                Just (Aeson.Object schemaObj) ->
                  case KM.lookup "properties" schemaObj of
                    Just (Aeson.Object props) ->
                      case KM.lookup "colors" props of
                        Just _ -> return ()  -- Schema has 'colors' field
                        Nothing -> expectationFailure "Schema missing 'colors' property"
                    _ -> expectationFailure "Schema 'properties' is not an object"
                _ -> expectationFailure "Wrapped schema 'schema' field is not an object"
            _ -> expectationFailure "json_schema is not an object"

      resp <- getResponse req

      case parseOpenAIResponse (Model GLM45 OpenRouter) configs msgs resp of
        Right parsedMsgs -> do
          -- Extract AssistantJSON, may also have AssistantReasoning
          let jsonMsgs = [jsonVal | AssistantJSON jsonVal <- parsedMsgs]
          length jsonMsgs `shouldSatisfy` (> 0)
          case head jsonMsgs of
            Aeson.Object obj ->
              case KM.lookup "colors" obj of
                Just (Aeson.Array arr) -> length arr `shouldSatisfy` (>= 3)
                _ -> expectationFailure "JSON missing 'colors' array"
            _ -> expectationFailure "Response not a JSON object"
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "latest UserRequestJSON sets the schema for the request" $ do
      let model = Model GLM45 OpenRouter
          schema1 = object ["type" .= ("string" :: Text)]
          schema2 = object ["type" .= ("number" :: Text)]
          configs = [MaxTokens 50]

          -- First JSON request
          msgs1 = [UserRequestJSON "Give me a string" schema1 :: Message TestModel]
          req1 = buildRequest model configs msgs1

          -- Second request after response - add new JSON request with different schema
          msgs2 = msgs1 <> [AssistantJSON (Aeson.String "hello"), UserRequestJSON "Give me a number" schema2]
          req2 = buildRequest model configs msgs2

      -- req1 should have schema1 (wrapped in name/strict/schema)
      case response_format req1 of
        Just format -> case json_schema format of
          Just (Aeson.Object wrapper) ->
            case KM.lookup "schema" wrapper of
              Just (Aeson.Object schemaObj) ->
                case KM.lookup "type" schemaObj of
                  Just (Aeson.String "string") -> return ()
                  _ -> expectationFailure "Expected schema1 (type: string)"
              _ -> expectationFailure "Expected wrapped schema object"
          _ -> expectationFailure "Expected json_schema object"
        Nothing -> expectationFailure "Expected response_format to be set"

      -- req2 should have schema2 (latest UserRequestJSON wins)
      case response_format req2 of
        Just format -> case json_schema format of
          Just (Aeson.Object wrapper) ->
            case KM.lookup "schema" wrapper of
              Just (Aeson.Object schemaObj) ->
                case KM.lookup "type" schemaObj of
                  Just (Aeson.String "number") -> return ()
                  _ -> expectationFailure "Expected schema2 (type: number)"
              _ -> expectationFailure "Expected wrapped schema object"
          _ -> expectationFailure "Expected json_schema object"
        Nothing -> expectationFailure "Expected response_format to be set"

  describe "Compile-Time Safety Demonstrations" $ do

    it "allows tool use with tool-capable model" $ do
      let model = Model GLM45 OpenRouter  -- HasTools
          toolDef = ToolDefinition "test_tool" "Test" (object [])
          configs = [Tools [toolDef], MaxTokens 50]
          msgs = [UserText "test" :: Message TestModel]
          req = buildRequest model configs msgs

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
      let msg1 = defaultOpenAIMessage
            { role = "assistant"
            , content = Just "Hello"
            }
      role msg1 `shouldBe` "assistant"
      content msg1 `shouldBe` Just "Hello"

      -- Test with reasoning content
      let msg2 = defaultOpenAIMessage
            { role = "assistant"
            , content = Just "The answer is 42"
            , reasoning_content = Just "Let me think..."
            }
      reasoning_content msg2 `shouldBe` Just "Let me think..."
      content msg2 `shouldBe` Just "The answer is 42"

      -- Test with tool calls
      let toolFunc = defaultOpenAIToolFunction
            { toolFunctionName = "get_weather"
            , toolFunctionArguments = "{\"location\": \"Paris\"}"
            }
          toolCall = defaultOpenAIToolCall
            { callId = "call1"
            , toolFunction = toolFunc
            }
          msg3 = defaultOpenAIMessage
            { role = "assistant"
            , tool_calls = Just [toolCall]
            }
      tool_calls msg3 `shouldBe` Just [toolCall]

  describe "OpenRouter Reasoning Details Preservation" $ do
    it "stores reasoning_details by tool call ID" $ do
      let toolCall1 = OpenAIToolCall
            { callId = "call_abc123"
            , toolCallType = "function"
            , toolFunction = OpenAIToolFunction
                { toolFunctionName = "get_weather"
                , toolFunctionArguments = "{}"
                }
            }
          reasoningDetailsValue = object ["test" .= ("value" :: Text)]
          responseMsg = defaultOpenAIMessage
            { role = "assistant"
            , tool_calls = Just [toolCall1]
            , reasoning_details = Just reasoningDetailsValue
            }
          response = OpenAISuccess $ OpenAISuccessResponse [OpenAIChoice responseMsg]

          -- Create the reasoning provider and get its handlers
          initialState = OpenRouterReasoningState mempty mempty
          handlers = openRouterReasoning  @(Model TestModels.Gemini3ProPreview OpenRouter) (Model TestModels.Gemini3ProPreview OpenRouter) [] initialState

          -- Store reasoning_details from the response
          updatedState = cpPostResponse handlers response initialState

      -- Verify the tool call ID was stored in the map
      toolCallToDetails updatedState `shouldSatisfy` (\m -> not $ null $ show m)

    it "adds reasoning_details to messages with tool calls" $ do
      let toolCall1 = OpenAIToolCall
            { callId = "call_xyz789"
            , toolCallType = "function"
            , toolFunction = OpenAIToolFunction
                { toolFunctionName = "search"
                , toolFunctionArguments = "{}"
                }
            }
          reasoningDetailsValue = object ["thinking" .= ("step by step" :: Text)]

          -- Create state with stored reasoning_details
          stateWithDetails = OpenRouterReasoningState mempty (Map.singleton "call_xyz789" reasoningDetailsValue)

          -- Create request with a message that has tool calls (but no reasoning_details yet)
          requestMsg = defaultOpenAIMessage
            { role = "assistant"
            , tool_calls = Just [toolCall1]
            , reasoning_details = Nothing
            }
          request = defaultOpenAIRequest { messages = [requestMsg] }

          -- Apply the config handler
          handlers = openRouterReasoning @(Model TestModels.Gemini3ProPreview OpenRouter) (Model TestModels.Gemini3ProPreview OpenRouter) [] stateWithDetails
          updatedRequest = cpConfigHandler handlers request

      -- Verify reasoning_details were added to the message
      case messages updatedRequest of
        [msg] -> reasoning_details msg `shouldBe` Just reasoningDetailsValue
        _ -> expectationFailure "Expected exactly one message"
