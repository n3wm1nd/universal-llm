{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module AnthropicComposableSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import qualified Data.Aeson as Aeson
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.Anthropic
import qualified UniversalLLM.Protocols.Anthropic as Proto
import qualified UniversalLLM.Providers.Anthropic as Provider

-- Helper to build request using the model's provider implementation
buildRequest :: forall model. ProviderImplementation Provider.Anthropic model
             => model
             -> [ModelConfig Provider.Anthropic model]
             -> [Message model Provider.Anthropic]
             -> AnthropicRequest
buildRequest = toProviderRequest Provider.Anthropic

-- Helper to parse response using the model's provider implementation
parseResponse :: forall model. (ProviderImplementation Provider.Anthropic model, ModelName Provider.Anthropic model)
              => model
              -> [ModelConfig Provider.Anthropic model]
              -> [Message model Provider.Anthropic]  -- history
              -> AnthropicResponse
              -> Either LLMError [Message model Provider.Anthropic]
parseResponse model configs history resp =
  let msgs = fromProviderResponse Provider.Anthropic model configs history resp
  in if null msgs
     then Left $ ParseError "No messages parsed from response"
     else Right msgs

spec :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
spec getResponse = do
  describe "Anthropic Request Building (no API calls)" $ do
    it "can build and evaluate a request without looping" $ do
      let model = ClaudeSonnet45
          configs = [Temperature 0.7, MaxTokens 200]
          msgs = [UserText "Test message"]
          req = buildRequest model configs msgs

      -- Force evaluation - should not loop
      req `seq` return ()

      -- Check basic properties
      Proto.model req `shouldBe` "claude-sonnet-4-5-20250929"
      Proto.max_tokens req `shouldBe` 200
      length (Proto.messages req) `shouldBe` 1

  describe "Anthropic Composable Provider - Basic Text" $ do

    it "sends message, receives response, and maintains conversation history" $ do
      let model = ClaudeSonnet45
          configs = [MaxTokens 100, SystemPrompt "You are a helpful assistant."]

          -- First exchange
          msgs1 = [UserText "What is 2+2?"]
          req1 = Provider.withMagicSystemPrompt $
                   buildRequest model configs msgs1

      resp1 <- getResponse req1

      case parseResponse @ClaudeSonnet45 model configs msgs1 resp1 of
        Right [AssistantText txt] -> do
          T.isInfixOf "4" txt `shouldBe` True

          -- Second exchange - append to history
          let msgs2 = msgs1 <> [AssistantText txt, UserText "What about 3+3?"]
              req2 = Provider.withMagicSystemPrompt $
                       buildRequest model configs msgs2

          resp2 <- getResponse req2

          case parseResponse @ClaudeSonnet45 model configs msgs2 resp2 of
            Right [AssistantText txt2] -> do
              T.isInfixOf "6" txt2 `shouldBe` True

              -- Verify request has full conversation history (alternating user/assistant)
              length (messages req2) `shouldBe` 3  -- user, assistant, user

        Right other -> expectationFailure $ "Expected [AssistantText], got: " ++ show other
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "applies system prompt from config" $ do
      let model = ClaudeSonnet45
          sysPrompt = "You are a mathematical assistant."
          configs = [MaxTokens 50, SystemPrompt sysPrompt]
          msgs = [UserText "Hello"]
          req = Provider.withMagicSystemPrompt $
                  buildRequest model configs msgs

      case system req of
        Just blocks -> do
          -- Should have both magic system prompt and user's system prompt
          length blocks `shouldSatisfy` (>= 2)
          -- Check that user's system prompt is included
          any (\b -> case b of
                       AnthropicSystemBlock txt _ -> txt == sysPrompt
                       _ -> False) blocks `shouldBe` True
        Nothing -> expectationFailure "Expected system prompt in request"

  describe "Anthropic Composable Provider - Tool Calling" $ do

    it "completes full tool calling conversation flow" $ do
      let model = ClaudeSonnet45
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the current weather in a given location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("The city and state, e.g. San Francisco, CA" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [MaxTokens 150, Tools [toolDef]]

          -- Step 1: Initial request with tools
          msgs1 = [UserText "What is the weather like in San Francisco?"]
          req1 = Provider.withMagicSystemPrompt $
                   buildRequest model configs msgs1

      -- Verify tools are in request
      case tools req1 of
        Just [tool] -> do
          anthropicToolName tool `shouldBe` "get_weather"
          anthropicToolDescription tool `shouldBe` "Get the current weather in a given location"
        _ -> expectationFailure "Expected exactly one tool in request"

      resp1 <- getResponse req1

      case parseResponse @ClaudeSonnet45 model configs msgs1 resp1 of
        Right [AssistantTool toolCall] -> do
          getToolCallName toolCall `shouldBe` "get_weather"

          -- Step 2: Execute tool (simulated) and send result
          let toolResult = ToolResult toolCall
                             (Right $ object
                               [ "temperature" .= ("72°F" :: Text)
                               , "conditions" .= ("sunny" :: Text)
                               ])
              msgs2 = msgs1 <> [AssistantTool toolCall, ToolResultMsg toolResult]
              req2 = Provider.withMagicSystemPrompt $
                       buildRequest model configs msgs2

          -- Verify history has all messages (grouped by direction)
          -- user -> assistant (tool_use) -> user (tool_result)
          length (messages req2) `shouldBe` 3

          resp2 <- getResponse req2

          case parseResponse @ClaudeSonnet45 model configs msgs2 resp2 of
            Right [AssistantText finalTxt] -> do
              -- Should incorporate the tool result
              (T.isInfixOf "72" finalTxt || T.isInfixOf "sunny" finalTxt) `shouldBe` True
            Right other -> expectationFailure $ "Expected [AssistantText], got: " ++ show other
            Left err -> expectationFailure $ "parseResponse failed: " ++ show err

        Right other -> expectationFailure $ "Expected [AssistantTool], got: " ++ show other
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "handles tool use with mixed text and tool blocks" $ do
      let model = ClaudeSonnet45
          toolDef = ToolDefinition "calculator" "Perform calculations"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 200, Tools [toolDef]]
          msgs = [UserText "What is 15 * 23?"]
          req = Provider.withMagicSystemPrompt $
                  buildRequest model configs msgs

      resp <- getResponse req

      case parseResponse @ClaudeSonnet45 model configs msgs resp of
        -- Could get just tool call, or text + tool call, or just text
        Right parsedMsgs -> do
          length parsedMsgs `shouldSatisfy` (> 0)
          -- At least one message should be present
          return ()
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

  describe "Anthropic Composable Provider - Message Grouping" $ do

    it "groups consecutive messages by direction (user/assistant)" $ do
      let model = ClaudeSonnet45
          configs = [MaxTokens 50]

          -- Multiple consecutive user messages
          msgs = [ UserText "First question"
                 , UserText "Second question"
                 , UserText "Third question"
                 ]
          req = Provider.withMagicSystemPrompt $
                  buildRequest model configs msgs

      -- Anthropic should group all consecutive user messages into one
      length (messages req) `shouldBe` 1
      case head (messages req) of
        AnthropicMessage "user" blocks -> do
          -- Should have 3 text blocks
          length blocks `shouldBe` 3
          all (\b -> case b of
                       AnthropicTextBlock _ -> True
                       _ -> False) blocks `shouldBe` True
        _ -> expectationFailure "Expected user message with multiple blocks"

    it "alternates user and assistant messages correctly" $ do
      let model = ClaudeSonnet45
          configs = [MaxTokens 50]
          toolCall = ToolCall "toolu_123" "test_tool" (object [])

          -- Alternating conversation
          msgs = [ UserText "Question 1"
                 , AssistantText "Answer 1"
                 , UserText "Question 2"
                 , AssistantTool toolCall
                 , ToolResultMsg (ToolResult toolCall (Right $ object []))
                 ]
          req = Provider.withMagicSystemPrompt $
                  buildRequest model configs msgs

      -- Should have: user, assistant, user, assistant (with tool_use), user (with tool_result)
      -- = 5 messages (tool call and tool result alternate directions)
      length (messages req) `shouldBe` 5

      case messages req of
        [m1, m2, m3, m4, m5] -> do
          role m1 `shouldBe` "user"       -- Question 1
          role m2 `shouldBe` "assistant"  -- Answer 1
          role m3 `shouldBe` "user"       -- Question 2
          role m4 `shouldBe` "assistant"  -- Tool call (assistant direction)
          role m5 `shouldBe` "user"       -- Tool result (user direction)
        _ -> expectationFailure "Expected 5 messages"

  describe "Compile-Time Safety Demonstrations" $ do

    it "allows tool use with tool-capable model" $ do
      let model = ClaudeSonnet45  -- HasTools
          toolDef = ToolDefinition "test_tool" "Test" (object [])
          configs = [Tools [toolDef], MaxTokens 50]
          msgs = [UserText "test"]
          req = buildRequest model configs msgs

      case tools req of
        Just [_] -> return ()
        _ -> expectationFailure "Expected tools in request"

    -- These tests demonstrate compile-time safety by being commented out
    -- Uncommenting them will cause compilation to fail

    {-
    it "SHOULD NOT COMPILE: tools without HasTools instance" $ do
      -- Define a model without HasTools
      data NoToolsModel = NoToolsModel deriving (Show, Eq)
      instance ModelName Anthropic NoToolsModel where
        modelName _ = "no-tools-model"

      let model = NoToolsModel  -- NO HasTools instance!
          toolDef = ToolDefinition "test_tool" "Test" (object [])
          -- This line will fail to compile:
          -- No instance for (HasTools NoToolsModel)
          configs = [Tools [toolDef], MaxTokens 50]
          msgs = [UserText "test"]
          req = buildRequest model configs msgs
      return ()
    -}

    {-
    it "SHOULD NOT COMPILE: AssistantTool message without HasTools" $ do
      data NoToolsModel = NoToolsModel deriving (Show, Eq)
      instance ModelName Anthropic NoToolsModel where
        modelName _ = "no-tools-model"

      let model = NoToolsModel  -- NO HasTools!
          toolCall = ToolCall "id" "name" (object [])
          -- This line will fail to compile:
          -- No instance for (HasTools NoToolsModel)
          msgs = [AssistantTool toolCall]
          req = buildRequest model configs msgs
      return ()
    -}
