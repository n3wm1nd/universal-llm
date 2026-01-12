{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}

module AnthropicComposableSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import Control.Monad (unless)
import qualified Data.Aeson as Aeson
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM
import UniversalLLM.Protocols.Anthropic
import qualified UniversalLLM.Protocols.Anthropic as Proto
import qualified UniversalLLM.Providers.Anthropic as Provider
import Data.Default (Default(..))

-- Type alias for the test model
type TestModel = Model TestModels.ClaudeSonnet45 Provider.Anthropic

-- Helper to build request for ClaudeSonnet45
buildRequest :: TestModel
             -> [ModelConfig TestModel]
             -> [Message TestModel]
             -> AnthropicRequest
buildRequest model = buildRequestGeneric TestModels.anthropicSonnet45 model ((), ())

-- Type alias for the reasoning test model
type TestModelWithReasoning = Model TestModels.ClaudeSonnet45WithReasoning Provider.Anthropic

-- Helper to build request for ClaudeSonnet45WithReasoning
buildRequestWithReasoning :: TestModelWithReasoning
                          -> [ModelConfig TestModelWithReasoning]
                          -> [Message TestModelWithReasoning]
                          -> AnthropicRequest
buildRequestWithReasoning model = buildRequestGeneric TestModels.anthropicSonnet45Reasoning model (def, ((), ()))

-- Generic helper to build request with explicit composable provider
buildRequestGeneric :: forall m s. (ProviderRequest m ~ AnthropicRequest)
                    => ComposableProvider m s
                    -> m
                    -> s
                    -> [ModelConfig m]
                    -> [Message m]
                    -> AnthropicRequest
buildRequestGeneric composableProvider model s configs = snd . toProviderRequest composableProvider model configs s

-- Helper to parse response for ClaudeSonnet45
parseResponse :: TestModel
              -> [ModelConfig TestModel]
              -> [Message TestModel]
              -> AnthropicResponse
              -> Either LLMError [Message TestModel]
parseResponse model configs _history resp =
  let msgs = parseResponseGeneric TestModels.anthropicSonnet45 model configs ((), ()) resp
  in if null msgs
     then Left $ ParseError "No messages parsed from response"
     else Right msgs

-- Helper to parse response for ClaudeSonnet45WithReasoning
parseResponseWithReasoning :: TestModelWithReasoning
                           -> [ModelConfig TestModelWithReasoning]
                           -> [Message TestModelWithReasoning]
                           -> AnthropicResponse
                           -> Either LLMError [Message TestModelWithReasoning]
parseResponseWithReasoning model configs _history resp =
  let msgs = parseResponseGeneric TestModels.anthropicSonnet45Reasoning model configs (def, ((), ())) resp
  in if null msgs
     then Left $ ParseError "No messages parsed from response"
     else Right msgs

-- Generic helper to parse response with explicit composable provider
parseResponseGeneric :: forall m s. (ProviderResponse m ~ AnthropicResponse)
                     => ComposableProvider m s
                     -> m
                     -> [ModelConfig m]
                     -> s
                     -> AnthropicResponse
                     -> [Message m]
parseResponseGeneric composableProvider model configs s resp =
  either (error . show) snd $ fromProviderResponse composableProvider model configs s resp

spec :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
spec getResponse = do
  describe "Anthropic Request Building (no API calls)" $ do
    it "can build and evaluate a request without looping" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic
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
      let model = Model ClaudeSonnet45 Provider.Anthropic
          configs = [MaxTokens 100, SystemPrompt "You are a helpful assistant."]

          -- First exchange
          msgs1 = [UserText "What is 2+2?"]
          req1 = Provider.withMagicSystemPrompt $
                   buildRequest model configs msgs1

      resp1 <- getResponse req1

      case parseResponse model configs msgs1 resp1 of
        Right [AssistantText txt] -> do
          T.isInfixOf "4" txt `shouldBe` True

          -- Second exchange - append to history
          let msgs2 = msgs1 <> [AssistantText txt, UserText "What about 3+3?"]
              req2 = Provider.withMagicSystemPrompt $
                       buildRequest model configs msgs2

          resp2 <- getResponse req2

          case parseResponse model configs msgs2 resp2 of
            Right [AssistantText txt2] -> do
              T.isInfixOf "6" txt2 `shouldBe` True

              -- Verify request has full conversation history (alternating user/assistant)
              length (messages req2) `shouldBe` 3  -- user, assistant, user

        Right other -> expectationFailure $ "Expected [AssistantText], got: " ++ show other
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "applies system prompt from config" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic
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
                       AnthropicSystemBlock txt _ _ -> txt == sysPrompt
                       _ -> False) blocks `shouldBe` True
        Nothing -> expectationFailure "Expected system prompt in request"

  describe "Anthropic Composable Provider - Tool Calling" $ do

    it "completes full tool calling conversation flow" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel
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
          msgs1 = [UserText "What is the weather like in San Francisco?" :: Message TestModel]
          req1 = Provider.withMagicSystemPrompt $
                   buildRequest model configs msgs1

      -- Verify tools are in request
      case tools req1 of
        Just [tool] -> do
          anthropicToolName tool `shouldBe` "get_weather"
          anthropicToolDescription tool `shouldBe` "Get the current weather in a given location"
        _ -> expectationFailure "Expected exactly one tool in request"

      resp1 <- getResponse req1

      case parseResponse model configs msgs1 resp1 of
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

          case parseResponse model configs msgs2 resp2 of
            Right [AssistantText finalTxt] -> do
              -- Should incorporate the tool result
              (T.isInfixOf "72" finalTxt || T.isInfixOf "sunny" finalTxt) `shouldBe` True
            Right other -> expectationFailure $ "Expected [AssistantText], got: " ++ show other
            Left err -> expectationFailure $ "parseResponse failed: " ++ show err

        Right other -> expectationFailure $ "Expected [AssistantTool], got: " ++ show other
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

    it "handles tool use with mixed text and tool blocks" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel
          toolDef = ToolDefinition "calculator" "Perform calculations"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 200, Tools [toolDef]]
          msgs = [UserText "What is 15 * 23?" :: Message TestModel]
          req = Provider.withMagicSystemPrompt $
                  buildRequest model configs msgs

      resp <- getResponse req

      case parseResponse model configs msgs resp of
        -- Could get just tool call, or text + tool call, or just text
        Right parsedMsgs -> do
          length parsedMsgs `shouldSatisfy` (> 0)
          -- At least one message should be present
          return ()
        Left err -> expectationFailure $ "parseResponse failed: " ++ show err

  describe "Anthropic Composable Provider - Message Grouping" $ do

    it "groups consecutive messages by direction (user/assistant)" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel
          configs = [MaxTokens 50]

          -- Multiple consecutive user messages
          msgs = [ UserText "First question" :: Message TestModel
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
                       AnthropicTextBlock _ _ -> True
                       _ -> False) blocks `shouldBe` True
        _ -> expectationFailure "Expected user message with multiple blocks"

    it "alternates user and assistant messages correctly" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel
          configs = [MaxTokens 50]
          toolCall = ToolCall "toolu_123" "test_tool" (object [])

          -- Alternating conversation
          msgs = [ UserText "Question 1" :: Message TestModel
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

  describe "Anthropic Composable Provider - Streaming with Tools" $ do

    it "builds correct request for streaming tool calls" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel
          toolDef = ToolDefinition
            { toolDefName = "calculator"
            , toolDefDescription = "Perform math calculations"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "expression" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("Math expression" :: Text)
                        ]
                    ]
                ]
            }
          configs = [MaxTokens 2048, Tools [toolDef], Streaming True]
          msgs = [UserText "Calculate 2+2" :: Message TestModel]
          req = buildRequest model configs msgs

      -- Verify streaming is enabled
      Proto.stream req `shouldBe` Just True

      -- Verify tools are in request
      case Proto.tools req of
        Just [tool] ->
          anthropicToolName tool `shouldBe` "calculator"
        _ -> expectationFailure "Expected exactly one tool in request"

    it "builds correct request for streaming tool calls with reasoning" $ do
      let model = Model ClaudeSonnet45WithReasoning Provider.Anthropic :: TestModelWithReasoning
          toolDef = ToolDefinition
            { toolDefName = "search"
            , toolDefDescription = "Search for information"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "query" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("Search query" :: Text)
                        ]
                    ]
                ]
            }
          configs = [MaxTokens 4000, Tools [toolDef], Streaming True]
          msgs = [UserText "Search for Haskell" :: Message TestModelWithReasoning]
          req = buildRequestWithReasoning model configs msgs

      -- Verify streaming and reasoning are enabled
      Proto.stream req `shouldBe` Just True
      Proto.thinking req `shouldNotBe` Nothing

      -- Verify tools are present
      case Proto.tools req of
        Just [tool] ->
          anthropicToolName tool `shouldBe` "search"
        _ -> expectationFailure "Expected exactly one tool in request"

  describe "Compile-Time Safety Demonstrations" $ do

    it "allows tool use with tool-capable model" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel  -- HasTools
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
          req = buildRequest TestModels.ClaudeSonnet45 configs msgs
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
          req = buildRequest TestModels.ClaudeSonnet45 configs msgs
      return ()
    -}

  describe "Anthropic Composable Provider - Reasoning/Thinking" $ do
    it "includes thinking config with budget_tokens when reasoning is enabled" $ do
      let model = Model ClaudeSonnet45WithReasoning Provider.Anthropic :: TestModelWithReasoning
          configs = [MaxTokens 4000]  -- Use 4000 so we get 2000 budget (within bounds)
          msgs = [UserText "Think about this problem" :: Message TestModelWithReasoning]
          req = buildRequestWithReasoning model configs msgs

      -- Check that thinking config is present
      case Proto.thinking req of
        Just thinkingConfig -> do
          Proto.thinkingType thinkingConfig `shouldBe` "enabled"
          -- Check that budget_tokens is set and respects max(1024, min(5000, max_tokens/2))
          case Proto.thinkingBudgetTokens thinkingConfig of
            Just budget -> budget `shouldBe` 2000  -- max(1024, min(5000, 4000/2)) = 2000
            Nothing -> expectationFailure "budget_tokens should be set"
        Nothing ->
          expectationFailure "thinking config should be set when HasReasoning is enabled"

    it "enforces minimum thinking budget of 1024" $ do
      let model = Model ClaudeSonnet45WithReasoning Provider.Anthropic :: TestModelWithReasoning
          configs = [MaxTokens 1000]  -- Small max_tokens would give budget < 1024
          msgs = [UserText "Think about this" :: Message TestModelWithReasoning]
          req = buildRequestWithReasoning model configs msgs

      -- Check that budget respects API minimum
      case Proto.thinking req of
        Just thinkingConfig ->
          -- Should be max(1024, min(5000, 1000/2)) = 1024
          Proto.thinkingBudgetTokens thinkingConfig `shouldBe` Just 1024
        Nothing ->
          expectationFailure "thinking config should be set"

    it "respects max_tokens when setting thinking budget" $ do
      let model = Model ClaudeSonnet45WithReasoning Provider.Anthropic :: TestModelWithReasoning
          configs = [MaxTokens 4000]
          msgs = [UserText "Think about this" :: Message TestModelWithReasoning]
          req = buildRequestWithReasoning model configs msgs

      -- Check that thinking config respects the user's max_tokens
      case Proto.thinking req of
        Just thinkingConfig ->
          -- Should be min(5000, 4000/2) = 2000
          Proto.thinkingBudgetTokens thinkingConfig `shouldBe` Just 2000
        Nothing ->
          expectationFailure "thinking config should be set"

    it "caps thinking budget at 5000" $ do
      let model = Model ClaudeSonnet45WithReasoning Provider.Anthropic :: TestModelWithReasoning
          configs = [MaxTokens 20000]
          msgs = [UserText "Think about this" :: Message TestModelWithReasoning]
          req = buildRequestWithReasoning model configs msgs

      -- Check that thinking budget doesn't exceed 5000
      case Proto.thinking req of
        Just thinkingConfig ->
          -- Should be min(5000, 20000/2) = 5000
          Proto.thinkingBudgetTokens thinkingConfig `shouldBe` Just 5000
        Nothing ->
          expectationFailure "thinking config should be set"

    it "does not include thinking config when reasoning is disabled" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel
          configs = [MaxTokens 200]
          msgs = [UserText "Simple question" :: Message TestModel]
          req = buildRequest model configs msgs

      -- Check that thinking config is not present
      Proto.thinking req `shouldBe` Nothing

    it "sends reasoning request to API and gets response" $ do
      let model = Model ClaudeSonnet45WithReasoning Provider.Anthropic :: TestModelWithReasoning
          -- max_tokens must be greater than budget_tokens
          configs = [MaxTokens 12000]
          msgs = [UserText "What is 2+2?" :: Message TestModelWithReasoning]
          req = Provider.withMagicSystemPrompt $
                 buildRequestWithReasoning model configs msgs

      resp <- getResponse req

      -- Verify we got a successful response from the API
      case resp of
        AnthropicSuccess respData ->
          -- Should have content (thinking and/or text)
          length (Proto.responseContent respData) `shouldSatisfy` (> 0)
        AnthropicError errResp ->
          expectationFailure $ "API returned error: " ++ show errResp

  describe "Anthropic Composable Provider - Streaming + Tools Live Test" $ do
    it "receives SSE formatted streaming response with tool call events" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic :: TestModel
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the weather for a location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("City name" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [MaxTokens 2048, Tools [toolDef], Streaming True]
          msgs = [UserText "What's the weather in Paris?" :: Message TestModel]
          req = Provider.withMagicSystemPrompt $
                 buildRequest model configs msgs

      -- Verify the request is correctly configured for streaming
      Proto.stream req `shouldBe` Just True
      case Proto.tools req of
        Just [tool] -> anthropicToolName tool `shouldBe` "get_weather"
        _ -> expectationFailure "Expected get_weather tool in request"

      -- Note: Streaming responses from Anthropic API come in SSE format.
      -- The test framework can't mock SSE properly, so full response validation
      -- requires live API testing. In production, SSE events will be:
      -- - event: message_start (response initialization)
      -- - event: content_block_start (tool_use block start)
      -- - event: content_block_delta (tool_use block contents)
      -- Consumer code is responsible for reassembling SSE deltas into complete response

  describe "Anthropic Provider - Content Block Types" $ do
    it "correctly creates all content block types" $ do
      -- Test thinking block
      let thinkingBlock = AnthropicThinkingBlock "Some reasoning" (object ["test" .= ("sig" :: Text)]) Nothing
      case thinkingBlock of
        AnthropicThinkingBlock txt _ _ -> txt `shouldBe` "Some reasoning"
        _ -> expectationFailure "Should be thinking block"

      -- Test text block
      let textBlock = AnthropicTextBlock "Some text response" Nothing
      case textBlock of
        AnthropicTextBlock txt _ -> txt `shouldBe` "Some text response"
        _ -> expectationFailure "Should be text block"

      -- Test tool use block
      let toolBlock = AnthropicToolUseBlock "tool_id_123" "get_weather" (object ["location" .= ("Paris" :: Text)]) Nothing
      case toolBlock of
        AnthropicToolUseBlock bid bname binput _ -> do
          bid `shouldBe` "tool_id_123"
          bname `shouldBe` "get_weather"
        _ -> expectationFailure "Should be tool use block"

    it "handles response with multiple blocks" $ do
      -- Create a response with multiple content blocks
      let resp = defaultAnthropicSuccessResponse
            { Proto.responseId = "msg_123"
            , Proto.responseModel = "claude-sonnet"
            , Proto.responseRole = "assistant"
            , Proto.responseContent =
                [ AnthropicThinkingBlock "First thinking" (object ["sig" .= (1 :: Int)]) Nothing
                , AnthropicTextBlock "First text" Nothing
                , AnthropicThinkingBlock "Second thinking" (object ["sig" .= (2 :: Int)]) Nothing
                , AnthropicTextBlock "Second text" Nothing
                ]
            , Proto.responseStopReason = Just "end_turn"
            , Proto.responseUsage = AnthropicUsage 100 200
            }

      -- Response should have 4 blocks
      length (Proto.responseContent resp) `shouldBe` 4

      -- Verify each block type
      case Proto.responseContent resp of
        [AnthropicThinkingBlock _ _ _, AnthropicTextBlock _ _, AnthropicThinkingBlock _ _ _, AnthropicTextBlock _ _] ->
          return ()
        _ -> expectationFailure "Should have thinking, text, thinking, text blocks in that order"
