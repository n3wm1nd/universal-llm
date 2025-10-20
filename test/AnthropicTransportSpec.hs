{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-x-partial -Wno-unused-imports -Wno-name-shadowing #-}

module AnthropicTransportSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as Aeson.KeyMap
import Autodocodec (toJSONViaCodec, eitherDecodeJSONViaCodec)
import UniversalLLM.Protocols.Anthropic
import UniversalLLM.Core.Types
import qualified UniversalLLM.Providers.Anthropic as Provider
import TestModels (ClaudeSonnet45(..))

spec :: Spec
spec = do
  describe "Anthropic OAuth Transport" $ do

    it "translates user message to correct Anthropic request via toRequest" $ do
      let testModel = ClaudeSonnet45
      let testMessages = [UserText "What is 2+2?"]
      let testConfigs = [MaxTokens 50, SystemPrompt "You are Claude Code, Anthropic's official CLI for Claude."]

      let request = Provider.toRequest' Provider.Anthropic testModel testConfigs testMessages

      model request `shouldBe` "claude-sonnet-4-5-20250929"
      length (messages request) `shouldBe` 1
      case system request of
        Just [AnthropicSystemBlock txt _] -> txt `shouldBe` "You are Claude Code, Anthropic's official CLI for Claude."
        Just _ -> expectationFailure "Expected single system block"
        Nothing -> expectationFailure "System prompt should be present"
      max_tokens request `shouldBe` 50

    it "parses text response and converts to AssistantText message" $ do
      -- Real response from: curl with OAuth token and magic system prompt
      -- Command: curl -X POST https://api.anthropic.com/v1/messages \
      --   -H "Authorization: Bearer sk-ant-oat01-..." \
      --   -H "anthropic-version: 2023-06-01" \
      --   -H "anthropic-beta: oauth-2025-04-20" \
      --   -H "User-Agent: hs-universal-llm (prerelease-dev)" \
      --   -d '{"model":"claude-sonnet-4-5-20250929","max_tokens":50,
      --        "system":"You are Claude Code, Anthropic'\''s official CLI for Claude.",
      --        "messages":[{"role":"user","content":"What is 2+2?"}]}'

      let realResponse = "{\
        \\"model\":\"claude-sonnet-4-5-20250929\",\
        \\"id\":\"msg_01URmAjiM8jrZagrPhD6Kk1U\",\
        \\"type\":\"message\",\
        \\"role\":\"assistant\",\
        \\"content\":[{\"type\":\"text\",\"text\":\"2 + 2 = 4\"}],\
        \\"stop_reason\":\"end_turn\",\
        \\"stop_sequence\":null,\
        \\"usage\":{\
        \  \"input_tokens\":28,\
        \  \"cache_creation_input_tokens\":0,\
        \  \"cache_read_input_tokens\":0,\
        \  \"output_tokens\":13\
        \}\
        \}"

      case eitherDecodeJSONViaCodec realResponse of
        Right response -> do
          -- Test that we can parse the response
          case response of
            AnthropicSuccess resp -> do
              responseId resp `shouldBe` "msg_01URmAjiM8jrZagrPhD6Kk1U"
              responseStopReason resp `shouldBe` Just "end_turn"
              length (responseContent resp) `shouldBe` 1

          -- Test that fromResponse correctly extracts the text
          case Provider.fromResponse' @ClaudeSonnet45 response of
            Right [AssistantText txt] -> txt `shouldBe` "2 + 2 = 4"
            Right _ -> expectationFailure "Expected single AssistantText message"
            Left err -> expectationFailure $ "fromResponse failed: " <> show err

        Left parseErr ->
          expectationFailure $ "Failed to parse response: " <> parseErr

    it "parses tool_use content blocks and converts to AssistantTool" $ do
      -- Real response from Anthropic API with tool_use content block
      -- Command: curl -X POST https://api.anthropic.com/v1/messages \
      --   -H "Authorization: Bearer sk-ant-oat01-..." \
      --   -H "anthropic-version: 2023-06-01" \
      --   -H "anthropic-beta: oauth-2025-04-20" \
      --   -H "User-Agent: hs-universal-llm (prerelease-dev)" \
      --   -d '{"model":"claude-sonnet-4-5-20250929","max_tokens":100,
      --        "system":"You are Claude Code, Anthropic's official CLI for Claude.",
      --        "messages":[{"role":"user","content":"What is the weather like in San Francisco?"}],
      --        "tools":[{"name":"get_weather","description":"Get the current weather in a given location",
      --                  "input_schema":{"type":"object","properties":{"location":{"type":"string",
      --                  "description":"The city and state, e.g. San Francisco, CA"}},"required":["location"]}}]}'

      let realToolResponse = "{\
        \\"model\":\"claude-sonnet-4-5-20250929\",\
        \\"id\":\"msg_01WgoVbY7fppryHFtZRwqs7E\",\
        \\"type\":\"message\",\
        \\"role\":\"assistant\",\
        \\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_01MiAg7qRTuJAYUjNBDBN9AS\",\"name\":\"get_weather\",\"input\":{\"location\":\"San Francisco, CA\"}}],\
        \\"stop_reason\":\"tool_use\",\
        \\"stop_sequence\":null,\
        \\"usage\":{\
        \  \"input_tokens\":601,\
        \  \"cache_creation_input_tokens\":0,\
        \  \"cache_read_input_tokens\":0,\
        \  \"cache_creation\":{\
        \    \"ephemeral_5m_input_tokens\":0,\
        \    \"ephemeral_1h_input_tokens\":0\
        \  },\
        \  \"output_tokens\":56,\
        \  \"service_tier\":\"standard\"\
        \}\
        \}"

      case eitherDecodeJSONViaCodec realToolResponse of
        Right response -> do
          -- Verify the tool_use content block is parsed correctly
          case response of
            AnthropicSuccess resp -> do
              responseId resp `shouldBe` "msg_01WgoVbY7fppryHFtZRwqs7E"
              responseStopReason resp `shouldBe` Just "tool_use"
              case responseContent resp of
                [AnthropicToolUseBlock toolId toolName toolInput] -> do
                  toolId `shouldBe` "toolu_01MiAg7qRTuJAYUjNBDBN9AS"
                  toolName `shouldBe` "get_weather"
                  case toolInput of
                    Aeson.Object obj -> do
                      case Aeson.KeyMap.lookup "location" obj of
                        Just (Aeson.String loc) -> loc `shouldBe` "San Francisco, CA"
                        _ -> expectationFailure "Expected location field in tool input"
                    _ -> expectationFailure "Expected tool input to be an object"
                _ -> expectationFailure "Expected single tool_use block"
            AnthropicError _ -> expectationFailure "Expected success response"

          -- Test that fromResponse correctly converts tool_use to AssistantTool
          case Provider.fromResponse' @ClaudeSonnet45 response of
            Right [AssistantTool (ToolCall tcId tcName _tcParams)] -> do
              tcId `shouldBe` "toolu_01MiAg7qRTuJAYUjNBDBN9AS"
              tcName `shouldBe` "get_weather"
            Right _ -> expectationFailure "Expected AssistantTool message with single ToolCall"
            Left err -> expectationFailure $ "fromResponse failed: " <> show err

        Left parseErr ->
          expectationFailure $ "Failed to parse response: " <> parseErr

    it "parses response after tool result and extracts via fromResponse" $ do
      -- Real response from: curl with tool result continuation
      -- Command: curl -X POST https://api.anthropic.com/v1/messages \
      --   -H "Authorization: Bearer sk-ant-oat01-..." \
      --   -H "anthropic-version: 2023-06-01" \
      --   -H "anthropic-beta: oauth-2025-04-20" \
      --   -H "User-Agent: hs-universal-llm (prerelease-dev)" \
      --   -d '{"model":"claude-sonnet-4-5-20250929","max_tokens":100,
      --        "system":"You are Claude Code, Anthropic's official CLI for Claude.",
      --        "messages":[
      --          {"role":"user","content":"What is the weather like in San Francisco?"},
      --          {"role":"assistant","content":[{"type":"tool_use","id":"toolu_01MiAg7qRTuJAYUjNBDBN9AS","name":"get_weather","input":{"location":"San Francisco, CA"}}]},
      --          {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01MiAg7qRTuJAYUjNBDBN9AS","content":"72°F, sunny with light fog in the morning"}]}
      --        ],
      --        "tools":[{"name":"get_weather","description":"Get the current weather in a given location",
      --                  "input_schema":{"type":"object","properties":{"location":{"type":"string",
      --                  "description":"The city and state, e.g. San Francisco, CA"}},"required":["location"]}}]}'

      let realToolResultResponseText = "{\
        \\"model\":\"claude-sonnet-4-5-20250929\",\
        \\"id\":\"msg_01JqZuDBAtXCijpFoT6t7cW9\",\
        \\"type\":\"message\",\
        \\"role\":\"assistant\",\
        \\"content\":[{\"type\":\"text\",\"text\":\"The weather in San Francisco is currently 72°F and sunny with light fog in the morning. Typical San Francisco weather!\"}],\
        \\"stop_reason\":\"end_turn\",\
        \\"stop_sequence\":null,\
        \\"usage\":{\
        \  \"input_tokens\":681,\
        \  \"cache_creation_input_tokens\":0,\
        \  \"cache_read_input_tokens\":0,\
        \  \"cache_creation\":{\
        \    \"ephemeral_5m_input_tokens\":0,\
        \    \"ephemeral_1h_input_tokens\":0\
        \  },\
        \  \"output_tokens\":29,\
        \  \"service_tier\":\"standard\"\
        \}\
        \}" :: Text
          realToolResultResponse = BSL.fromStrict $ TE.encodeUtf8 realToolResultResponseText

      case eitherDecodeJSONViaCodec realToolResultResponse of
        Right response -> do
          case response of
            AnthropicSuccess resp -> do
              responseId resp `shouldBe` "msg_01JqZuDBAtXCijpFoT6t7cW9"
              responseStopReason resp `shouldBe` Just "end_turn"

          -- Test that fromResponse extracts the text correctly
          case Provider.fromResponse' @ClaudeSonnet45 response of
            Right [AssistantText txt] ->
              T.isInfixOf "72°F" txt `shouldBe` True
            Right _ -> expectationFailure "Expected single AssistantText message"
            Left err -> expectationFailure $ "fromResponse failed: " <> show err

        Left parseErr ->
          expectationFailure $ "Failed to parse response: " <> parseErr

    it "simulates full tool calling conversation flow" $ do
      let testModel = ClaudeSonnet45

      -- Step 1: Create initial request with tools
      let toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the current weather in a given location"
            , toolDefParameters = Aeson.object
                [ ("type", Aeson.String "object")
                , ("properties", Aeson.object
                    [ ("location", Aeson.object
                        [ ("type", Aeson.String "string")
                        , ("description", Aeson.String "The city and state, e.g. San Francisco, CA")
                        ])
                    ])
                , ("required", Aeson.toJSON ["location" :: Text])
                ]
            }
      let initialMessages = [UserText "What is the weather like in San Francisco?"]
      let configs = [MaxTokens 100, Tools [toolDef]]
      let initialRequest = Provider.toRequest' Provider.Anthropic testModel configs initialMessages

      -- Verify initial request has tools
      case tools initialRequest of
        Just [anthropicTool] -> do
          anthropicToolName anthropicTool `shouldBe` "get_weather"
          anthropicToolDescription anthropicTool `shouldBe` "Get the current weather in a given location"
        _ -> expectationFailure "Expected exactly one tool in request"

      -- Step 2: Simulate LLM response with tool_use (from real API response)
      let toolUseResponse = "{\
        \\"model\":\"claude-sonnet-4-5-20250929\",\
        \\"id\":\"msg_01WgoVbY7fppryHFtZRwqs7E\",\
        \\"type\":\"message\",\
        \\"role\":\"assistant\",\
        \\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_01MiAg7qRTuJAYUjNBDBN9AS\",\"name\":\"get_weather\",\"input\":{\"location\":\"San Francisco, CA\"}}],\
        \\"stop_reason\":\"tool_use\",\
        \\"stop_sequence\":null,\
        \\"usage\":{\
        \  \"input_tokens\":601,\
        \  \"cache_creation_input_tokens\":0,\
        \  \"cache_read_input_tokens\":0,\
        \  \"cache_creation\":{\
        \    \"ephemeral_5m_input_tokens\":0,\
        \    \"ephemeral_1h_input_tokens\":0\
        \  },\
        \  \"output_tokens\":56,\
        \  \"service_tier\":\"standard\"\
        \}\
        \}"

      -- Decode response and verify we get AssistantTool message
      case eitherDecodeJSONViaCodec toolUseResponse of
        Right response -> do
          case Provider.fromResponse' @ClaudeSonnet45 response of
            Right [AssistantTool toolCall] -> do
              getToolCallId toolCall `shouldBe` "toolu_01MiAg7qRTuJAYUjNBDBN9AS"
              getToolCallName toolCall `shouldBe` "get_weather"

              -- Step 3: Simulate tool execution
              let toolResult = ToolResult
                    { toolResultCall = toolCall
                    , toolResultOutput = Right $ Aeson.object
                        [ ("temperature", Aeson.String "72°F")
                        , ("conditions", Aeson.String "sunny")
                        ]
                    }

              -- Step 4: Build history and create follow-up request
              let history = [ UserText "What is the weather like in San Francisco?"
                            , AssistantTool toolCall
                            , ToolResultMsg toolResult
                            ]
              let followUpRequest = Provider.toRequest' Provider.Anthropic testModel [MaxTokens 100] history

              -- Verify follow-up request has all messages in correct format
              length (messages followUpRequest) `shouldBe` 3

              case messages followUpRequest of
                [userMsg, assistantMsg, toolResultMsg] -> do
                  -- Check user message
                  role userMsg `shouldBe` "user"
                  case content userMsg of
                    [AnthropicTextBlock txt] -> txt `shouldBe` "What is the weather like in San Francisco?"
                    _ -> expectationFailure "Expected text block in user message"

                  -- Check assistant message has tool_use block
                  role assistantMsg `shouldBe` "assistant"
                  case content assistantMsg of
                    [AnthropicToolUseBlock tid tname tinput] -> do
                      tid `shouldBe` "toolu_01MiAg7qRTuJAYUjNBDBN9AS"
                      tname `shouldBe` "get_weather"
                      case Aeson.KeyMap.lookup "location" (case tinput of Aeson.Object obj -> obj; _ -> Aeson.KeyMap.empty) of
                        Just (Aeson.String loc) -> loc `shouldBe` "San Francisco, CA"
                        _ -> expectationFailure "Expected location in tool input"
                    _ -> expectationFailure "Expected single tool_use block in assistant message"

                  -- Check tool result message
                  role toolResultMsg `shouldBe` "user"
                  case content toolResultMsg of
                    [AnthropicToolResultBlock rid _rcontent] ->
                      rid `shouldBe` "toolu_01MiAg7qRTuJAYUjNBDBN9AS"
                    _ -> expectationFailure "Expected tool_result block"

                _ -> expectationFailure "Expected exactly 3 messages in history"

            Right _ -> expectationFailure "Expected AssistantTool with one tool call"
            Left err -> expectationFailure $ "fromResponse failed: " <> show err

        Left parseErr -> expectationFailure $ "Failed to parse tool_use response: " <> parseErr

    it "handles OAuth authentication error" $ do
      -- Real error response from curl without magic system prompt:
      let realError = "{\
        \  \"type\": \"error\",\
        \  \"error\": {\
        \    \"type\": \"invalid_request_error\",\
        \    \"message\": \"This credential is only authorized for use with Claude Code and cannot be used for other API requests.\"\
        \  },\
        \  \"request_id\": \"req_011CUGqTJ1GG8CX2KFgCGSv1\"\
        \}"

      case eitherDecodeJSONViaCodec realError of
        Right (AnthropicError err) -> do
          errorType err `shouldBe` "invalid_request_error"
          T.isInfixOf "Claude Code" (errorMessage err) `shouldBe` True
        Right (AnthropicSuccess _) -> expectationFailure "Expected error response"
        Left parseErr -> expectationFailure $ "Parse error: " ++ parseErr

