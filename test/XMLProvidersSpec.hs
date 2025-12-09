{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module XMLProvidersSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (Value, object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KeyMap
import UniversalLLM.Core.Types
import UniversalLLM.ToolCall.XML
import UniversalLLM.Providers.XMLToolCalls
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Protocols.OpenAI
import TestModels (GLM45)

-- ============================================================================
-- Test Helpers
-- ============================================================================

-- Mock model that uses XML response parsing (Strategy A)
data MockXMLResponseModel = MockXMLResponseModel deriving (Show, Eq)

instance ModelName (MockXMLResponseModel `Via` OpenAI.LlamaCpp) where
  modelName _ = "mock-xml-response"

instance HasTools (MockXMLResponseModel `Via` OpenAI.LlamaCpp) where
  withTools = OpenAI.openAITools

-- Mock model that uses full XML support (Strategy B)
data MockFullXMLModel = MockFullXMLModel deriving (Show, Eq)

instance ModelName (MockFullXMLModel `Via` OpenAI.OpenAICompatible) where
  modelName _ = "mock-full-xml"

instance HasTools (MockFullXMLModel `Via` OpenAI.OpenAICompatible) where
  withTools = OpenAI.openAITools

-- Helper to build a request from messages (with explicit composable provider)
buildRequestGeneric :: (Monoid (ProviderRequest model))
                   => ComposableProvider model s
                   -> model
                   -> [ModelConfig model]
                   -> s
                   -> [Message model]
                   -> ProviderRequest model
buildRequestGeneric composableProvider model configs s = snd . toProviderRequest composableProvider model configs s

-- TypeClass to support buildRequest with implicit composable provider/state
class BuildRequest model where
  buildRequest :: model
               -> [ModelConfig model]
               -> [Message model]
               -> ProviderRequest model

-- Instance for MockXMLResponseModel
instance BuildRequest (MockXMLResponseModel `Via` OpenAI.LlamaCpp) where
  buildRequest model configs msgs =
    let base = OpenAI.baseComposableProvider @(MockXMLResponseModel `Via` OpenAI.LlamaCpp)
        withToolsProvider = chainProviders OpenAI.openAITools base
        composableProvider = withXMLResponseParsing withToolsProvider
    in snd $ toProviderRequest composableProvider model configs ((), ((), ())) msgs

-- Instance for MockFullXMLModel
instance BuildRequest (MockFullXMLModel `Via` OpenAI.OpenAICompatible) where
  buildRequest model configs msgs =
    let base = OpenAI.baseComposableProvider @(MockFullXMLModel `Via` OpenAI.OpenAICompatible)
        withToolsProvider = chainProviders OpenAI.openAITools base
        composableProvider = withFullXMLToolSupport withToolsProvider
    in snd $ toProviderRequest composableProvider model configs ((), ((), ())) msgs

-- Helper to parse a response (with explicit composable provider)
parseResponseGeneric :: ComposableProvider model s
                    -> model
                    -> [ModelConfig model]
                    -> s
                    -> [Message model]  -- history
                    -> ProviderResponse model
                    -> [Message model]
parseResponseGeneric composableProvider model configs s history resp =
  let msgs = either (error . show) snd $ fromProviderResponse composableProvider model configs s resp
  in msgs

-- TypeClass to support parseResponse with implicit composable provider/state
class ParseResponse model where
  parseResponse :: model
                -> [ModelConfig model]
                -> [Message model]  -- history
                -> ProviderResponse model
                -> [Message model]

-- Instance for MockXMLResponseModel
instance ParseResponse (MockXMLResponseModel `Via` OpenAI.LlamaCpp) where
  parseResponse model configs history resp =
    let base = OpenAI.baseComposableProvider @(MockXMLResponseModel `Via` OpenAI.LlamaCpp)
        withToolsProvider = chainProviders OpenAI.openAITools base
        composableProvider = withXMLResponseParsing withToolsProvider
    in either (error . show) snd $ fromProviderResponse composableProvider model configs ((), ((), ())) resp

-- Instance for MockFullXMLModel
instance ParseResponse (MockFullXMLModel `Via` OpenAI.OpenAICompatible) where
  parseResponse model configs history resp =
    let base = OpenAI.baseComposableProvider @(MockFullXMLModel `Via` OpenAI.OpenAICompatible)
        withToolsProvider = chainProviders OpenAI.openAITools base
        composableProvider = withFullXMLToolSupport withToolsProvider
    in either (error . show) snd $ fromProviderResponse composableProvider model configs ((), ((), ())) resp

-- Create a mock OpenAI response with text content
mockTextResponse :: Text -> OpenAIResponse
mockTextResponse txt = OpenAISuccess $ defaultOpenAISuccessResponse
  { choices = [defaultOpenAIChoice
      { message = defaultOpenAIMessage
          { role = "assistant"
          , content = Just txt
          }
      }]
  }

-- Create a mock OpenAI response with tool calls
mockToolCallResponse :: [(Text, Text, Value)] -> OpenAIResponse
mockToolCallResponse calls = OpenAISuccess $ defaultOpenAISuccessResponse
  { choices = [defaultOpenAIChoice
      { message = defaultOpenAIMessage
          { role = "assistant"
          , tool_calls = Just openAICalls
          }
      }]
  }
  where
    openAICalls = [defaultOpenAIToolCall
                    { callId = tcId
                    , toolCallType = "function"
                    , toolFunction = defaultOpenAIToolFunction
                        { toolFunctionName = tcName
                        , toolFunctionArguments = T.pack $ show tcArgs
                        }
                    }
                  | (tcId, tcName, tcArgs) <- calls]

-- ============================================================================
-- Strategy A: XML Response Parsing Tests
-- ============================================================================

spec :: Spec
spec = do
  describe "XML Provider Integration Tests" $ do
    describe "Strategy A: withXMLResponseParsing (native tool support + XML responses)" $ do
      it "preserves message history when parsing XML responses" $ do
        let model = Model MockXMLResponseModel OpenAI.LlamaCpp
            toolDef = ToolDefinition "test_tool" "A test tool" (object ["param" .= ("string" :: Text)])
            configs = [Tools [toolDef]]

            -- Initial conversation
            history = [ UserText "Hello"
                      , AssistantText "Hi there!"
                      , UserText "Use the test_tool"
                      ]

            -- Build request with history
            req = buildRequest model configs history
            reqMsgs = messages req

        -- Check that all messages are in the request
        length reqMsgs `shouldBe` 3

        -- Mock response with XML tool call
        let xmlToolCall = "<tool_call>test_tool\n<arg_key>param</arg_key>\n<arg_value>value</arg_value>\n</tool_call>"
            response = mockTextResponse xmlToolCall

            -- Parse response
            parsedMsgs = parseResponse model configs history response

        -- Should parse into a tool call message
        case parsedMsgs of
          [AssistantTool (ToolCall _ name _)] -> name `shouldBe` "test_tool"
          _ -> expectationFailure $ "Expected tool call, got: " ++ show parsedMsgs

      it "handles mixed text and XML tool calls" $ do
        let model = Model MockXMLResponseModel OpenAI.LlamaCpp
            toolDef = ToolDefinition "calc" "Calculator" (object [])
            configs = [Tools [toolDef]]
            history = [UserText "Calculate 2+2"]

            -- Response with text before and after tool call
            xmlToolCall = "Let me calculate that:\n<tool_call>calc\n<arg_key>expr</arg_key>\n<arg_value>2+2</arg_value>\n</tool_call>\nDone!"
            response = mockTextResponse xmlToolCall

            parsedMsgs = parseResponse model configs history response

        -- Should have both text and tool call
        length parsedMsgs `shouldBe` 2
        case parsedMsgs of
          [AssistantText txt, AssistantTool (ToolCall _ "calc" _)] ->
            T.isInfixOf "Let me calculate" txt `shouldBe` True
          _ -> expectationFailure $ "Expected text + tool call, got: " ++ show parsedMsgs

      it "preserves tool results in native OpenAI format (not converted to XML)" $ do
        let model = Model MockXMLResponseModel OpenAI.LlamaCpp
            toolDef = ToolDefinition "echo" "Echo tool" (object [])
            configs = [Tools [toolDef]]

            -- Conversation with tool result
            toolCall = ToolCall "call_123" "echo" (object ["text" .= ("hello" :: Text)])
            toolResult = ToolResult toolCall (Right (Aeson.String "hello"))
            history = [ UserText "Echo hello"
                          , AssistantTool toolCall
                          , ToolResultMsg toolResult
                          ]

            -- Build request - tool results should be in native format
            req = buildRequest model configs history
            reqMsgs = messages req

        -- Should have 3 messages, last one is tool result in OpenAI format
        length reqMsgs `shouldBe` 3
        let msg = last reqMsgs
        role msg `shouldBe` "tool"
        case content msg of
          Just c -> T.isInfixOf "hello" c `shouldBe` True
          Nothing -> expectationFailure "Expected content in tool message"
        tool_call_id msg `shouldBe` Just "call_123"

    describe "Strategy B: withFullXMLToolSupport (no native tool support)" $ do
      it "converts tool definitions to system prompt text" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            toolDef = ToolDefinition "search" "Search the web" (object ["query" .= ("string" :: Text)])
            configs = [Tools [toolDef], SystemPrompt "You are helpful."]
            history = [UserText "Search for cats"]

            req = buildRequest model configs history
            reqMsgs = messages req

        -- First message should be system with tool definitions
        let msg = head reqMsgs
        role msg `shouldBe` "system"
        case content msg of
          Just sysContent -> do
            T.isInfixOf "You are helpful" sysContent `shouldBe` True
            T.isInfixOf "search" sysContent `shouldBe` True
            T.isInfixOf "<tool_call>" sysContent `shouldBe` True
          Nothing -> expectationFailure "Expected system message content"

      it "converts tool calls to XML text in assistant messages" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            configs = []

            toolCall = ToolCall "call_456" "multiply" (object ["a" .= (2 :: Int), "b" .= (3 :: Int)])
            history = [ UserText "What is 2*3?"
                          , AssistantTool toolCall
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req

        -- Check that tool call is converted to XML in an assistant message
        let assistantMsgs = [msg | msg <- reqMsgs, role msg == "assistant"]
        length assistantMsgs `shouldSatisfy` (>= 1)
        let msg = last assistantMsgs
        role msg `shouldBe` "assistant"
        case content msg of
          Just c -> do
            T.isInfixOf "<tool_call>" c `shouldBe` True
            T.isInfixOf "multiply" c `shouldBe` True
          Nothing -> expectationFailure "Expected assistant message content"

      it "converts tool results to XML text in user messages" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            configs = []

            toolCall = ToolCall "call_789" "divide" (object [])
            toolResult = ToolResult toolCall (Right (Aeson.Number 42))
            history = [ UserText "Calculate"
                          , AssistantTool toolCall
                          , ToolResultMsg toolResult
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req

        -- Check that tool result is converted to XML in a user message
        let userMsgs = [msg | msg <- reqMsgs, role msg == "user"]
        length userMsgs `shouldSatisfy` (>= 2)  -- At least initial user + tool result
        let msg = last userMsgs
        role msg `shouldBe` "user"
        case content msg of
          Just c -> do
            T.isInfixOf "<tool_result>" c `shouldBe` True
            T.isInfixOf "call_789" c `shouldBe` True
            T.isInfixOf "divide" c `shouldBe` True
          Nothing -> expectationFailure "Expected user message content"

      it "preserves full message history through transformations" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            toolDef = ToolDefinition "fetch" "Fetch data" (object [])
            configs = [Tools [toolDef]]

            -- Complete conversation flow
            toolCall = ToolCall "call_abc" "fetch" (object ["url" .= ("http://example.com" :: Text)])
            toolResult = ToolResult toolCall (Right (Aeson.String "data"))
            history = [ UserText "First message"
                          , AssistantText "Acknowledged"
                          , UserText "Fetch something"
                          , AssistantTool toolCall
                          , ToolResultMsg toolResult
                          , UserText "What did you get?"
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req

            -- System + 6 messages = 7 total
            -- (system with tools, then all 6 conversation messages)
            contentTexts = [c | msg <- reqMsgs, Just c <- [content msg]]

        -- Check that messages are in order
        length reqMsgs `shouldSatisfy` (>= 6)  -- At least 6 conversation messages
        any (T.isInfixOf "First message") contentTexts `shouldBe` True
        any (T.isInfixOf "Acknowledged") contentTexts `shouldBe` True
        any (T.isInfixOf "What did you get") contentTexts `shouldBe` True

      it "parses XML tool calls from responses" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            configs = []
            history = [UserText "Do something"]

            -- Response with XML tool call
            xmlResponse = "<tool_call>process\n<arg_key>input</arg_key>\n<arg_value>data</arg_value>\n</tool_call>"
            response = mockTextResponse xmlResponse

            parsedMsgs = parseResponse model configs history response

        -- Should parse into tool call
        case parsedMsgs of
          [AssistantTool (ToolCall tcId "process" params)] -> do
            T.isPrefixOf "xml-" tcId `shouldBe` True
            case params of
              Aeson.Object obj -> KeyMap.lookup (Key.fromText "input") obj `shouldBe` Just (Aeson.String "data")
              _ -> expectationFailure "Expected object params"
          _ -> expectationFailure $ "Expected tool call, got: " ++ show parsedMsgs

    describe "Error Handling" $ do
      it "handles tool result errors correctly (Strategy A)" $ do
        let model = Model MockXMLResponseModel OpenAI.LlamaCpp
            configs = []

            toolCall = ToolCall "call_err" "broken_tool" (object [])
            errorResult = ToolResult toolCall (Left "Tool execution failed: timeout")
            history = [ UserText "Run broken tool"
                          , AssistantTool toolCall
                          , ToolResultMsg errorResult
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req

        -- Tool result should preserve error message
        let msg = last reqMsgs
        role msg `shouldBe` "tool"
        case content msg of
          Just c -> T.isInfixOf "Tool execution failed" c `shouldBe` True
          Nothing -> expectationFailure "Expected tool message content"

      it "handles tool result errors correctly (Strategy B - XML format)" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            configs = []

            toolCall = ToolCall "call_err2" "failed_tool" (object [])
            errorResult = ToolResult toolCall (Left "Permission denied")
            history = [ UserText "Try something"
                          , AssistantTool toolCall
                          , ToolResultMsg errorResult
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req

        -- Tool result should be XML with error
        let msg = last reqMsgs
        role msg `shouldBe` "user"
        case content msg of
          Just c -> do
            T.isInfixOf "<tool_result>" c `shouldBe` True
            T.isInfixOf "Error: Permission denied" c `shouldBe` True
          Nothing -> expectationFailure "Expected user message content"

      it "handles invalid tool calls (non-existent tool)" $ do
        -- For Strategy A (native OpenAI protocol)
        let model1 = Model MockXMLResponseModel OpenAI.LlamaCpp
            invalidCall1 = InvalidToolCall "call_inv" "nonexistent_tool" "some args" "Tool not found"
            history1 = [UserText "test", AssistantTool invalidCall1]
            req1 = buildRequest model1 [] history1
            reqMsgs1 = messages req1

        -- InvalidToolCall cannot be represented in OpenAI's native format,
        -- so it should be converted to text or handled specially
        length reqMsgs1 `shouldBe` 2

        -- For Strategy B (full XML)
        let model2 = Model MockFullXMLModel OpenAI.OpenAICompatible
            invalidCall2 = InvalidToolCall "call_inv2" "nonexistent_tool" "some args" "Tool not found"
            history2 = [UserText "test", AssistantTool invalidCall2]
            req2 = buildRequest model2 [] history2
            reqMsgs2 = messages req2

        -- Should convert to text representation
        let msg = last reqMsgs2
        role msg `shouldBe` "assistant"
        case content msg of
          Just c -> T.isInfixOf "nonexistent_tool" c `shouldBe` True
          Nothing -> expectationFailure "Expected assistant message content"

      it "handles malformed XML in responses gracefully" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            configs = []
            history = [UserText "test"]

            -- Malformed XML (missing closing tag)
            malformedXML = "<tool_call>broken_tool\n<arg_key>key</arg_key>"
            response = mockTextResponse malformedXML

            parsedMsgs = parseResponse model configs history response

        -- Should return text as-is when XML can't be parsed
        case parsedMsgs of
          [AssistantText txt] -> T.isInfixOf "tool_call" txt `shouldBe` True
          _ -> expectationFailure $ "Expected text fallback, got: " ++ show parsedMsgs

    describe "Message History Preservation" $ do
      it "does not drop messages during transformation (Strategy A)" $ do
        let model = Model MockXMLResponseModel OpenAI.LlamaCpp
            configs = []

            -- Complex history with various message types
            history = [ UserText "Message 1"
                          , AssistantText "Response 1"
                          , UserText "Message 2"
                          , AssistantText "Response 2"
                          , UserText "Message 3"
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req

            -- All messages should be preserved
        length reqMsgs `shouldBe` 5

      it "does not drop messages during transformation (Strategy B)" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            toolDef = ToolDefinition "tool1" "Tool" (object [])
            configs = [Tools [toolDef]]

            toolCall = ToolCall "call_1" "tool1" (object [])
            toolResult = ToolResult toolCall (Right (Aeson.String "ok"))
            history = [ UserText "Start"
                          , AssistantText "Ready"
                          , UserText "Use tool"
                          , AssistantTool toolCall
                          , ToolResultMsg toolResult
                          , UserText "Continue"
                          , AssistantText "Done"
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req

            -- System + 7 conversation messages
        length reqMsgs `shouldSatisfy` (>= 7)

      it "preserves message order during transformation" $ do
        let model = Model MockFullXMLModel OpenAI.OpenAICompatible
            configs = []

            history = [ UserText "First"
                          , AssistantText "Second"
                          , UserText "Third"
                          ]

            req = buildRequest model configs history
            reqMsgs = messages req
            roles = [role msg | msg <- reqMsgs]

            -- Check alternating pattern (may have system first)
            conversationRoles = filter (/= "system") roles

        conversationRoles `shouldBe` ["user", "assistant", "user"]
