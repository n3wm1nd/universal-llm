{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

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

instance ModelName OpenAI.LlamaCpp MockXMLResponseModel where
  modelName _ = "mock-xml-response"

instance HasTools MockXMLResponseModel OpenAI.LlamaCpp where
  withTools = withXMLResponseParsing . OpenAI.openAIWithTools

instance ProviderImplementation OpenAI.LlamaCpp MockXMLResponseModel where
  getComposableProvider = withTools $ OpenAI.baseComposableProvider

-- Mock model that uses full XML support (Strategy B)
data MockFullXMLModel = MockFullXMLModel deriving (Show, Eq)

instance ModelName OpenAI.OpenAICompatible MockFullXMLModel where
  modelName _ = "mock-full-xml"

instance HasTools MockFullXMLModel OpenAI.OpenAICompatible where
  withTools = withFullXMLToolSupport . OpenAI.openAIWithTools

instance ProviderImplementation OpenAI.OpenAICompatible MockFullXMLModel where
  getComposableProvider = withTools $ OpenAI.baseComposableProvider

-- Helper to build a request from messages
buildRequest :: (ProviderImplementation provider model, Monoid (ProviderRequest provider))
             => provider
             -> model
             -> [ModelConfig provider model]
             -> [Message model provider]
             -> ProviderRequest provider
buildRequest = toProviderRequest

-- Helper to parse a response
parseResponse :: ProviderImplementation provider model
              => provider
              -> model
              -> [ModelConfig provider model]
              -> [Message model provider]  -- history
              -> ProviderResponse provider
              -> [Message model provider]
parseResponse provider model configs history resp =
  let (_provider, _model, msgs) = fromProviderResponse provider model configs history resp
  in msgs

-- Create a mock OpenAI response with text content
mockTextResponse :: Text -> OpenAIResponse
mockTextResponse txt = OpenAISuccess $ OpenAISuccessResponse
  [OpenAIChoice $ OpenAIMessage "assistant" (Just txt) Nothing Nothing Nothing]

-- Create a mock OpenAI response with tool calls
mockToolCallResponse :: [(Text, Text, Value)] -> OpenAIResponse
mockToolCallResponse calls = OpenAISuccess $ OpenAISuccessResponse
  [OpenAIChoice $ OpenAIMessage "assistant" Nothing Nothing (Just openAICalls) Nothing]
  where
    openAICalls = [OpenAIToolCall tcId "function" (OpenAIToolFunction tcName (T.pack $ show tcArgs))
                  | (tcId, tcName, tcArgs) <- calls]

-- ============================================================================
-- Strategy A: XML Response Parsing Tests
-- ============================================================================

spec :: Spec
spec = do
  describe "XML Provider Integration Tests" $ do
    describe "Strategy A: withXMLResponseParsing (native tool support + XML responses)" $ do
      it "preserves message history when parsing XML responses" $ do
        let provider = OpenAI.LlamaCpp
            model = MockXMLResponseModel
            toolDef = ToolDefinition "test_tool" "A test tool" (object ["param" .= ("string" :: Text)])
            configs = [Tools [toolDef]]

            -- Initial conversation
            history = [ UserText "Hello"
                      , AssistantText "Hi there!"
                      , UserText "Use the test_tool"
                      ]

            -- Build request with history
            req = buildRequest provider model configs history
            reqMsgs = messages req

        -- Check that all messages are in the request
        length reqMsgs `shouldBe` 3

        -- Mock response with XML tool call
        let xmlToolCall = "<tool_call>test_tool\n<arg_key>param</arg_key>\n<arg_value>value</arg_value>\n</tool_call>"
            response = mockTextResponse xmlToolCall

            -- Parse response
            parsedMsgs = parseResponse provider model configs history response

        -- Should parse into a tool call message
        case parsedMsgs of
          [AssistantTool (ToolCall _ name _)] -> name `shouldBe` "test_tool"
          _ -> expectationFailure $ "Expected tool call, got: " ++ show parsedMsgs

      it "handles mixed text and XML tool calls" $ do
        let provider = OpenAI.LlamaCpp
            model = MockXMLResponseModel
            toolDef = ToolDefinition "calc" "Calculator" (object [])
            configs = [Tools [toolDef]]
            history = [UserText "Calculate 2+2"]

            -- Response with text before and after tool call
            xmlToolCall = "Let me calculate that:\n<tool_call>calc\n<arg_key>expr</arg_key>\n<arg_value>2+2</arg_value>\n</tool_call>\nDone!"
            response = mockTextResponse xmlToolCall

            parsedMsgs = parseResponse provider model configs history response

        -- Should have both text and tool call
        length parsedMsgs `shouldBe` 2
        case parsedMsgs of
          [AssistantText txt, AssistantTool (ToolCall _ "calc" _)] ->
            T.isInfixOf "Let me calculate" txt `shouldBe` True
          _ -> expectationFailure $ "Expected text + tool call, got: " ++ show parsedMsgs

      it "preserves tool results in native OpenAI format (not converted to XML)" $ do
        let provider = OpenAI.LlamaCpp
            model = MockXMLResponseModel
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
            req = buildRequest provider model configs history
            reqMsgs = messages req

        -- Should have 3 messages, last one is tool result in OpenAI format
        length reqMsgs `shouldBe` 3
        case last reqMsgs of
          OpenAIMessage "tool" (Just content) _ _ (Just callId) -> do
            callId `shouldBe` "call_123"
            T.isInfixOf "hello" content `shouldBe` True
          _ -> expectationFailure $ "Expected tool message, got: " ++ show (last reqMsgs)

    describe "Strategy B: withFullXMLToolSupport (no native tool support)" $ do
      it "converts tool definitions to system prompt text" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
            toolDef = ToolDefinition "search" "Search the web" (object ["query" .= ("string" :: Text)])
            configs = [Tools [toolDef], SystemPrompt "You are helpful."]
            history = [UserText "Search for cats"]

            req = buildRequest provider model configs history
            reqMsgs = messages req

        -- First message should be system with tool definitions
        case head reqMsgs of
          OpenAIMessage "system" (Just sysContent) _ _ _ -> do
            T.isInfixOf "You are helpful" sysContent `shouldBe` True
            T.isInfixOf "search" sysContent `shouldBe` True
            T.isInfixOf "<tool_call>" sysContent `shouldBe` True
          _ -> expectationFailure $ "Expected system message, got: " ++ show (head reqMsgs)

      it "converts tool calls to XML text in assistant messages" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
            configs = []

            toolCall = ToolCall "call_456" "multiply" (object ["a" .= (2 :: Int), "b" .= (3 :: Int)])
            history = [ UserText "What is 2*3?"
                          , AssistantTool toolCall
                          ]

            req = buildRequest provider model configs history
            reqMsgs = messages req

        -- Check that tool call is converted to XML in an assistant message
        let assistantMsgs = [msg | msg@(OpenAIMessage "assistant" _ _ _ _) <- reqMsgs]
        length assistantMsgs `shouldSatisfy` (>= 1)
        case last assistantMsgs of
          OpenAIMessage "assistant" (Just content) _ _ _ -> do
            T.isInfixOf "<tool_call>" content `shouldBe` True
            T.isInfixOf "multiply" content `shouldBe` True
          _ -> expectationFailure $ "Expected assistant message with XML, got: " ++ show (last assistantMsgs)

      it "converts tool results to XML text in user messages" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
            configs = []

            toolCall = ToolCall "call_789" "divide" (object [])
            toolResult = ToolResult toolCall (Right (Aeson.Number 42))
            history = [ UserText "Calculate"
                          , AssistantTool toolCall
                          , ToolResultMsg toolResult
                          ]

            req = buildRequest provider model configs history
            reqMsgs = messages req

        -- Check that tool result is converted to XML in a user message
        let userMsgs = [msg | msg@(OpenAIMessage "user" _ _ _ _) <- reqMsgs]
        length userMsgs `shouldSatisfy` (>= 2)  -- At least initial user + tool result
        case last userMsgs of
          OpenAIMessage "user" (Just content) _ _ _ -> do
            T.isInfixOf "<tool_result>" content `shouldBe` True
            T.isInfixOf "call_789" content `shouldBe` True
            T.isInfixOf "divide" content `shouldBe` True
          _ -> expectationFailure $ "Expected user message with XML result, got: " ++ show (last userMsgs)

      it "preserves full message history through transformations" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
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

            req = buildRequest provider model configs history
            reqMsgs = messages req

            -- System + 6 messages = 7 total
            -- (system with tools, then all 6 conversation messages)
            contentTexts = [content | OpenAIMessage _ (Just content) _ _ _ <- reqMsgs]

        -- Check that messages are in order
        length reqMsgs `shouldSatisfy` (>= 6)  -- At least 6 conversation messages
        any (T.isInfixOf "First message") contentTexts `shouldBe` True
        any (T.isInfixOf "Acknowledged") contentTexts `shouldBe` True
        any (T.isInfixOf "What did you get") contentTexts `shouldBe` True

      it "parses XML tool calls from responses" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
            configs = []
            history = [UserText "Do something"]

            -- Response with XML tool call
            xmlResponse = "<tool_call>process\n<arg_key>input</arg_key>\n<arg_value>data</arg_value>\n</tool_call>"
            response = mockTextResponse xmlResponse

            parsedMsgs = parseResponse provider model configs history response

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
        let provider = OpenAI.LlamaCpp
            model = MockXMLResponseModel
            configs = []

            toolCall = ToolCall "call_err" "broken_tool" (object [])
            errorResult = ToolResult toolCall (Left "Tool execution failed: timeout")
            history = [ UserText "Run broken tool"
                          , AssistantTool toolCall
                          , ToolResultMsg errorResult
                          ]

            req = buildRequest provider model configs history
            reqMsgs = messages req

        -- Tool result should preserve error message
        case last reqMsgs of
          OpenAIMessage "tool" (Just content) _ _ _ ->
            T.isInfixOf "Tool execution failed" content `shouldBe` True
          _ -> expectationFailure "Expected tool message with error"

      it "handles tool result errors correctly (Strategy B - XML format)" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
            configs = []

            toolCall = ToolCall "call_err2" "failed_tool" (object [])
            errorResult = ToolResult toolCall (Left "Permission denied")
            history = [ UserText "Try something"
                          , AssistantTool toolCall
                          , ToolResultMsg errorResult
                          ]

            req = buildRequest provider model configs history
            reqMsgs = messages req

        -- Tool result should be XML with error
        case last reqMsgs of
          OpenAIMessage "user" (Just content) _ _ _ -> do
            T.isInfixOf "<tool_result>" content `shouldBe` True
            T.isInfixOf "Error: Permission denied" content `shouldBe` True
          _ -> expectationFailure $ "Expected user message with XML error, got: " ++ show (last reqMsgs)

      it "handles invalid tool calls (non-existent tool)" $ do
        -- For Strategy A (native OpenAI protocol)
        let provider1 = OpenAI.LlamaCpp
            model1 = MockXMLResponseModel
            invalidCall1 = InvalidToolCall "call_inv" "nonexistent_tool" "some args" "Tool not found"
            history1 = [UserText "test", AssistantTool invalidCall1]
            req1 = buildRequest provider1 model1 [] history1
            reqMsgs1 = messages req1

        -- InvalidToolCall cannot be represented in OpenAI's native format,
        -- so it should be converted to text or handled specially
        length reqMsgs1 `shouldBe` 2

        -- For Strategy B (full XML)
        let provider2 = OpenAI.OpenAICompatible
            model2 = MockFullXMLModel
            invalidCall2 = InvalidToolCall "call_inv2" "nonexistent_tool" "some args" "Tool not found"
            history2 = [UserText "test", AssistantTool invalidCall2]
            req2 = buildRequest provider2 model2 [] history2
            reqMsgs2 = messages req2

        -- Should convert to text representation
        case last reqMsgs2 of
          OpenAIMessage "assistant" (Just content) _ _ _ ->
            T.isInfixOf "nonexistent_tool" content `shouldBe` True
          _ -> expectationFailure "Expected assistant message with error"

      it "handles malformed XML in responses gracefully" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
            configs = []
            history = [UserText "test"]

            -- Malformed XML (missing closing tag)
            malformedXML = "<tool_call>broken_tool\n<arg_key>key</arg_key>"
            response = mockTextResponse malformedXML

            parsedMsgs = parseResponse provider model configs history response

        -- Should return text as-is when XML can't be parsed
        case parsedMsgs of
          [AssistantText txt] -> T.isInfixOf "tool_call" txt `shouldBe` True
          _ -> expectationFailure $ "Expected text fallback, got: " ++ show parsedMsgs

    describe "Message History Preservation" $ do
      it "does not drop messages during transformation (Strategy A)" $ do
        let provider = OpenAI.LlamaCpp
            model = MockXMLResponseModel
            configs = []

            -- Complex history with various message types
            history = [ UserText "Message 1"
                          , AssistantText "Response 1"
                          , UserText "Message 2"
                          , AssistantText "Response 2"
                          , UserText "Message 3"
                          ]

            req = buildRequest provider model configs history
            reqMsgs = messages req

            -- All messages should be preserved
        length reqMsgs `shouldBe` 5

      it "does not drop messages during transformation (Strategy B)" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
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

            req = buildRequest provider model configs history
            reqMsgs = messages req

            -- System + 7 conversation messages
        length reqMsgs `shouldSatisfy` (>= 7)

      it "preserves message order during transformation" $ do
        let provider = OpenAI.OpenAICompatible
            model = MockFullXMLModel
            configs = []

            history = [ UserText "First"
                          , AssistantText "Second"
                          , UserText "Third"
                          ]

            req = buildRequest provider model configs history
            reqMsgs = messages req
            roles = [role | OpenAIMessage role _ _ _ _ <- reqMsgs]

            -- Check alternating pattern (may have system first)
            conversationRoles = filter (/= "system") roles

        conversationRoles `shouldBe` ["user", "assistant", "user"]
