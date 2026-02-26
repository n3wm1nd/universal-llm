{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedRecordDot #-}

{- |
Module: Protocol.OpenAI

Low-level protocol helpers for OpenAI wire protocol testing.

= Purpose

This module provides building blocks for protocol-level capability probes.
It is NOT for testing our abstractions (use StandardTests for that).

Provides:
- Helper functions to build requests (hides Aeson/JSON complexity)
- Functions to extract data from responses
- Composable error checking (catch API errors early)
- Assertions for protocol tests

= Design Principles

* Protocol-agnostic naming - functions should work similarly across protocols
* Hide complexity - no Aeson manipulation in tests
* Use mempty with record updates - resilient to field additions
* Descriptive error messages - failures should be immediately clear
* Keep functions minimal - add more as needed, don't over-engineer

= Usage Pattern

Building a request:
@
let req = (simpleUserRequest "What is 2+2?") { model = "gpt-4" }
    withTools = req { tools = Just [simpleTool "calculator" "Do math"] }
@

Checking responses:
@
resp <- makeRequest req
let text = getAssistantText . checkError $ resp  -- checkError catches API errors first
assertHasAssistantText resp  -- use in protocol tests
@

-}

module Protocol.OpenAI where

import UniversalLLM.Protocols.OpenAI
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import Test.Hspec (Spec, it, shouldSatisfy, Expectation, HasCallStack)

-- ============================================================================
-- Empty/Default Objects
--
-- Internal defaults for building objects with record updates.
-- These ensure we handle all fields and can swap to mempty if instances added.
-- ============================================================================

-- | Empty message - replace with mempty if OpenAIMessage gets Monoid instance
emptyMessage :: OpenAIMessage
emptyMessage = OpenAIMessage
  { role = ""
  , content = Nothing
  , reasoning_content = Nothing
  , reasoning_details = Nothing
  , tool_calls = Nothing
  , tool_call_id = Nothing
  }

-- | Empty reasoning config - internal, use 'enableReasoning' instead
emptyReasoningConfig :: OpenAIReasoningConfig
emptyReasoningConfig = OpenAIReasoningConfig
  { reasoning_enabled = Nothing
  , reasoning_effort = Nothing
  , reasoning_max_tokens = Nothing
  , reasoning_exclude = Nothing
  }

-- ============================================================================
-- Request Helpers
--
-- Functions to build requests without dealing with protocol details.
-- Named to be protocol-agnostic where possible.
-- ============================================================================

-- | Create a user message
userMessage :: Text -> OpenAIMessage
userMessage txt = emptyMessage { role = "user", content = Just txt }

-- | Create a simple request with a user message
--
-- Common pattern: start with this, then set model field
-- >>> simpleUserRequest "What is 2+2?" & \r -> r { model = "gpt-4" }
simpleUserRequest :: Text -> OpenAIRequest
simpleUserRequest txt = mempty { messages = [userMessage txt] }

-- | Create a request with consecutive user messages
--
-- Used to test if the API accepts multiple user messages in a row
consecutiveUserMessages :: Text -> Text -> OpenAIRequest
consecutiveUserMessages msg1 msg2 = mempty { messages = [userMessage msg1, userMessage msg2] }

-- | Create an assistant message
assistantMessage :: Text -> OpenAIMessage
assistantMessage txt = emptyMessage { role = "assistant", content = Just txt }

-- | Create a request starting with an assistant message
--
-- Used to test if the API accepts history starting with assistant (no initial user message)
startsWithAssistant :: OpenAIRequest
startsWithAssistant = mempty
  { messages =
      [ assistantMessage "I'm a helpful assistant."
      , userMessage "What is 2+2?"
      ]
  }

-- | Create a simple tool definition with parameters
--
-- Example: simpleTool "get_weather" "Get current weather" [("location", "string", "City name")]
--
-- This creates a complete, valid tool definition with properly specified parameters.
-- Each parameter is (name, type, description), and all parameters are marked as required.
simpleTool :: Text -> Text -> [(Text, Text, Text)] -> OpenAIToolDefinition
simpleTool name description params = OpenAIToolDefinition
  { tool_type = "function"
  , function = OpenAIFunction
      { name = name
      , description = description
      , parameters = Aeson.object
          [ "type" Aeson..= ("object" :: Text)
          , "properties" Aeson..= Aeson.object
              [ Key.fromText paramName Aeson..= Aeson.object
                  [ "type" Aeson..= paramType
                  , "description" Aeson..= paramDesc
                  ]
              | (paramName, paramType, paramDesc) <- params
              ]
          , "required" Aeson..= ([paramName | (paramName, _, _) <- params] :: [Text])
          ]
      }
  }

-- | Create a weather tool definition with proper parameters
--
-- This creates a complete, valid tool definition that models can actually use.
-- The location parameter is properly specified with type and description.
weatherTool :: OpenAIToolDefinition
weatherTool = simpleTool "get_weather" "Get current weather for a location"
  [("location", "string", "City name")]

-- | Enable reasoning on a request
enableReasoning :: OpenAIRequest -> OpenAIRequest
enableReasoning req = req
  { reasoning = Just $ emptyReasoningConfig { reasoning_enabled = Just True }
  }

-- | Disable reasoning on a request
disableReasoning :: OpenAIRequest -> OpenAIRequest
disableReasoning req = req
  { reasoning = Just $ emptyReasoningConfig { reasoning_enabled = Just False }
  }

-- | Create an assistant message with a tool call
--
-- Example: assistantToolCallMessage "call_123" "get_weather" "{\"location\": \"London\"}"
assistantToolCallMessage :: Text -> Text -> Text -> OpenAIMessage
assistantToolCallMessage callId functionName arguments = emptyMessage
  { role = "assistant"
  , tool_calls = Just [OpenAIToolCall
      { callId = callId
      , toolCallType = "function"
      , toolFunction = OpenAIToolFunction
          { toolFunctionName = functionName
          , toolFunctionArguments = arguments
          }
      }]
  }

-- | Create a tool response message
--
-- Example: toolResponseMessage "call_123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
toolResponseMessage :: Text -> Text -> OpenAIMessage
toolResponseMessage callId result = emptyMessage
  { role = "tool"
  , tool_call_id = Just callId
  , content = Just result
  }

-- | Create a system message
systemMsg :: Text -> OpenAIMessage
systemMsg txt = emptyMessage { role = "system", content = Just txt }

-- | Create a request with system message mid-conversation
--
-- History: user message, then system message, then user message.
-- Tests if the API accepts system messages that are not at the beginning.
requestWithSystemMidConversation :: OpenAIRequest
requestWithSystemMidConversation = mempty
  { messages =
      [ userMessage "What is 2+2?"
      , assistantMessage "4"
      , systemMsg "You are a helpful assistant."
      , userMessage "And what is 3+3?"
      ]
  }

-- | Create a request with system message at the beginning
--
-- History: system message, then user message.
-- This is the expected/standard positioning.
requestWithSystemAtStart :: OpenAIRequest
requestWithSystemAtStart = mempty
  { messages =
      [ systemMsg "You are a helpful assistant."
      , userMessage "What is 2+2?"
      ]
  }

-- | Create a request with multiple system messages
--
-- History: three system messages, then user message.
-- Tests if the API accepts more than one system message.
requestWithMultipleSystemMessages :: OpenAIRequest
requestWithMultipleSystemMessages = mempty
  { messages =
      [ systemMsg "You are a helpful assistant."
      , systemMsg "Always respond concisely."
      , systemMsg "Use plain language."
      , userMessage "What is 2+2?"
      ]
  }

-- | Create a request with fabricated tool call history
--
-- This creates a conversation with:
-- 1. User asks question that should trigger tool
-- 2. Assistant makes tool call
-- 3. Tool returns result (automated, from us)
-- (Next message should be assistant responding to the tool result)
--
-- Used to test if the API accepts tool results in the expected format.
-- Includes the tool definition since the tool call references it.
requestWithToolCallHistory :: OpenAIRequest
requestWithToolCallHistory = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      ]
  , tools = Just [weatherTool]
  }

-- | Create a request with tool call history but tool no longer available
--
-- This tests if the API accepts tool calls in history where the tool definition
-- is no longer in the current tool set. This happens when tools are added/removed
-- between requests.
--
-- History contains a get_weather tool call, but only calculator tool is available now.
requestWithStaleToolInHistory :: OpenAIRequest
requestWithStaleToolInHistory = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      , assistantMessage "It's 72 degrees and sunny in London."
      , userMessage "Now calculate 5 * 7"
      ]
  , tools = Just [simpleTool "calculator" "Perform calculations" [("expression", "string", "Math expression")]]
  }

-- | Create a request with tool call result but no tools defined
--
-- This tests if the API accepts tool results when tools field is empty/null.
-- Some APIs (like Nova) require tool definitions even when processing results.
requestWithToolResultNoTools :: OpenAIRequest
requestWithToolResultNoTools = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      ]
  , tools = Nothing  -- No tools defined
  }

-- | Create a request with tool call result but the specific tool no longer available
--
-- This tests if the API accepts tool results when the tool that was called
-- is no longer in the tool set. Different from requestWithStaleToolInHistory
-- which has the tool call + result deep in history - this is immediately after
-- the tool call returns.
requestWithToolResultToolGone :: OpenAIRequest
requestWithToolResultToolGone = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      ]
  , tools = Just [simpleTool "calculator" "Perform calculations" [("expression", "string", "Math expression")]]
  }

-- | Create a request with old tool call in history, new user message, tool still available
--
-- This tests if tools in history must remain in the tool set even after
-- the conversation has moved on. The get_weather call is in history but conversation
-- has continued - does the tool need to stay available?
requestWithOldToolCallStillAvailable :: OpenAIRequest
requestWithOldToolCallStillAvailable = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      , assistantMessage "It's 72 degrees and sunny in London."
      , userMessage "Now calculate 5 * 7"
      ]
  , tools = Just [weatherTool]  -- weather still available, though conversation moved to math
  }

-- | Create a request with tool result from a previous response
--
-- This extracts the assistant message from the response (preserving all fields
-- including reasoning_details), adds a mock tool result, and creates a new request.
--
-- Used to test if reasoning/metadata is preserved through tool call chains.
requestWithToolResult :: OpenAIResponse -> OpenAIRequest
requestWithToolResult resp =
  let assistantMsg = getFirstMessage . expectSuccess $ resp
  in mempty
    { messages =
        [ userMessage "Use the get_weather function to check the weather in London."
        , assistantMsg  -- Preserve all fields (including reasoning_details)
        , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
        ]
    }

-- ============================================================================
-- Response Helpers
--
-- Functions to extract data from responses and check for errors.
-- Use 'expectSuccess' when you expect a successful response.
-- Use 'expectError' when testing error handling.
-- ============================================================================

-- | Expect a success response - unwrap and return OpenAISuccessResponse
--
-- Use this when you expect the request to succeed:
-- >>> text <- getAssistantText . expectSuccess $ resp
--
-- Note: This throws on OpenAIError, which is a VALID protocol response.
-- If you want to test error handling, use 'expectError' instead.
expectSuccess :: HasCallStack => OpenAIResponse -> OpenAISuccessResponse
expectSuccess (OpenAIError (OpenAIErrorResponse details)) =
  let typeInfo = case errorType details of
        Just t -> " (" <> t <> ")"
        Nothing -> ""
  in error $ T.unpack $ "Expected success but got provider error: " <> errorMessage details <> typeInfo
expectSuccess (OpenAISuccess success) = success

-- | Expect an error response - unwrap and return OpenAIErrorResponse
--
-- Use this when testing error handling:
-- >>> errResp <- expectError resp
-- >>> errorMessage (errorDetail errResp) `shouldContain` "invalid"
expectError :: OpenAIResponse -> OpenAIErrorResponse
expectError (OpenAISuccess _) =
  error "Expected provider error but got success response"
expectError (OpenAIError err) = err

-- | Check if response is an error - throw if so, otherwise pass through
--
-- DEPRECATED: Use 'expectSuccess' instead for clarity.
-- This name is confusing because it doesn't "check" - it throws on error.
{-# DEPRECATED checkError "Use expectSuccess instead" #-}
checkError :: OpenAIResponse -> OpenAIResponse
checkError resp = OpenAISuccess (expectSuccess resp)

-- | Extract the single message from success response
--
-- Throws if no choices or multiple choices. Use this to avoid repeated pattern matching:
-- >>> msg <- getFirstMessage . expectSuccess $ resp
getFirstMessage :: OpenAISuccessResponse -> OpenAIMessage
getFirstMessage (OpenAISuccessResponse []) =
  error "No choices in response"
getFirstMessage (OpenAISuccessResponse [OpenAIChoice msg]) = msg
getFirstMessage (OpenAISuccessResponse choices) =
  error $ "Expected exactly one choice, got " ++ show (length choices)

-- | Extract assistant text from success response
--
-- Throws if no choices or no content. Use after expectSuccess:
-- >>> getAssistantText . expectSuccess $ resp
getAssistantText :: OpenAISuccessResponse -> Text
getAssistantText success =
  case (getFirstMessage success).content of
    Just txt -> txt
    Nothing -> error "Assistant message has no content"

-- | Extract error details from error response
--
-- Throws if response is not an error. Use to test error handling:
-- >>> details <- getErrorDetail . expectError $ resp
-- >>> errorMessage details `shouldContain` "invalid"
getErrorDetail :: OpenAIErrorResponse -> OpenAIErrorDetail
getErrorDetail (OpenAIErrorResponse details) = details

-- ============================================================================
-- Protocol Assertions
--
-- Assertions for use in protocol capability probes.
-- These check specific protocol behaviors and throw descriptive errors.
-- ============================================================================

-- | Assert that response was successful (not an error)
--
-- Use this when you just need to verify no error occurred.
-- Throws if provider returned error, succeeds if response is valid.
-- Useful for tests that just need to check acceptance without inspecting content.
wasSuccessful :: HasCallStack => OpenAIResponse -> Expectation
wasSuccessful resp = expectSuccess resp `seq` return ()

-- | Assert that response is a provider error (not a success)
--
-- Use this to test error handling - verifies we got an OpenAIError response
assertIsProviderError :: HasCallStack => OpenAIResponse -> Expectation
assertIsProviderError resp = do
  let details = getErrorDetail . expectError $ resp
  -- Verify it has error details
  T.length (errorMessage details) `shouldSatisfy` (> 0)

-- | Assert that response contains assistant text
--
-- Use this in protocol tests to verify basic text responses
assertHasAssistantText :: HasCallStack => OpenAIResponse -> Expectation
assertHasAssistantText resp = do
  let text = getAssistantText . expectSuccess $ resp
  T.length text `shouldSatisfy` (> 0)

-- | Assert that response contains tool calls
assertHasToolCalls :: HasCallStack => OpenAIResponse -> Expectation
assertHasToolCalls resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.tool_calls of
    Just (_:_) -> return ()
    Just [] -> error "Message has empty tool_calls list"
    Nothing -> error "Message has no tool_calls field"

-- | Assert that response contains reasoning content
assertHasReasoningContent :: HasCallStack => OpenAIResponse -> Expectation
assertHasReasoningContent resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.reasoning_content of
    Just txt | T.length txt > 0 -> return ()
    Just _ -> error "Message has empty reasoning_content"
    Nothing -> error "Message has no reasoning_content field"

-- | Assert that response contains reasoning_details
--
-- Some providers (like OpenRouter) put reasoning in reasoning_details instead
assertHasReasoningDetails :: HasCallStack => OpenAIResponse -> Expectation
assertHasReasoningDetails resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.reasoning_details of
    Just _ -> return ()  -- It's a Value, just check it exists
    Nothing -> error "Message has no reasoning_details field"

-- | Assert that response contains an XML-style tool call in the content
--
-- Some models (like GLM-4.5 via llama.cpp) return tool calls as XML in the content field
-- instead of using the proper tool_calls field. This checks for the presence of
-- <tool_call>function_name in the response body.
assertHasXMLToolCall :: HasCallStack => Text -> OpenAIResponse -> Expectation
assertHasXMLToolCall functionName resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.content of
    Just txt | T.isInfixOf ("<tool_call>" <> functionName) txt -> return ()
    Just txt -> error $ "Message content does not contain <tool_call>" <> T.unpack functionName <> ", got: " <> T.unpack txt
    Nothing -> error "Message has no content field"
