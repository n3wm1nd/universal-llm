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
import Test.Hspec (Spec, it, shouldSatisfy, Expectation)

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

-- | Create a request with fabricated tool call history
--
-- This creates a conversation with:
-- 1. User asks question that should trigger tool
-- 2. Assistant makes tool call
-- 3. Tool returns result (automated, from us)
-- (Next message should be assistant responding to the tool result)
--
-- Used to test if the API accepts tool results in the expected format.
requestWithToolCallHistory :: OpenAIRequest
requestWithToolCallHistory = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      ]
  }

-- | Create a request with tool result from a previous response
--
-- This extracts the assistant message from the response (preserving all fields
-- including reasoning_details), adds a mock tool result, and creates a new request.
--
-- Used to test if reasoning/metadata is preserved through tool call chains.
requestWithToolResult :: OpenAIResponse -> OpenAIRequest
requestWithToolResult resp = case checkError resp of
  OpenAISuccess (OpenAISuccessResponse (OpenAIChoice assistantMsg : _)) ->
    mempty
      { messages =
          [ userMessage "Use the get_weather function to check the weather in London."
          , assistantMsg  -- Preserve all fields (including reasoning_details)
          , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
          ]
      }
  _ -> error "Response doesn't contain assistant message"

-- ============================================================================
-- Response Helpers
--
-- Functions to extract data from responses and check for errors.
-- Use 'checkError' first in composition chains to catch API errors early.
-- ============================================================================

-- | Check if response is an error - throw if so, otherwise pass through
--
-- Use this first in a composition chain to catch API errors:
-- >>> getAssistantText . checkError $ resp
checkError :: OpenAIResponse -> OpenAIResponse
checkError (OpenAIError (OpenAIErrorResponse details)) =
  error $ T.unpack $ "API error: " <> errorMessage details <> " (" <> errorType details <> ")"
checkError resp = resp

-- | Extract assistant text from response
--
-- Throws if no choices or no content. Use after checkError:
-- >>> getAssistantText . checkError $ resp
getAssistantText :: OpenAIResponse -> Text
getAssistantText (OpenAISuccess (OpenAISuccessResponse [])) =
  error "No choices in response"
getAssistantText (OpenAISuccess (OpenAISuccessResponse (OpenAIChoice msg : _))) =
  case msg.content of
    Just txt -> txt
    Nothing -> error "Assistant message has no content"
getAssistantText (OpenAIError _) =
  error "Cannot get assistant text from error response (use checkError first)"

-- ============================================================================
-- Protocol Assertions
--
-- Assertions for use in protocol capability probes.
-- These check specific protocol behaviors and throw descriptive errors.
-- ============================================================================

-- | Assert that response contains assistant text
--
-- Use this in protocol tests to verify basic text responses
assertHasAssistantText :: OpenAIResponse -> Expectation
assertHasAssistantText resp = do
  let text = getAssistantText . checkError $ resp
  T.length text `shouldSatisfy` (> 0)

-- | Assert that response contains tool calls
assertHasToolCalls :: OpenAIResponse -> Expectation
assertHasToolCalls resp = do
  case checkError resp of
    OpenAISuccess (OpenAISuccessResponse []) ->
      error "No choices in response"
    OpenAISuccess (OpenAISuccessResponse (OpenAIChoice msg : _)) ->
      case msg.tool_calls of
        Just (_:_) -> return ()
        Just [] -> error "Message has empty tool_calls list"
        Nothing -> error "Message has no tool_calls field"
    OpenAIError _ ->
      error "Cannot check tool calls on error response (use checkError first)"

-- | Assert that response contains reasoning content
assertHasReasoningContent :: OpenAIResponse -> Expectation
assertHasReasoningContent resp = do
  case checkError resp of
    OpenAISuccess (OpenAISuccessResponse []) ->
      error "No choices in response"
    OpenAISuccess (OpenAISuccessResponse (OpenAIChoice msg : _)) ->
      case msg.reasoning_content of
        Just txt | T.length txt > 0 -> return ()
        Just _ -> error "Message has empty reasoning_content"
        Nothing -> error "Message has no reasoning_content field"
    OpenAIError _ ->
      error "Cannot check reasoning content on error response (use checkError first)"

-- | Assert that response contains reasoning_details
--
-- Some providers (like OpenRouter) put reasoning in reasoning_details instead
assertHasReasoningDetails :: OpenAIResponse -> Expectation
assertHasReasoningDetails resp = do
  case checkError resp of
    OpenAISuccess (OpenAISuccessResponse []) ->
      error "No choices in response"
    OpenAISuccess (OpenAISuccessResponse (OpenAIChoice msg : _)) ->
      case msg.reasoning_details of
        Just _ -> return ()  -- It's a Value, just check it exists
        Nothing -> error "Message has no reasoning_details field"
    OpenAIError _ ->
      error "Cannot check reasoning details on error response (use checkError first)"

-- | Assert that response contains an XML-style tool call in the content
--
-- Some models (like GLM-4.5 via llama.cpp) return tool calls as XML in the content field
-- instead of using the proper tool_calls field. This checks for the presence of
-- <tool_call>function_name in the response body.
assertHasXMLToolCall :: Text -> OpenAIResponse -> Expectation
assertHasXMLToolCall functionName resp = do
  case checkError resp of
    OpenAISuccess (OpenAISuccessResponse []) ->
      error "No choices in response"
    OpenAISuccess (OpenAISuccessResponse (OpenAIChoice msg : _)) ->
      case msg.content of
        Just txt | T.isInfixOf ("<tool_call>" <> functionName) txt -> return ()
        Just txt -> error $ "Message content does not contain <tool_call>" <> T.unpack functionName <> ", got: " <> T.unpack txt
        Nothing -> error "Message has no content field"
    OpenAIError _ ->
      error "Cannot check XML tool call on error response (use checkError first)"
