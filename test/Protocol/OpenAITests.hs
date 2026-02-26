{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedRecordDot #-}

{- |
Module: Protocol.OpenAITests

Protocol capability probes for OpenAI wire protocol.

= Purpose

This module provides __capability probes__ - tests that discover what a
model/provider combination supports at the protocol level. These are not
traditional pass/fail tests, but rather __discovery tools__.

Each probe tests ONE specific capability or quirk:
- Does it respond to text?
- Can it make tool calls?
- Does it use reasoning_content or reasoning_details?

= How Probes Work

Run ALL probes against an unknown model/provider:
- Some will pass → capabilities we can use
- Some will fail → features not supported or quirks to work around

Results inform:
1. Which probes to __enshrine__ for that model (see Models.GLM for example)
2. How to build the ComposableProvider (what handlers to include)
3. What StandardTests can be run (what abstractions are supported)

= What Probes Test

Probes test __protocol behavior__, NOT our abstractions:
- ✓ "Does the wire protocol return tool_calls in the response?"
- ✗ "Does our Message type conversion work?" (that's StandardTests)

= Design Guidelines for Probes

* __Focus__ - Test ONE thing. Name clearly states what.
* __Simple__ - Minimal setup. No complex logic.
* __Direct__ - Call assertions from Protocol.OpenAI, don't implement inline
* __Clear failure__ - When it fails, immediately obvious what's missing
* __Quirk discovery__ - Variants for provider-specific behaviors (e.g., reasoningViaDetails)

= Usage

In a model-specific test suite:
@
import Protocol.OpenAITests

modelTests :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
modelTests provider = do
  describe "My Model" $ do
    basicText provider "model-name"
    toolCalling provider "model-name"
    reasoningViaDetails provider "model-name"  -- Uses quirk variant
@

In discovery (testing unknown model):
@
-- Run ALL probes, see which pass
basicText provider "unknown-model"           -- ✓ passes
toolCalling provider "unknown-model"         -- ✓ passes
reasoning provider "unknown-model"           -- ✗ fails
reasoningViaDetails provider "unknown-model" -- ✓ passes (quirk!)
@

-}

module Protocol.OpenAITests where

import UniversalLLM.Protocols.OpenAI
import Protocol.OpenAI  -- unqualified - we mainly call these
import Data.Text (Text)
import qualified Data.Text as T
import Test.Hspec (Spec, describe, it, shouldSatisfy, HasCallStack)

-- ============================================================================
-- Capability Probes
--
-- Each probe tests ONE specific protocol capability or quirk.
-- Keep them focused, simple, and use assertions from Protocol.OpenAI.
-- ============================================================================

-- | Probe: Basic text response
--
-- __Tests:__ Can the model respond to a simple text question?
--
-- __Checks:__ Response contains non-empty assistant text
--
-- __Expected to pass:__ Almost all models
basicText :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
basicText makeRequest modelName = do
  it "returns assistant text for simple question" $ do
    let req = (simpleUserRequest "What is 2+2?") { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Tool calling support
--
-- __Tests:__ Can the model make tool calls when tools are provided?
--
-- __Checks:__ Response contains tool_calls field with at least one call
--
-- __Expected to pass:__ Models that support function calling
--
-- __Note:__ This only tests if the model CAN call tools, not if it does
-- so correctly or appropriately. Use StandardTests for that.
toolCalling :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
toolCalling makeRequest modelName = do
  it "makes tool calls when tools are available" $ do
    let req = (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName
          , tools = Just [weatherTool]
          }
    resp <- makeRequest req
    assertHasToolCalls resp

-- | Probe: Reasoning content (standard field)
--
-- __Tests:__ Does the model return reasoning in reasoning_content field?
--
-- __Checks:__ Response contains non-empty reasoning_content
--
-- __Expected to pass:__ Native OpenAI reasoning models
--
-- __Expected to fail:__ OpenRouter (uses reasoning_details instead)
--
-- __See also:__ 'reasoningViaDetails' for the OpenRouter variant
reasoning :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
reasoning makeRequest modelName = do
  it "returns reasoning content when enabled" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
          { model = modelName }
    resp <- makeRequest req
    assertHasReasoningContent resp

-- | Probe: Reasoning via reasoning_details field (OpenRouter quirk)
--
-- __Tests:__ Does the provider put reasoning in reasoning_details?
--
-- __Checks:__ Response contains reasoning_details field
--
-- __Expected to pass:__ OpenRouter with reasoning models
--
-- __Expected to fail:__ Native OpenAI API
--
-- __Provider quirk:__ OpenRouter uses reasoning_details instead of
-- reasoning_content. This requires a handler in ComposableProvider
-- to translate the field for our Message abstraction.
reasoningViaDetails :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
reasoningViaDetails makeRequest modelName = do
  it "returns reasoning in reasoning_details field" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
          { model = modelName }
    resp <- makeRequest req
    assertHasReasoningDetails resp

-- | Probe: Tool calling via XML in content (model quirk)
--
-- __Tests:__ Does the model return tool calls as XML in content field?
--
-- __Checks:__ Response contains <tool_call>function_name in content
--
-- __Expected to pass:__ Models that don't support native tool_calls format
-- (like GLM-4.5 via llama.cpp)
--
-- __Expected to fail:__ Models with proper tool_calls field support
--
-- __Model quirk:__ Some models trained for XML tool format return tools in
-- the content field instead of using the tool_calls field. This requires
-- withXMLResponseParsing handler in ComposableProvider.
toolCallingViaXML :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
toolCallingViaXML makeRequest modelName = do
  it "returns tool calls as XML in content field" $ do
    let req = (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName
          , tools = Just [weatherTool]
          }
    resp <- makeRequest req
    assertHasXMLToolCall "get_weather" resp

-- | Probe: Accepts tool call responses
--
-- __Tests:__ Does the API accept tool results in conversation history?
--
-- __Checks:__ Response succeeds with fabricated tool call + result in history
--
-- __Expected to pass:__ All models that support tool calling
--
-- __Expected to fail:__ Models without tool support, or models requiring reasoning preservation (Gemini)
--
-- __Note:__ This tests if the wire protocol accepts the format we send,
-- not the full conversation flow (that's covered by StandardTests).
-- We use fabricated history because we only care about format acceptance.
-- Some models (Gemini) may reject this - use acceptsToolResultsWithoutReasoning instead.
acceptsToolResults :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
acceptsToolResults makeRequest modelName = do
  it "accepts tool results in conversation history" $ do
    let req = requestWithToolCallHistory { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Accepts tool call responses with reasoning disabled
--
-- __Tests:__ Does the API accept tool results when reasoning is explicitly disabled?
--
-- __Checks:__ Response succeeds with fabricated tool call + result, reasoning disabled
--
-- __Expected to pass:__ Models that support tool calling without reasoning
--
-- __Expected to fail:__ Models that always require reasoning (if any), or no tool support
--
-- __Note:__ This explicitly disables reasoning to test if fabricated tool histories
-- work when we're not in reasoning mode. Some models (Gemini) require reasoning_details
-- to be preserved in history, but might accept fabricated history if reasoning is disabled.
acceptsToolResultsWithoutReasoning :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
acceptsToolResultsWithoutReasoning makeRequest modelName = do
  it "accepts tool results with reasoning disabled" $ do
    let req = disableReasoning requestWithToolCallHistory { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Tool calling with reasoning preservation
--
-- __Tests:__ Does the model preserve reasoning through tool call chains?
--
-- __Checks:__ After tool call + result, model still responds correctly
--
-- __Expected to pass:__ Models that use reasoning_details and require preservation (Gemini via OpenRouter)
--
-- __Expected to fail:__ Models without reasoning support
--
-- __Note:__ This uses a real conversation flow (can't fabricate reasoning_details).
-- Some models (like Gemini) require reasoning_details to be preserved in history
-- or they fail/behave incorrectly on subsequent responses.
toolCallingWithReasoning :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
toolCallingWithReasoning makeRequest modelName = do
  it "preserves reasoning through tool call chains" $ do
    -- Step 1: Get model to make a tool call with reasoning
    let req1 = enableReasoning (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName
          , tools = Just [weatherTool]
          }
    resp1 <- makeRequest req1

    -- Verify it has reasoning and tool calls
    assertHasReasoningDetails resp1

    -- Step 2: Build request with tool result (helper preserves assistant message)
    let req2 = (requestWithToolResult resp1) { model = modelName }
    resp2 <- makeRequest req2

    -- Step 3: Verify we get a valid response
    assertHasAssistantText resp2

-- | Probe: Consecutive user messages
--
-- __Tests:__ Does the API accept multiple user messages in a row?
--
-- __Checks:__ Response succeeds with two consecutive user messages
--
-- __Expected to pass:__ Most models (semantically valid)
--
-- __Expected to fail:__ Models/APIs with strict alternating message requirements
--
-- __Note:__ Semantically, consecutive user messages make sense (user adds context
-- or asks follow-up before assistant responds), but some APIs enforce strict
-- user/assistant alternation.
consecutiveUserMessages :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
consecutiveUserMessages makeRequest modelName = do
  it "accepts consecutive user messages" $ do
    let req = (Protocol.OpenAI.consecutiveUserMessages "Here is some context." "Now answer this question: what is 2+2?")
          { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: History starting with assistant message
--
-- __Tests:__ Does the API accept conversation history starting with assistant?
--
-- __Checks:__ Response succeeds with assistant message before user message
--
-- __Expected to pass:__ Some models/APIs
--
-- __Expected to fail:__ Models/APIs requiring user message first
--
-- __Note:__ Some APIs/templates require conversation to start with user message.
-- Others accept assistant-first messages for system-like introductions or priming.
startsWithAssistant :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
startsWithAssistant makeRequest modelName = do
  it "accepts history starting with assistant message" $ do
    let req = Protocol.OpenAI.startsWithAssistant { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: System message mid-conversation
--
-- __Tests:__ Does the API accept a system message after user/assistant messages?
--
-- __Checks:__ Response succeeds with system message in the middle of history
--
-- __Expected to pass:__ APIs that allow system messages anywhere
--
-- __Expected to fail:__ Models with chat templates that require system at beginning
-- (e.g. Qwen3.5 raises "System message must be at the beginning")
--
-- __Note:__ If this fails but systemMessageAtStart passes, the model requires
-- system messages to be hoisted to the front. Use systemMessagesFirst provider handler.
systemMessageMidConversation :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
systemMessageMidConversation makeRequest modelName = do
  it "accepts system message mid-conversation" $ do
    let req = requestWithSystemMidConversation { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: System message at start of conversation
--
-- __Tests:__ Does the API accept a system message at the beginning?
--
-- __Checks:__ Response succeeds with system message before user message
--
-- __Expected to pass:__ Almost all models
--
-- __Expected to fail:__ Models that don't support system messages at all
systemMessageAtStart :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
systemMessageAtStart makeRequest modelName = do
  it "accepts system message at start" $ do
    let req = requestWithSystemAtStart { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Multiple system messages
--
-- __Tests:__ Does the API accept multiple system messages at the start?
--
-- __Checks:__ Response succeeds with three system messages before user message
--
-- __Expected to pass:__ APIs/templates that accept multiple system messages
--
-- __Expected to fail:__ Templates that only accept a single system message
-- (e.g. Qwen3.5 raises "System message must be at the beginning")
--
-- __Note:__ If this fails, the model needs mergeSystemMessages provider handler
-- to collapse multiple SystemPrompt configs into one.
multipleSystemMessages :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
multipleSystemMessages makeRequest modelName = do
  it "accepts multiple system messages" $ do
    let req = requestWithMultipleSystemMessages { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Tool result with no tools defined
--
-- __Tests:__ Does the API accept tool results when tools field is None/empty?
--
-- __Checks:__ Response succeeds with tool call + result in history but no tools defined
--
-- __Expected to pass:__ Flexible APIs that don't require tool definitions for historical calls
--
-- __Expected to fail:__ APIs that require tools field when tool results are present (Nova)
--
-- __Note:__ This is the most restrictive case - no tools at all.
-- Tests if tool definitions are required even for completed tool interactions.
acceptsToolResultNoTools :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
acceptsToolResultNoTools makeRequest modelName = do
  it "accepts tool result when no tools defined" $ do
    let req = requestWithToolResultNoTools { model = modelName }
    resp <- makeRequest req
    -- Just verify no error - response might be text or empty
    wasSuccessful resp

-- | Probe: Tool result but the called tool no longer available
--
-- __Tests:__ Does the API accept tool results when the specific tool is gone?
--
-- __Checks:__ Response succeeds with get_weather call+result but only calculator available
--
-- __Expected to pass:__ Flexible APIs
--
-- __Expected to fail:__ APIs requiring the called tool to be present (Nova)
--
-- __Note:__ This tests immediate removal - tool call just returned but tool is gone.
-- Different from acceptsStaleToolInHistory which has assistant message after.
acceptsToolResultToolGone :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
acceptsToolResultToolGone makeRequest modelName = do
  it "accepts tool result when called tool no longer available" $ do
    let req = requestWithToolResultToolGone { model = modelName }
    resp <- makeRequest req
    -- Just verify no error - response might be text or empty
    wasSuccessful resp

-- | Probe: Tool call in history but tool no longer available (further back)
--
-- __Tests:__ Does the API accept tool calls in history when tool is no longer in tool set?
--
-- __Checks:__ Model responds successfully (might call tool, or respond with text)
--
-- __Expected to pass:__ Flexible APIs that don't validate historical tool calls
--
-- __Expected to fail:__ APIs that require all tools referenced in history to be available
--
-- __Note:__ This tests tool call further back in history (after assistant responded).
-- Informs how careful we need to be when modifying tool sets during conversations.
-- We ask "calculate 5 * 7" with calculator tool, but the key test is API accepts
-- the request with stale get_weather tool in history.
acceptsStaleToolInHistory :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
acceptsStaleToolInHistory makeRequest modelName = do
  it "accepts tool call in history when tool no longer available" $ do
    let req = requestWithStaleToolInHistory { model = modelName }
    resp <- makeRequest req
    -- Just verify request succeeds - model behavior may vary
    wasSuccessful resp

-- | Probe: Old tool call in history with tool still available
--
-- __Tests:__ Does conversation work when tool from history is still available but conversation moved on?
--
-- __Checks:__ Response succeeds with old get_weather in history, weather tool available, asking about math
--
-- __Expected to pass:__ Most models (tool still available)
--
-- __Expected to fail:__ Models that get confused when available tool doesn't match current intent
--
-- __Note:__ This is the "safe" case - tool is still available even though conversation moved on.
-- Contrasts with acceptsStaleToolInHistory where tool is removed.
acceptsOldToolCallStillAvailable :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
acceptsOldToolCallStillAvailable makeRequest modelName = do
  it "accepts old tool call in history with tool still available" $ do
    let req = requestWithOldToolCallStillAvailable { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Provider error responses are valid protocol responses
--
-- __Tests:__ Does the provider properly return error responses in OpenAI format?
--
-- __Checks:__ Error response has proper structure with error details
--
-- __Expected to pass:__ All providers (error responses are part of the protocol)
--
-- __Expected to fail:__ Never (this tests our protocol handling, not provider behavior)
--
-- __Note:__ This verifies that OpenAIError is treated as a VALID protocol response,
-- not a failure. We test this by triggering an error condition (invalid model name)
-- and verifying we get a well-formed error response.
providerErrorResponse :: HasCallStack => (OpenAIRequest -> IO OpenAIResponse) -> Spec
providerErrorResponse makeRequest = do
  it "returns well-formed error response for invalid model" $ do
    let req = (simpleUserRequest "What is 2+2?") { model = "invalid-model-name-that-does-not-exist" }
    resp <- makeRequest req
    assertIsProviderError resp
