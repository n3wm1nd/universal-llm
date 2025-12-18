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
1. Which probes to __enshrine__ for that model (see Models.GLM45Air for example)
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
import Test.Hspec (Spec, describe, it, shouldSatisfy)

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
basicText :: (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
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
toolCalling :: (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
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
reasoning :: (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
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
reasoningViaDetails :: (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
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
toolCallingViaXML :: (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
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
-- __Expected to fail:__ Models without tool support
--
-- __Note:__ This tests if the wire protocol accepts the format we send,
-- not the full conversation flow (that's covered by StandardTests).
-- We use fabricated history because we only care about format acceptance.
acceptsToolResults :: (OpenAIRequest -> IO OpenAIResponse) -> Text -> Spec
acceptsToolResults makeRequest modelName = do
  it "accepts tool results in conversation history" $ do
    let req = requestWithToolCallHistory { model = modelName }
    resp <- makeRequest req
    assertHasAssistantText resp
