{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Protocol.AnthropicTests

Protocol capability probes for Anthropic wire protocol.

= Purpose

This module provides __capability probes__ - tests that discover what a
model/provider combination supports at the protocol level. These are not
traditional pass/fail tests, but rather __discovery tools__.

Each probe tests ONE specific capability or quirk:
- Does it respond to text?
- Can it make tool calls?
- Does it support extended thinking?

= How Probes Work

Run ALL probes against an unknown model/provider:
- Some will pass → capabilities we can use
- Some will fail → features not supported or quirks to work around

Results inform:
1. Which probes to __enshrine__ for that model
2. How to build the ComposableProvider (what handlers to include)
3. What StandardTests can be run (what abstractions are supported)

= What Probes Test

Probes test __protocol behavior__, NOT our abstractions:
- ✓ "Does the wire protocol return tool_use content blocks?"
- ✗ "Does our Message type conversion work?" (that's StandardTests)

= Design Guidelines for Probes

* __Focus__ - Test ONE thing. Name clearly states what.
* __Simple__ - Minimal setup. No complex logic.
* __Direct__ - Call assertions from Protocol.Anthropic, don't implement inline
* __Clear failure__ - When it fails, immediately obvious what's missing
* __Quirk discovery__ - Variants for provider-specific behaviors

= Usage

In a model-specific test suite:
@
import Protocol.AnthropicTests

modelTests :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
modelTests provider = do
  describe "My Model" $ do
    basicText provider
    toolCalling provider
    reasoning provider
@

In discovery (testing unknown model):
@
-- Run ALL probes, see which pass
basicText provider           -- ✓ passes
toolCalling provider         -- ✓ passes
reasoning provider           -- ✓ passes (if model supports it)
@

-}

module Protocol.AnthropicTests where

import UniversalLLM.Protocols.Anthropic
import Protocol.Anthropic  -- unqualified - we mainly call these
import Data.Text (Text)
import Test.Hspec (Spec, describe, it, HasCallStack)

-- ============================================================================
-- Capability Probes
--
-- Each probe tests ONE specific protocol capability or quirk.
-- Keep them focused, simple, and use assertions from Protocol.Anthropic.
-- ============================================================================

-- | Probe: Basic text response
--
-- __Tests:__ Can the model respond to a simple text question?
--
-- __Checks:__ Response contains non-empty assistant text
--
-- __Expected to pass:__ All Anthropic models
basicText :: HasCallStack => (AnthropicRequest -> IO AnthropicResponse) -> Spec
basicText makeRequest = do
  it "returns assistant text for simple question" $ do
    let req = simpleUserRequest "What is 2+2?"
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Tool calling support
--
-- __Tests:__ Can the model make tool calls when tools are provided?
--
-- __Checks:__ Response contains tool_use content blocks
--
-- __Expected to pass:__ All modern Claude models
--
-- __Note:__ This only tests if the model CAN call tools, not if it does
-- so correctly or appropriately. Use StandardTests for that.
toolCalling :: HasCallStack => (AnthropicRequest -> IO AnthropicResponse) -> Spec
toolCalling makeRequest = do
  it "makes tool calls when tools are available" $ do
    let req = (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { tools = Just [weatherTool]
          }
    resp <- makeRequest req
    assertHasToolCalls resp

-- | Probe: Reasoning (extended thinking)
--
-- __Tests:__ Does the model return reasoning in thinking content blocks?
--
-- __Checks:__ Response contains thinking content blocks
--
-- __Expected to pass:__ Claude models with extended thinking enabled
--
-- __Note:__ Requires thinking.type = "enabled" in request
reasoning :: HasCallStack => (AnthropicRequest -> IO AnthropicResponse) -> Spec
reasoning makeRequest = do
  it "returns thinking content when enabled" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
    resp <- makeRequest req
    assertHasReasoning resp

-- | Probe: Reasoning with tools
--
-- __Tests:__ Can the model combine reasoning and tool use in one response?
--
-- __Checks:__ Response contains both thinking blocks AND tool_use blocks
--
-- __Expected to pass:__ Claude models with extended thinking + tool support
toolCallingWithReasoning :: HasCallStack => (AnthropicRequest -> IO AnthropicResponse) -> Spec
toolCallingWithReasoning makeRequest = do
  it "combines reasoning with tool calls" $ do
    let req = enableReasoning (simpleUserRequest "Think carefully, then use get_weather to check London.")
          { tools = Just [weatherTool]
          }
    resp <- makeRequest req
    assertHasReasoning resp
    assertHasToolCalls resp

-- | Probe: Adaptive reasoning (Opus 4.6+)
--
-- __Tests:__ Does the model support adaptive thinking with effort parameter?
--
-- __Checks:__ Response contains thinking content blocks when using adaptive mode
--
-- __Expected to pass:__ Opus 4.6 and newer models with adaptive thinking
adaptiveReasoning :: HasCallStack => (AnthropicRequest -> IO AnthropicResponse) -> Spec
adaptiveReasoning makeRequest = do
  it "returns thinking content when using adaptive mode" $ do
    let req = enableAdaptiveReasoning (simpleUserRequest "Think step by step: What is 17 * 29?")
    resp <- makeRequest req
    assertHasReasoning resp

-- | Probe: Consecutive user messages
--
-- __Tests:__ Does the API accept multiple user messages in a row?
--
-- __Checks:__ Request with consecutive user messages succeeds
--
-- __Expected to pass:__ Most models (though not recommended practice)
consecutiveUserMessages :: HasCallStack => (AnthropicRequest -> IO AnthropicResponse) -> Spec
consecutiveUserMessages makeRequest = do
  it "accepts consecutive user messages" $ do
    let req = Protocol.Anthropic.consecutiveUserMessages "First message" "Second message"
    resp <- makeRequest req
    assertHasAssistantText resp

-- | Probe: Starts with assistant
--
-- __Tests:__ Does the API accept history starting with assistant message?
--
-- __Checks:__ Request starting with assistant message succeeds
--
-- __Expected to pass:__ Most models (used for few-shot prompting)
startsWithAssistant :: HasCallStack => (AnthropicRequest -> IO AnthropicResponse) -> Spec
startsWithAssistant makeRequest = do
  it "accepts history starting with assistant message" $ do
    let req = Protocol.Anthropic.startsWithAssistant
    resp <- makeRequest req
    assertHasAssistantText resp
