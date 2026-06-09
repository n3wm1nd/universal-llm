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
import TestCache (request, ResponseProvider)
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

import TestCache (request, ResponseProvider)
import UniversalLLM.Protocols.Anthropic
import Protocol.Anthropic  -- unqualified - we mainly call these
import Data.Text (Text)
import Test.Hspec (Spec, describe, it, HasCallStack, runIO)
import TestFixtures (glassbottlePng, glassbottleMirroredJpeg)

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
basicText :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Spec
basicText makeRequest = do
  it "returns assistant text for simple question" $ do
    let req = simpleUserRequest "What is 2+2?"
    resp <- request makeRequest req
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
toolCalling :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Spec
toolCalling makeRequest = do
  it "makes tool calls when tools are available" $ do
    let req = (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { tools = Just [weatherTool]
          }
    resp <- request makeRequest req
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
reasoning :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Spec
reasoning makeRequest = do
  it "returns thinking content when enabled" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
    resp <- request makeRequest req
    assertHasReasoning resp

-- | Probe: Reasoning with tools
--
-- __Tests:__ Can the model combine reasoning and tool use in one response?
--
-- __Checks:__ Response contains both thinking blocks AND tool_use blocks
--
-- __Expected to pass:__ Claude models with extended thinking + tool support
toolCallingWithReasoning :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Spec
toolCallingWithReasoning makeRequest = do
  it "combines reasoning with tool calls" $ do
    let req = enableReasoning (simpleUserRequest "Think carefully, then use get_weather to check London.")
          { tools = Just [weatherTool]
          }
    resp <- request makeRequest req
    assertHasReasoning resp
    assertHasToolCalls resp

-- | Probe: Adaptive reasoning (Opus 4.6+)
--
-- __Tests:__ Does the model support adaptive thinking with effort parameter?
--
-- __Checks:__ Response contains thinking content blocks when using adaptive mode
--
-- __Expected to pass:__ Opus 4.6 and newer models with adaptive thinking
adaptiveReasoning :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Spec
adaptiveReasoning makeRequest = do
  it "returns thinking content when using adaptive mode" $ do
    let req = enableAdaptiveReasoning (simpleUserRequest "Think step by step: What is 17 * 29?")
    resp <- request makeRequest req
    assertHasReasoning resp

-- | Probe: Vision / PNG format support
--
-- __Tests:__ Can the model accept a PNG image and identify its subject?
--
-- __Checks:__ Response mentions "bottle" when shown the glassbottle image
--
-- __Expected to pass:__ All Claude models (all support vision natively)
visionPng :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Text -> Spec
visionPng makeRequest modelName = do
  (mediaType, b64Data) <- runIO glassbottlePng
  it "accepts PNG image and identifies subject" $ do
    let req = visionIdentifyRequest modelName mediaType b64Data
    resp <- request makeRequest req
    assertMentions "bottle" resp

-- | Probe: Vision / JPEG format support
--
-- __Tests:__ Can the model accept a JPEG image and identify its subject?
--
-- __Checks:__ Response mentions "bottle" when shown the mirrored glassbottle JPEG
--
-- __Expected to pass:__ All Claude models
visionJpeg :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Text -> Spec
visionJpeg makeRequest modelName = do
  (mediaType, b64Data) <- runIO glassbottleMirroredJpeg
  it "accepts JPEG image and identifies subject" $ do
    let req = visionIdentifyRequest modelName mediaType b64Data
    resp <- request makeRequest req
    assertMentions "bottle" resp

-- | Probe: Vision / multiple images in one prompt
--
-- __Tests:__ Can the model receive two images in a single message and compare them?
--
-- __Checks:__ Model confirms the second image is a mirrored version of the first
--
-- __Expected to pass:__ All Claude models
visionMultipleImages :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Text -> Spec
visionMultipleImages makeRequest modelName = do
  (mt1, b64Png) <- runIO glassbottlePng
  (mt2, b64Jpg) <- runIO glassbottleMirroredJpeg
  it "accepts multiple images and compares them" $ do
    let req = visionCompareRequest modelName mt1 b64Png mt2 b64Jpg
    resp <- request makeRequest req
    assertConfirmsYes resp

-- | Probe: Vision / multiple images with disambiguation hint
--
-- __Tests:__ Can the model compare two images when told they are either mirrored or completely different?
--
-- __Checks:__ Model confirms the second image is a mirrored version of the first
--
-- __Expected to pass:__ All Claude models (more reliable than visionMultipleImages)
visionMultipleImagesHinted :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Text -> Spec
visionMultipleImagesHinted makeRequest modelName = do
  (mt1, b64Png) <- runIO glassbottlePng
  (mt2, b64Jpg) <- runIO glassbottleMirroredJpeg
  it "accepts multiple images and compares them (with disambiguation hint)" $ do
    let req = visionCompareRequestHinted modelName mt1 b64Png mt2 b64Jpg
    resp <- request makeRequest req
    assertConfirmsYes resp

-- | Probe: Consecutive user messages
--
-- __Tests:__ Does the API accept multiple user messages in a row?
--
-- __Checks:__ Request with consecutive user messages succeeds
--
-- __Expected to pass:__ Most models (though not recommended practice)
consecutiveUserMessages :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Spec
consecutiveUserMessages makeRequest = do
  it "accepts consecutive user messages" $ do
    let req = Protocol.Anthropic.consecutiveUserMessages "First message" "Second message"
    resp <- request makeRequest req
    assertHasAssistantText resp

-- | Probe: Starts with assistant
--
-- __Tests:__ Does the API accept history starting with assistant message?
--
-- __Checks:__ Request starting with assistant message succeeds
--
-- __Expected to pass:__ Most models (used for few-shot prompting)
startsWithAssistant :: HasCallStack => ResponseProvider AnthropicRequest AnthropicResponse -> Spec
startsWithAssistant makeRequest = do
  it "accepts history starting with assistant message" $ do
    let req = Protocol.Anthropic.startsWithAssistant
    resp <- request makeRequest req
    assertHasAssistantText resp
