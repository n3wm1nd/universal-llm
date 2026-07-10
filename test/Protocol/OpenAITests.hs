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

Probes test __protocol and feature support__, NOT model performance:
- ✓ "Does the wire protocol return tool_calls in the response?"
- ✗ "Does our Message type conversion work?" (that's StandardTests)
- ✗ "Is the model smart enough to pick the right tool / write correct code / phrase a
  good answer?" (never - that's not our library's concern, and it isn't testable
  reliably anyway)

A probe answers "does this model/provider combination support X," not "how good is
this model at X." The two are easy to conflate because both are checked by sending a
request and looking at the response - the discipline is in what the assertion checks
for, and how the prompt is worded.

= Probes Must Be Straightforward, Not Tricky

The prompt and setup in a probe should be as easy as reasonably possible for a
__genuinely capable__ model to satisfy - a probe is not an obstacle course, and a weak
but honestly-supporting model should still pass it reliably. Don't handicap a probe with
convoluted phrasing, adversarial framing, or unnecessary constraints "to make sure it's
really working" - if the feature is supported, a plain, direct prompt should be enough.
(Composable-provider implementations, by contrast, are free and encouraged to add
compliance hints, restated schemas, etc. to get the most out of a model in practice -
see 'UniversalLLM.Providers.OpenAI.handleJSONMessageWithComplianceHint'. That's a
different concern from what a probe should test.)

At the same time, a probe __must not be guessable__: it must be impossible (or as close
to it as practical) for a model that does NOT support the feature to produce a passing
result by hallucination, pattern-matching the prompt, or coincidence. If a model can
"pass" a probe just by confabulating plausible-looking output without actually
exercising the capability being probed, the probe isn't testing anything. Concretely:

* __Vision probes__ - ask what's distinctively IN the image (a specific object, a
  count, an unusual color) rather than asking the model to describe/interpret it in
  depth. The point is to tell "the model saw something" from "the model is
  hallucinating a plausible-sounding description," not to grade how well it describes
  what it sees.
* __JSON schema probes__ - use a schema shape that has nothing to do with the natural
  phrasing of the prompt (see 'oddShapeColorsSchema' / 'jsonSchemaShapeCompliance'). A
  schema like @{"colors": [...]}@ for "list 3 colors" is guessable even by a model with
  zero real schema-following capability, because it's also the obvious freeform answer.
  A schema like @{"result": {"hue_list": [...]}}@ is not guessable - matching it is only
  possible by actually reading @response_format@.
* __Tool-calling probes__ - check that a tool call was made with recognizably-derived
  arguments, not that the tool choice or argument phrasing was "good."

= Design Guidelines for Probes

* __Focus__ - Test ONE thing. Name clearly states what.
* __Simple__ - Minimal setup. No complex logic. Prompt should be the easiest reasonable
  ask for a model that genuinely supports the feature (see above).
* __Unguessable__ - A model without the capability must not be able to pass by luck,
  hallucination, or pattern-matching the prompt (see above).
* __Direct__ - Call assertions from Protocol.OpenAI, don't implement inline
* __Clear failure__ - When it fails, immediately obvious what's missing
* __Quirk discovery__ - Variants for provider-specific behaviors (e.g., reasoningViaDetails)
* __Negative probes__ - When a quirk or missing capability is confirmed, encode it as a
  probe that asserts the quirk is still present (e.g. 'wrapsJSONInFence',
  'ignoresJSONSchemaShape') rather than leaving a probe permanently red. This keeps
  cabal test green while still flagging it automatically if the provider's behavior
  ever changes - see each probe's haddock for what "the provider fixed it" looks like.

= Usage

In a model-specific test suite:
@
import TestCache (request, ResponseProvider)
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

import TestCache (request, ResponseProvider)
import UniversalLLM.Protocols.OpenAI
import Protocol.OpenAI  -- unqualified - we mainly call these
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import Test.Hspec (Spec, describe, it, shouldSatisfy, HasCallStack, runIO)
import TestFixtures (loadImageBase64, glassbottlePng, glassbottleMirroredJpeg)

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
basicText :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
basicText makeRequest modelName = do
  it "returns assistant text for simple question" $ do
    let req = (simpleUserRequest "What is 2+2?") { model = modelName }
    resp <- request makeRequest req
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
toolCalling :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolCalling makeRequest modelName = do
  it "makes tool calls when tools are available" $ do
    let req = (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName
          , tools = Just [weatherTool]
          }
    resp <- request makeRequest req
    assertHasToolCalls resp

-- | Probe: JSON mode support
--
-- __Tests:__ Does the model return bare, valid JSON when @response_format: json_schema@
-- is requested?
--
-- __Checks:__ Response content parses directly as a JSON object (no markdown fences,
-- no preamble text)
--
-- __Expected to fail:__ Providers/models that accept the request but wrap structured
-- output in prose or code fences - use 'wrapsJSONInFence' to document that quirk
-- explicitly instead of leaving this probe red.
jsonMode :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
jsonMode makeRequest modelName = do
  it "returns bare JSON content when JSON mode requested" $ do
    let req = (jsonRequest "List 3 primary colors as a JSON object with a 'colors' array." colorsSchema)
          { model = modelName, max_tokens = Just 4096 }
    resp <- request makeRequest req
    assertHasValidJSONContent resp

-- | Probe: real JSON schema-following, not just "the model's natural answer happened
-- to look plausible"
--
-- __Tests:__ Does the backend actually enforce/follow an arbitrary @response_format@
-- schema, or does it just produce a "natural" answer to the prompt that happens to
-- satisfy an obvious schema? Uses a deliberately unusual, nested schema
-- ('oddShapeColorsSchema') that nothing in the prompt text suggests - the only way to
-- match it is to actually read @response_format@.
--
-- __No compliance hint__: this probe intentionally does NOT restate the schema in the
-- prompt text (that's a job for the composable-provider implementation, not for a
-- protocol probe - see 'UniversalLLM.Providers.OpenAI.handleJSONMessageWithComplianceHint').
-- It tests the raw protocol/backend capability in isolation.
--
-- __Expected to pass:__ Backends with real grammar-enforced structured output
-- (llama.cpp, AlibabaCloud)
--
-- __Expected to fail:__ Backends where @response_format@/@strict: true@ is accepted but
-- not actually enforced - failing here means the model has no real schema-following
-- capability at the protocol level, regardless of how well-behaved it looks on an
-- obvious schema like 'colorsSchema'.
jsonSchemaShapeCompliance :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
jsonSchemaShapeCompliance makeRequest modelName = do
  it "follows an unusual, nested response_format schema (not just a guessable shape)" $ do
    let req = (jsonRequest "List 3 primary colors, output in json." oddShapeColorsSchema)
          { model = modelName, max_tokens = Just 4096 }
    resp <- request makeRequest req
    assertMatchesOddShapeSchema resp

-- | Negative probe: the backend accepts response_format/schema but never follows it
--
-- __Tests:__ Does the backend still ignore an arbitrary, unusual @response_format@
-- schema and answer with its own guessed shape instead?
--
-- __Expected to pass:__ GLM-4.5-Air (and other GLM models) via the ZAI backend
-- (directly or via OpenRouter) - confirmed by 'jsonSchemaShapeCompliance': the model
-- returns its own natural JSON shape (e.g. a bare array), not the requested nested
-- @{"result": {"hue_list": [...]}}@ shape. This is why these models don't claim
-- 'UniversalLLM.HasJSON' - see GLM-Specific Quirks in
-- "UniversalLLM.Models.ZhipuAI.GLM".
--
-- __Expected to fail:__ If this probe fails, the backend started honoring
-- @response_format@'s schema and 'UniversalLLM.HasJSON' could potentially be
-- re-enabled for this model.
ignoresJSONSchemaShape :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
ignoresJSONSchemaShape makeRequest modelName = do
  it "ignores an unusual, nested response_format schema (known lack of JSON support)" $ do
    let req = (jsonRequest "List 3 primary colors, output in json." oddShapeColorsSchema)
          { model = modelName, max_tokens = Just 4096 }
    resp <- request makeRequest req
    let result = expectSuccess resp
    -- Should fail to match the odd schema - if it succeeds, the backend actually
    -- honored response_format and we have real JSON support to reconsider.
    (case Aeson.decode (BSL.fromStrict (TE.encodeUtf8 (T.strip (getAssistantText result)))) of
      Just (Aeson.Object obj) | Just (Aeson.Object resultObj) <- KM.lookup "result" obj
                              , Just (Aeson.Array _) <- KM.lookup "hue_list" resultObj ->
        error "Expected the odd schema to be ignored (known quirk) but it was matched - backend may have gained real JSON schema support"
      _ -> return ())

-- | Negative probe: JSON mode output arrives wrapped in a markdown code fence
--
-- __Tests:__ Does the model still fail to return bare JSON in response to
-- @response_format: json_schema@ with @strict: true@?
--
-- __Expected to pass:__ GLM-4.5-Air via OpenRouter (paid slug) - confirmed quirk,
-- worked around by 'UniversalLLM.Providers.OpenAI.openAIJSONWithFencedFallback'
--
-- __Expected to fail:__ If this probe fails, the provider started returning bare JSON
-- and the fenced-fallback workaround for this model may no longer be needed - drop it
-- and switch the model back to plain 'UniversalLLM.Providers.OpenAI.openAIJSON'.
wrapsJSONInFence :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
wrapsJSONInFence makeRequest modelName = do
  it "wraps JSON-mode output in a markdown code fence (known quirk)" $ do
    let req = (jsonRequest "List 3 primary colors as a JSON object with a 'colors' array." colorsSchema)
          { model = modelName, max_tokens = Just 4096 }
    resp <- request makeRequest req
    assertHasFencedJSONContent resp

-- | Negative probe: JSON mode output stays in reasoning_details, content never populated
--
-- __Tests:__ Does the model still leave @content: null@ and put the whole JSON answer
-- in @reasoning_details@ in response to @response_format: json_schema@?
--
-- __Expected to pass:__ GLM-4.7 and GLM-4.7-Flash via OpenRouter - confirmed quirk,
-- distinct from GLM-4.5-Air's fenced-content quirk (see 'wrapsJSONInFence'). Not caused
-- by token-budget truncation (finish_reason is "stop", verified with TEST_MODE=update
-- to rule out a one-off sampling fluke).
--
-- __Expected to fail:__ If this probe fails, the provider started populating content
-- and 'UniversalLLM.Providers.OpenAI.HasJSON' could potentially be re-enabled for this
-- model.
jsonStaysInReasoningDetails :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
jsonStaysInReasoningDetails makeRequest modelName = do
  it "leaves content empty and puts JSON-mode output in reasoning_details (known quirk)" $ do
    let req = (jsonRequest "List 3 primary colors as a JSON object with a 'colors' array." colorsSchema)
          { model = modelName, max_tokens = Just 4096 }
    resp <- request makeRequest req
    assertJSONStaysInReasoningDetails resp

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
reasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
reasoning makeRequest modelName = do
  it "returns reasoning content when enabled" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
          { model = modelName }
    resp <- request makeRequest req
    assertHasReasoningContent resp

-- | Probe: Reasoning toggle ON actually produces reasoning output
--
-- __Tests:__ Does reasoning_enabled=true actually cause the model to emit reasoning?
--
-- __Checks:__ Two requests - one with reasoning on, one off. The ON request must
-- have reasoning_content and the OFF request must not.
--
-- __Purpose:__ The field being sent correctly doesn't mean the model honours it.
-- Template-based models (llama.cpp) may ignore the reasoning field entirely and
-- always think (or never think). This probe catches that.
--
-- __Expected to pass:__ Models that genuinely toggle reasoning via this field
--
-- __Expected to fail:__ Models that always reason regardless of the flag
reasoningTogglesOn :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
reasoningTogglesOn makeRequest modelName = do
  it "reasoning_enabled=true produces reasoning_content" $ do
    let req = enableReasoning (simpleUserRequest "What is 15 * 23?") { model = modelName }
    resp <- request makeRequest req
    assertHasReasoningContent resp

-- | Probe: Reasoning toggle OFF actually suppresses reasoning output
--
-- __Tests:__ Does reasoning_enabled=false actually suppress reasoning_content?
--
-- __Checks:__ Request with reasoning disabled must have no reasoning_content
-- and no reasoning_details.
--
-- __Purpose:__ Some models (e.g. hosted models behind a provider) ignore disable
-- requests and still return reasoning output (or silently consume tokens/time
-- for hidden reasoning). Some template-based models require /nothink or an
-- empty <think></think> prefix instead.
--
-- __Expected to pass:__ Models that honour the disable flag
--
-- __Expected to fail:__ Models that always reason, or where the field has no effect
reasoningTogglesOff :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
reasoningTogglesOff makeRequest modelName = do
  it "reasoning_enabled=false suppresses reasoning_content" $ do
    let req = disableReasoning (simpleUserRequest "What is 15 * 23?") { model = modelName }
    resp <- request makeRequest req
    assertNoReasoningData resp

-- | Probe: /nothink token suppresses reasoning (Qwen3 chat template)
--
-- __Tests:__ Does appending /nothink to the user message suppress reasoning_content?
--
-- __Purpose:__ llama.cpp Qwen3 chat templates honour the /nothink token to
-- skip the <think> block. The reasoning_enabled field alone has no effect on
-- template-rendered models.
--
-- __Expected to pass:__ Qwen3 models via llama.cpp
--
-- __Expected to fail:__ Models that don't use the Qwen3 chat template
noThinkSuppressesReasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
noThinkSuppressesReasoning makeRequest modelName = do
  it "/nothink token suppresses reasoning_content" $ do
    let req = (simpleUserRequest "What is 15 * 23? /nothink") { model = modelName }
    resp <- request makeRequest req
    assertNoReasoningData resp

-- | Probe: <think></think> assistant prefill suppresses reasoning (Qwen3)
--
-- __Tests:__ Does prefilling the assistant turn with <think></think> suppress reasoning?
--
-- __Purpose:__ An alternative to /nothink: force an empty think block by
-- prefilling the assistant message. The model then continues past it without
-- generating reasoning tokens.
--
-- __Expected to pass:__ Qwen3 models via llama.cpp that support assistant prefill
--
-- __Expected to fail:__ Models that don't support prefill or ignore the tag
emptyThinkPrefillSuppressesReasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
emptyThinkPrefillSuppressesReasoning makeRequest modelName = do
  it "<think></think> assistant prefill suppresses reasoning_content" $ do
    let req = mempty
          { model = modelName
          , messages =
              [ userMessage "What is 15 * 23?"
              , assistantMessage "<think></think>"
              ]
          }
    resp <- request makeRequest req
    assertNoReasoningData resp

-- | Probe: chat_template_kwargs enable_thinking=false suppresses reasoning (llama.cpp)
--
-- __Tests:__ Does chat_template_kwargs={"enable_thinking":false} suppress reasoning_content?
--
-- __Purpose:__ llama.cpp ignores reasoning_enabled but exposes enable_thinking via
-- chat_template_kwargs. This is the documented way to disable thinking on template-based
-- models. Note: known to be silently ignored in some llama.cpp builds.
--
-- __Expected to pass:__ Qwen3 models via a llama.cpp build that honours this param
--
-- __Expected to fail:__ OpenRouter and other providers; buggy llama.cpp builds
chatTemplateKwargsDisablesThinking :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
chatTemplateKwargsDisablesThinking makeRequest modelName = do
  it "chat_template_kwargs enable_thinking=false suppresses reasoning_content" $ do
    let req = disableThinkingLlamaCpp (simpleUserRequest "What is 15 * 23?") { model = modelName }
    resp <- request makeRequest req
    assertNoReasoningData resp

-- | Probe: Empty reasoning prefill suppresses further reasoning (Qwen3)
--
-- __Tests:__ Does sending an AssistantReasoning "" message before the final
-- turn suppress reasoning output?
--
-- __Purpose:__ If the model sees a prior (empty) reasoning turn for this
-- question, it may skip generating a new <think> block and proceed directly
-- to the answer. This is the semantic equivalent of <think></think> prefill
-- but expressed through our Message abstraction rather than raw content.
--
-- __Expected to pass:__ Models that treat a prior reasoning message as
-- "already thought, just answer now"
--
-- __Expected to fail:__ Models that ignore the reasoning history or always
-- re-reason regardless
emptyReasoningPrefillSuppressesReasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
emptyReasoningPrefillSuppressesReasoning makeRequest modelName = do
  it "empty AssistantReasoning prefill suppresses reasoning_content" $ do
    let req = mempty
          { model = modelName
          , messages =
              [ userMessage "What is 15 * 23?"
              , emptyMessage { role = "assistant", reasoning_content = Just "" }
              ]
          }
    resp <- request makeRequest req
    assertNoReasoningData resp

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
reasoningViaDetails :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
reasoningViaDetails makeRequest modelName = do
  it "returns reasoning in reasoning_details field" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
          { model = modelName }
    resp <- request makeRequest req
    assertHasReasoningDetails resp

-- | Probe: Hidden reasoning (accepts config, no response fields)
--
-- __Tests:__ Does the model accept reasoning config but not expose reasoning?
--
-- __Checks:__ Request with reasoning succeeds, but response has neither
-- reasoning_content nor reasoning_details
--
-- __Expected to pass:__ Models with internal/hidden reasoning (e.g., Kimi K2.5
-- via AlibabaCloud)
--
-- __Expected to fail:__ Models that expose reasoning in responses
--
-- __Model behavior:__ Some models accept reasoning parameters and use them
-- internally to improve response quality, but don't expose the reasoning
-- process in the API response. This is "hidden reasoning" - the model thinks
-- but doesn't show its work.
acceptsHiddenReasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsHiddenReasoning makeRequest modelName = do
  it "accepts reasoning config without returning reasoning data" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
          { model = modelName }
    resp <- request makeRequest req
    -- Should succeed (not error)
    assertHasAssistantText resp
    -- Should NOT have reasoning_content or reasoning_details
    assertNoReasoningData resp

-- | Probe: Encrypted reasoning (reasoning_details with encrypted data)
--
-- __Tests:__ Does the model return encrypted reasoning in reasoning_details?
--
-- __Checks:__ Request with reasoning succeeds, response has reasoning_details
-- with encrypted data (type: "reasoning.encrypted")
--
-- __Expected to pass:__ Models with encrypted reasoning (e.g., GPT models via
-- OpenRouter, which return reasoning.encrypted type)
--
-- __Expected to fail:__ Models with readable reasoning or no reasoning
--
-- __Model behavior:__ Some models return reasoning_details but the content
-- is encrypted/signed by the provider. The field is present for round-tripping
-- in conversation history but not readable as tokens.
encryptedReasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
encryptedReasoning makeRequest modelName = do
  it "returns encrypted reasoning in reasoning_details field" $ do
    let req = enableReasoning (simpleUserRequest "Think step by step: What is 15 * 23?")
          { model = modelName }
    resp <- request makeRequest req
    -- Should succeed with assistant text
    assertHasAssistantText resp
    -- Should have reasoning_details with encrypted data
    assertHasEncryptedReasoning resp

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
toolCallingViaXML :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolCallingViaXML makeRequest modelName = do
  it "returns tool calls as XML in content field" $ do
    let req = (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName
          , tools = Just [weatherTool]
          }
    resp <- request makeRequest req
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
acceptsToolResults :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsToolResults makeRequest modelName = do
  it "accepts tool results in conversation history" $ do
    let req = requestWithToolCallHistory { model = modelName }
    resp <- request makeRequest req
    assertHasAssistantText resp

-- | Probe: Tool result follow-up completes within a modest token budget
--
-- __Tests:__ Does the model still produce assistant text after a tool result when
-- @max_tokens@ is capped (800), rather than left unbounded?
--
-- __Checks:__ Response contains non-empty assistant text
--
-- __Expected to pass:__ Non-reasoning models, or reasoning models with cheap reasoning
--
-- __Expected to fail:__ Reasoning models that spend the entire token budget on
-- reasoning_content/reasoning_details before emitting any content - the response comes
-- back with content: null. This is a capacity problem (budget too small for this model's
-- reasoning verbosity), not a parsing bug - see 'jsonMode' for the same failure mode in
-- JSON output.
acceptsToolResultsUnderTightBudget :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsToolResultsUnderTightBudget makeRequest modelName = do
  it "produces assistant text after tool result within an 800 token budget" $ do
    let req = requestWithToolCallHistory { model = modelName, max_tokens = Just 800 }
    resp <- request makeRequest req
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
acceptsToolResultsWithoutReasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsToolResultsWithoutReasoning makeRequest modelName = do
  it "accepts tool results with reasoning disabled" $ do
    let req = disableReasoning requestWithToolCallHistory { model = modelName }
    resp <- request makeRequest req
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
toolCallingWithReasoning :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolCallingWithReasoning makeRequest modelName = do
  it "preserves reasoning through tool call chains" $ do
    -- Step 1: Get model to make a tool call with reasoning
    let req1 = enableReasoning (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName
          , tools = Just [weatherTool]
          }
    resp1 <- request makeRequest req1

    -- Verify it has reasoning and tool calls
    assertHasReasoningDetails resp1

    -- Step 2: Build request with tool result (helper preserves assistant message)
    let req2 = (requestWithToolResult resp1) { model = modelName }
    resp2 <- request makeRequest req2

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
consecutiveUserMessages :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
consecutiveUserMessages makeRequest modelName = do
  it "accepts consecutive user messages" $ do
    let req = (Protocol.OpenAI.consecutiveUserMessages "Here is some context." "Now answer this question: what is 2+2?")
          { model = modelName }
    resp <- request makeRequest req
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
startsWithAssistant :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
startsWithAssistant makeRequest modelName = do
  it "accepts history starting with assistant message" $ do
    let req = Protocol.OpenAI.startsWithAssistant { model = modelName }
    resp <- request makeRequest req
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
systemMessageMidConversation :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
systemMessageMidConversation makeRequest modelName = do
  it "accepts system message mid-conversation" $ do
    let req = requestWithSystemMidConversation { model = modelName }
    resp <- request makeRequest req
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
systemMessageAtStart :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
systemMessageAtStart makeRequest modelName = do
  it "accepts system message at start" $ do
    let req = requestWithSystemAtStart { model = modelName }
    resp <- request makeRequest req
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
multipleSystemMessages :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
multipleSystemMessages makeRequest modelName = do
  it "accepts multiple system messages" $ do
    let req = requestWithMultipleSystemMessages { model = modelName }
    resp <- request makeRequest req
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
acceptsToolResultNoTools :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsToolResultNoTools makeRequest modelName = do
  it "accepts tool result when no tools defined" $ do
    let req = requestWithToolResultNoTools { model = modelName }
    resp <- request makeRequest req
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
acceptsToolResultToolGone :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsToolResultToolGone makeRequest modelName = do
  it "accepts tool result when called tool no longer available" $ do
    let req = requestWithToolResultToolGone { model = modelName }
    resp <- request makeRequest req
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
acceptsStaleToolInHistory :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsStaleToolInHistory makeRequest modelName = do
  it "accepts tool call in history when tool no longer available" $ do
    let req = requestWithStaleToolInHistory { model = modelName }
    resp <- request makeRequest req
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
acceptsOldToolCallStillAvailable :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsOldToolCallStillAvailable makeRequest modelName = do
  it "accepts old tool call in history with tool still available" $ do
    let req = requestWithOldToolCallStillAvailable { model = modelName }
    resp <- request makeRequest req
    assertHasAssistantText resp

-- | Probe: Tools + reasoning_enabled=true (no effort, no exclude)
--
-- __Tests:__ Does the model accept tools alongside minimal reasoning config?
--
-- __Checks:__ Response succeeds (no 500)
--
-- __Use when:__ Investigating why tools+reasoning fails - isolates whether
-- reasoning_enabled alone triggers the issue
toolsWithReasoningEnabled :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolsWithReasoningEnabled makeRequest modelName = do
  it "accepts tools with reasoning_enabled=true (no effort/exclude fields)" $ do
    let req = enableReasoning (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName, tools = Just [weatherTool] }
    resp <- request makeRequest req
    wasSuccessful resp

-- | Probe: Tools + full reasoning config (enabled=true, effort=low, exclude=false)
--
-- __Tests:__ Does the model accept tools alongside the full reasoning config
-- that openAIReasoning sends?
--
-- __Checks:__ Response succeeds (no 500)
--
-- __Use when:__ Investigating which specific reasoning field causes crashes
toolsWithReasoningFull :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolsWithReasoningFull makeRequest modelName = do
  it "accepts tools with full reasoning config (enabled+effort+exclude)" $ do
    let req = enableReasoningFull (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName, tools = Just [weatherTool] }
    resp <- request makeRequest req
    wasSuccessful resp

-- | Probe: Tools + effort field only (no enabled flag)
--
-- __Tests:__ Does the model accept tools alongside reasoning effort without enabled flag?
--
-- __Checks:__ Response succeeds (no 500)
--
-- __Use when:__ Isolating whether reasoning_effort alone (without enabled=true) triggers crashes
toolsWithReasoningEffortOnly :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolsWithReasoningEffortOnly makeRequest modelName = do
  it "accepts tools with reasoning effort field only (no enabled flag)" $ do
    let req = enableReasoningEffortOnly (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName, tools = Just [weatherTool] }
    resp <- request makeRequest req
    wasSuccessful resp

-- | Probe: Tools + reasoning with list_files tool (same as ST.reasoningWithTools)
--
-- __Tests:__ Does the model accept tools+reasoning with the exact same tool def
-- and prompt used in ST.reasoningWithTools?
--
-- __Checks:__ Response succeeds (no 500)
--
-- __Use when:__ Isolating whether the list_files tool schema or the specific
-- prompt triggers the chat template crash seen in ST.reasoningWithTools
toolsWithReasoningListFiles :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolsWithReasoningListFiles makeRequest modelName = do
  it "accepts tools+reasoning with list_files tool and reasoning prompt" $ do
    let listFilesTool = simpleTool "list_files" "List files matching a pattern" []
        req = enableReasoningFull
                (simpleUserRequest "Think carefully: What files match the pattern '*.md'? Use the available tools.")
                  { model = modelName, tools = Just [listFilesTool] }
    resp <- request makeRequest req
    wasSuccessful resp

-- | Probe: Tools + full reasoning config + max_tokens (exact ST.reasoningWithTools shape)
--
-- __Tests:__ Does adding max_tokens to the tools+reasoning request trigger the 500?
--
-- __Use when:__ Isolating whether max_tokens is what triggers the chat template crash
toolsWithReasoningAndMaxTokens :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolsWithReasoningAndMaxTokens makeRequest modelName = do
  it "accepts tools+reasoning+max_tokens (exact shape of ST.reasoningWithTools)" $ do
    let listFilesTool = simpleTool "list_files" "List files matching a pattern" []
        req = enableReasoningFull
                (simpleUserRequest "Think carefully: What files match the pattern '*.md'? Use the available tools.")
                  { model = modelName
                  , tools = Just [listFilesTool]
                  , max_tokens = Just 4096
                  }
    resp <- request makeRequest req
    wasSuccessful resp

-- | Probe: Tools + reasoning_enabled=false
--
-- __Tests:__ Does explicitly disabling reasoning while providing tools work?
--
-- __Checks:__ Response succeeds and contains tool calls or assistant text
--
-- __Use when:__ Checking if the reasoning field presence itself (regardless of value)
-- is what triggers crashes
toolsWithReasoningDisabled :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
toolsWithReasoningDisabled makeRequest modelName = do
  it "accepts tools with reasoning explicitly disabled" $ do
    let req = disableReasoning (simpleUserRequest "Use the get_weather function to check the weather in London.")
          { model = modelName, tools = Just [weatherTool] }
    resp <- request makeRequest req
    wasSuccessful resp

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
providerErrorResponse :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Spec
providerErrorResponse makeRequest = do
  it "returns well-formed error response for invalid model" $ do
    let req = (simpleUserRequest "What is 2+2?") { model = "invalid-model-name-that-does-not-exist" }
    resp <- request makeRequest req
    assertIsProviderError resp

-- | Probe: Vision / PNG format support
--
-- __Tests:__ Can the model accept a PNG image and identify its subject?
--
-- __Checks:__ Response mentions "bottle" when shown the glassbottle image
--
-- __Expected to pass:__ Vision-capable models
--
-- __Expected to fail:__ Text-only models
visionPng :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
visionPng makeRequest modelName = do
  (mediaType, b64Data) <- runIO glassbottlePng
  it "accepts PNG and identifies image subject" $ do
    let req = visionIdentifyRequest modelName mediaType b64Data
    resp <- request makeRequest req
    assertMentions "bottle" resp

-- | Probe: Vision / JPEG format support
--
-- __Tests:__ Can the model accept a JPEG image and identify its subject?
--
-- __Checks:__ Response mentions "bottle" when shown the mirrored glassbottle JPEG
--
-- __Expected to pass:__ Vision-capable models that support JPEG
--
-- __Expected to fail:__ Text-only models or models limited to PNG
visionJpeg :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
visionJpeg makeRequest modelName = do
  (mediaType, b64Data) <- runIO glassbottleMirroredJpeg
  it "accepts JPEG and identifies image subject" $ do
    let req = visionIdentifyRequest modelName mediaType b64Data
    resp <- request makeRequest req
    assertMentions "bottle" resp

-- | Probe: Vision / content accuracy (no hallucination)
--
-- __Tests:__ Does the model accurately describe image content without hallucinating?
--
-- __Checks:__ Response mentions "bottle", does NOT mention "cat"
--
-- __Expected to pass:__ Vision-capable models with accurate perception
--
-- __Expected to fail:__ Text-only models, or models that hallucinate common objects
visionAccuracy :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
visionAccuracy makeRequest modelName = do
  (mediaType, b64Data) <- runIO glassbottlePng
  it "identifies what is in the image and does not hallucinate" $ do
    let req = visionIdentifyRequest modelName mediaType b64Data
    resp <- request makeRequest req
    assertMentions "bottle" resp
    assertDoesNotMention "cat" resp

-- | Probe: Vision / multiple images in one prompt
--
-- __Tests:__ Can the model receive two images in a single message and compare them?
--
-- __Checks:__ Model confirms that the second image is a mirrored version of the first
--
-- __Expected to pass:__ Vision models with multi-image support
--
-- __Expected to fail:__ Models limited to a single image per message
visionMultipleImages :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
visionMultipleImages makeRequest modelName = do
  (mt1, b64Png) <- runIO glassbottlePng
  (mt2, b64Jpg) <- runIO glassbottleMirroredJpeg
  it "compares two images and identifies mirroring" $ do
    let req = visionCompareRequest modelName mt1 b64Png mt2 b64Jpg
    resp <- request makeRequest req
    assertConfirmsYes resp

-- ============================================================================
-- Negative Probes
--
-- These probes verify that certain requests are REJECTED by the model/provider
-- with a specific, known error. Passing means the rejection is still happening
-- as expected. Failing means the API behaviour changed — a workaround may no
-- longer be needed, or a new kind of failure appeared.
--
-- IsExpected predicates are precise: they match the exact error shape observed,
-- so transient errors (rate limits, server outages) are never cached.
-- ============================================================================

-- | True if the response is the Qwen3.5 chat template system-message rejection:
-- HTTP 500, server_error, code 500, "System message must be at the beginning" in the body.
isQwenSystemMessageRejection :: Int -> OpenAIResponse -> Bool
isQwenSystemMessageRejection _ (OpenAISuccess _) = False
isQwenSystemMessageRejection sc (OpenAIError err) =
  let d = getErrorDetail err
  in sc == 500
  && code d == 500
  && errorType d == Just "server_error"
  && "System message must be at the beginning" `T.isInfixOf` errorMessage d

-- | True if the response is the llama.cpp think-prefill rejection:
-- HTTP 400, invalid_request_error, "prefill is incompatible with enable_thinking".
isLlamaCppThinkPrefillRejection :: Int -> OpenAIResponse -> Bool
isLlamaCppThinkPrefillRejection sc (OpenAIError err) =
  let d = getErrorDetail err
  in sc == 400
  && code d == 400
  && errorType d == Just "invalid_request_error"
  && "prefill is incompatible with enable_thinking" `T.isInfixOf` errorMessage d
isLlamaCppThinkPrefillRejection _ _ = False

-- | True if the response is the llama.cpp empty-assistant-message rejection:
-- HTTP 400, invalid_request_error, "must contain either 'content' or 'tool_calls'".
isLlamaCppEmptyAssistantRejection :: Int -> OpenAIResponse -> Bool
isLlamaCppEmptyAssistantRejection sc (OpenAIError err) =
  let d = getErrorDetail err
  in sc == 400
  && code d == 400
  && errorType d == Just "invalid_request_error"
  && "must contain either 'content' or 'tool_calls'" `T.isInfixOf` errorMessage d
isLlamaCppEmptyAssistantRejection _ _ = False

-- | Negative probe: system message mid-conversation is rejected
--
-- __Tests:__ Does the model reject a system message after user/assistant turns?
--
-- __Expected to pass:__ Qwen3.5 via llama.cpp (chat template constraint)
--
-- __Expected to fail:__ If this probe fails, the systemMessagesFirst workaround may no longer be needed.
rejectsSystemMessageMidConversation :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
rejectsSystemMessageMidConversation makeRequest modelName = do
  it "rejects system message mid-conversation (chat template constraint)" $ do
    let req = requestWithSystemMidConversation { model = modelName }
    resp <- makeRequest (\sc r -> isQwenSystemMessageRejection sc r) req
    resp `shouldSatisfy` \r -> case r of
      OpenAIError err -> "System message must be at the beginning" `T.isInfixOf` errorMessage (getErrorDetail err)
      OpenAISuccess _ -> False

-- | Negative probe: multiple system messages are rejected
--
-- __Tests:__ Does the model reject more than one system message?
--
-- __Expected to pass:__ Qwen3.5 via llama.cpp (chat template constraint)
--
-- __Expected to fail:__ If this probe fails, the mergeSystemMessages workaround may no longer be needed.
rejectsMultipleSystemMessages :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
rejectsMultipleSystemMessages makeRequest modelName = do
  it "rejects multiple system messages (chat template constraint)" $ do
    let req = requestWithMultipleSystemMessages { model = modelName }
    resp <- makeRequest (\sc r -> isQwenSystemMessageRejection sc r) req
    resp `shouldSatisfy` \r -> case r of
      OpenAIError err -> "System message must be at the beginning" `T.isInfixOf` errorMessage (getErrorDetail err)
      OpenAISuccess _ -> False

-- | Negative probe: reasoning_enabled=false does not suppress reasoning
--
-- __Tests:__ Does the model ignore reasoning_enabled=false and still return reasoning?
--
-- __Expected to pass:__ llama.cpp models that ignore the reasoning_enabled field
--
-- __Expected to fail:__ If this probe fails, the model now honours reasoning_enabled=false.
ignoresReasoningDisable :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
ignoresReasoningDisable makeRequest modelName = do
  it "ignores reasoning_enabled=false (still returns reasoning_content)" $ do
    let req = disableReasoning (simpleUserRequest "What is 15 * 23?") { model = modelName }
    resp <- request makeRequest req
    assertHasReasoningContent resp

-- | Negative probe: /nothink token does not suppress reasoning
--
-- __Tests:__ Does the model ignore the /nothink token and still return reasoning?
--
-- __Expected to pass:__ llama.cpp builds where /nothink is stripped before the template
--
-- __Expected to fail:__ If this probe fails, the build now honours /nothink.
ignoresNoThink :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
ignoresNoThink makeRequest modelName = do
  it "ignores /nothink token (still returns reasoning_content)" $ do
    let req = (simpleUserRequest "What is 15 * 23? /nothink") { model = modelName }
    resp <- request makeRequest req
    assertHasReasoningContent resp

-- | Negative probe: <think></think> assistant prefill is rejected
--
-- __Tests:__ Does the model reject an assistant prefill of <think></think>?
--
-- __Expected to pass:__ llama.cpp where prefill is incompatible with enable_thinking
--
-- __Expected to fail:__ If this probe fails, the model now accepts this prefill.
rejectsEmptyThinkPrefill :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
rejectsEmptyThinkPrefill makeRequest modelName = do
  it "rejects <think></think> assistant prefill (incompatible with enable_thinking)" $ do
    let req = mempty
          { model = modelName
          , messages = [ userMessage "What is 15 * 23?", assistantMessage "<think></think>" ]
          }
    resp <- makeRequest (\sc r -> isLlamaCppThinkPrefillRejection sc r) req
    resp `shouldSatisfy` \r -> case r of
      OpenAIError err -> "prefill is incompatible with enable_thinking" `T.isInfixOf` errorMessage (getErrorDetail err)
      OpenAISuccess _ -> False

-- | Negative probe: empty reasoning_content prefill is rejected
--
-- __Tests:__ Does the model reject an assistant message with only reasoning_content = ""?
--
-- __Expected to pass:__ llama.cpp where assistant messages require content or tool_calls
--
-- __Expected to fail:__ If this probe fails, the model now accepts empty reasoning prefill.
rejectsEmptyReasoningPrefill :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
rejectsEmptyReasoningPrefill makeRequest modelName = do
  it "rejects empty reasoning prefill (assistant message needs content/tool_calls)" $ do
    let req = mempty
          { model = modelName
          , messages = [ userMessage "What is 15 * 23?", emptyMessage { role = "assistant", reasoning_content = Just "" } ]
          }
    resp <- makeRequest (\sc r -> isLlamaCppEmptyAssistantRejection sc r) req
    resp `shouldSatisfy` \r -> case r of
      OpenAIError err -> "must contain either 'content' or 'tool_calls'" `T.isInfixOf` errorMessage (getErrorDetail err)
      OpenAISuccess _ -> False

-- | Negative probe: tool result in history without toolConfig is rejected (Nova)
--
-- __Tests:__ Does Nova reject fabricated tool history when no tools are provided?
--
-- __Checks:__ Request with tool call + result but tools=Nothing is rejected
--
-- __Expected to pass:__ Amazon Nova via OpenRouter/Bedrock (toolConfig required)
--
-- __Expected to fail:__ If this probe fails, Nova no longer requires toolConfig
rejectsToolResultWithoutToolConfig :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
rejectsToolResultWithoutToolConfig makeRequest modelName = do
  it "rejects tool result in history when toolConfig field is missing" $ do
    let req = requestWithToolResultNoTools { model = modelName }
    resp <- makeRequest (\_ _ -> False) req
    resp `shouldSatisfy` \r -> case r of
      OpenAIError _ -> True
      OpenAISuccess _ -> False

-- | Probe: Accepts fabricated tool call history
--
-- __Tests:__ Does the model accept a fabricated tool call + result in history?
--
-- __Checks:__ Response succeeds with full tool call + result + matching tools present
--
-- __Expected to pass:__ Nova, Gemini (both accept fabricated tool history)
--
-- __Note:__ This was previously thought to require thought_signature (Gemini) or toolConfig
-- (Nova), but testing shows both accept fabricated history when tools are provided.
acceptsFabricatedToolHistory :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
acceptsFabricatedToolHistory makeRequest modelName = do
  it "accepts fabricated tool call history" $ do
    let req = requestWithToolCallHistory { model = modelName }
    resp <- request makeRequest req
    assertHasAssistantText resp
