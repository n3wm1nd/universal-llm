{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Gemini3Flash

Model test suite for Google Gemini 3 Flash Preview

This module tests Gemini 3 Flash when accessed through OpenRouter.
Gemini requires special handling of reasoning_details preservation.

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (proper tool_calls format)
✓ Reasoning (via reasoning_details)
✓ Reasoning preservation through tool call chains

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Requires reasoning_details to be preserved in conversation history
  Tool call chains break if reasoning_details is stripped

-}

module Models.Gemini3Flash (testsOpenRouter) where

import UniversalLLM.Core.Types (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import qualified TestModels
import Test.Hspec (Spec, describe)

-- | Test Gemini 3 Flash via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "Gemini 3 Flash (OpenRouter)" $ do
    describe "Protocol" $ do
      basicText provider "google/gemini-3-flash-preview"
      toolCalling provider "google/gemini-3-flash-preview"

      -- Note: acceptsToolResults fails - Gemini requires thought_signature in function calls.
      -- Fabricated tool history is rejected because it lacks the required signatures.
      -- Error: "Function call is missing a thought_signature in functionCall parts"
      -- acceptsToolResults provider "google/gemini-3-flash-preview"

      -- Note: acceptsToolResultsWithoutReasoning also fails - even with reasoning disabled,
      -- Gemini still requires thought_signature in tool calls. Cannot use fabricated history.
      -- acceptsToolResultsWithoutReasoning provider "google/gemini-3-flash-preview"

      reasoningViaDetails provider "google/gemini-3-flash-preview"
      toolCallingWithReasoning provider "google/gemini-3-flash-preview"

    describe "Standard Tests" $
      testModel TestModels.openRouterGemini3FlashPreview (Model TestModels.Gemini3FlashPreview OpenRouter) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]
