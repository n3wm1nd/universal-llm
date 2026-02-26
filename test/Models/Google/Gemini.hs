{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Google.Gemini

Model test suite for Google Gemini models

This module tests Gemini models when accessed through OpenRouter.
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
  Requires thought_signature in function calls (cannot use fabricated tool history)

-}

module Models.Google.Gemini (testsGemini3FlashOpenRouter, testsGemini3ProOpenRouter) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.Google.Gemini
  ( Gemini3FlashPreview(..)
  , Gemini3ProPreview(..)
  , gemini3FlashPreview
  , gemini3ProPreview
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)

-- | Test Gemini 3 Flash via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGemini3FlashOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGemini3FlashOpenRouter provider = do
  describe "Gemini 3 Flash via OpenRouter" $ do
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

      -- Note: All tool result/history probes would fail - requires thought_signature
      -- acceptsToolResultNoTools provider "google/gemini-3-flash-preview"
      -- acceptsToolResultToolGone provider "google/gemini-3-flash-preview"
      -- acceptsStaleToolInHistory provider "google/gemini-3-flash-preview"
      -- acceptsOldToolCallStillAvailable provider "google/gemini-3-flash-preview"

      consecutiveUserMessages provider "google/gemini-3-flash-preview"
      startsWithAssistant provider "google/gemini-3-flash-preview"
      reasoningViaDetails provider "google/gemini-3-flash-preview"
      toolCallingWithReasoning provider "google/gemini-3-flash-preview"

    describe "Standard Tests" $
      testModel gemini3FlashPreview (Model Gemini3FlashPreview OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.openAIReasoningDetailsPreservation ]

-- | Test Gemini 3 Pro via OpenRouter
--
-- Same protocol quirks as Flash (thought_signature, reasoning_details).
-- More capable but slower; standard tests only (protocol behaviour identical to Flash).
testsGemini3ProOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGemini3ProOpenRouter provider = do
  describe "Gemini 3 Pro via OpenRouter" $ do
    describe "Standard Tests" $
      testModel gemini3ProPreview (Model Gemini3ProPreview OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]
