{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Moonshot.KimiK25

Model test suite for Moonshot AI Kimi K2.5

This module tests Kimi K2.5 when accessed through OpenRouter.

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (native tool_calls format)
✓ Reasoning (via reasoning_details, OpenRouter style)
✓ Reasoning preservation through tool call chains

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Requires reasoning_details to be preserved in conversation history

-}

module Models.Moonshot.KimiK25 (testsOpenRouter) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.Moonshot.Kimi
  ( KimiK25(..)
  , kimiK25
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)

-- | Test Kimi K2.5 via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "Moonshot AI Kimi K2.5 via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "moonshotai/kimi-k2.5"
      toolCalling provider "moonshotai/kimi-k2.5"
      acceptsToolResults provider "moonshotai/kimi-k2.5"
      acceptsToolResultNoTools provider "moonshotai/kimi-k2.5"
      acceptsToolResultToolGone provider "moonshotai/kimi-k2.5"
      acceptsStaleToolInHistory provider "moonshotai/kimi-k2.5"
      acceptsOldToolCallStillAvailable provider "moonshotai/kimi-k2.5"
      consecutiveUserMessages provider "moonshotai/kimi-k2.5"
      startsWithAssistant provider "moonshotai/kimi-k2.5"
      -- Note: reasoning probe fails - Kimi K2.5 uses reasoning_details (OpenRouter style),
      -- not the standard reasoning_content field.
      -- reasoning provider "moonshotai/kimi-k2.5"
      reasoningViaDetails provider "moonshotai/kimi-k2.5"
      toolCallingWithReasoning provider "moonshotai/kimi-k2.5"

    describe "Standard Tests" $
      testModel kimiK25 (Model KimiK25 OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]
