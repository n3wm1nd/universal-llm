{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.OpenAI.GPT54

Model test suites for GPT-5.4 family models via OpenRouter.

= Models Covered

* GPT-5.4-Pro - Advanced capabilities
* GPT-5.4 - Latest generation

= Expected Capabilities

All models are expected to support:
✓ Basic text responses
✓ Tool calling (native tool_calls format)
✓ Reasoning (via reasoning_details on OpenRouter)
✓ System messages
✓ Multi-turn conversations

= Provider-Specific Notes

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Proper tool_calls field support

-}

module Models.OpenAI.GPT54
  ( testsGPT54ProOpenRouter
  , testsGPT54OpenRouter
  ) where

import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.OpenAI.GPT (GPT54Pro(..), GPT54(..))
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)

-- | Test GPT-5.4-Pro via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT54ProOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT54ProOpenRouter provider = do
  describe "GPT-5.4-Pro via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.4-pro"
      toolCalling provider "openai/gpt-5.4-pro"
      acceptsToolResults provider "openai/gpt-5.4-pro"
      acceptsToolResultNoTools provider "openai/gpt-5.4-pro"
      acceptsToolResultToolGone provider "openai/gpt-5.4-pro"
      acceptsStaleToolInHistory provider "openai/gpt-5.4-pro"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.4-pro"
      consecutiveUserMessages provider "openai/gpt-5.4-pro"
      startsWithAssistant provider "openai/gpt-5.4-pro"
      encryptedReasoning provider "openai/gpt-5.4-pro"

    describe "Standard Tests" $
      testModel route (GPT54Pro `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.hiddenReasoning ]

-- | Test GPT-5.4 via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT54OpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT54OpenRouter provider = do
  describe "GPT-5.4 via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.4"
      toolCalling provider "openai/gpt-5.4"
      acceptsToolResults provider "openai/gpt-5.4"
      acceptsToolResultNoTools provider "openai/gpt-5.4"
      acceptsToolResultToolGone provider "openai/gpt-5.4"
      acceptsStaleToolInHistory provider "openai/gpt-5.4"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.4"
      consecutiveUserMessages provider "openai/gpt-5.4"
      startsWithAssistant provider "openai/gpt-5.4"
      encryptedReasoning provider "openai/gpt-5.4"

    describe "Standard Tests" $
      testModel route (GPT54 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.hiddenReasoning ]
