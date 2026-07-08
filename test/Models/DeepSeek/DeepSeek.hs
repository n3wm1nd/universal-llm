{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.DeepSeek.DeepSeek

Test suites for DeepSeek's V4 model family.

= Models Covered

* DeepSeek V4 Flash (OpenRouter)
* DeepSeek V4 Pro (OpenRouter)

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Proper tool_calls field support

-}

module Models.DeepSeek.DeepSeek
  ( testsDeepSeekV4FlashOpenRouter
  , testsDeepSeekV4ProOpenRouter
  ) where

import UniversalLLM (via, route)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.DeepSeek.DeepSeek (DeepSeekV4Flash(..), DeepSeekV4Pro(..))
import Protocol.OpenAITests
import qualified StandardTests as ST
import qualified ComposableProviderTests as CPT
import TestCache (ResponseProvider)
import TestHelpers (testModel, testModelOffline)
import Test.Hspec (Spec, describe)

-- | Test DeepSeek V4 Flash via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsDeepSeekV4FlashOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsDeepSeekV4FlashOpenRouter provider = do
  describe "DeepSeek V4 Flash via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "deepseek/deepseek-v4-flash"
      toolCalling provider "deepseek/deepseek-v4-flash"
      acceptsToolResults provider "deepseek/deepseek-v4-flash"
      acceptsToolResultNoTools provider "deepseek/deepseek-v4-flash"
      acceptsToolResultToolGone provider "deepseek/deepseek-v4-flash"
      acceptsStaleToolInHistory provider "deepseek/deepseek-v4-flash"
      acceptsOldToolCallStillAvailable provider "deepseek/deepseek-v4-flash"
      consecutiveUserMessages provider "deepseek/deepseek-v4-flash"
      startsWithAssistant provider "deepseek/deepseek-v4-flash"
      reasoningViaDetails provider "deepseek/deepseek-v4-flash"
      toolCallingWithReasoning provider "deepseek/deepseek-v4-flash"
      providerErrorResponse provider

    describe "Standard Tests" $
      testModel route (DeepSeekV4Flash `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

    describe "Composable Provider Tests" $
      testModelOffline route (DeepSeekV4Flash `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

-- | Test DeepSeek V4 Pro via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsDeepSeekV4ProOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsDeepSeekV4ProOpenRouter provider = do
  describe "DeepSeek V4 Pro via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "deepseek/deepseek-v4-pro"
      toolCalling provider "deepseek/deepseek-v4-pro"
      acceptsToolResults provider "deepseek/deepseek-v4-pro"
      acceptsToolResultNoTools provider "deepseek/deepseek-v4-pro"
      acceptsToolResultToolGone provider "deepseek/deepseek-v4-pro"
      acceptsStaleToolInHistory provider "deepseek/deepseek-v4-pro"
      acceptsOldToolCallStillAvailable provider "deepseek/deepseek-v4-pro"
      consecutiveUserMessages provider "deepseek/deepseek-v4-pro"
      startsWithAssistant provider "deepseek/deepseek-v4-pro"
      reasoningViaDetails provider "deepseek/deepseek-v4-pro"
      toolCallingWithReasoning provider "deepseek/deepseek-v4-pro"
      providerErrorResponse provider

    describe "Standard Tests" $
      testModel route (DeepSeekV4Pro `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

    describe "Composable Provider Tests" $
      testModelOffline route (DeepSeekV4Pro `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]
