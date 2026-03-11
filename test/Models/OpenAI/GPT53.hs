{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.OpenAI.GPT53

Model test suites for GPT-5.3 family models via OpenRouter.

= Models Covered

* GPT-5.3-Codex - Specialized for code generation
* GPT-5.3-Chat - Optimized for conversation

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

module Models.OpenAI.GPT53
  ( testsGPT53CodexOpenRouter
  , testsGPT53ChatOpenRouter
  ) where

import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.OpenAI.GPT (GPT53Codex(..), GPT53Chat(..))
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)

-- | Test GPT-5.3-Codex via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT53CodexOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT53CodexOpenRouter provider = do
  describe "GPT-5.3-Codex via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.3-codex"
      toolCalling provider "openai/gpt-5.3-codex"
      acceptsToolResults provider "openai/gpt-5.3-codex"
      acceptsToolResultNoTools provider "openai/gpt-5.3-codex"
      acceptsToolResultToolGone provider "openai/gpt-5.3-codex"
      acceptsStaleToolInHistory provider "openai/gpt-5.3-codex"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.3-codex"
      consecutiveUserMessages provider "openai/gpt-5.3-codex"
      startsWithAssistant provider "openai/gpt-5.3-codex"
      encryptedReasoning provider "openai/gpt-5.3-codex"

    describe "Standard Tests" $
      testModel route (GPT53Codex `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.hiddenReasoning ]

-- | Test GPT-5.3-Chat via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT53ChatOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT53ChatOpenRouter provider = do
  describe "GPT-5.3-Chat via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.3-chat"
      toolCalling provider "openai/gpt-5.3-chat"
      acceptsToolResults provider "openai/gpt-5.3-chat"
      acceptsToolResultNoTools provider "openai/gpt-5.3-chat"
      acceptsToolResultToolGone provider "openai/gpt-5.3-chat"
      acceptsStaleToolInHistory provider "openai/gpt-5.3-chat"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.3-chat"
      consecutiveUserMessages provider "openai/gpt-5.3-chat"
      startsWithAssistant provider "openai/gpt-5.3-chat"
      -- Note: GPT-5.3-Chat does NOT support reasoning via OpenRouter (no reasoning_details returned)

    describe "Standard Tests" $
      testModel route (GPT53Chat `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]
