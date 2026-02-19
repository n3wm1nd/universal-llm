{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.MinimaxM25

Model test suite for MiniMax M2.5

This module tests MiniMax M2.5 when accessed through OpenRouter.

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

module Models.MinimaxM25 (testsOpenRouter) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.MinimaxM25
  ( MinimaxM25(..)
  , minimaxM25
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)

-- | Test MiniMax M2.5 via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "MiniMax M2.5 via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "minimax/minimax-m2.5"
      toolCalling provider "minimax/minimax-m2.5"
      acceptsToolResults provider "minimax/minimax-m2.5"
      acceptsToolResultNoTools provider "minimax/minimax-m2.5"
      acceptsToolResultToolGone provider "minimax/minimax-m2.5"
      acceptsStaleToolInHistory provider "minimax/minimax-m2.5"
      acceptsOldToolCallStillAvailable provider "minimax/minimax-m2.5"
      consecutiveUserMessages provider "minimax/minimax-m2.5"
      startsWithAssistant provider "minimax/minimax-m2.5"
      -- Note: reasoning probe fails - MiniMax M2.5 uses reasoning_details (OpenRouter style),
      -- not the standard reasoning_content field.
      -- reasoning provider "minimax/minimax-m2.5"
      reasoningViaDetails provider "minimax/minimax-m2.5"
      toolCallingWithReasoning provider "minimax/minimax-m2.5"

    describe "Standard Tests" $
      testModel minimaxM25 (Model MinimaxM25 OpenRouter) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]
