{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.Minimax.MinimaxM25

Model test suite for MiniMax M2.5

This module tests MiniMax M2.5 when accessed through OpenRouter and llama.cpp.

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (native tool_calls format)
✓ Reasoning (via reasoning_details on OpenRouter, reasoning_content on llama.cpp)
✓ Reasoning preservation through tool call chains

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Requires reasoning_details to be preserved in conversation history

__LlamaCpp:__
  Uses standard reasoning_content field

-}

module Models.Minimax.MinimaxM25 (testsOpenRouter, testsLlamaCpp, testsAlibabaCloud) where

import Data.Text (Text)
import qualified Data.Text as T
import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..), AlibabaCloud(..))
import UniversalLLM.Models.Minimax.M (MinimaxM25(..))
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
      testModel route (MinimaxM25 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

-- | Test MiniMax M2.5 via llama.cpp
--
-- Uses standard reasoning_content (not OpenRouter's reasoning_details).
testsLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCpp provider modelName = do
  describe ("MiniMax M2.5 via llama.cpp with " <> T.unpack modelName) $ do
    describe "Protocol" $ do
      basicText provider modelName
      toolCalling provider modelName
      acceptsToolResults provider modelName
      acceptsToolResultNoTools provider modelName
      acceptsToolResultToolGone provider modelName
      acceptsStaleToolInHistory provider modelName
      acceptsOldToolCallStillAvailable provider modelName
      consecutiveUserMessages provider modelName
      startsWithAssistant provider modelName
      reasoning provider modelName
      -- Note: toolCallingWithReasoning asserts reasoning_details (OpenRouter style).
      -- On llama.cpp, reasoning goes through standard reasoning_content.
      -- Reasoning + tools is covered by ST.reasoningWithTools in Standard Tests.

    describe "Standard Tests" $
      testModel route (MinimaxM25 `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools ]

-- | Test MiniMax M2.5 via AlibabaCloud
testsAlibabaCloud :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsAlibabaCloud provider = do
  describe "MiniMax M2.5 via AlibabaCloud" $ do
    describe "Protocol" $ do
      basicText provider "MiniMax-M2.5"
      toolCalling provider "MiniMax-M2.5"
      acceptsToolResults provider "MiniMax-M2.5"
      acceptsToolResultNoTools provider "MiniMax-M2.5"
      acceptsToolResultToolGone provider "MiniMax-M2.5"
      acceptsStaleToolInHistory provider "MiniMax-M2.5"
      acceptsOldToolCallStillAvailable provider "MiniMax-M2.5"
      consecutiveUserMessages provider "MiniMax-M2.5"
      startsWithAssistant provider "MiniMax-M2.5"
      systemMessageAtStart provider "MiniMax-M2.5"
      systemMessageMidConversation provider "MiniMax-M2.5"
      multipleSystemMessages provider "MiniMax-M2.5"

    describe "Standard Tests" $
      testModel route (MinimaxM25 `via` AlibabaCloud) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]
