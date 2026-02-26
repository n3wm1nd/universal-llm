{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Alibaba.Qwen35

Model test suite for Qwen 3.5 122B

This module tests Qwen 3.5 122B when accessed through OpenRouter and llama.cpp.

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (proper tool_calls format)
✓ Reasoning (via reasoning_content on llama.cpp, reasoning_details on OpenRouter)

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content

__llama.cpp:__
  Model name is dynamically determined from loaded GGUF file
  Uses proper tool_calls field (not XML like GLM-4.5)
  Reasoning via standard reasoning_content field

-}

module Models.Alibaba.Qwen35 (testsLlamaCpp, testsOpenRouter) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import UniversalLLM.Models.Alibaba.Qwen
  ( Qwen35_122B(..)
  , qwen35_122B
  , qwen35_122BOpenRouter
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)
import Data.Text (Text)

-- | Test Qwen 3.5 122B via OpenRouter
testsOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "Qwen 3.5 122B via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "qwen/qwen3.5-122b-a10b"
      toolCalling provider "qwen/qwen3.5-122b-a10b"
      acceptsToolResults provider "qwen/qwen3.5-122b-a10b"
      acceptsToolResultNoTools provider "qwen/qwen3.5-122b-a10b"
      acceptsToolResultToolGone provider "qwen/qwen3.5-122b-a10b"
      acceptsStaleToolInHistory provider "qwen/qwen3.5-122b-a10b"
      acceptsOldToolCallStillAvailable provider "qwen/qwen3.5-122b-a10b"
      consecutiveUserMessages provider "qwen/qwen3.5-122b-a10b"
      startsWithAssistant provider "qwen/qwen3.5-122b-a10b"
      reasoningViaDetails provider "qwen/qwen3.5-122b-a10b"
      toolCallingWithReasoning provider "qwen/qwen3.5-122b-a10b"

    describe "Standard Tests" $
      testModel qwen35_122BOpenRouter (Model Qwen35_122B OpenRouter) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

-- | Test Qwen 3.5 122B via llama.cpp
testsLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCpp provider modelName = do
  describe "Qwen 3.5 122B via llama.cpp" $ do
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

    describe "Standard Tests" $
      testModel qwen35_122B (Model Qwen35_122B LlamaCpp) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]
