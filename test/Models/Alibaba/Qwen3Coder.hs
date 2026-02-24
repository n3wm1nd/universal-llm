{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Alibaba.Qwen3Coder

Model test suite for Qwen 3 Coder variants

This module tests Qwen 3 Coder models when served through llama.cpp's
OpenAI-compatible endpoint. Unlike GLM-4.5, Qwen uses proper tool_calls
field (not XML).

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (proper tool_calls format)

= Provider-Specific Quirks

__llama.cpp:__
  Model name is dynamically determined from loaded GGUF file
  Uses proper tool_calls field (not XML like GLM-4.5)

-}

module Models.Alibaba.Qwen3Coder (testsLlamaCppNext, testsLlamaCpp30bInstruct) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..))
import UniversalLLM.Models.Alibaba.Qwen
  ( Qwen3CoderNext(..)
  , qwen3CoderNext
  , Qwen3Coder30bInstruct(..)
  , qwen3Coder30bInstruct
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)
import Data.Text (Text)

-- | Test Qwen 3 Coder Next via llama.cpp
testsLlamaCppNext :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCppNext provider modelName = do
  describe "Qwen 3 Coder Next via llama.cpp" $ do
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

    describe "Standard Tests" $
      testModel qwen3CoderNext (Model Qwen3CoderNext LlamaCpp) provider
        [ ST.text, ST.tools ]

-- | Test Qwen 3 Coder 30B Instruct via llama.cpp
testsLlamaCpp30bInstruct :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCpp30bInstruct provider modelName = do
  describe "Qwen 3 Coder 30B Instruct via llama.cpp" $ do
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

    describe "Standard Tests" $
      testModel qwen3Coder30bInstruct (Model Qwen3Coder30bInstruct LlamaCpp) provider
        [ ST.text, ST.tools ]
