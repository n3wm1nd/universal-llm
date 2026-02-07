{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Qwen3Coder

Model test suite for Qwen 3 Coder

This module tests Qwen 3 Coder when served through llama.cpp's OpenAI-compatible
endpoint. Unlike GLM-4.5, Qwen uses proper tool_calls field (not XML).

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (proper tool_calls format)

= Provider-Specific Quirks

__llama.cpp:__
  Model name is dynamically determined from loaded GGUF file
  Uses proper tool_calls field (not XML like GLM-4.5)

-}

module Models.Qwen3Coder (testsLlamaCpp) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..))
import UniversalLLM.Models.Qwen
  ( Qwen3Coder(..)
  , qwen3Coder
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)
import qualified Data.Text as T
import Data.Text (Text)

-- | Test Qwen 3 Coder via llama.cpp
--
-- Takes the canonicalized model name as determined by querying the llama.cpp
-- server. The model name is extracted from the loaded GGUF file.
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCpp provider modelName = do
  describe ("Qwen 3 Coder via llama.cpp with " <> T.unpack modelName) $ do
    describe "Protocol" $ do
      basicText provider modelName
      toolCalling provider modelName  -- Uses proper tool_calls (not XML)
      acceptsToolResults provider modelName
      acceptsToolResultNoTools provider modelName
      acceptsToolResultToolGone provider modelName
      acceptsStaleToolInHistory provider modelName
      acceptsOldToolCallStillAvailable provider modelName
      consecutiveUserMessages provider modelName
      startsWithAssistant provider modelName

    describe "Standard Tests" $
      testModel qwen3Coder (Model Qwen3Coder LlamaCpp) provider
        [ ST.text, ST.tools ]
