{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Qwen3Coder

Model test suite for Qwen 3 Coder

This module tests Qwen 3 Coder when served through llama.cpp's OpenAI-compatible
endpoint. Unlike GLM-4.5, Qwen may use proper tool_calls field.

= Discovered Capabilities

To be determined through capability probes

= Provider-Specific Quirks

__llama.cpp:__
  Model name is dynamically determined from loaded GGUF file
  Tool calling format TBD (may use proper tool_calls or XML)

-}

module Models.Qwen3Coder (testsLlamaCpp) where

import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import Protocol.OpenAITests
import TestCache (ResponseProvider)
import Test.Hspec (Spec, describe)
import Data.Text (Text)

-- | Test Qwen 3 Coder via llama.cpp
--
-- Takes the canonicalized model name as determined by querying the llama.cpp
-- server. The model name is extracted from the loaded GGUF file.
--
-- Currently only includes protocol probes to discover capabilities.
-- Standard tests can be added once we know what the model supports.
testsLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCpp provider modelName = do
  describe ("Qwen 3 Coder (llama.cpp with " <> show modelName <> ")") $ do
    describe "Protocol" $ do
      basicText provider modelName
      toolCalling provider modelName  -- Test if it uses proper tool_calls field
