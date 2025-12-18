{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.GLM45Air

Model test suite for GLM 4.5 Air

This module demonstrates the pattern for model-specific test suites.
Each model package should have a similar structure:
- Protocol probes that define the model's capabilities
- Enshrined tests (only the probes we expect to pass)
- Clear documentation of quirks and limitations

The model can be accessed through multiple providers:
- OpenRouter (z-ai/glm-4.5-air:free)
- llama.cpp (local inference)

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling
✓ Reasoning

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Proper tool_calls field support

__llama.cpp:__
  Uses standard reasoning_content field
  Returns tool calls as XML in content field (not tool_calls)
  Model name is dynamically determined from loaded GGUF file

-}

module Models.GLM45Air (modelTestsOpenRouter, modelTestsLlamaCpp) where

import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import Protocol.OpenAITests
import TestCache (ResponseProvider)
import Test.Hspec (Spec, describe)
import Data.Text (Text)

-- | Run all enshrined tests for GLM 4.5 Air via OpenRouter
--
-- These are the capability probes we EXPECT to pass for this model.
-- If any fail, it indicates a regression or API change.
modelTestsOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
modelTestsOpenRouter provider = do
  describe "GLM 4.5 Air (z-ai/glm-4.5-air:free via OpenRouter)" $ do
    basicText provider "z-ai/glm-4.5-air:free"
    toolCalling provider "z-ai/glm-4.5-air:free"
    reasoningViaDetails provider "z-ai/glm-4.5-air:free"

-- | Run all enshrined tests for GLM 4.5 Air via llama.cpp
--
-- Takes the canonicalized model name as determined by querying the llama.cpp
-- server. The model name is extracted from the loaded GGUF file.
modelTestsLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
modelTestsLlamaCpp provider modelName = do
  describe ("GLM 4.5 Air (" <> show modelName <> " via llama.cpp)") $ do
    basicText provider modelName
    toolCallingViaXML provider modelName  -- GLM-4.5 returns XML tool calls in content
    reasoning provider modelName  -- llama.cpp uses standard reasoning_content
