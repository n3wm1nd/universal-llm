{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.GLM45Air

Model test suite for GLM 4.5 Air

This module demonstrates the pattern for model-specific test suites.
Each model package should have a similar structure:
- Model definition and capability instances
- Composable provider with all handlers
- Protocol probes that define the model's capabilities
- Standard tests using the high-level API
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
  Requires withXMLResponseParsing handler for tool calls

-}

module Models.GLM45Air
  ( testsOpenRouter
  , testsLlamaCpp
  ) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Models.GLM
  ( GLM45Air(..)
  , glm45AirLlamaCpp
  , glm45AirOpenRouter
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)
import qualified Data.Text as T
import Data.Text (Text)

-- | Test GLM 4.5 Air via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "GLM 4.5 Air via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "z-ai/glm-4.5-air:free"
      toolCalling provider "z-ai/glm-4.5-air:free"
      acceptsToolResults provider "z-ai/glm-4.5-air:free"
      acceptsToolResultNoTools provider "z-ai/glm-4.5-air:free"
      acceptsToolResultToolGone provider "z-ai/glm-4.5-air:free"
      acceptsStaleToolInHistory provider "z-ai/glm-4.5-air:free"
      acceptsOldToolCallStillAvailable provider "z-ai/glm-4.5-air:free"
      consecutiveUserMessages provider "z-ai/glm-4.5-air:free"
      startsWithAssistant provider "z-ai/glm-4.5-air:free"
      reasoningViaDetails provider "z-ai/glm-4.5-air:free"
      providerErrorResponse provider

    describe "Standard Tests" $
      testModel glm45AirOpenRouter (Model GLM45Air OpenRouter) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]

-- | Test GLM 4.5 Air via llama.cpp
--
-- Takes the canonicalized model name as determined by querying the llama.cpp
-- server. The model name is extracted from the loaded GGUF file.
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCpp provider modelName = do
  describe ("GLM 4.5 Air via llama.cpp with " <> T.unpack modelName) $ do
    describe "Protocol" $ do
      basicText provider modelName
      toolCallingViaXML provider modelName  -- GLM-4.5 returns XML tool calls in content
      acceptsToolResults provider modelName
      acceptsToolResultNoTools provider modelName
      acceptsToolResultToolGone provider modelName
      acceptsStaleToolInHistory provider modelName
      acceptsOldToolCallStillAvailable provider modelName
      consecutiveUserMessages provider modelName
      startsWithAssistant provider modelName
      reasoning provider modelName  -- llama.cpp uses standard reasoning_content

    describe "Standard Tests" $
      testModel glm45AirLlamaCpp (Model GLM45Air LlamaCpp) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning ]
