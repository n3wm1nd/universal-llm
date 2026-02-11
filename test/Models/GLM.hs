{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.GLM

Test suites for Zhipu AI's GLM model family.

= Models Covered

* GLM 4.5 Air (OpenRouter, llama.cpp)
* GLM 5 (ZAI)

= GLM 4.5 Air Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Proper tool_calls field support

__llama.cpp:__
  Uses standard reasoning_content field
  Returns tool calls as XML in content field (not tool_calls)
  Model name is dynamically determined from loaded GGUF file
  Requires withXMLResponseParsing handler for tool calls

-}

module Models.GLM
  ( testsGLM45AirOpenRouter
  , testsGLM45AirLlamaCpp
  , testsGLM5ZAI
  ) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import UniversalLLM.Models.GLM
  ( GLM45Air(..)
  , GLM5(..)
  , ZAI(..)
  , glm45AirLlamaCpp
  , glm45AirOpenRouter
  , glm5
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe, xdescribe)
import qualified Data.Text as T
import Data.Text (Text)

-- | Test GLM 4.5 Air via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGLM45AirOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGLM45AirOpenRouter provider = do
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
testsGLM45AirLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsGLM45AirLlamaCpp provider modelName = do
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

-- | Test GLM 5 via ZAI
--
-- Includes standard tests via the official ZAI API.
testsGLM5ZAI :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGLM5ZAI provider = do
  xdescribe "GLM 5 via ZAI (requires higher-tier ZAI subscription)" $ do
    describe "Protocol" $ do
      basicText provider "glm-5"
      toolCalling provider "glm-5"
      acceptsToolResults provider "glm-5"
      acceptsToolResultNoTools provider "glm-5"
      acceptsToolResultToolGone provider "glm-5"
      acceptsStaleToolInHistory provider "glm-5"
      acceptsOldToolCallStillAvailable provider "glm-5"
      consecutiveUserMessages provider "glm-5"
      startsWithAssistant provider "glm-5"
      reasoning provider "glm-5"

    describe "Standard Tests" $
      testModel glm5 (Model GLM5 ZAI) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]
