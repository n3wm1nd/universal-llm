{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.OpenAI.GPTOSS

Model test suite for GPT-OSS

This module tests GPT-OSS when accessed through OpenRouter and llama.cpp.

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (native tool_calls format)
✓ Reasoning (via reasoning_details on OpenRouter, reasoning_content on llama.cpp)
✓ Reasoning preservation through tool call chains

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content

__llama.cpp:__
  Model name is dynamically determined from loaded GGUF file
  Uses standard reasoning_content field

-}

module Models.OpenAI.GPTOSS (testsOpenRouter, testsLlamaCpp) where

import Data.Text (Text)
import qualified Data.Text as T
import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import UniversalLLM.Models.OpenAI.GPT
  ( GPTOSS(..)
  , gptOSSOpenRouter
  , gptOSS
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)

-- | Test GPT-OSS via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "GPT-OSS via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-oss-120b"
      toolCalling provider "openai/gpt-oss-120b"
      acceptsToolResults provider "openai/gpt-oss-120b"
      acceptsToolResultNoTools provider "openai/gpt-oss-120b"
      acceptsToolResultToolGone provider "openai/gpt-oss-120b"
      acceptsStaleToolInHistory provider "openai/gpt-oss-120b"
      acceptsOldToolCallStillAvailable provider "openai/gpt-oss-120b"
      consecutiveUserMessages provider "openai/gpt-oss-120b"
      startsWithAssistant provider "openai/gpt-oss-120b"
      reasoningViaDetails provider "openai/gpt-oss-120b"
      toolCallingWithReasoning provider "openai/gpt-oss-120b"

    describe "Standard Tests" $
      testModel gptOSSOpenRouter (Model GPTOSS OpenRouter) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

-- | Test GPT-OSS via llama.cpp
--
-- Takes the canonicalized model name as determined by querying the llama.cpp
-- server. The model name is extracted from the loaded GGUF file.
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLlamaCpp provider modelName = do
  describe ("GPT-OSS via llama.cpp with " <> T.unpack modelName) $ do
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
      testModel gptOSS (Model GPTOSS LlamaCpp) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]
