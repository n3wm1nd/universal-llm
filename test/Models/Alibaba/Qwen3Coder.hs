{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

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

module Models.Alibaba.Qwen3Coder (testsLlamaCppNext, testsLlamaCpp30bInstruct, testsQwen3CoderNextAlibabaCloud, testsQwen3CoderPlusAlibabaCloud) where

import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), AlibabaCloud(..))
import UniversalLLM.Models.Alibaba.Qwen
  ( Qwen3CoderNext(..)
  , Qwen3Coder30bInstruct(..)
  , Qwen3CoderPlus(..)
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
      testModel route (Qwen3CoderNext `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]

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
      testModel route (Qwen3Coder30bInstruct `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]

-- | Test Qwen 3 Coder Next via AlibabaCloud
testsQwen3CoderNextAlibabaCloud :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen3CoderNextAlibabaCloud provider = do
  describe "Qwen 3 Coder Next via AlibabaCloud" $ do
    describe "Protocol" $ do
      basicText provider "qwen3-coder-next"
      toolCalling provider "qwen3-coder-next"
      acceptsToolResults provider "qwen3-coder-next"
      acceptsToolResultNoTools provider "qwen3-coder-next"
      acceptsToolResultToolGone provider "qwen3-coder-next"
      acceptsStaleToolInHistory provider "qwen3-coder-next"
      acceptsOldToolCallStillAvailable provider "qwen3-coder-next"
      consecutiveUserMessages provider "qwen3-coder-next"
      startsWithAssistant provider "qwen3-coder-next"
      systemMessageAtStart provider "qwen3-coder-next"
      systemMessageMidConversation provider "qwen3-coder-next"
      multipleSystemMessages provider "qwen3-coder-next"

    describe "Standard Tests" $
      testModel route (Qwen3CoderNext `via` AlibabaCloud) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]

-- | Test Qwen 3 Coder Plus via AlibabaCloud
testsQwen3CoderPlusAlibabaCloud :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen3CoderPlusAlibabaCloud provider = do
  describe "Qwen 3 Coder Plus via AlibabaCloud" $ do
    describe "Protocol" $ do
      basicText provider "qwen3-coder-plus"
      toolCalling provider "qwen3-coder-plus"
      acceptsToolResults provider "qwen3-coder-plus"
      acceptsToolResultNoTools provider "qwen3-coder-plus"
      acceptsToolResultToolGone provider "qwen3-coder-plus"
      acceptsStaleToolInHistory provider "qwen3-coder-plus"
      acceptsOldToolCallStillAvailable provider "qwen3-coder-plus"
      consecutiveUserMessages provider "qwen3-coder-plus"
      startsWithAssistant provider "qwen3-coder-plus"
      systemMessageAtStart provider "qwen3-coder-plus"
      systemMessageMidConversation provider "qwen3-coder-plus"
      multipleSystemMessages provider "qwen3-coder-plus"

    describe "Standard Tests" $
      testModel route (Qwen3CoderPlus `via` AlibabaCloud) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]
