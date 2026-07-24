{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.Alibaba.Qwen

Model test suite for all Qwen models (Qwen 3.5, Qwen 3.6, Qwen 3.7, Qwen 3.8, Qwen 3 Coder)

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (proper tool_calls format, not XML)
✓ Reasoning (via reasoning_content on llama.cpp/AlibabaCloudTokenPlan, reasoning_details on OpenRouter)
✓ Vision (Qwen 3.5 40B)

= Provider-Specific Quirks

__llama.cpp:__
  Reasoning via standard reasoning_content field
  Qwen3.5 chat template requires exactly one system message at the start
    (mitigated by mergeSystemMessages + systemMessagesFirst in route)

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content

__AlibabaCloudTokenPlan:__
  Uses standard openAIReasoning (reasoning_content field)

-}

module Models.Alibaba.Qwen
  ( -- * Qwen 3.5 122B
    testsQwen35_122BLlamaCpp
  , testsQwen35_122BOpenRouter
    -- * Qwen 3.5 40B
  , testsQwen35_40BLlamaCpp
    -- * Qwen 3.6 Plus
  , testsQwen36PlusOpenRouter
    -- * Qwen 3.6 Flash
  , testsQwen36FlashOpenRouter
  , testsQwen36FlashAlibabaCloudTokenPlan
    -- * Qwen 3.7 Plus
  , testsQwen37PlusOpenRouter
  , testsQwen37PlusAlibabaCloudTokenPlan
    -- * Qwen 3.7 Max
  , testsQwen37MaxOpenRouter
  , testsQwen37MaxAlibabaCloudTokenPlan
    -- * Qwen 3.8 Max Preview
  , testsQwen38MaxPreviewAlibabaCloudTokenPlan
    -- * Qwen 3 Coder Next
  , testsQwen3CoderNextLlamaCpp
    -- * Qwen 3 Coder 30B Instruct
  , testsQwen3Coder30bInstructLlamaCpp
  ) where

import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..), AlibabaCloudTokenPlan(..))
import UniversalLLM.Models.Alibaba.Qwen
  ( Qwen35_122B(..)
  , Qwen35_40B(..)
  , Qwen36Plus(..)
  , Qwen36Flash(..)
  , Qwen37Plus(..)
  , Qwen37Max(..)
  , Qwen38MaxPreview(..)
  , Qwen3CoderNext(..)
  , Qwen3Coder30bInstruct(..)
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import qualified ComposableProviderTests as CPT
import TestCache (ResponseProvider)
import TestHelpers (testModel, testModelOffline)
import Test.Hspec (Spec, describe)
import Data.Text (Text)

--------------------------------------------------------------------------------
-- Qwen 3.5 122B
--------------------------------------------------------------------------------

testsQwen35_122BLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsQwen35_122BLlamaCpp provider modelName = do
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
      systemMessageAtStart provider modelName
      rejectsSystemMessageMidConversation provider modelName
      rejectsMultipleSystemMessages provider modelName

    describe "Standard Tests" $
      testModel route (Qwen35_122B `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.json ]

testsQwen35_122BOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen35_122BOpenRouter provider = do
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
      systemMessageAtStart provider "qwen/qwen3.5-122b-a10b"
      systemMessageMidConversation provider "qwen/qwen3.5-122b-a10b"
      multipleSystemMessages provider "qwen/qwen3.5-122b-a10b"

    describe "Standard Tests" $
      testModel route (Qwen35_122B `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation, ST.json ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen35_122B `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

--------------------------------------------------------------------------------
-- Qwen 3.5 40B
--------------------------------------------------------------------------------

testsQwen35_40BLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsQwen35_40BLlamaCpp provider modelName = do
  describe "Qwen 3.5 40B via llama.cpp" $ do
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
      reasoningTogglesOn provider modelName
      chatTemplateKwargsDisablesThinking provider modelName
      ignoresReasoningDisable provider modelName
      ignoresNoThink provider modelName
      rejectsEmptyThinkPrefill provider modelName
      rejectsEmptyReasoningPrefill provider modelName
      visionPng provider modelName
      visionJpeg provider modelName
      visionAccuracy provider modelName
      visionMultipleImages provider modelName
      systemMessageAtStart provider modelName
      -- Probes to investigate tools+reasoning interaction (500 from chat template):
      toolsWithReasoningDisabled provider modelName
      toolsWithReasoningEnabled provider modelName
      toolsWithReasoningEffortOnly provider modelName
      toolsWithReasoningFull provider modelName
      toolsWithReasoningListFiles provider modelName
      toolsWithReasoningAndMaxTokens provider modelName

    describe "Standard Tests" $
      testModel route (Qwen35_40B `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningDisabled, ST.reasoningWithTools, ST.vision, ST.visionJpeg, ST.visionMultipleImages, ST.json ]

--------------------------------------------------------------------------------
-- Qwen 3.6 Plus (OpenRouter)
--------------------------------------------------------------------------------

testsQwen36PlusOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen36PlusOpenRouter provider = do
  describe "Qwen 3.6 Plus via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "qwen/qwen3.6-plus"
      toolCalling provider "qwen/qwen3.6-plus"
      acceptsToolResults provider "qwen/qwen3.6-plus"
      acceptsToolResultNoTools provider "qwen/qwen3.6-plus"
      acceptsToolResultToolGone provider "qwen/qwen3.6-plus"
      acceptsStaleToolInHistory provider "qwen/qwen3.6-plus"
      acceptsOldToolCallStillAvailable provider "qwen/qwen3.6-plus"
      consecutiveUserMessages provider "qwen/qwen3.6-plus"
      startsWithAssistant provider "qwen/qwen3.6-plus"
      reasoningViaDetails provider "qwen/qwen3.6-plus"
      toolCallingWithReasoning provider "qwen/qwen3.6-plus"
      systemMessageAtStart provider "qwen/qwen3.6-plus"
      systemMessageMidConversation provider "qwen/qwen3.6-plus"
      multipleSystemMessages provider "qwen/qwen3.6-plus"

    describe "Standard Tests" $
      testModel route (Qwen36Plus `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation, ST.json ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen36Plus `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

--------------------------------------------------------------------------------
-- Qwen 3.7 Max (OpenRouter)
--------------------------------------------------------------------------------

testsQwen37MaxOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen37MaxOpenRouter provider = do
  describe "Qwen 3.7 Max via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "qwen/qwen3.7-max"
      toolCalling provider "qwen/qwen3.7-max"
      acceptsToolResults provider "qwen/qwen3.7-max"
      acceptsToolResultNoTools provider "qwen/qwen3.7-max"
      acceptsToolResultToolGone provider "qwen/qwen3.7-max"
      acceptsStaleToolInHistory provider "qwen/qwen3.7-max"
      acceptsOldToolCallStillAvailable provider "qwen/qwen3.7-max"
      consecutiveUserMessages provider "qwen/qwen3.7-max"
      startsWithAssistant provider "qwen/qwen3.7-max"
      reasoningViaDetails provider "qwen/qwen3.7-max"
      toolCallingWithReasoning provider "qwen/qwen3.7-max"
      systemMessageAtStart provider "qwen/qwen3.7-max"
      systemMessageMidConversation provider "qwen/qwen3.7-max"
      multipleSystemMessages provider "qwen/qwen3.7-max"

    describe "Standard Tests" $
      testModel route (Qwen37Max `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation, ST.json ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen37Max `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

--------------------------------------------------------------------------------
-- Qwen 3.7 Max (AlibabaCloudTokenPlan)
--------------------------------------------------------------------------------

testsQwen37MaxAlibabaCloudTokenPlan :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen37MaxAlibabaCloudTokenPlan provider = do
  describe "Qwen 3.7 Max via AlibabaCloudTokenPlan" $ do
    describe "Protocol" $ do
      basicText provider "qwen3.7-max"
      toolCalling provider "qwen3.7-max"
      acceptsToolResults provider "qwen3.7-max"
      acceptsToolResultNoTools provider "qwen3.7-max"
      acceptsToolResultToolGone provider "qwen3.7-max"
      acceptsStaleToolInHistory provider "qwen3.7-max"
      acceptsOldToolCallStillAvailable provider "qwen3.7-max"
      consecutiveUserMessages provider "qwen3.7-max"
      startsWithAssistant provider "qwen3.7-max"
      systemMessageAtStart provider "qwen3.7-max"
      systemMessageMidConversation provider "qwen3.7-max"
      multipleSystemMessages provider "qwen3.7-max"
      reasoning provider "qwen3.7-max"

    describe "Standard Tests" $
      testModel route (Qwen37Max `via` AlibabaCloudTokenPlan) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.json ]

--------------------------------------------------------------------------------
-- Qwen 3.7 Plus (OpenRouter)
--------------------------------------------------------------------------------

testsQwen37PlusOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen37PlusOpenRouter provider = do
  describe "Qwen 3.7 Plus via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "qwen/qwen3.7-plus"
      toolCalling provider "qwen/qwen3.7-plus"
      acceptsToolResults provider "qwen/qwen3.7-plus"
      acceptsToolResultNoTools provider "qwen/qwen3.7-plus"
      acceptsToolResultToolGone provider "qwen/qwen3.7-plus"
      acceptsStaleToolInHistory provider "qwen/qwen3.7-plus"
      acceptsOldToolCallStillAvailable provider "qwen/qwen3.7-plus"
      consecutiveUserMessages provider "qwen/qwen3.7-plus"
      startsWithAssistant provider "qwen/qwen3.7-plus"
      reasoningViaDetails provider "qwen/qwen3.7-plus"
      toolCallingWithReasoning provider "qwen/qwen3.7-plus"
      systemMessageAtStart provider "qwen/qwen3.7-plus"
      systemMessageMidConversation provider "qwen/qwen3.7-plus"
      multipleSystemMessages provider "qwen/qwen3.7-plus"

    describe "Standard Tests" $
      testModel route (Qwen37Plus `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation, ST.json ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen37Plus `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

--------------------------------------------------------------------------------
-- Qwen 3.7 Plus (AlibabaCloudTokenPlan)
--------------------------------------------------------------------------------

testsQwen37PlusAlibabaCloudTokenPlan :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen37PlusAlibabaCloudTokenPlan provider = do
  describe "Qwen 3.7 Plus via AlibabaCloudTokenPlan" $ do
    describe "Protocol" $ do
      basicText provider "qwen3.7-plus"
      toolCalling provider "qwen3.7-plus"
      acceptsToolResults provider "qwen3.7-plus"
      acceptsToolResultNoTools provider "qwen3.7-plus"
      acceptsToolResultToolGone provider "qwen3.7-plus"
      acceptsStaleToolInHistory provider "qwen3.7-plus"
      acceptsOldToolCallStillAvailable provider "qwen3.7-plus"
      consecutiveUserMessages provider "qwen3.7-plus"
      startsWithAssistant provider "qwen3.7-plus"
      systemMessageAtStart provider "qwen3.7-plus"
      systemMessageMidConversation provider "qwen3.7-plus"
      multipleSystemMessages provider "qwen3.7-plus"
      reasoning provider "qwen3.7-plus"

    describe "Standard Tests" $
      testModel route (Qwen37Plus `via` AlibabaCloudTokenPlan) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.json ]

--------------------------------------------------------------------------------
-- Qwen 3.6 Flash (OpenRouter)
--------------------------------------------------------------------------------

testsQwen36FlashOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen36FlashOpenRouter provider = do
  describe "Qwen 3.6 Flash via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "qwen/qwen3.6-flash"
      toolCalling provider "qwen/qwen3.6-flash"
      acceptsToolResults provider "qwen/qwen3.6-flash"
      acceptsToolResultNoTools provider "qwen/qwen3.6-flash"
      acceptsToolResultToolGone provider "qwen/qwen3.6-flash"
      acceptsStaleToolInHistory provider "qwen/qwen3.6-flash"
      acceptsOldToolCallStillAvailable provider "qwen/qwen3.6-flash"
      consecutiveUserMessages provider "qwen/qwen3.6-flash"
      startsWithAssistant provider "qwen/qwen3.6-flash"
      reasoningViaDetails provider "qwen/qwen3.6-flash"
      toolCallingWithReasoning provider "qwen/qwen3.6-flash"
      systemMessageAtStart provider "qwen/qwen3.6-flash"
      systemMessageMidConversation provider "qwen/qwen3.6-flash"
      multipleSystemMessages provider "qwen/qwen3.6-flash"

    describe "Standard Tests" $
      testModel route (Qwen36Flash `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation, ST.json ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen36Flash `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

--------------------------------------------------------------------------------
-- Qwen 3.6 Flash (AlibabaCloudTokenPlan)
--------------------------------------------------------------------------------

testsQwen36FlashAlibabaCloudTokenPlan :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen36FlashAlibabaCloudTokenPlan provider = do
  describe "Qwen 3.6 Flash via AlibabaCloudTokenPlan" $ do
    describe "Protocol" $ do
      basicText provider "qwen3.6-flash"
      toolCalling provider "qwen3.6-flash"
      acceptsToolResults provider "qwen3.6-flash"
      acceptsToolResultNoTools provider "qwen3.6-flash"
      acceptsToolResultToolGone provider "qwen3.6-flash"
      acceptsStaleToolInHistory provider "qwen3.6-flash"
      acceptsOldToolCallStillAvailable provider "qwen3.6-flash"
      consecutiveUserMessages provider "qwen3.6-flash"
      startsWithAssistant provider "qwen3.6-flash"
      systemMessageAtStart provider "qwen3.6-flash"
      systemMessageMidConversation provider "qwen3.6-flash"
      multipleSystemMessages provider "qwen3.6-flash"
      reasoning provider "qwen3.6-flash"

    describe "Standard Tests" $
      testModel route (Qwen36Flash `via` AlibabaCloudTokenPlan) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.json ]

--------------------------------------------------------------------------------
-- Qwen 3.8 Max Preview (AlibabaCloudTokenPlan)
--------------------------------------------------------------------------------

testsQwen38MaxPreviewAlibabaCloudTokenPlan :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen38MaxPreviewAlibabaCloudTokenPlan provider = do
  describe "Qwen 3.8 Max Preview via AlibabaCloudTokenPlan" $ do
    describe "Protocol" $ do
      basicText provider "qwen3.8-max-preview"
      toolCalling provider "qwen3.8-max-preview"
      acceptsToolResults provider "qwen3.8-max-preview"
      acceptsToolResultNoTools provider "qwen3.8-max-preview"
      acceptsToolResultToolGone provider "qwen3.8-max-preview"
      acceptsStaleToolInHistory provider "qwen3.8-max-preview"
      acceptsOldToolCallStillAvailable provider "qwen3.8-max-preview"
      consecutiveUserMessages provider "qwen3.8-max-preview"
      startsWithAssistant provider "qwen3.8-max-preview"
      systemMessageAtStart provider "qwen3.8-max-preview"
      systemMessageMidConversation provider "qwen3.8-max-preview"
      multipleSystemMessages provider "qwen3.8-max-preview"
      reasoning provider "qwen3.8-max-preview"

    describe "Standard Tests" $
      testModel route (Qwen38MaxPreview `via` AlibabaCloudTokenPlan) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.json ]

--------------------------------------------------------------------------------
-- Qwen 3 Coder Next (llama.cpp)
--------------------------------------------------------------------------------

testsQwen3CoderNextLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsQwen3CoderNextLlamaCpp provider modelName = do
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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.json ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen3CoderNext `via` LlamaCpp)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

--------------------------------------------------------------------------------
-- Qwen 3 Coder 30B Instruct (llama.cpp)
--------------------------------------------------------------------------------

testsQwen3Coder30bInstructLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsQwen3Coder30bInstructLlamaCpp provider modelName = do
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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.json ]
