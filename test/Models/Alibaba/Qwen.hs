{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.Alibaba.Qwen

Model test suite for all Qwen models (Qwen 3.5, Qwen 3.6, Qwen 3.7, Qwen 3 Coder)

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (proper tool_calls format, not XML)
✓ Reasoning (via reasoning_content on llama.cpp/AlibabaCloud, reasoning_details on OpenRouter)
✓ Vision (Qwen 3.5 40B, Qwen 3.5 Plus, Qwen 3.6 Plus)

= Provider-Specific Quirks

__llama.cpp:__
  Reasoning via standard reasoning_content field
  Qwen3.5 chat template requires exactly one system message at the start
    (mitigated by mergeSystemMessages + systemMessagesFirst in route)

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content

__AlibabaCloud:__
  Uses standard openAIReasoning (reasoning_content field)

-}

module Models.Alibaba.Qwen
  ( -- * Qwen 3.5 122B
    testsQwen35_122BLlamaCpp
  , testsQwen35_122BOpenRouter
    -- * Qwen 3.5 40B
  , testsQwen35_40BLlamaCpp
    -- * Qwen 3.5 Plus
  , testsQwen35PlusAlibabaCloud
    -- * Qwen 3.6 Plus
  , testsQwen36PlusAlibabaCloud
  , testsQwen36PlusOpenRouter
    -- * Qwen 3.7 Max
  , testsQwen37MaxOpenRouter
    -- * Qwen 3 Coder Next
  , testsQwen3CoderNextLlamaCpp
  , testsQwen3CoderNextAlibabaCloud
    -- * Qwen 3 Coder 30B Instruct
  , testsQwen3Coder30bInstructLlamaCpp
    -- * Qwen 3 Coder Plus
  , testsQwen3CoderPlusAlibabaCloud
  ) where

import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..), AlibabaCloud(..))
import UniversalLLM.Models.Alibaba.Qwen
  ( Qwen35_122B(..)
  , Qwen35_40B(..)
  , Qwen36Plus(..)
  , Qwen37Max(..)
  , Qwen3CoderNext(..)
  , Qwen3Coder30bInstruct(..)
  , Qwen3CoderPlus(..)
  , Qwen35Plus(..)
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
      -- multipleSystemMessages and systemMessageMidConversation fail —
      -- Qwen3.5 chat template requires exactly one system message at the start.
      -- Mitigated by mergeSystemMessages + systemMessagesFirst in route.

    describe "Standard Tests" $
      testModel route (Qwen35_122B `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools ]

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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

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
      -- reasoningTogglesOff: fails - reasoning_enabled=false is ignored by llama.cpp
      -- noThinkSuppressesReasoning: fails - /nothink stripped before reaching template
      -- emptyThinkPrefillSuppressesReasoning: rejected - prefill incompatible with enable_thinking
      -- emptyReasoningPrefillSuppressesReasoning: rejected - assistant message needs content/tool_calls
      chatTemplateKwargsDisablesThinking provider modelName
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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningDisabled, ST.reasoningWithTools, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

--------------------------------------------------------------------------------
-- Qwen 3.5 Plus (AlibabaCloud)
--------------------------------------------------------------------------------

testsQwen35PlusAlibabaCloud :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen35PlusAlibabaCloud provider = do
  describe "Qwen 3.5 Plus via AlibabaCloud" $ do
    describe "Protocol" $ do
      basicText provider "qwen3.5-plus"
      toolCalling provider "qwen3.5-plus"
      acceptsToolResults provider "qwen3.5-plus"
      acceptsToolResultNoTools provider "qwen3.5-plus"
      acceptsToolResultToolGone provider "qwen3.5-plus"
      acceptsStaleToolInHistory provider "qwen3.5-plus"
      acceptsOldToolCallStillAvailable provider "qwen3.5-plus"
      consecutiveUserMessages provider "qwen3.5-plus"
      startsWithAssistant provider "qwen3.5-plus"
      systemMessageAtStart provider "qwen3.5-plus"
      systemMessageMidConversation provider "qwen3.5-plus"
      multipleSystemMessages provider "qwen3.5-plus"
      reasoning provider "qwen3.5-plus"
      visionPng provider "qwen3.5-plus"
      visionJpeg provider "qwen3.5-plus"

    describe "Standard Tests" $
      testModel route (Qwen35Plus `via` AlibabaCloud) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.vision, ST.visionJpeg ]

--------------------------------------------------------------------------------
-- Qwen 3.6 Plus (AlibabaCloud + OpenRouter)
--------------------------------------------------------------------------------

testsQwen36PlusAlibabaCloud :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsQwen36PlusAlibabaCloud provider = do
  describe "Qwen 3.6 Plus via AlibabaCloud" $ do
    describe "Protocol" $ do
      basicText provider "qwen3.6-plus"
      toolCalling provider "qwen3.6-plus"
      acceptsToolResults provider "qwen3.6-plus"
      acceptsToolResultNoTools provider "qwen3.6-plus"
      acceptsToolResultToolGone provider "qwen3.6-plus"
      acceptsStaleToolInHistory provider "qwen3.6-plus"
      acceptsOldToolCallStillAvailable provider "qwen3.6-plus"
      consecutiveUserMessages provider "qwen3.6-plus"
      startsWithAssistant provider "qwen3.6-plus"
      systemMessageAtStart provider "qwen3.6-plus"
      systemMessageMidConversation provider "qwen3.6-plus"
      multipleSystemMessages provider "qwen3.6-plus"
      reasoning provider "qwen3.6-plus"
      visionPng provider "qwen3.6-plus"
      visionJpeg provider "qwen3.6-plus"

    describe "Standard Tests" $
      testModel route (Qwen36Plus `via` AlibabaCloud) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.vision, ST.visionJpeg ]

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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen37Max `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

--------------------------------------------------------------------------------
-- Qwen 3 Coder Next (llama.cpp + AlibabaCloud)
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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]

    describe "Composable Provider Tests" $
      testModelOffline route (Qwen3CoderNext `via` LlamaCpp)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

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
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools ]

--------------------------------------------------------------------------------
-- Qwen 3 Coder Plus (AlibabaCloud)
--------------------------------------------------------------------------------

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
