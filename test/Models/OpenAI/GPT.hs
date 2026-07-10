{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.OpenAI.GPT

Model test suites for all GPT models.

= Models Covered

* GPT-OSS - OpenAI's open-weight model (OpenRouter + llama.cpp)
* GPT-5.3-Codex - Specialized for code generation (OpenRouter)
* GPT-5.3-Chat - Optimized for conversation (OpenRouter)
* GPT-5.4-Pro - Advanced capabilities (OpenRouter)
* GPT-5.4 - Latest generation (OpenRouter)

= Expected Capabilities

All models are expected to support:
✓ Basic text responses
✓ Tool calling (native tool_calls format)
✓ System messages
✓ Multi-turn conversations

= Provider-Specific Notes

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content
  Proper tool_calls field support

__llama.cpp:__
  Model name is dynamically determined from loaded GGUF file
  Uses standard reasoning_content field

-}

module Models.OpenAI.GPT
  ( testsGPTOSSOpenRouter
  , testsGPTOSS20BOpenRouter
  , testsGPTOSSLlamaCpp
  , testsGPT53CodexOpenRouter
  , testsGPT53ChatOpenRouter
  , testsGPT54ProOpenRouter
  , testsGPT54OpenRouter
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import UniversalLLM.Models.OpenAI.GPT (GPTOSS(..), GPTOSS20B(..), GPT53Codex(..), GPT53Chat(..), GPT54Pro(..), GPT54(..))
import Protocol.OpenAITests
import qualified StandardTests as ST
import qualified ComposableProviderTests as CPT
import TestCache (ResponseProvider)
import TestHelpers (testModel, testModelOffline)
import Test.Hspec (Spec, describe)

-- | Test GPT-OSS via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPTOSSOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPTOSSOpenRouter provider = do
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
      -- Note: GPT-OSS does not support vision (no official multimodal weights, no mmproj).

    describe "Standard Tests" $
      testModel route (GPTOSS `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

    describe "Composable Provider Tests" $
      testModelOffline route (GPTOSS `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

-- | Test GPT-OSS-20B via OpenRouter -- the smaller model in the same
-- family as 'GPTOSS' (120B). Unlike 'GPTOSS' via OpenRouter, this claims
-- 'UniversalLLM.HasJSON' (added alongside 'UniversalLLM.Models.OpenAI.GPT.GPTOSS20B'
-- itself specifically so it could be verified, not left unclaimed like the
-- 120B OpenRouter instance) -- 'ST.json' below is what verifies that claim.
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPTOSS20BOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPTOSS20BOpenRouter provider = do
  describe "GPT-OSS-20B via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-oss-20b"
      toolCalling provider "openai/gpt-oss-20b"
      acceptsToolResults provider "openai/gpt-oss-20b"
      acceptsToolResultNoTools provider "openai/gpt-oss-20b"
      acceptsToolResultToolGone provider "openai/gpt-oss-20b"
      acceptsStaleToolInHistory provider "openai/gpt-oss-20b"
      acceptsOldToolCallStillAvailable provider "openai/gpt-oss-20b"
      consecutiveUserMessages provider "openai/gpt-oss-20b"
      startsWithAssistant provider "openai/gpt-oss-20b"
      reasoningViaDetails provider "openai/gpt-oss-20b"
      toolCallingWithReasoning provider "openai/gpt-oss-20b"
      -- Note: GPT-OSS does not support vision (no official multimodal weights, no mmproj).
      -- Diagnosing 'ST.json's StandardTests failure (returned valid JSON,
      -- but only 1 of 3 requested colors): jsonMode confirms the wire-level
      -- shape is fine; jsonSchemaShapeCompliance/ignoresJSONSchemaShape
      -- distinguish "doesn't really honor response_format's schema" (like
      -- GLM, see 'ignoresJSONSchemaShape's Haddock) from "honors the schema
      -- but under-delivers on requested content," which point at very
      -- different fixes (drop 'UniversalLLM.HasJSON' entirely vs. a content
      -- quality caveat).
      jsonMode provider "openai/gpt-oss-20b"
      jsonSchemaShapeCompliance provider "openai/gpt-oss-20b"

    describe "Standard Tests" $
      testModel route (GPTOSS20B `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation, ST.json ]

    describe "Composable Provider Tests" $
      testModelOffline route (GPTOSS20B `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

-- | Test GPT-OSS via llama.cpp
--
-- Takes the canonicalized model name as determined by querying the llama.cpp
-- server. The model name is extracted from the loaded GGUF file.
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPTOSSLlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsGPTOSSLlamaCpp provider modelName = do
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
      -- Note: GPT-OSS does not support vision (no official multimodal weights, no mmproj).

    describe "Standard Tests" $
      testModel route (GPTOSS `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.json ]

-- | Test GPT-5.3-Codex via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT53CodexOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT53CodexOpenRouter provider = do
  describe "GPT-5.3-Codex via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.3-codex"
      toolCalling provider "openai/gpt-5.3-codex"
      acceptsToolResults provider "openai/gpt-5.3-codex"
      acceptsToolResultNoTools provider "openai/gpt-5.3-codex"
      acceptsToolResultToolGone provider "openai/gpt-5.3-codex"
      acceptsStaleToolInHistory provider "openai/gpt-5.3-codex"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.3-codex"
      consecutiveUserMessages provider "openai/gpt-5.3-codex"
      startsWithAssistant provider "openai/gpt-5.3-codex"
      encryptedReasoning provider "openai/gpt-5.3-codex"
      visionPng provider "openai/gpt-5.3-codex"
      visionJpeg provider "openai/gpt-5.3-codex"
      visionMultipleImages provider "openai/gpt-5.3-codex"

    describe "Standard Tests" $
      testModel route (GPT53Codex `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.hiddenReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages, ST.json ]

-- | Test GPT-5.3-Chat via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT53ChatOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT53ChatOpenRouter provider = do
  describe "GPT-5.3-Chat via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.3-chat"
      toolCalling provider "openai/gpt-5.3-chat"
      acceptsToolResults provider "openai/gpt-5.3-chat"
      acceptsToolResultNoTools provider "openai/gpt-5.3-chat"
      acceptsToolResultToolGone provider "openai/gpt-5.3-chat"
      acceptsStaleToolInHistory provider "openai/gpt-5.3-chat"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.3-chat"
      consecutiveUserMessages provider "openai/gpt-5.3-chat"
      startsWithAssistant provider "openai/gpt-5.3-chat"
      -- Note: GPT-5.3-Chat does NOT support reasoning via OpenRouter (no reasoning_details returned)
      visionPng provider "openai/gpt-5.3-chat"
      visionJpeg provider "openai/gpt-5.3-chat"
      visionMultipleImages provider "openai/gpt-5.3-chat"

    describe "Standard Tests" $
      testModel route (GPT53Chat `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.vision, ST.visionJpeg, ST.visionMultipleImages, ST.json ]

-- | Test GPT-5.4-Pro via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT54ProOpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT54ProOpenRouter provider = do
  describe "GPT-5.4-Pro via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.4-pro"
      toolCalling provider "openai/gpt-5.4-pro"
      acceptsToolResults provider "openai/gpt-5.4-pro"
      acceptsToolResultNoTools provider "openai/gpt-5.4-pro"
      acceptsToolResultToolGone provider "openai/gpt-5.4-pro"
      acceptsStaleToolInHistory provider "openai/gpt-5.4-pro"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.4-pro"
      consecutiveUserMessages provider "openai/gpt-5.4-pro"
      startsWithAssistant provider "openai/gpt-5.4-pro"
      encryptedReasoning provider "openai/gpt-5.4-pro"
      visionPng provider "openai/gpt-5.4-pro"
      visionJpeg provider "openai/gpt-5.4-pro"
      visionMultipleImages provider "openai/gpt-5.4-pro"

    describe "Standard Tests" $
      testModel route (GPT54Pro `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.hiddenReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages, ST.json ]

-- | Test GPT-5.4 via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsGPT54OpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsGPT54OpenRouter provider = do
  describe "GPT-5.4 via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "openai/gpt-5.4"
      toolCalling provider "openai/gpt-5.4"
      acceptsToolResults provider "openai/gpt-5.4"
      acceptsToolResultNoTools provider "openai/gpt-5.4"
      acceptsToolResultToolGone provider "openai/gpt-5.4"
      acceptsStaleToolInHistory provider "openai/gpt-5.4"
      acceptsOldToolCallStillAvailable provider "openai/gpt-5.4"
      consecutiveUserMessages provider "openai/gpt-5.4"
      startsWithAssistant provider "openai/gpt-5.4"
      encryptedReasoning provider "openai/gpt-5.4"
      visionPng provider "openai/gpt-5.4"
      visionJpeg provider "openai/gpt-5.4"
      visionMultipleImages provider "openai/gpt-5.4"

    describe "Standard Tests" $
      testModel route (GPT54 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.hiddenReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages, ST.json ]
