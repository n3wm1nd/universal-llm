{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{- |
Module: Models.Amazon.Nova2Lite

Model test suite for Amazon Nova 2 Lite

This module tests Amazon Nova 2 Lite when accessed through OpenRouter.
Nova has specific quirks around tool usage.

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (proper tool_calls format)
✓ Consecutive user messages
✓ History starting with assistant message
✓ Fabricated tool history (accepts with tools present)
✓ Vision (PNG, JPEG, multiple images)
✗ Tool result with no tools field (toolConfig required)
✗ Reasoning exposure (reasons internally, no reasoning_details in response)

= Provider-Specific Quirks

__OpenRouter/Amazon Bedrock:__
  Requires toolConfig (tools) field to be present when tool calls/results in history
  The field just needs to exist - doesn't validate tools match the history
  Cannot use fabricated tool history without setting tools field
  Has reasoning capabilities but doesn't expose reasoning_details via API
  Requires empty content fields to be normalized (cannot be null)

-}

module Models.Amazon.Nova2Lite (testsOpenRouter) where

import UniversalLLM (route, via)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.Amazon.Nova (Nova2Lite(..))
import Protocol.OpenAITests
import Protocol.CacheCoherency (CacheNormalized(..), normalizeOpenAIRequestWith, normalizeNovaMessage)
import qualified UniversalLLM.Protocols.OpenAI as OP
import qualified StandardTests as ST
import qualified ComposableProviderTests as CPT
import TestCache (ResponseProvider)
import TestHelpers (testModel, testModelOffline)
import Test.Hspec (Spec, describe, HasCallStack)

-- | Nova normalizes @content: ""@ to @content: null@ via @normalizeEmptyContent@
-- in its provider chain.  Override the default OpenAI message instance to match.
instance {-# OVERLAPPING #-} CacheNormalized Nova2Lite OpenRouter OP.OpenAIMessage where
  cacheNormalize = normalizeNovaMessage

instance {-# OVERLAPPING #-} CacheNormalized Nova2Lite OpenRouter OP.OpenAIRequest where
  cacheNormalize = normalizeOpenAIRequestWith normalizeNovaMessage

-- | Test Amazon Nova 2 Lite via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpenRouter :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "Amazon Nova 2 Lite via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "amazon/nova-2-lite-v1"
      toolCalling provider "amazon/nova-2-lite-v1"
      acceptsFabricatedToolHistory provider "amazon/nova-2-lite-v1"
      rejectsToolResultWithoutToolConfig provider "amazon/nova-2-lite-v1"
      acceptsToolResultToolGone provider "amazon/nova-2-lite-v1"
      acceptsStaleToolInHistory provider "amazon/nova-2-lite-v1"
      acceptsOldToolCallStillAvailable provider "amazon/nova-2-lite-v1"
      consecutiveUserMessages provider "amazon/nova-2-lite-v1"
      startsWithAssistant provider "amazon/nova-2-lite-v1"
      acceptsHiddenReasoning provider "amazon/nova-2-lite-v1"
      visionPng provider "amazon/nova-2-lite-v1"
      visionJpeg provider "amazon/nova-2-lite-v1"
      visionMultipleImages provider "amazon/nova-2-lite-v1"

    describe "Standard Tests" $
      testModel route (Nova2Lite `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "Composable Provider Tests" $
      testModelOffline route (Nova2Lite `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]
