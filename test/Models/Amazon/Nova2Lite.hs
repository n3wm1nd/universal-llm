{-# LANGUAGE OverloadedStrings #-}

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
✗ Reasoning exposure (has reasoning but not accessible via API)

= Provider-Specific Quirks

__OpenRouter/Amazon Bedrock:__
  Requires toolConfig (tools) field to be present when tool calls/results in history
  The field just needs to exist - doesn't validate tools match the history
  Cannot use fabricated tool history without setting tools field
  Has reasoning capabilities but doesn't expose reasoning_details via API
  Requires empty content fields to be normalized (cannot be null)

-}

module Models.Amazon.Nova2Lite (testsOpenRouter) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.Amazon.Nova
  ( Nova2Lite(..)
  , nova2Lite
  )
import Protocol.OpenAITests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe, HasCallStack)

-- | Test Amazon Nova 2 Lite via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpenRouter :: HasCallStack => ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpenRouter provider = do
  describe "Amazon Nova 2 Lite via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "amazon/nova-2-lite-v1"
      toolCalling provider "amazon/nova-2-lite-v1"

      -- Note: acceptsToolResults fails - Nova requires toolConfig field when using tool results
      -- Error: "The toolConfig field must be defined when using toolUse and toolResult content blocks."
      -- Cannot use fabricated tool history without proper toolConfig
      -- acceptsToolResults provider "amazon/nova-2-lite-v1"

      -- Note: acceptsToolResultsWithoutReasoning also fails - same toolConfig requirement
      -- acceptsToolResultsWithoutReasoning provider "amazon/nova-2-lite-v1"

      -- Note: acceptsToolResultNoTools fails - Nova requires toolConfig field to be present
      -- even if empty. The field must exist when tool calls/results are in history.
      -- acceptsToolResultNoTools provider "amazon/nova-2-lite-v1"

      -- Note: acceptsToolResultToolGone fails - Nova requires toolConfig field when tool
      -- results are present. Doesn't validate that tools match history, just needs field set.
      -- acceptsToolResultToolGone provider "amazon/nova-2-lite-v1"

      -- Note: acceptsStaleToolInHistory would also fail - same toolConfig requirement
      -- acceptsStaleToolInHistory provider "amazon/nova-2-lite-v1"

      -- Note: acceptsOldToolCallStillAvailable would also fail - same toolConfig requirement
      -- acceptsOldToolCallStillAvailable provider "amazon/nova-2-lite-v1"

      consecutiveUserMessages provider "amazon/nova-2-lite-v1"
      startsWithAssistant provider "amazon/nova-2-lite-v1"

      -- Note: reasoningViaDetails fails - Nova has reasoning but doesn't expose it via API
      -- The model has reasoning capabilities but OpenRouter/Bedrock doesn't provide reasoning_details
      -- reasoningViaDetails provider "amazon/nova-2-lite-v1"

      -- Note: toolCallingWithReasoning fails - requires reasoning_details which Nova doesn't provide
      -- toolCallingWithReasoning provider "amazon/nova-2-lite-v1"

    describe "Standard Tests" $
      testModel nova2Lite (Model Nova2Lite OpenRouter) provider
        [ ST.text, ST.tools ]
