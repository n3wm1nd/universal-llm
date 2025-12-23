{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Models.Claude

Model test suite for Claude models (Sonnet, Haiku, Opus)

This module tests Claude Sonnet 4.5 when accessed through the Anthropic API.
Claude Sonnet 4.5 is a reasoning model with extended thinking capabilities.

= Discovered Capabilities

✓ Basic text responses
✓ Tool calling (native Anthropic tool format)
✓ Extended thinking (reasoning via thinking content blocks)
✓ Reasoning with tools (combines extended thinking with tool use)
✓ Consecutive user messages
✓ History starting with assistant message

= Provider-Specific Quirks

__Anthropic API:__
  Uses native Anthropic protocol (not OpenAI-compatible)
  Supports tools natively via tool_use content blocks
  Extended thinking exposed via thinking content blocks
  Requires system prompts in separate system field
  OAuth authentication supported via ANTHROPIC_OAUTH_TOKEN

-}

module Models.Claude (testsSonnet45) where

import UniversalLLM.Core.Types (Model(..))
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import UniversalLLM.Providers.Anthropic (Anthropic(..))
import Protocol.AnthropicTests
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import qualified TestModels
import Test.Hspec (Spec, describe)

-- | Test Claude Sonnet 4.5 via Anthropic API
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsSonnet45 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsSonnet45 provider = do
  describe "Claude Sonnet 4.5 (Anthropic)" $ do
    describe "Protocol" $ do
      basicText provider
      toolCalling provider
      consecutiveUserMessages provider
      startsWithAssistant provider
      reasoning provider
      toolCallingWithReasoning provider

    describe "Standard Tests" $
      testModel TestModels.anthropicSonnet45Reasoning (Model TestModels.ClaudeSonnet45WithReasoning Anthropic) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]
