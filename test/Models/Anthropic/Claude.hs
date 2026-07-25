{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.Anthropic.Claude

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

module Models.Anthropic.Claude
  ( testsSonnet45
  , testsSonnet46
  , testsSonnet5
  , testsHaiku45
  , testsOpus46
  , testsOpus48
  , testsOpus5
  , testsFable5
  , testsSonnet46OpenRouter
  , testsSonnet5OpenRouter
  , testsOpus48OpenRouter
  , testsOpus5OpenRouter
  ) where

import UniversalLLM (route, via)
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse, model)
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import UniversalLLM.Providers.Anthropic (Anthropic(..), AnthropicOAuth(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Models.Anthropic.Claude
  ( ClaudeSonnet45(..)
  , ClaudeSonnet46(..)
  , ClaudeSonnet5(..)
  , ClaudeHaiku45(..)
  , ClaudeOpus46(..)
  , ClaudeOpus48(..)
  , ClaudeOpus5(..)
  , ClaudeFable5(..)
  )
import Protocol.AnthropicTests
import qualified Protocol.AnthropicOAuthBlacklist as Blacklist
import qualified Protocol.OpenAITests as OAI
import qualified StandardTests as ST
import qualified ComposableProviderTests as CPT
import TestCache (ResponseProvider)
import TestHelpers (testModel, testModelOffline)
import Test.Hspec (Spec, describe)

-- | Test Claude Sonnet 4.5 via Anthropic API
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsSonnet45 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsSonnet45 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-sonnet-4-5-20250929" })
  describe "Claude Sonnet 4.5 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-sonnet-4-5-20250929"
      visionJpeg oauthProvider "claude-sonnet-4-5-20250929"
      visionMultipleImages oauthProvider "claude-sonnet-4-5-20250929"

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel route (ClaudeSonnet45 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "Composable Provider Tests" $
      testModelOffline route (ClaudeSonnet45 `via` AnthropicOAuth)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeSonnet45 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"      -- Previously blacklisted, now works directly
        , ST.toolWithName "read_file" -- Previously blacklisted, now works directly
        , ST.toolWithName "echo"      -- Always worked normally
        ]

-- | Test Claude Sonnet 4.6 via Anthropic API
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsSonnet46 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsSonnet46 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-sonnet-4-6" })
  describe "Claude Sonnet 4.6 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-sonnet-4-6"
      visionJpeg oauthProvider "claude-sonnet-4-6"
      visionMultipleImages oauthProvider "claude-sonnet-4-6"

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel route (ClaudeSonnet46 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeSonnet46 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"      -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "read_file" -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "echo"      -- Not blacklisted, should work normally
        ]

-- | Test Claude Sonnet 5 via Anthropic API
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsSonnet5 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsSonnet5 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-sonnet-5" })
  describe "Claude Sonnet 5 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-sonnet-5"
      visionJpeg oauthProvider "claude-sonnet-5"
      visionMultipleImages oauthProvider "claude-sonnet-5"

    describe "Standard Tests" $
      testModel route (ClaudeSonnet5 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeSonnet5 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"
        , ST.toolWithName "read_file"
        , ST.toolWithName "echo"
        ]

-- | Test Claude Haiku 4.5 via Anthropic API
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsHaiku45 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsHaiku45 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-haiku-4-5-20251001" })
  describe "Claude Haiku 4.5 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-haiku-4-5-20251001"
      visionJpeg oauthProvider "claude-haiku-4-5-20251001"

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel route (ClaudeHaiku45 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeHaiku45 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"      -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "read_file" -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "echo"      -- Not blacklisted, should work normally
        ]

-- | Test Claude Opus 4.6 via Anthropic API
--
-- Opus 4.6 uses adaptive thinking with effort parameter instead of budget_tokens.
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpus46 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsOpus46 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-opus-4-6" })
  describe "Claude Opus 4.6 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      adaptiveReasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-opus-4-6"
      visionJpeg oauthProvider "claude-opus-4-6"
      visionMultipleImages oauthProvider "claude-opus-4-6"

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel route (ClaudeOpus46 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeOpus46 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"      -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "read_file" -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "echo"      -- Not blacklisted, should work normally
        ]

-- | Test Claude Opus 4.8 via Anthropic API
--
-- Opus 4.8 uses adaptive thinking (always-on) with effort parameter.
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpus48 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsOpus48 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-opus-4-8" })
  describe "Claude Opus 4.8 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      adaptiveReasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-opus-4-8"
      visionJpeg oauthProvider "claude-opus-4-8"
      visionMultipleImages oauthProvider "claude-opus-4-8"

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel route (ClaudeOpus48 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImagesHinted ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeOpus48 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"
        , ST.toolWithName "read_file"
        , ST.toolWithName "echo"
        ]

-- | Test Claude Opus 5 via Anthropic API
--
-- Opus 5 uses adaptive thinking (always-on) with effort parameter.
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsOpus5 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsOpus5 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-opus-5" })
  describe "Claude Opus 5 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      adaptiveReasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-opus-5"
      visionJpeg oauthProvider "claude-opus-5"
      visionMultipleImages oauthProvider "claude-opus-5"

    describe "Standard Tests" $
      testModel route (ClaudeOpus5 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeOpus5 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"
        , ST.toolWithName "read_file"
        , ST.toolWithName "echo"
        ]

-- | Test Claude Fable 5 via Anthropic API
--
-- Fable 5 uses adaptive thinking (always-on) and is Anthropic's most capable widely released model.
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsFable5 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsFable5 provider = do
  let oauthProvider isExpected req = provider isExpected (Anthropic.withMagicSystemPrompt req { model = "claude-fable-5" })
  describe "Claude Fable 5 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      adaptiveReasoning oauthProvider
      toolCallingWithReasoning oauthProvider
      visionPng oauthProvider "claude-fable-5"
      visionJpeg oauthProvider "claude-fable-5"
      visionMultipleImagesHinted oauthProvider "claude-fable-5"

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel route (ClaudeFable5 `via` AnthropicOAuth) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.vision, ST.visionJpeg, ST.visionMultipleImagesHinted ]

    describe "OAuth Provider Tests" $
      testModel route (ClaudeFable5 `via` AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"
        , ST.toolWithName "read_file"
        , ST.toolWithName "echo"
        ]

-- | Test Claude Sonnet 4.6 via OpenRouter
--
-- Unlike the native Anthropic route, this goes over the OpenAI-compatible
-- protocol, so reasoning is exposed/threaded via reasoning_details (the
-- same OpenRouter mechanism used by e.g. Gemini and GPT-OSS), not native
-- thinking content blocks. No 'UniversalLLM.HasJSON' claim -- see the
-- Haddock above the OpenRouter instances in the model module.
testsSonnet46OpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsSonnet46OpenRouter provider = do
  describe "Claude Sonnet 4.6 via OpenRouter" $ do
    describe "Protocol" $ do
      OAI.basicText provider "anthropic/claude-sonnet-4-6"
      OAI.toolCalling provider "anthropic/claude-sonnet-4-6"
      OAI.acceptsToolResultNoTools provider "anthropic/claude-sonnet-4-6"
      OAI.acceptsToolResultToolGone provider "anthropic/claude-sonnet-4-6"
      OAI.acceptsStaleToolInHistory provider "anthropic/claude-sonnet-4-6"
      OAI.acceptsOldToolCallStillAvailable provider "anthropic/claude-sonnet-4-6"
      OAI.consecutiveUserMessages provider "anthropic/claude-sonnet-4-6"
      OAI.startsWithAssistant provider "anthropic/claude-sonnet-4-6"
      -- No OAI.reasoningViaDetails / OAI.toolCallingWithReasoning. Unlike
      -- Nova2Lite (OAI.acceptsHiddenReasoning: reasoning_details is
      -- deterministically NEVER exposed), Claude's adaptive reasoning at
      -- this library's fixed reasoning_effort="low" (OpenAI.openRouterReasoning)
      -- sometimes surfaces reasoning_details and sometimes doesn't -- e.g.
      -- Claude Sonnet 4.6 returned it (9 reasoning tokens) for "Use the
      -- get_weather function..." while Claude Sonnet 5 skipped reasoning
      -- entirely (no reasoning_content/reasoning_details/tool_calls, just a
      -- direct worked answer) on the plain "What is 15*23?" prompt --
      -- confirmed via assertHasReasoningDetails's diagnostic output. There's
      -- no fixed "always"/"never" to assert; it's the model's own per-call
      -- judgment, not a protocol guarantee.
      OAI.visionPng provider "anthropic/claude-sonnet-4-6"
      OAI.visionJpeg provider "anthropic/claude-sonnet-4-6"
      OAI.visionMultipleImages provider "anthropic/claude-sonnet-4-6"

    describe "Standard Tests" $
      testModel route (ClaudeSonnet46 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.openAIReasoningDetailsPreservation, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "Composable Provider Tests" $
      testModelOffline route (ClaudeSonnet46 `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

-- | Test Claude Sonnet 5 via OpenRouter
--
-- Same OpenAI-compatible reasoning_details mechanism as 'testsSonnet46OpenRouter'.
-- No 'UniversalLLM.HasJSON' claim -- see the Haddock above the OpenRouter
-- instances in the model module.
testsSonnet5OpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsSonnet5OpenRouter provider = do
  describe "Claude Sonnet 5 via OpenRouter" $ do
    describe "Protocol" $ do
      OAI.basicText provider "anthropic/claude-sonnet-5"
      OAI.toolCalling provider "anthropic/claude-sonnet-5"
      OAI.acceptsToolResultNoTools provider "anthropic/claude-sonnet-5"
      OAI.acceptsToolResultToolGone provider "anthropic/claude-sonnet-5"
      OAI.acceptsStaleToolInHistory provider "anthropic/claude-sonnet-5"
      OAI.acceptsOldToolCallStillAvailable provider "anthropic/claude-sonnet-5"
      OAI.consecutiveUserMessages provider "anthropic/claude-sonnet-5"
      OAI.startsWithAssistant provider "anthropic/claude-sonnet-5"
      OAI.rejectsSystemMessageMidConversationAnthropic provider "anthropic/claude-sonnet-5"
      -- No OAI.reasoningViaDetails / OAI.toolCallingWithReasoning: Claude's
      -- adaptive reasoning is not deterministically exposed -- see the
      -- comment on 'testsSonnet46OpenRouter'.
      OAI.visionPng provider "anthropic/claude-sonnet-5"
      OAI.visionJpeg provider "anthropic/claude-sonnet-5"
      OAI.visionMultipleImages provider "anthropic/claude-sonnet-5"

    describe "Standard Tests" $
      -- ST.systemMessageMidConversation passes here via the model's
      -- systemMessagesFirst hoist (see the 'Routing' instance), not because
      -- Claude Sonnet 5 natively accepts a system message directly after a
      -- plain assistant turn -- it doesn't (see the
      -- rejectsSystemMessageMidConversationAnthropic Protocol probe above).
      testModel route (ClaudeSonnet5 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.openAIReasoningDetailsPreservation, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "Composable Provider Tests" $
      testModelOffline route (ClaudeSonnet5 `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

-- | Test Claude Opus 4.8 via OpenRouter
--
-- Same OpenAI-compatible reasoning_details mechanism as 'testsSonnet46OpenRouter'.
-- No 'UniversalLLM.HasJSON' claim -- see the Haddock above the OpenRouter
-- instances in the model module.
testsOpus48OpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpus48OpenRouter provider = do
  describe "Claude Opus 4.8 via OpenRouter" $ do
    describe "Protocol" $ do
      OAI.basicText provider "anthropic/claude-opus-4-8"
      OAI.toolCalling provider "anthropic/claude-opus-4-8"
      OAI.acceptsToolResultNoTools provider "anthropic/claude-opus-4-8"
      OAI.acceptsToolResultToolGone provider "anthropic/claude-opus-4-8"
      OAI.acceptsStaleToolInHistory provider "anthropic/claude-opus-4-8"
      OAI.acceptsOldToolCallStillAvailable provider "anthropic/claude-opus-4-8"
      OAI.consecutiveUserMessages provider "anthropic/claude-opus-4-8"
      OAI.startsWithAssistant provider "anthropic/claude-opus-4-8"
      OAI.rejectsSystemMessageMidConversationAnthropic provider "anthropic/claude-opus-4-8"
      -- No OAI.reasoningViaDetails / OAI.toolCallingWithReasoning: Claude's
      -- adaptive reasoning is not deterministically exposed -- see the
      -- comment on 'testsSonnet46OpenRouter'.
      OAI.visionPng provider "anthropic/claude-opus-4-8"
      OAI.visionJpeg provider "anthropic/claude-opus-4-8"
      OAI.visionMultipleImages provider "anthropic/claude-opus-4-8"

    describe "Standard Tests" $
      -- ST.systemMessageMidConversation passes here via the model's
      -- systemMessagesFirst hoist (see the 'Routing' instance), not because
      -- Claude Opus 4.8 natively accepts a system message directly after a
      -- plain assistant turn -- it doesn't (see the
      -- rejectsSystemMessageMidConversationAnthropic Protocol probe above).
      testModel route (ClaudeOpus48 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.openAIReasoningDetailsPreservation, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "Composable Provider Tests" $
      testModelOffline route (ClaudeOpus48 `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

-- | Test Claude Opus 5 via OpenRouter
--
-- Same OpenAI-compatible reasoning_details mechanism as 'testsSonnet46OpenRouter'.
-- No 'UniversalLLM.HasJSON' claim -- see the Haddock above the OpenRouter
-- instances in the model module.
testsOpus5OpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsOpus5OpenRouter provider = do
  describe "Claude Opus 5 via OpenRouter" $ do
    describe "Protocol" $ do
      OAI.basicText provider "anthropic/claude-opus-5"
      OAI.toolCalling provider "anthropic/claude-opus-5"
      OAI.acceptsToolResultNoTools provider "anthropic/claude-opus-5"
      OAI.acceptsToolResultToolGone provider "anthropic/claude-opus-5"
      OAI.acceptsStaleToolInHistory provider "anthropic/claude-opus-5"
      OAI.acceptsOldToolCallStillAvailable provider "anthropic/claude-opus-5"
      OAI.consecutiveUserMessages provider "anthropic/claude-opus-5"
      OAI.startsWithAssistant provider "anthropic/claude-opus-5"
      -- OAI.rejectsSystemMessageMidConversationAnthropic carried over as a
      -- pending probe rather than asserted: Opus 48/Sonnet 5 confirmed this
      -- quirk independently, but it hasn't been run against Opus 5 yet. The
      -- systemMessagesFirst hoist in the 'Routing' instance is applied
      -- either way, so ST.systemMessageMidConversation below passes
      -- regardless of the outcome once recorded.
      OAI.rejectsSystemMessageMidConversationAnthropic provider "anthropic/claude-opus-5"
      -- No OAI.reasoningViaDetails / OAI.toolCallingWithReasoning: Claude's
      -- adaptive reasoning is not deterministically exposed -- see the
      -- comment on 'testsSonnet46OpenRouter'.
      OAI.visionPng provider "anthropic/claude-opus-5"
      OAI.visionJpeg provider "anthropic/claude-opus-5"
      OAI.visionMultipleImages provider "anthropic/claude-opus-5"

    describe "Standard Tests" $
      -- ST.systemMessageMidConversation passes here via the model's
      -- systemMessagesFirst hoist (see the 'Routing' instance).
      testModel route (ClaudeOpus5 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning, ST.openAIReasoningDetailsPreservation, ST.vision, ST.visionJpeg, ST.visionMultipleImages ]

    describe "Composable Provider Tests" $
      testModelOffline route (ClaudeOpus5 `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]
