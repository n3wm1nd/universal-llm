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

module Models.Claude (testsSonnet45, testsSonnet46, testsHaiku45, testsOpus46) where

import UniversalLLM (Model(..))
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse, model)
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import UniversalLLM.Providers.Anthropic (Anthropic(..), AnthropicOAuth(..))
import UniversalLLM.Models.Anthropic
  ( ClaudeSonnet45(..)
  , ClaudeSonnet46(..)
  , ClaudeHaiku45(..)
  , ClaudeOpus46(..)
  , claudeSonnet45OAuth
  , claudeSonnet46OAuth
  , claudeHaiku45OAuth
  , claudeOpus46OAuth
  )
import Protocol.AnthropicTests
import qualified Protocol.AnthropicOAuthBlacklist as Blacklist
import qualified StandardTests as ST
import TestCache (ResponseProvider)
import TestHelpers (testModel)
import Test.Hspec (Spec, describe)

-- | Test Claude Sonnet 4.5 via Anthropic API
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsSonnet45 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsSonnet45 provider = do
  let oauthProvider req = provider (Anthropic.withMagicSystemPrompt req { model = "claude-sonnet-4-5-20250929" })
  describe "Claude Sonnet 4.5 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      toolCallingWithReasoning oauthProvider

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel claudeSonnet45OAuth (Model ClaudeSonnet45 Anthropic.AnthropicOAuth) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning ]

    describe "OAuth Provider Tests" $
      testModel claudeSonnet45OAuth (Model ClaudeSonnet45 Anthropic.AnthropicOAuth) provider
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
  let oauthProvider req = provider (Anthropic.withMagicSystemPrompt req { model = "claude-sonnet-4-6" })
  describe "Claude Sonnet 4.6 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      toolCallingWithReasoning oauthProvider

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel claudeSonnet46OAuth (Model ClaudeSonnet46 Anthropic.AnthropicOAuth) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning ]

    describe "OAuth Provider Tests" $
      testModel claudeSonnet46OAuth (Model ClaudeSonnet46 Anthropic.AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"      -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "read_file" -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "echo"      -- Not blacklisted, should work normally
        ]

-- | Test Claude Haiku 4.5 via Anthropic API
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsHaiku45 :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
testsHaiku45 provider = do
  let oauthProvider req = provider (Anthropic.withMagicSystemPrompt req { model = "claude-haiku-4-5-20251001" })
  describe "Claude Haiku 4.5 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      toolCallingWithReasoning oauthProvider

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel claudeHaiku45OAuth (Model ClaudeHaiku45 Anthropic.AnthropicOAuth) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning ]

    describe "OAuth Provider Tests" $
      testModel claudeHaiku45OAuth (Model ClaudeHaiku45 Anthropic.AnthropicOAuth) provider
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
  let oauthProvider req = provider (Anthropic.withMagicSystemPrompt req { model = "claude-opus-4-6" })
  describe "Claude Opus 4.6 via Anthropic" $ do
    describe "Protocol" $ do
      basicText oauthProvider
      toolCalling oauthProvider
      consecutiveUserMessages oauthProvider
      startsWithAssistant oauthProvider
      reasoning oauthProvider
      adaptiveReasoning oauthProvider
      toolCallingWithReasoning oauthProvider

    -- NOTE: Blacklist removed as of 2025 - keeping test structure for potential future use
    describe "OAuth Tool Name Blacklist" $ do
      -- Blacklist.blacklistProbes oauthProvider
      return ()

    describe "Standard Tests" $
      testModel claudeOpus46OAuth (Model ClaudeOpus46 Anthropic.AnthropicOAuth) provider
        [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.reasoningWithToolsModifiedReasoning ]

    describe "OAuth Provider Tests" $
      testModel claudeOpus46OAuth (Model ClaudeOpus46 Anthropic.AnthropicOAuth) provider
        [ ST.text
        , ST.tools
        , ST.toolWithName "grep"      -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "read_file" -- Blacklisted, should work via prefix/unprefix
        , ST.toolWithName "echo"      -- Not blacklisted, should work normally
        ]
