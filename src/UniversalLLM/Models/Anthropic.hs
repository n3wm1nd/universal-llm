{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{- |
Module: UniversalLLM.Models.Anthropic
Description: Production-ready Anthropic model definitions

This module provides tested, production-ready definitions for Anthropic's Claude models.
These models can be used directly in applications without redefinition.

= Available Models

* 'ClaudeSonnet45' - Claude Sonnet 4.5 with reasoning and tool support (default)
* 'ClaudeSonnet45NoReason' - Sonnet 4.5 with tools only, no reasoning state (exception)

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Anthropic

-- Use with regular Anthropic API (with reasoning)
let model = Model ClaudeSonnet45 Anthropic
let provider = claudeSonnet45

-- Use with OAuth (with reasoning)
let oauthModel = Model ClaudeSonnet45 AnthropicOAuth
let oauthProvider = claudeSonnet45OAuth

-- Use without reasoning (rare)
let noReasonModel = Model ClaudeSonnet45NoReason Anthropic
let noReasonProvider = claudeSonnet45NoReason
@

= Authentication

Set environment variable:
- Regular API: @ANTHROPIC_API_KEY@
- OAuth: @ANTHROPIC_OAUTH_TOKEN@
-}

module UniversalLLM.Models.Anthropic
  ( -- * Model Types
    ClaudeSonnet45(..)
  , ClaudeSonnet45NoReason(..)
  , ClaudeOpus46(..)
    -- * Composable Providers
  , claudeSonnet45
  , claudeSonnet45NoReason
  , claudeSonnet45OAuth
  , claudeSonnet45NoReasonOAuth
  , claudeOpus46
  , claudeOpus46OAuth
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import UniversalLLM.Providers.Anthropic (Anthropic(..), AnthropicOAuth(..), OAuthToolsState)

--------------------------------------------------------------------------------
-- Claude Sonnet 4.5
--------------------------------------------------------------------------------

-- | Claude Sonnet 4.5 - Anthropic's latest reasoning model
--
-- Capabilities:
-- - Extended thinking via reasoning blocks
-- - Native tool support
-- - High-quality text generation
-- - Streaming responses
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet45 Anthropic) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45 Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeSonnet45 Anthropic) where
  type ReasoningState (Model ClaudeSonnet45 Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

-- | Composable provider for Claude Sonnet 4.5 with reasoning and tools
--
-- This is the default provider with full reasoning state tracking.
claudeSonnet45 :: ComposableProvider (Model ClaudeSonnet45 Anthropic) (Anthropic.AnthropicReasoningState, ((), ()))
claudeSonnet45 = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 Anthropic)

--------------------------------------------------------------------------------
-- Claude Sonnet 4.5 without Reasoning
--------------------------------------------------------------------------------

-- | Claude Sonnet 4.5 without explicit reasoning state
--
-- Use this variant when you want tool support but don't need reasoning state tracking.
-- This is the exception - most use cases should use ClaudeSonnet45 with reasoning.
data ClaudeSonnet45NoReason = ClaudeSonnet45NoReason deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet45NoReason Anthropic) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45NoReason Anthropic) where
  withTools = Anthropic.anthropicTools

-- | Composable provider for Claude Sonnet 4.5 without reasoning state
--
-- Provides tools only, without explicit reasoning state tracking.
claudeSonnet45NoReason :: ComposableProvider (Model ClaudeSonnet45NoReason Anthropic) ((), ())
claudeSonnet45NoReason = withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45NoReason Anthropic)

--------------------------------------------------------------------------------
-- Claude Opus 4.6
--------------------------------------------------------------------------------

-- | Claude Opus 4.6 - Anthropic's most capable model
--
-- Capabilities:
-- - Extended thinking via reasoning blocks
-- - Native tool support
-- - High-quality text generation
-- - Streaming responses
-- - Superior reasoning and analysis
data ClaudeOpus46 = ClaudeOpus46 deriving (Show, Eq)

instance ModelName (Model ClaudeOpus46 Anthropic) where
  modelName (Model _ _) = "claude-opus-4-6"

instance HasTools (Model ClaudeOpus46 Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeOpus46 Anthropic) where
  type ReasoningState (Model ClaudeOpus46 Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

-- | Composable provider for Claude Opus 4.6 with adaptive reasoning and tools
--
-- Uses Opus 4.6's adaptive thinking with effort parameter instead of budget_tokens.
claudeOpus46 :: ComposableProvider (Model ClaudeOpus46 Anthropic) (Anthropic.AnthropicReasoningState, ((), ()))
claudeOpus46 = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus46 Anthropic)

--------------------------------------------------------------------------------
-- OAuth Versions
--------------------------------------------------------------------------------

-- OAuth version for ClaudeSonnet45 (with reasoning and tool name workarounds)
instance ModelName (Model ClaudeSonnet45 AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45 AnthropicOAuth) where
  type ToolState (Model ClaudeSonnet45 AnthropicOAuth) = OAuthToolsState
  withTools = Anthropic.anthropicOAuthBlacklistedTools

instance HasReasoning (Model ClaudeSonnet45 AnthropicOAuth) where
  type ReasoningState (Model ClaudeSonnet45 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

-- | Composable provider for Claude Sonnet 4.5 via OAuth with reasoning
--
-- Combines reasoning state, OAuth tool workarounds, and the magic system prompt.
claudeSonnet45OAuth :: ComposableProvider (Model ClaudeSonnet45 AnthropicOAuth) (Anthropic.AnthropicReasoningState, (OAuthToolsState, ((), ())))
claudeSonnet45OAuth = withReasoning `chainProviders` withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 AnthropicOAuth)

-- OAuth version without reasoning
instance ModelName (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  type ToolState (Model ClaudeSonnet45NoReason AnthropicOAuth) = OAuthToolsState
  withTools = Anthropic.anthropicOAuthBlacklistedTools

-- | Composable provider for Claude Sonnet 4.5 via OAuth without reasoning
--
-- Includes OAuth tool workarounds and the magic system prompt, but no reasoning state.
claudeSonnet45NoReasonOAuth :: ComposableProvider (Model ClaudeSonnet45NoReason AnthropicOAuth) (OAuthToolsState, ((), ()))
claudeSonnet45NoReasonOAuth = withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45NoReason AnthropicOAuth)

-- OAuth version for ClaudeOpus46 (with reasoning and tool name workarounds)
instance ModelName (Model ClaudeOpus46 AnthropicOAuth) where
  modelName (Model _ _) = "claude-opus-4-6"

instance HasTools (Model ClaudeOpus46 AnthropicOAuth) where
  type ToolState (Model ClaudeOpus46 AnthropicOAuth) = OAuthToolsState
  withTools = Anthropic.anthropicOAuthBlacklistedTools

instance HasReasoning (Model ClaudeOpus46 AnthropicOAuth) where
  type ReasoningState (Model ClaudeOpus46 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

-- | Composable provider for Claude Opus 4.6 via OAuth with adaptive reasoning
--
-- Combines adaptive reasoning, OAuth tool workarounds, and the magic system prompt.
claudeOpus46OAuth :: ComposableProvider (Model ClaudeOpus46 AnthropicOAuth) (Anthropic.AnthropicReasoningState, (OAuthToolsState, ((), ())))
claudeOpus46OAuth = withReasoning `chainProviders` withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus46 AnthropicOAuth)
