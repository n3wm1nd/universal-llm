{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{- |
Module: UniversalLLM.Models.Anthropic.Claude
Description: Production-ready Anthropic Claude model definitions

This module provides tested, production-ready definitions for Anthropic's Claude models.
These models can be used directly in applications without redefinition.

= Available Models

* 'ClaudeSonnet45' - Claude Sonnet 4.5 with reasoning and tool support (default)
* 'ClaudeSonnet45NoReason' - Sonnet 4.5 with tools only, no reasoning state (exception)
* 'ClaudeSonnet46' - Claude Sonnet 4.6 with reasoning and tool support
* 'ClaudeHaiku45' - Claude Haiku 4.5 with reasoning and tool support
* 'ClaudeOpus46' - Claude Opus 4.6 with adaptive reasoning and tool support

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Anthropic.Claude

-- Use with regular Anthropic API (with reasoning)
let model = Model ClaudeSonnet45 Anthropic
let provider = route

-- Use with OAuth (with reasoning)
let oauthModel = Model ClaudeSonnet45 AnthropicOAuth
let provider = route

-- Use without reasoning (rare)
let noReasonModel = Model ClaudeSonnet45NoReason Anthropic
let provider = route
@

= Authentication

Set environment variable:
- Regular API: @ANTHROPIC_API_KEY@
- OAuth: @ANTHROPIC_OAUTH_TOKEN@
-}

module UniversalLLM.Models.Anthropic.Claude
  ( -- * Model Types
    ClaudeSonnet45(..)
  , ClaudeSonnet45NoReason(..)
  , ClaudeSonnet46(..)
  , ClaudeHaiku45(..)
  , ClaudeOpus46(..)
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import UniversalLLM.Providers.Anthropic (Anthropic(..), AnthropicOAuth(..))

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

instance Routing (Model ClaudeSonnet45 Anthropic) where
  type RoutingState (Model ClaudeSonnet45 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 Anthropic)

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

instance Routing (Model ClaudeSonnet45NoReason Anthropic) where
  type RoutingState (Model ClaudeSonnet45NoReason Anthropic) = ((), ())
  route = withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45NoReason Anthropic)

--------------------------------------------------------------------------------
-- Claude Sonnet 4.6
--------------------------------------------------------------------------------

-- | Claude Sonnet 4.6 - Anthropic's latest Sonnet model
--
-- Capabilities:
-- - Extended thinking via reasoning blocks
-- - Native tool support
-- - High-quality text generation
-- - Streaming responses
-- - Improved performance over 4.5
data ClaudeSonnet46 = ClaudeSonnet46 deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet46 Anthropic) where
  modelName (Model _ _) = "claude-sonnet-4-6"

instance HasTools (Model ClaudeSonnet46 Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeSonnet46 Anthropic) where
  type ReasoningState (Model ClaudeSonnet46 Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

instance Routing (Model ClaudeSonnet46 Anthropic) where
  type RoutingState (Model ClaudeSonnet46 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet46 Anthropic)

--------------------------------------------------------------------------------
-- Claude Haiku 4.5
--------------------------------------------------------------------------------

-- | Claude Haiku 4.5 - Fast and efficient model
--
-- Capabilities:
-- - Extended thinking via reasoning blocks
-- - Native tool support
-- - Fast responses
-- - Cost-effective
data ClaudeHaiku45 = ClaudeHaiku45 deriving (Show, Eq)

instance ModelName (Model ClaudeHaiku45 Anthropic) where
  modelName (Model _ _) = "claude-haiku-4-5-20251001"

instance HasTools (Model ClaudeHaiku45 Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeHaiku45 Anthropic) where
  type ReasoningState (Model ClaudeHaiku45 Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

instance Routing (Model ClaudeHaiku45 Anthropic) where
  type RoutingState (Model ClaudeHaiku45 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeHaiku45 Anthropic)

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

instance Routing (Model ClaudeOpus46 Anthropic) where
  type RoutingState (Model ClaudeOpus46 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus46 Anthropic)

--------------------------------------------------------------------------------
-- OAuth Versions
--------------------------------------------------------------------------------

-- OAuth version for ClaudeSonnet45 (with reasoning)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeSonnet45 AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeSonnet45 AnthropicOAuth) where
  type ReasoningState (Model ClaudeSonnet45 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

instance Routing (Model ClaudeSonnet45 AnthropicOAuth) where
  type RoutingState (Model ClaudeSonnet45 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 AnthropicOAuth)

-- OAuth version without reasoning
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance Routing (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  type RoutingState (Model ClaudeSonnet45NoReason AnthropicOAuth) = ((), ((), ()))
  route = withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45NoReason AnthropicOAuth)

-- OAuth version for ClaudeSonnet46 (with reasoning)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeSonnet46 AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-6"

instance HasTools (Model ClaudeSonnet46 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeSonnet46 AnthropicOAuth) where
  type ReasoningState (Model ClaudeSonnet46 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

instance Routing (Model ClaudeSonnet46 AnthropicOAuth) where
  type RoutingState (Model ClaudeSonnet46 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet46 AnthropicOAuth)

-- OAuth version for ClaudeHaiku45 (with reasoning)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeHaiku45 AnthropicOAuth) where
  modelName (Model _ _) = "claude-haiku-4-5-20251001"

instance HasTools (Model ClaudeHaiku45 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeHaiku45 AnthropicOAuth) where
  type ReasoningState (Model ClaudeHaiku45 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

instance Routing (Model ClaudeHaiku45 AnthropicOAuth) where
  type RoutingState (Model ClaudeHaiku45 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeHaiku45 AnthropicOAuth)

-- OAuth version for ClaudeOpus46 (with adaptive reasoning)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeOpus46 AnthropicOAuth) where
  modelName (Model _ _) = "claude-opus-4-6"

instance HasTools (Model ClaudeOpus46 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeOpus46 AnthropicOAuth) where
  type ReasoningState (Model ClaudeOpus46 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

instance Routing (Model ClaudeOpus46 AnthropicOAuth) where
  type RoutingState (Model ClaudeOpus46 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus46 AnthropicOAuth)
