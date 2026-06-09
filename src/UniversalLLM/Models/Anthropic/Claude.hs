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
* 'ClaudeOpus46' - Claude Opus 4.6 with adaptive reasoning and tool support (legacy)
* 'ClaudeOpus48' - Claude Opus 4.8 with adaptive reasoning and tool support
* 'ClaudeFable5' - Claude Fable 5 with adaptive reasoning and tool support (most capable widely released)

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
  , ClaudeOpus48(..)
  , ClaudeFable5(..)
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

instance HasVision (Model ClaudeSonnet45 Anthropic) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeSonnet45 Anthropic) where
  type RoutingState (Model ClaudeSonnet45 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 Anthropic)

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

instance HasVision (Model ClaudeSonnet45NoReason Anthropic) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeSonnet45NoReason Anthropic) where
  type RoutingState (Model ClaudeSonnet45NoReason Anthropic) = ((), ((), ()))
  route = withTools `chainProviders` withVision `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45NoReason Anthropic)

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

instance HasVision (Model ClaudeSonnet46 Anthropic) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeSonnet46 Anthropic) where
  type RoutingState (Model ClaudeSonnet46 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet46 Anthropic)

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

instance HasVision (Model ClaudeHaiku45 Anthropic) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeHaiku45 Anthropic) where
  type RoutingState (Model ClaudeHaiku45 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeHaiku45 Anthropic)

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

instance HasVision (Model ClaudeOpus46 Anthropic) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeOpus46 Anthropic) where
  type RoutingState (Model ClaudeOpus46 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus46 Anthropic)

--------------------------------------------------------------------------------
-- Claude Opus 4.8
--------------------------------------------------------------------------------

-- | Claude Opus 4.8 - Anthropic's most capable Opus-tier model
--
-- Capabilities:
-- - Adaptive thinking (always-on, via effort parameter)
-- - Native tool support
-- - High-quality text generation
-- - Streaming responses
-- - Superior reasoning for complex, long-horizon agentic work
data ClaudeOpus48 = ClaudeOpus48 deriving (Show, Eq)

instance ModelName (Model ClaudeOpus48 Anthropic) where
  modelName (Model _ _) = "claude-opus-4-8"

instance HasTools (Model ClaudeOpus48 Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeOpus48 Anthropic) where
  type ReasoningState (Model ClaudeOpus48 Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

instance HasVision (Model ClaudeOpus48 Anthropic) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeOpus48 Anthropic) where
  type RoutingState (Model ClaudeOpus48 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus48 Anthropic)

--------------------------------------------------------------------------------
-- Claude Fable 5
--------------------------------------------------------------------------------

-- | Claude Fable 5 - Anthropic's most capable widely released model
--
-- Capabilities:
-- - Adaptive thinking (always-on, via effort parameter)
-- - Native tool support
-- - High-quality text generation
-- - Streaming responses
-- - Most demanding reasoning and long-horizon agentic work
data ClaudeFable5 = ClaudeFable5 deriving (Show, Eq)

instance ModelName (Model ClaudeFable5 Anthropic) where
  modelName (Model _ _) = "claude-fable-5"

instance HasTools (Model ClaudeFable5 Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeFable5 Anthropic) where
  type ReasoningState (Model ClaudeFable5 Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

instance HasVision (Model ClaudeFable5 Anthropic) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeFable5 Anthropic) where
  type RoutingState (Model ClaudeFable5 Anthropic) = (Anthropic.AnthropicReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeFable5 Anthropic)

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

instance HasVision (Model ClaudeSonnet45 AnthropicOAuth) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeSonnet45 AnthropicOAuth) where
  type RoutingState (Model ClaudeSonnet45 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ((), ()))))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 AnthropicOAuth)

-- OAuth version without reasoning
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasVision (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeSonnet45NoReason AnthropicOAuth) where
  type RoutingState (Model ClaudeSonnet45NoReason AnthropicOAuth) = ((), ((), ((), ())))
  route = withTools `chainProviders` withVision `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45NoReason AnthropicOAuth)

-- OAuth version for ClaudeSonnet46 (with reasoning)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeSonnet46 AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-6"

instance HasTools (Model ClaudeSonnet46 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeSonnet46 AnthropicOAuth) where
  type ReasoningState (Model ClaudeSonnet46 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

instance HasVision (Model ClaudeSonnet46 AnthropicOAuth) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeSonnet46 AnthropicOAuth) where
  type RoutingState (Model ClaudeSonnet46 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ((), ()))))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet46 AnthropicOAuth)

-- OAuth version for ClaudeHaiku45 (with reasoning and vision)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeHaiku45 AnthropicOAuth) where
  modelName (Model _ _) = "claude-haiku-4-5-20251001"

instance HasTools (Model ClaudeHaiku45 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeHaiku45 AnthropicOAuth) where
  type ReasoningState (Model ClaudeHaiku45 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

instance HasVision (Model ClaudeHaiku45 AnthropicOAuth) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeHaiku45 AnthropicOAuth) where
  type RoutingState (Model ClaudeHaiku45 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ((), ()))))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeHaiku45 AnthropicOAuth)

-- OAuth version for ClaudeOpus48 (with adaptive reasoning)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeOpus48 AnthropicOAuth) where
  modelName (Model _ _) = "claude-opus-4-8"

instance HasTools (Model ClaudeOpus48 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeOpus48 AnthropicOAuth) where
  type ReasoningState (Model ClaudeOpus48 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

instance HasVision (Model ClaudeOpus48 AnthropicOAuth) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeOpus48 AnthropicOAuth) where
  type RoutingState (Model ClaudeOpus48 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ((), ()))))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus48 AnthropicOAuth)

-- OAuth version for ClaudeFable5 (with adaptive reasoning)
instance ModelName (Model ClaudeFable5 AnthropicOAuth) where
  modelName (Model _ _) = "claude-fable-5"

instance HasTools (Model ClaudeFable5 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeFable5 AnthropicOAuth) where
  type ReasoningState (Model ClaudeFable5 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

instance HasVision (Model ClaudeFable5 AnthropicOAuth) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeFable5 AnthropicOAuth) where
  type RoutingState (Model ClaudeFable5 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ((), ()))))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeFable5 AnthropicOAuth)

-- OAuth version for ClaudeOpus46 (with adaptive reasoning)
-- NOTE: Tool name blacklist workaround removed as of 2025 - API no longer blocks tool names
instance ModelName (Model ClaudeOpus46 AnthropicOAuth) where
  modelName (Model _ _) = "claude-opus-4-6"

instance HasTools (Model ClaudeOpus46 AnthropicOAuth) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeOpus46 AnthropicOAuth) where
  type ReasoningState (Model ClaudeOpus46 AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicAdaptiveReasoning

instance HasVision (Model ClaudeOpus46 AnthropicOAuth) where
  withVision = Anthropic.anthropicVision

instance Routing (Model ClaudeOpus46 AnthropicOAuth) where
  type RoutingState (Model ClaudeOpus46 AnthropicOAuth) = (Anthropic.AnthropicReasoningState, ((), ((), ((), ()))))
  route = withReasoning `chainProviders` withTools `chainProviders` withVision `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeOpus46 AnthropicOAuth)
