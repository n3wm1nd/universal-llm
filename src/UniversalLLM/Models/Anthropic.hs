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

* 'ClaudeSonnet45' - Claude Sonnet 4.5 with extended thinking and tool support
* 'ClaudeSonnet45WithReasoning' - Sonnet 4.5 with explicit reasoning state

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Anthropic

-- Use with regular Anthropic API
let model = Model ClaudeSonnet45 Anthropic
let provider = claudeSonnet45

-- Use with OAuth
let oauthModel = Model ClaudeSonnet45 AnthropicOAuth
let oauthProvider = claudeSonnet45OAuth
@

= Authentication

Set environment variable:
- Regular API: @ANTHROPIC_API_KEY@
- OAuth: @ANTHROPIC_OAUTH_TOKEN@
-}

module UniversalLLM.Models.Anthropic
  ( -- * Model Types
    ClaudeSonnet45(..)
  , ClaudeSonnet45WithReasoning(..)
    -- * Composable Providers
  , claudeSonnet45
  , claudeSonnet45Reasoning
  , claudeSonnet45OAuth
  , claudeSonnet45ReasoningOAuth
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

-- | Composable provider for Claude Sonnet 4.5 (tools only)
--
-- Use this when you want tool support but don't need explicit reasoning state tracking.
claudeSonnet45 :: ComposableProvider (Model ClaudeSonnet45 Anthropic) ((), ())
claudeSonnet45 = withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 Anthropic)

--------------------------------------------------------------------------------
-- Claude Sonnet 4.5 with Reasoning
--------------------------------------------------------------------------------

-- | Claude Sonnet 4.5 with explicit reasoning state
--
-- Use this variant when you need to track and access the model's reasoning process.
data ClaudeSonnet45WithReasoning = ClaudeSonnet45WithReasoning deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet45WithReasoning Anthropic) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45WithReasoning Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeSonnet45WithReasoning Anthropic) where
  type ReasoningState (Model ClaudeSonnet45WithReasoning Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

-- | Composable provider for Claude Sonnet 4.5 with reasoning state
--
-- Provides access to the model's extended thinking process via reasoning blocks.
claudeSonnet45Reasoning :: ComposableProvider (Model ClaudeSonnet45WithReasoning Anthropic) (Anthropic.AnthropicReasoningState, ((), ()))
claudeSonnet45Reasoning = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45WithReasoning Anthropic)

--------------------------------------------------------------------------------
-- OAuth Versions
--------------------------------------------------------------------------------

-- OAuth version for ClaudeSonnet45 (with tool name workarounds)
instance ModelName (Model ClaudeSonnet45 AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45 AnthropicOAuth) where
  type ToolState (Model ClaudeSonnet45 AnthropicOAuth) = OAuthToolsState
  withTools = Anthropic.anthropicOAuthBlacklistedTools

-- | Composable provider for Claude Sonnet 4.5 via OAuth
--
-- Includes workarounds for OAuth API restrictions and the magic system prompt.
claudeSonnet45OAuth :: ComposableProvider (Model ClaudeSonnet45 AnthropicOAuth) (OAuthToolsState, ((), ()))
claudeSonnet45OAuth = withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 AnthropicOAuth)

-- OAuth version with reasoning
instance ModelName (Model ClaudeSonnet45WithReasoning AnthropicOAuth) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45WithReasoning AnthropicOAuth) where
  type ToolState (Model ClaudeSonnet45WithReasoning AnthropicOAuth) = OAuthToolsState
  withTools = Anthropic.anthropicOAuthBlacklistedTools

instance HasReasoning (Model ClaudeSonnet45WithReasoning AnthropicOAuth) where
  type ReasoningState (Model ClaudeSonnet45WithReasoning AnthropicOAuth) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

-- | Composable provider for Claude Sonnet 4.5 via OAuth with reasoning
--
-- Combines reasoning state, OAuth tool workarounds, and the magic system prompt.
claudeSonnet45ReasoningOAuth :: ComposableProvider (Model ClaudeSonnet45WithReasoning AnthropicOAuth) (Anthropic.AnthropicReasoningState, (OAuthToolsState, ((), ())))
claudeSonnet45ReasoningOAuth = withReasoning `chainProviders` withTools `chainProviders` Anthropic.anthropicOAuthMagicPrompt `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45WithReasoning AnthropicOAuth)
