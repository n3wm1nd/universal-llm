{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.OpenAI.GPT
Description: Production-ready GPT-OSS model definitions

This module provides definitions for OpenAI's open-source GPT model,
accessed through OpenRouter or a local llama.cpp server.

= Available Models

* 'GPTOSS' - GPT-OSS, OpenAI's open-weight model
* 'GPT53Codex' - GPT-5.3-Codex, specialized for code generation
* 'GPT54Pro' - GPT-5.4-Pro, advanced capabilities
* 'GPT54' - GPT-5.4, latest generation
* 'GPT53Chat' - GPT-5.3-Chat, optimized for conversation

= Usage

@
import UniversalLLM
import UniversalLLM.Models.OpenAI.GPT

-- Via OpenRouter
let model = Model GPTOSS OpenRouter
let provider = route

-- Via llama.cpp
let model = Model GPTOSS LlamaCpp
let provider = route
@

= Authentication

OpenRouter: Set @OPENROUTER_API_KEY@ environment variable.
LlamaCpp: Set @LLAMACPP_URL@ environment variable.
-}

module UniversalLLM.Models.OpenAI.GPT
  ( -- * Model Types
    GPTOSS(..)
  , GPT53Codex(..)
  , GPT54Pro(..)
  , GPT54(..)
  , GPT53Chat(..)
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))

--------------------------------------------------------------------------------
-- GPT-OSS via LlamaCpp
--------------------------------------------------------------------------------

-- | GPT-OSS - OpenAI's open-weight model
--
-- Capabilities:
-- - Tool calling
-- - Reasoning
data GPTOSS = GPTOSS deriving (Show, Eq)

-- | GPT-5.3-Codex - Specialized for code generation
--
-- Capabilities:
-- - Tool calling
-- - Reasoning
data GPT53Codex = GPT53Codex deriving (Show, Eq)

-- | GPT-5.4-Pro - Advanced capabilities
--
-- Capabilities:
-- - Tool calling
-- - Reasoning
data GPT54Pro = GPT54Pro deriving (Show, Eq)

-- | GPT-5.4 - Latest generation
--
-- Capabilities:
-- - Tool calling
-- - Reasoning
data GPT54 = GPT54 deriving (Show, Eq)

-- | GPT-5.3-Chat - Optimized for conversation
--
-- Capabilities:
-- - Tool calling
-- - No reasoning support (via OpenRouter)
data GPT53Chat = GPT53Chat deriving (Show, Eq)

--------------------------------------------------------------------------------
-- GPT-OSS via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model GPTOSS OpenRouter) where
  modelName (Model _ _) = "openai/gpt-oss-120b"

instance HasTools (Model GPTOSS OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GPTOSS OpenRouter) where
  type ReasoningState (Model GPTOSS OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model GPTOSS OpenRouter) where
  type RoutingState (Model GPTOSS OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPTOSS OpenRouter)

--------------------------------------------------------------------------------
-- GPT-OSS via LlamaCpp
--------------------------------------------------------------------------------

instance ModelName (Model GPTOSS LlamaCpp) where
  modelName (Model _ _) = "gpt-oss-120b"

instance HasTools (Model GPTOSS LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GPTOSS LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model GPTOSS LlamaCpp) where
  type RoutingState (Model GPTOSS LlamaCpp) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPTOSS LlamaCpp)

--------------------------------------------------------------------------------
-- GPT-5.3-Codex via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model GPT53Codex OpenRouter) where
  modelName (Model _ _) = "openai/gpt-5.3-codex"

instance HasTools (Model GPT53Codex OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GPT53Codex OpenRouter) where
  type ReasoningState (Model GPT53Codex OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model GPT53Codex OpenRouter) where
  type RoutingState (Model GPT53Codex OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPT53Codex OpenRouter)

--------------------------------------------------------------------------------
-- GPT-5.4-Pro via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model GPT54Pro OpenRouter) where
  modelName (Model _ _) = "openai/gpt-5.4-pro"

instance HasTools (Model GPT54Pro OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GPT54Pro OpenRouter) where
  type ReasoningState (Model GPT54Pro OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model GPT54Pro OpenRouter) where
  type RoutingState (Model GPT54Pro OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPT54Pro OpenRouter)

--------------------------------------------------------------------------------
-- GPT-5.4 via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model GPT54 OpenRouter) where
  modelName (Model _ _) = "openai/gpt-5.4"

instance HasTools (Model GPT54 OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GPT54 OpenRouter) where
  type ReasoningState (Model GPT54 OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model GPT54 OpenRouter) where
  type RoutingState (Model GPT54 OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPT54 OpenRouter)

--------------------------------------------------------------------------------
-- GPT-5.3-Chat via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model GPT53Chat OpenRouter) where
  modelName (Model _ _) = "openai/gpt-5.3-chat"

instance HasTools (Model GPT53Chat OpenRouter) where
  withTools = OpenAI.openAITools

-- Note: GPT-5.3-Chat does NOT support reasoning via OpenRouter
-- (no reasoning_details returned in responses)

instance Routing (Model GPT53Chat OpenRouter) where
  type RoutingState (Model GPT53Chat OpenRouter) = ((), ())
  route = withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPT53Chat OpenRouter)
