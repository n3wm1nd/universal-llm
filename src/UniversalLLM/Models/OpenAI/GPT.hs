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

= Usage

@
import UniversalLLM
import UniversalLLM.Models.OpenAI.GPT

-- Via OpenRouter
let model = Model GPTOSS OpenRouter
let provider = gptOSSOpenRouter

-- Via llama.cpp
let model = Model GPTOSS LlamaCpp
let provider = gptOSS
@

= Authentication

OpenRouter: Set @OPENROUTER_API_KEY@ environment variable.
LlamaCpp: Set @LLAMACPP_URL@ environment variable.
-}

module UniversalLLM.Models.OpenAI.GPT
  ( -- * Model Types
    GPTOSS(..)
    -- * Composable Providers
  , gptOSSOpenRouter
  , gptOSS
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

-- | Composable provider for GPT-OSS via OpenRouter
--
-- Includes reasoning support via reasoning_details (OpenRouter style).
gptOSSOpenRouter :: ComposableProvider (Model GPTOSS OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
gptOSSOpenRouter = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPTOSS OpenRouter)

--------------------------------------------------------------------------------
-- GPT-OSS via LlamaCpp
--------------------------------------------------------------------------------

instance ModelName (Model GPTOSS LlamaCpp) where
  modelName (Model _ _) = "gpt-oss-120b"

instance HasTools (Model GPTOSS LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GPTOSS LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

-- | Composable provider for GPT-OSS via llama.cpp
--
-- Uses standard reasoning_content (not OpenRouter's reasoning_details).
gptOSS :: ComposableProvider (Model GPTOSS LlamaCpp) ((), ((), ()))
gptOSS = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GPTOSS LlamaCpp)
