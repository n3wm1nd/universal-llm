{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.Minimax.M
Description: Production-ready MiniMax model definitions

This module provides tested, production-ready definitions for MiniMax's
M2.5 model, accessed through OpenRouter or a local llama.cpp server.

= Available Models

* 'MinimaxM25' - MiniMax M2.5, a high-capability coding and reasoning model

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Minimax.M

-- Via OpenRouter
let model = Model MinimaxM25 OpenRouter
let provider = minimaxM25

-- Via llama.cpp
let model = Model MinimaxM25 LlamaCpp
let provider = minimaxM25LlamaCpp
@

= Authentication

OpenRouter: Set @OPENROUTER_API_KEY@ environment variable.
LlamaCpp: Set @LLAMACPP_URL@ environment variable.
-}

module UniversalLLM.Models.Minimax.M
  ( -- * Model Types
    MinimaxM25(..)
    -- * Composable Providers
  , minimaxM25
  , minimaxM25LlamaCpp
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))

--------------------------------------------------------------------------------
-- MiniMax M2.5
--------------------------------------------------------------------------------

-- | MiniMax M2.5 - High-capability coding and reasoning model
--
-- Capabilities:
-- - Tool calling
-- - High-quality code generation and reasoning
-- - Reasoning via reasoning_details (OpenRouter style)
--
-- Accessed via OpenRouter.
data MinimaxM25 = MinimaxM25 deriving (Show, Eq)

instance ModelName (Model MinimaxM25 OpenRouter) where
  modelName (Model _ _) = "minimax/minimax-m2.5"

instance HasTools (Model MinimaxM25 OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model MinimaxM25 OpenRouter) where
  type ReasoningState (Model MinimaxM25 OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- | Composable provider for MiniMax M2.5 via OpenRouter
--
-- Includes reasoning support via reasoning_details (OpenRouter style).
minimaxM25 :: ComposableProvider (Model MinimaxM25 OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
minimaxM25 = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model MinimaxM25 OpenRouter)

--------------------------------------------------------------------------------
-- MiniMax M2.5 via LlamaCpp
--------------------------------------------------------------------------------

instance ModelName (Model MinimaxM25 LlamaCpp) where
  modelName (Model _ _) = "MiniMax-M2.5"

instance HasTools (Model MinimaxM25 LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model MinimaxM25 LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

-- | Composable provider for MiniMax M2.5 via llama.cpp
--
-- Uses standard reasoning_content (not OpenRouter's reasoning_details).
minimaxM25LlamaCpp :: ComposableProvider (Model MinimaxM25 LlamaCpp) ((), ((), ()))
minimaxM25LlamaCpp = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model MinimaxM25 LlamaCpp)
