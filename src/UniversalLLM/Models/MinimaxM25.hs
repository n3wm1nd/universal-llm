{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.MinimaxM25
Description: Production-ready MiniMax M2.5 model definition

This module provides a tested, production-ready definition for MiniMax's
M2.5 model, accessed through OpenRouter.

= Available Models

* 'MinimaxM25' - MiniMax M2.5, a high-capability coding and reasoning model

= Usage

@
import UniversalLLM
import UniversalLLM.Models.MinimaxM25

let model = Model MinimaxM25 OpenRouter
let provider = minimaxM25
@

= Authentication

Set @OPENROUTER_API_KEY@ environment variable.
-}

module UniversalLLM.Models.MinimaxM25
  ( -- * Model Types
    MinimaxM25(..)
    -- * Composable Providers
  , minimaxM25
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..))

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

-- | Composable provider for MiniMax M2.5
--
-- Includes reasoning support via reasoning_details (OpenRouter style).
minimaxM25 :: ComposableProvider (Model MinimaxM25 OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
minimaxM25 = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model MinimaxM25 OpenRouter)
