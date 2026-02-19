{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.KimiK25
Description: Production-ready Kimi K2.5 model definition

This module provides a tested, production-ready definition for Moonshot AI's
Kimi K2.5 model, accessed through OpenRouter.

= Available Models

* 'KimiK25' - Moonshot AI's Kimi K2.5, a high-capability coding and agentic model

= Usage

@
import UniversalLLM
import UniversalLLM.Models.KimiK25

let model = Model KimiK25 OpenRouter
let provider = kimiK25
@

= Authentication

Set @OPENROUTER_API_KEY@ environment variable.
-}

module UniversalLLM.Models.KimiK25
  ( -- * Model Types
    KimiK25(..)
    -- * Composable Providers
  , kimiK25
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..))

--------------------------------------------------------------------------------
-- Kimi K2.5
--------------------------------------------------------------------------------

-- | Moonshot AI Kimi K2.5 - High-capability coding and agentic model
--
-- Capabilities:
-- - Tool calling
-- - High-quality code generation and reasoning
-- - Long context (128K tokens)
--
-- Accessed via OpenRouter.
data KimiK25 = KimiK25 deriving (Show, Eq)

instance ModelName (Model KimiK25 OpenRouter) where
  modelName (Model _ _) = "moonshotai/kimi-k2.5"

instance HasTools (Model KimiK25 OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model KimiK25 OpenRouter) where
  type ReasoningState (Model KimiK25 OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- | Composable provider for Kimi K2.5
--
-- Includes reasoning support via reasoning_details (OpenRouter style).
kimiK25 :: ComposableProvider (Model KimiK25 OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
kimiK25 = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model KimiK25 OpenRouter)
