{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.DeepSeek.DeepSeek
Description: Production-ready DeepSeek V4 model definitions

This module provides tested, production-ready definitions for DeepSeek's
V4 model family, accessed through OpenRouter.

= Available Models

* 'DeepSeekV4Flash' - DeepSeek V4 Flash, fast and low-cost variant
* 'DeepSeekV4Pro' - DeepSeek V4 Pro, high-capability variant

= Usage

@
import UniversalLLM
import UniversalLLM.Models.DeepSeek.DeepSeek

let model = Model DeepSeekV4Flash OpenRouter
let provider = route
@

= Authentication

- OpenRouter: Set @OPENROUTER_API_KEY@ environment variable
-}

module UniversalLLM.Models.DeepSeek.DeepSeek
  ( -- * Model Types
    DeepSeekV4Flash(..)
  , DeepSeekV4Pro(..)
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..))

--------------------------------------------------------------------------------
-- DeepSeek V4 Flash
--------------------------------------------------------------------------------

-- | DeepSeek V4 Flash - Fast, low-cost variant of DeepSeek V4
--
-- Capabilities:
-- - Tool calling
-- - Reasoning via reasoning_details (OpenRouter style)
-- - JSON mode
--
-- Accessed via OpenRouter.
data DeepSeekV4Flash = DeepSeekV4Flash deriving (Show, Eq)

instance ModelName (Model DeepSeekV4Flash OpenRouter) where
  modelName (Model _ _) = "deepseek/deepseek-v4-flash"

instance HasTools (Model DeepSeekV4Flash OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model DeepSeekV4Flash OpenRouter) where
  type ReasoningState (Model DeepSeekV4Flash OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance HasJSON (Model DeepSeekV4Flash OpenRouter) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model DeepSeekV4Flash OpenRouter) where
  type RoutingState (Model DeepSeekV4Flash OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model DeepSeekV4Flash OpenRouter)

--------------------------------------------------------------------------------
-- DeepSeek V4 Pro
--------------------------------------------------------------------------------

-- | DeepSeek V4 Pro - High-capability variant of DeepSeek V4
--
-- Capabilities:
-- - Tool calling
-- - Reasoning via reasoning_details (OpenRouter style)
-- - JSON mode
--
-- Accessed via OpenRouter.
data DeepSeekV4Pro = DeepSeekV4Pro deriving (Show, Eq)

instance ModelName (Model DeepSeekV4Pro OpenRouter) where
  modelName (Model _ _) = "deepseek/deepseek-v4-pro"

instance HasTools (Model DeepSeekV4Pro OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model DeepSeekV4Pro OpenRouter) where
  type ReasoningState (Model DeepSeekV4Pro OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance HasJSON (Model DeepSeekV4Pro OpenRouter) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model DeepSeekV4Pro OpenRouter) where
  type RoutingState (Model DeepSeekV4Pro OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model DeepSeekV4Pro OpenRouter)
