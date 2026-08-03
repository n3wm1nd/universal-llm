{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.DeepSeek.DeepSeek
Description: Production-ready DeepSeek V4 model definitions

This module provides tested, production-ready definitions for DeepSeek's
V4 model family, accessed through OpenRouter, llama.cpp (V4 Flash only), and
(for V4 Pro) Alibaba's Token Plan.

= Available Models

* 'DeepSeekV4Flash' - DeepSeek V4 Flash, fast and low-cost variant
* 'DeepSeekV4Pro' - DeepSeek V4 Pro, high-capability variant

= Usage

@
import UniversalLLM
import UniversalLLM.Models.DeepSeek.DeepSeek

let model = Model DeepSeekV4Flash OpenRouter
let provider = route

-- DeepSeek V4 Flash via llama.cpp (local inference)
let model = Model DeepSeekV4Flash LlamaCpp
let provider = route

-- DeepSeek V4 Pro via Alibaba's Token Plan
let model = Model DeepSeekV4Pro AlibabaCloudTokenPlan
let provider = route
@

= Authentication

- OpenRouter: Set @OPENROUTER_API_KEY@ environment variable
- AlibabaCloudTokenPlan: Set @ALIBABACLOUD_API_KEY@ environment variable
- llama.cpp: No auth needed (local)
-}

module UniversalLLM.Models.DeepSeek.DeepSeek
  ( -- * Model Types
    DeepSeekV4Flash(..)
  , DeepSeekV4Pro(..)
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..), AlibabaCloudTokenPlan(..), LlamaCpp(..))

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
-- DeepSeek V4 Flash via llama.cpp
--------------------------------------------------------------------------------

instance ModelName (Model DeepSeekV4Flash LlamaCpp) where
  modelName (Model _ _) = "DeepSeek-V4-Flash"

instance HasTools (Model DeepSeekV4Flash LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model DeepSeekV4Flash LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model DeepSeekV4Flash LlamaCpp) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model DeepSeekV4Flash LlamaCpp) where
  type RoutingState (Model DeepSeekV4Flash LlamaCpp) = ((), ((), ((), ())))
  route = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model DeepSeekV4Flash LlamaCpp)

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

--------------------------------------------------------------------------------
-- DeepSeek V4 Pro via AlibabaCloudTokenPlan
--------------------------------------------------------------------------------

instance ModelName (Model DeepSeekV4Pro AlibabaCloudTokenPlan) where
  modelName (Model _ _) = "deepseek-v4-pro"

instance HasTools (Model DeepSeekV4Pro AlibabaCloudTokenPlan) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model DeepSeekV4Pro AlibabaCloudTokenPlan) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model DeepSeekV4Pro AlibabaCloudTokenPlan) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model DeepSeekV4Pro AlibabaCloudTokenPlan) where
  type RoutingState (Model DeepSeekV4Pro AlibabaCloudTokenPlan) = ((), ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model DeepSeekV4Pro AlibabaCloudTokenPlan)
