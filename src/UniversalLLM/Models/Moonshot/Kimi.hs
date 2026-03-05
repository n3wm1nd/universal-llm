{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.Moonshot.Kimi
Description: Production-ready Kimi model definitions

This module provides a tested, production-ready definition for Moonshot AI's
Kimi K2.5 model, accessed through OpenRouter or AlibabaCloud.

= Available Models

* 'KimiK25' - Moonshot AI's Kimi K2.5, a high-capability coding and agentic model

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Moonshot.Kimi

let model = Model KimiK25 OpenRouter
let provider = route

-- Or via AlibabaCloud
let model = Model KimiK25 AlibabaCloud
let provider = route
@

= Authentication

- OpenRouter: Set @OPENROUTER_API_KEY@ environment variable
- AlibabaCloud: Set @ALIBABACLOUD_API_KEY@ environment variable
-}

module UniversalLLM.Models.Moonshot.Kimi
  ( -- * Model Types
    KimiK25(..)
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..), AlibabaCloud(..))

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

instance Routing (Model KimiK25 OpenRouter) where
  type RoutingState (Model KimiK25 OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model KimiK25 OpenRouter)

--------------------------------------------------------------------------------
-- Kimi K2.5 via AlibabaCloud
--------------------------------------------------------------------------------

instance ModelName (Model KimiK25 AlibabaCloud) where
  modelName (Model _ _) = "kimi-k2.5"

instance HasTools (Model KimiK25 AlibabaCloud) where
  withTools = OpenAI.openAITools

-- Note: Kimi K2.5 via AlibabaCloud supports "Deep Thinking" but the reasoning
-- is completely hidden - it accepts the reasoning parameter and performs internal
-- reasoning to improve response quality, but does not expose reasoning_content
-- or reasoning_details in API responses.
instance HasReasoning (Model KimiK25 AlibabaCloud) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model KimiK25 AlibabaCloud) where
  type RoutingState (Model KimiK25 AlibabaCloud) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model KimiK25 AlibabaCloud)
