{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{- |
Module: UniversalLLM.Models.Alibaba.Qwen
Description: Production-ready Qwen model definitions

This module provides tested, production-ready definitions for Alibaba's Qwen models.

= Available Models

* 'Qwen35_122B' - Qwen 3.5 122B, large reasoning model
* 'Qwen3CoderNext' - Latest Qwen 3 Coder variant (aliased as 'Qwen3Coder')
* 'Qwen3Coder30bInstruct' - Qwen 3 Coder 30B Instruct
* 'Qwen36Flash' - Qwen 3.6 Flash (via OpenRouter and AlibabaCloudTokenPlan)
* 'Qwen37Plus' - Qwen 3.7 Plus (via OpenRouter and AlibabaCloudTokenPlan)
* 'Qwen37Max' - Qwen 3.7 Max (via OpenRouter and AlibabaCloudTokenPlan)
* 'Qwen38MaxPreview' - Qwen 3.8 Max Preview (via AlibabaCloudTokenPlan only, not yet on OpenRouter)

= Provider Support

Qwen models are currently available through:

- __llama.cpp__: Local inference with native tool support
- __OpenRouter__: Qwen 3.5 122B, Qwen 3.6 Plus, Qwen 3.6 Flash, Qwen 3.7 Plus, Qwen 3.7 Max
- __AlibabaCloudTokenPlan__: Qwen 3.6 Flash, Qwen 3.7 Plus, Qwen 3.7 Max, Qwen 3.8 Max Preview

Unlike GLM models, Qwen's chat template properly converts XML tool calls to OpenAI format,
so no special XML parsing is needed.

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Alibaba.Qwen

-- Qwen 3.5 122B via OpenRouter
let model = Model Qwen35_122B OpenRouter
let provider = route

-- Qwen 3.5 122B via llama.cpp
let model = Model Qwen35_122B LlamaCpp
let provider = route

-- Latest Qwen 3 Coder (currently Qwen3CoderNext)
let model = Model Qwen3CoderNext LlamaCpp
let provider = route

-- Qwen 3.8 Max Preview via Alibaba's Token Plan
let model = Model Qwen38MaxPreview AlibabaCloudTokenPlan
let provider = route
@

= Authentication

- llama.cpp: No auth needed (local)
- OpenRouter: Set @OPENROUTER_API_KEY@ environment variable
- AlibabaCloudTokenPlan: Set @ALIBABACLOUD_API_KEY@ environment variable
-}

module UniversalLLM.Models.Alibaba.Qwen
  ( -- * Model Types
    Qwen35_122B(..)
  , Qwen35_40B(..)
  , Qwen36Plus(..)
  , Qwen36Flash(..)
  , Qwen37Plus(..)
  , Qwen37Max(..)
  , Qwen38MaxPreview(..)
  , Qwen3CoderNext(..)
  , Qwen3Coder30bInstruct(..)
    -- * Aliases
  , Qwen3Coder
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..), AlibabaCloudTokenPlan(..))

--------------------------------------------------------------------------------
-- Qwen 3.5 122B
--------------------------------------------------------------------------------

-- | Qwen 3.5 122B - Large reasoning model
--
-- Capabilities:
-- - Tool calling (native OpenAI format via chat template)
-- - JSON mode
-- - Reasoning (via reasoning_content)
data Qwen35_122B = Qwen35_122B deriving (Show, Eq)

instance ModelName (Model Qwen35_122B LlamaCpp) where
  modelName (Model _ _) = "Qwen3.5-122B"

instance HasTools (Model Qwen35_122B LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen35_122B LlamaCpp) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen35_122B LlamaCpp) where
  withReasoning = OpenAI.llamaCppReasoning

instance Routing (Model Qwen35_122B LlamaCpp) where
  type RoutingState (Model Qwen35_122B LlamaCpp) = ((), ((), ((), ((), ((), ())))))
  route = OpenAI.mergeSystemMessages `chainProviders` OpenAI.systemMessagesFirst `chainProviders` withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen35_122B LlamaCpp)

--------------------------------------------------------------------------------
-- Qwen 3.5 40B (llama.cpp)
--------------------------------------------------------------------------------

-- | Qwen 3.5 40B - Mid-size reasoning model
--
-- Same capabilities as Qwen35_122B, smaller parameter count.
data Qwen35_40B = Qwen35_40B deriving (Show, Eq)

instance ModelName (Model Qwen35_40B LlamaCpp) where
  modelName (Model _ _) = "Qwen3.5-40B"

instance HasTools (Model Qwen35_40B LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen35_40B LlamaCpp) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen35_40B LlamaCpp) where
  withReasoning = OpenAI.llamaCppReasoning

instance HasVision (Model Qwen35_40B LlamaCpp) where
  withVision = OpenAI.openAIVision

instance Routing (Model Qwen35_40B LlamaCpp) where
  type RoutingState (Model Qwen35_40B LlamaCpp) = ((), ((), ((), ((), ((), ((), ()))))))
  route = OpenAI.mergeSystemMessages `chainProviders` OpenAI.systemMessagesFirst `chainProviders` withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` withVision `chainProviders` OpenAI.baseComposableProvider @(Model Qwen35_40B LlamaCpp)

--------------------------------------------------------------------------------
-- Qwen 3.5 122B via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model Qwen35_122B OpenRouter) where
  modelName (Model _ _) = "qwen/qwen3.5-122b-a10b"

instance HasTools (Model Qwen35_122B OpenRouter) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen35_122B OpenRouter) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen35_122B OpenRouter) where
  type ReasoningState (Model Qwen35_122B OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model Qwen35_122B OpenRouter) where
  type RoutingState (Model Qwen35_122B OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen35_122B OpenRouter)

--------------------------------------------------------------------------------
-- Qwen 3 Coder Next
--------------------------------------------------------------------------------

-- | Qwen 3 Coder Next - Latest Qwen 3 Coder variant
--
-- Capabilities:
-- - Tool calling (native OpenAI format via chat template)
-- - JSON mode
-- - High-quality code generation
data Qwen3CoderNext = Qwen3CoderNext deriving (Show, Eq)

instance ModelName (Model Qwen3CoderNext LlamaCpp) where
  modelName (Model _ _) = "Qwen3-Coder-Next"

instance HasTools (Model Qwen3CoderNext LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen3CoderNext LlamaCpp) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model Qwen3CoderNext LlamaCpp) where
  type RoutingState (Model Qwen3CoderNext LlamaCpp) = ((), ((), ()))
  route = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3CoderNext LlamaCpp)

--------------------------------------------------------------------------------
-- Qwen 3 Coder 30B Instruct
--------------------------------------------------------------------------------

-- | Qwen 3 Coder 30B Instruct - Original Qwen 3 Coder release
--
-- Capabilities:
-- - Tool calling (native OpenAI format via chat template)
-- - JSON mode
-- - High-quality code generation
data Qwen3Coder30bInstruct = Qwen3Coder30bInstruct deriving (Show, Eq)

instance ModelName (Model Qwen3Coder30bInstruct LlamaCpp) where
  modelName (Model _ _) = "Qwen3-Coder-30B-Instruct"

instance HasTools (Model Qwen3Coder30bInstruct LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen3Coder30bInstruct LlamaCpp) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model Qwen3Coder30bInstruct LlamaCpp) where
  type RoutingState (Model Qwen3Coder30bInstruct LlamaCpp) = ((), ((), ()))
  route = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3Coder30bInstruct LlamaCpp)

--------------------------------------------------------------------------------
-- Qwen 3.6 Plus (OpenRouter)
--------------------------------------------------------------------------------

-- | Qwen 3.6 Plus - Cloud-based model via OpenRouter
--
-- Capabilities:
-- - Tool calling
-- - JSON mode
-- - Reasoning (Deep Thinking)
data Qwen36Plus = Qwen36Plus deriving (Show, Eq)

instance ModelName (Model Qwen36Plus OpenRouter) where
  modelName (Model _ _) = "qwen/qwen3.6-plus"

instance HasTools (Model Qwen36Plus OpenRouter) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen36Plus OpenRouter) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen36Plus OpenRouter) where
  type ReasoningState (Model Qwen36Plus OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model Qwen36Plus OpenRouter) where
  type RoutingState (Model Qwen36Plus OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen36Plus OpenRouter)

--------------------------------------------------------------------------------
-- Qwen 3.7 Max (OpenRouter)
--------------------------------------------------------------------------------

-- | Qwen 3.7 Max - Large frontier model via OpenRouter
--
-- Capabilities:
-- - Tool calling
-- - JSON mode
-- - Reasoning
data Qwen37Max = Qwen37Max deriving (Show, Eq)

instance ModelName (Model Qwen37Max OpenRouter) where
  modelName (Model _ _) = "qwen/qwen3.7-max"

instance HasTools (Model Qwen37Max OpenRouter) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen37Max OpenRouter) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen37Max OpenRouter) where
  type ReasoningState (Model Qwen37Max OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model Qwen37Max OpenRouter) where
  type RoutingState (Model Qwen37Max OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen37Max OpenRouter)

--------------------------------------------------------------------------------
-- Qwen 3.7 Max via AlibabaCloudTokenPlan
--------------------------------------------------------------------------------

instance ModelName (Model Qwen37Max AlibabaCloudTokenPlan) where
  modelName (Model _ _) = "qwen3.7-max"

instance HasTools (Model Qwen37Max AlibabaCloudTokenPlan) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen37Max AlibabaCloudTokenPlan) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen37Max AlibabaCloudTokenPlan) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model Qwen37Max AlibabaCloudTokenPlan) where
  type RoutingState (Model Qwen37Max AlibabaCloudTokenPlan) = ((), ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen37Max AlibabaCloudTokenPlan)

--------------------------------------------------------------------------------
-- Qwen 3.7 Plus (AlibabaCloudTokenPlan)
--------------------------------------------------------------------------------

-- | Qwen 3.7 Plus - Cloud-based model via Alibaba's Token Plan
--
-- Capabilities:
-- - Tool calling
-- - JSON mode
-- - Reasoning (Deep Thinking)
data Qwen37Plus = Qwen37Plus deriving (Show, Eq)

instance ModelName (Model Qwen37Plus AlibabaCloudTokenPlan) where
  modelName (Model _ _) = "qwen3.7-plus"

instance HasTools (Model Qwen37Plus AlibabaCloudTokenPlan) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen37Plus AlibabaCloudTokenPlan) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen37Plus AlibabaCloudTokenPlan) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model Qwen37Plus AlibabaCloudTokenPlan) where
  type RoutingState (Model Qwen37Plus AlibabaCloudTokenPlan) = ((), ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen37Plus AlibabaCloudTokenPlan)

--------------------------------------------------------------------------------
-- Qwen 3.7 Plus via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model Qwen37Plus OpenRouter) where
  modelName (Model _ _) = "qwen/qwen3.7-plus"

instance HasTools (Model Qwen37Plus OpenRouter) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen37Plus OpenRouter) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen37Plus OpenRouter) where
  type ReasoningState (Model Qwen37Plus OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model Qwen37Plus OpenRouter) where
  type RoutingState (Model Qwen37Plus OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen37Plus OpenRouter)

--------------------------------------------------------------------------------
-- Qwen 3.6 Flash (AlibabaCloudTokenPlan)
--------------------------------------------------------------------------------

-- | Qwen 3.6 Flash - Fast, low-cost model via Alibaba's Token Plan
--
-- Capabilities:
-- - Tool calling
-- - JSON mode
-- - Reasoning (Deep Thinking)
data Qwen36Flash = Qwen36Flash deriving (Show, Eq)

instance ModelName (Model Qwen36Flash AlibabaCloudTokenPlan) where
  modelName (Model _ _) = "qwen3.6-flash"

instance HasTools (Model Qwen36Flash AlibabaCloudTokenPlan) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen36Flash AlibabaCloudTokenPlan) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen36Flash AlibabaCloudTokenPlan) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model Qwen36Flash AlibabaCloudTokenPlan) where
  type RoutingState (Model Qwen36Flash AlibabaCloudTokenPlan) = ((), ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen36Flash AlibabaCloudTokenPlan)

--------------------------------------------------------------------------------
-- Qwen 3.6 Flash via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model Qwen36Flash OpenRouter) where
  modelName (Model _ _) = "qwen/qwen3.6-flash"

instance HasTools (Model Qwen36Flash OpenRouter) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen36Flash OpenRouter) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen36Flash OpenRouter) where
  type ReasoningState (Model Qwen36Flash OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance Routing (Model Qwen36Flash OpenRouter) where
  type RoutingState (Model Qwen36Flash OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen36Flash OpenRouter)

--------------------------------------------------------------------------------
-- Qwen 3.8 Max Preview (AlibabaCloudTokenPlan)
--------------------------------------------------------------------------------

-- | Qwen 3.8 Max Preview - Preview of Alibaba's next frontier model, via Alibaba's Token Plan
--
-- Capabilities:
-- - Tool calling
-- - JSON mode
-- - Reasoning (Deep Thinking)
data Qwen38MaxPreview = Qwen38MaxPreview deriving (Show, Eq)

instance ModelName (Model Qwen38MaxPreview AlibabaCloudTokenPlan) where
  modelName (Model _ _) = "qwen3.8-max-preview"

instance HasTools (Model Qwen38MaxPreview AlibabaCloudTokenPlan) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen38MaxPreview AlibabaCloudTokenPlan) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen38MaxPreview AlibabaCloudTokenPlan) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model Qwen38MaxPreview AlibabaCloudTokenPlan) where
  type RoutingState (Model Qwen38MaxPreview AlibabaCloudTokenPlan) = ((), ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen38MaxPreview AlibabaCloudTokenPlan)

--------------------------------------------------------------------------------
-- Aliases (Qwen3Coder = latest = Qwen3CoderNext)
--------------------------------------------------------------------------------

-- | Alias for the latest Qwen 3 Coder variant ('Qwen3CoderNext')
type Qwen3Coder = Qwen3CoderNext
