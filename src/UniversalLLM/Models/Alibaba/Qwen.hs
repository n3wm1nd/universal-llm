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
* 'Qwen35Plus' - Qwen 3.5 Plus (via AlibabaCloud)
* 'Qwen3CoderPlus' - Qwen 3 Coder Plus (via AlibabaCloud)

= Provider Support

Qwen models are currently available through:

- __llama.cpp__: Local inference with native tool support
- __OpenRouter__: Qwen 3.5 122B
- __AlibabaCloud__: Qwen 3.5 Plus, Qwen 3 Coder variants

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
@

= Authentication

- llama.cpp: No auth needed (local)
- OpenRouter: Set @OPENROUTER_API_KEY@ environment variable
- AlibabaCloud: Set @ALIBABACLOUD_API_KEY@ environment variable
-}

module UniversalLLM.Models.Alibaba.Qwen
  ( -- * Model Types
    Qwen35_122B(..)
  , Qwen3CoderNext(..)
  , Qwen3Coder30bInstruct(..)
  , Qwen3CoderPlus(..)
  , Qwen35Plus(..)
    -- * Aliases
  , Qwen3Coder
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..), AlibabaCloud(..))

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
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model Qwen35_122B LlamaCpp) where
  type RoutingState (Model Qwen35_122B LlamaCpp) = ((), ((), ((), ((), ((), ())))))
  route = OpenAI.mergeSystemMessages `chainProviders` OpenAI.systemMessagesFirst `chainProviders` withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen35_122B LlamaCpp)

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
-- Qwen 3.5 Plus (AlibabaCloud)
--------------------------------------------------------------------------------

-- | Qwen 3.5 Plus - Cloud-based model via Alibaba Cloud
--
-- Capabilities:
-- - Tool calling
-- - JSON mode
-- - Reasoning (Deep Thinking)
data Qwen35Plus = Qwen35Plus deriving (Show, Eq)

instance ModelName (Model Qwen35Plus AlibabaCloud) where
  modelName (Model _ _) = "qwen3.5-plus"

instance HasTools (Model Qwen35Plus AlibabaCloud) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen35Plus AlibabaCloud) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model Qwen35Plus AlibabaCloud) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model Qwen35Plus AlibabaCloud) where
  type RoutingState (Model Qwen35Plus AlibabaCloud) = ((), ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen35Plus AlibabaCloud)

--------------------------------------------------------------------------------
-- Qwen 3 Coder Next via AlibabaCloud
--------------------------------------------------------------------------------

instance ModelName (Model Qwen3CoderNext AlibabaCloud) where
  modelName (Model _ _) = "qwen3-coder-next"

instance HasTools (Model Qwen3CoderNext AlibabaCloud) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen3CoderNext AlibabaCloud) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model Qwen3CoderNext AlibabaCloud) where
  type RoutingState (Model Qwen3CoderNext AlibabaCloud) = ((), ((), ()))
  route = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3CoderNext AlibabaCloud)

--------------------------------------------------------------------------------
-- Qwen 3 Coder Plus (AlibabaCloud)
--------------------------------------------------------------------------------

-- | Qwen 3 Coder Plus - Enhanced coding model
--
-- Capabilities:
-- - Tool calling
-- - JSON mode
data Qwen3CoderPlus = Qwen3CoderPlus deriving (Show, Eq)

instance ModelName (Model Qwen3CoderPlus AlibabaCloud) where
  modelName (Model _ _) = "qwen3-coder-plus"

instance HasTools (Model Qwen3CoderPlus AlibabaCloud) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen3CoderPlus AlibabaCloud) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model Qwen3CoderPlus AlibabaCloud) where
  type RoutingState (Model Qwen3CoderPlus AlibabaCloud) = ((), ((), ()))
  route = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3CoderPlus AlibabaCloud)

--------------------------------------------------------------------------------
-- Aliases (Qwen3Coder = latest = Qwen3CoderNext)
--------------------------------------------------------------------------------

-- | Alias for the latest Qwen 3 Coder variant ('Qwen3CoderNext')
type Qwen3Coder = Qwen3CoderNext
