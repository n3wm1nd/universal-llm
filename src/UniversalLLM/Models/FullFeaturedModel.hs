{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleContexts #-}

module UniversalLLM.Models.FullFeaturedModel where

import UniversalLLM.Core.Types
import UniversalLLM.Providers.OpenAI
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.Anthropic
import qualified UniversalLLM.Providers.Anthropic as Anthropic

-- FullFeaturedModel - example model demonstrating all implemented capabilities
-- This serves as a reference implementation showing how to declare a model
-- that supports every capability the library offers
data FullFeaturedModel = FullFeaturedModel deriving (Show, Eq)

-- OpenAI provider support - all capabilities
instance ModelName OpenAI FullFeaturedModel where
  modelName _ = "full-featured-model"

instance BaseComposableProvider FullFeaturedModel OpenAI where
  baseProvider = OpenAI.baseComposableProvider

instance HasTools FullFeaturedModel OpenAI where
  withTools = openAITools

instance HasJSON FullFeaturedModel OpenAI where
  withJSON = openAIJSON

instance HasReasoning FullFeaturedModel OpenAI where
  withReasoning = openAIReasoning

-- Note: HasVision intentionally NOT implemented yet (no provider supports it)
-- instance HasVision FullFeaturedModel OpenAI where
--   withVision = UniversalLLM.Providers.OpenAI.openAIWithVision


-- Anthropic provider support - subset of capabilities
instance ModelName Anthropic FullFeaturedModel where
  modelName _ = "full-featured-model"

instance BaseComposableProvider FullFeaturedModel Anthropic where
  baseProvider = Anthropic.baseComposableProvider

instance HasTools FullFeaturedModel Anthropic where
  withTools :: ComposableProvider
     Anthropic
     FullFeaturedModel
     (ToolState FullFeaturedModel Anthropic)
  withTools = anthropicTools

fullprovider :: (HasReasoning provider model, HasJSON provider model, HasTools provider model,
  BaseComposableProvider provider model) =>
  ComposableProvider model provider
    (ReasoningState provider model, (JSONState provider model, (ToolState provider model, BaseState provider model)))
fullprovider = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` baseProvider

