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
instance ModelName (Model FullFeaturedModel OpenAI) where
  modelName _ = "full-featured-model"

instance BaseComposableProvider (Model FullFeaturedModel OpenAI) where
  baseProvider = OpenAI.baseComposableProvider

instance HasTools (Model FullFeaturedModel OpenAI) where
  withTools = openAITools

instance HasJSON (Model FullFeaturedModel OpenAI) where
  withJSON = openAIJSON

instance HasReasoning (Model FullFeaturedModel OpenAI) where
  withReasoning = openAIReasoning

-- Note: HasVision intentionally NOT implemented yet (no provider supports it)
-- instance HasVision (Model FullFeaturedModel OpenAI) where
--   withVision = UniversalLLM.Providers.OpenAI.openAIWithVision


-- Anthropic provider support - subset of capabilities
instance ModelName (Model FullFeaturedModel Anthropic) where
  modelName _ = "full-featured-model"

instance BaseComposableProvider (Model FullFeaturedModel Anthropic) where
  baseProvider = Anthropic.baseComposableProvider

instance HasTools (Model FullFeaturedModel Anthropic) where
  withTools = anthropicTools

fullprovider :: (HasReasoning m, HasJSON m, HasTools m, BaseComposableProvider m) =>
  ComposableProvider m (ReasoningState m, (JSONState m, (ToolState m, BaseState m)))
fullprovider = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` baseProvider

