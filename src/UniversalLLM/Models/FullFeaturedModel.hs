{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module UniversalLLM.Models.FullFeaturedModel where

import UniversalLLM.Core.Types
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Providers.Anthropic

-- FullFeaturedModel - example model demonstrating all implemented capabilities
-- This serves as a reference implementation showing how to declare a model
-- that supports every capability the library offers
data FullFeaturedModel = FullFeaturedModel deriving (Show, Eq)

-- OpenAI provider support - all capabilities
instance ModelName OpenAI FullFeaturedModel where
  modelName _ = "full-featured-model"

instance HasTools FullFeaturedModel OpenAI where
  withTools = openAITools

instance HasJSON FullFeaturedModel OpenAI where
  withJSON = openAIJSON

instance HasReasoning FullFeaturedModel OpenAI where
  withReasoning = openAIReasoning

-- Note: HasVision intentionally NOT implemented yet (no provider supports it)
-- instance HasVision FullFeaturedModel OpenAI where
--   withVision = UniversalLLM.Providers.OpenAI.openAIWithVision


openai :: ComposableProvider OpenAI FullFeaturedModel ((), ((), ((), ())))
openai = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` UniversalLLM.Providers.OpenAI.baseComposableProvider @OpenAI @FullFeaturedModel

-- Anthropic provider support - subset of capabilities
instance ModelName Anthropic FullFeaturedModel where
  modelName _ = "full-featured-model"

instance HasTools FullFeaturedModel Anthropic where
  withTools :: ComposableProvider
     Anthropic
     FullFeaturedModel
     (ToolState FullFeaturedModel Anthropic)
  withTools = anthropicTools

anthropic :: ComposableProvider Anthropic FullFeaturedModel ((), ())
anthropic = withTools `chainProviders` UniversalLLM.Providers.Anthropic.baseComposableProvider @FullFeaturedModel

-- Note: This is a phantom model for examples and documentation
-- Real model definitions in external packages should use specific model names
-- e.g., "gpt-4o", "claude-3-5-sonnet-20241022", etc.
