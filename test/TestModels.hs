{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module TestModels where

import UniversalLLM.Core.Types
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.Anthropic (Anthropic(..))
import UniversalLLM.Providers.OpenAI (OpenAI(..), LlamaCpp(..), OpenRouter(..))

-- ============================================================================
-- Anthropic Models
-- ============================================================================

-- Test model for Claude Sonnet 4.5 (tools only)
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance ModelName Anthropic ClaudeSonnet45 where
  modelName _ = "claude-sonnet-4-5-20250929"

instance HasTools ClaudeSonnet45 Anthropic where
  withTools = Anthropic.anthropicTools

anthropicSonnet45 :: ComposableProvider Anthropic ClaudeSonnet45 ((), ())
anthropicSonnet45 = withTools `chainProviders` Anthropic.baseComposableProvider @ClaudeSonnet45

-- Test model for Claude Sonnet 4.5 with reasoning enabled (for testing extended thinking)
data ClaudeSonnet45WithReasoning = ClaudeSonnet45WithReasoning deriving (Show, Eq)

instance ModelName Anthropic ClaudeSonnet45WithReasoning where
  modelName _ = "claude-sonnet-4-5-20250929"

instance HasTools ClaudeSonnet45WithReasoning Anthropic where
  withTools = Anthropic.anthropicTools

instance HasReasoning ClaudeSonnet45WithReasoning Anthropic where
  withReasoning = Anthropic.anthropicReasoning

anthropicSonnet45Reasoning :: ComposableProvider   Anthropic ClaudeSonnet45WithReasoning ((), ((), ()))
anthropicSonnet45Reasoning = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @ClaudeSonnet45WithReasoning

-- ============================================================================
-- OpenAI-Compatible Models (GLM4.5 available via multiple backends)
-- ============================================================================

-- GLM4.5 - supports tools, reasoning, and JSON
-- Available via llama.cpp, OpenRouter, and other OpenAI-compatible providers
data GLM45 = GLM45 deriving (Show, Eq)

-- Model name varies by provider
instance ModelName OpenAI GLM45 where
  modelName _ = "glm-4-plus"  -- Generic fallback

instance ModelName LlamaCpp GLM45 where
  modelName _ = "GLM-4.5-Air"  -- Canonicalized from GGUF filename

instance ModelName OpenRouter GLM45 where
  modelName _ = "z-ai/glm-4.5-air:free"

-- Capability instances (same across all providers)
instance HasTools GLM45 OpenAI where
  withTools = OpenAI.openAITools

instance HasReasoning GLM45 OpenAI where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON GLM45 OpenAI where
  withJSON = OpenAI.openAIJSON

instance HasTools GLM45 LlamaCpp where
  withTools = OpenAI.openAITools

instance HasReasoning GLM45 LlamaCpp where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON GLM45 LlamaCpp where
  withJSON = OpenAI.openAIJSON

instance HasTools GLM45 OpenRouter where
  withTools = OpenAI.openAITools

instance HasReasoning GLM45 OpenRouter where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON GLM45 OpenRouter where
  withJSON = OpenAI.openAIJSON

-- Composable providers for each backend
openAIGLM45 :: ComposableProvider OpenAI GLM45 ((), ((), ((), ())))
openAIGLM45 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @OpenAI @GLM45

llamaCppGLM45 :: ComposableProvider LlamaCpp GLM45 ((), ((), ((), ())))
llamaCppGLM45 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @LlamaCpp @GLM45

openRouterGLM45 :: ComposableProvider OpenRouter GLM45 ((), ((), ((), ())))
openRouterGLM45 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @OpenRouter @GLM45

-- Basic text-only model (for compile-time safety tests)
data BasicTextModel = BasicTextModel deriving (Show, Eq)

instance ModelName OpenAI BasicTextModel where
  modelName _ = "basic-text-model"

-- BasicTextModel uses the default overlappable instance (just baseComposableProvider)

-- Tools-only model (no JSON, no reasoning)
data ToolsOnlyModel = ToolsOnlyModel deriving (Show, Eq)

instance ModelName OpenAI ToolsOnlyModel where
  modelName _ = "tools-only-model"

instance HasTools ToolsOnlyModel OpenAI where
  withTools = OpenAI.openAITools

openAIToolsonly :: ComposableProvider OpenAI ToolsOnlyModel ((), ())
openAIToolsonly = withTools `chainProviders` OpenAI.baseComposableProvider @OpenAI @ToolsOnlyModel

-- JSON-capable model (no tools, no reasoning)
data JSONModel = JSONModel deriving (Show, Eq)

instance ModelName OpenAI JSONModel where
  modelName _ = "json-model"

instance HasJSON JSONModel OpenAI where
  withJSON = OpenAI.openAIJSON

openaiJSON :: ComposableProvider OpenAI JSONModel ((), ())
openaiJSON = withJSON `chainProviders` OpenAI.baseComposableProvider @OpenAI @JSONModel
