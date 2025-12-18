{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module TestModels where

import UniversalLLM.Core.Types
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.Anthropic (Anthropic(..))
import UniversalLLM.Providers.OpenAI (OpenAI(..), LlamaCpp(..), OpenRouter(..), OpenAICompatible(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse, OpenAICompletionRequest, OpenAICompletionResponse)

-- New provider for ZAI coding endpoint
data ZAI = ZAI deriving (Show, Eq)

-- Provider instances for ZAI (uses OpenAI protocol)
instance ModelName (Model GLM45 ZAI) where
  modelName (Model _ _) = "GLM-4.5-Air"

instance Provider (Model aiModel ZAI) where
  type ProviderRequest (Model aiModel ZAI) = OpenAIRequest
  type ProviderResponse (Model aiModel ZAI) = OpenAIResponse

instance Provider (Model aiModel ZAI) => CompletionProvider (Model aiModel ZAI) where
  type CompletionRequest (Model aiModel ZAI) = OpenAICompletionRequest
  type CompletionResponse (Model aiModel ZAI) = OpenAICompletionResponse

instance {-# OVERLAPPABLE #-} ModelName (Model aiModel ZAI) => CompletionProviderImplementation (Model aiModel ZAI) where
  getComposableCompletionProvider = OpenAI.baseCompletionProvider

-- Supporting capability instances for ZAI
instance SupportsTemperature ZAI
instance SupportsMaxTokens ZAI
instance SupportsSeed ZAI
instance SupportsSystemPrompt ZAI
instance SupportsStop ZAI
instance SupportsStreaming ZAI

-- ============================================================================
-- Anthropic Models
-- ============================================================================

-- Test model for Claude Sonnet 4.5 (tools only)
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet45 Anthropic) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45 Anthropic) where
  withTools = Anthropic.anthropicTools

anthropicSonnet45 :: ComposableProvider (Model ClaudeSonnet45 Anthropic) ((), ())
anthropicSonnet45 = withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45 Anthropic)

-- Test model for Claude Sonnet 4.5 with reasoning enabled (for testing extended thinking)
data ClaudeSonnet45WithReasoning = ClaudeSonnet45WithReasoning deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet45WithReasoning Anthropic) where
  modelName (Model _ _) = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45WithReasoning Anthropic) where
  withTools = Anthropic.anthropicTools

instance HasReasoning (Model ClaudeSonnet45WithReasoning Anthropic) where
  type ReasoningState (Model ClaudeSonnet45WithReasoning Anthropic) = Anthropic.AnthropicReasoningState
  withReasoning = Anthropic.anthropicReasoning

anthropicSonnet45Reasoning :: ComposableProvider (Model ClaudeSonnet45WithReasoning Anthropic) (Anthropic.AnthropicReasoningState, ((), ()))
anthropicSonnet45Reasoning = withReasoning `chainProviders` withTools `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet45WithReasoning Anthropic)

-- ============================================================================
-- OpenAI-Compatible Models (GLM4.5 available via multiple backends)
-- ============================================================================

-- GLM4.5 - supports tools, reasoning, and JSON
-- Available via llama.cpp, OpenRouter, and other OpenAI-compatible providers
data GLM45 = GLM45 deriving (Show, Eq)

-- Model name varies by provider
instance ModelName (Model GLM45 OpenAI) where
  modelName (Model _ _) = "glm-4-plus"  -- Generic fallback

instance ModelName (Model GLM45 LlamaCpp) where
  modelName (Model _ _) = "GLM-4.5-Air"  -- Canonicalized from GGUF filename

instance ModelName (Model GLM45 OpenRouter) where
  modelName (Model _ _) = "z-ai/glm-4.5-air:free"

-- Capability instances (same across all providers)
instance HasTools (Model GLM45 OpenAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45 OpenAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM45 OpenAI) where
  withJSON = OpenAI.openAIJSON

instance HasTools (Model GLM45 LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45 LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM45 LlamaCpp) where
  withJSON = OpenAI.openAIJSON

instance HasTools (Model GLM45 OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45 OpenRouter) where
  type ReasoningState (Model GLM45 OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance HasJSON (Model GLM45 OpenRouter) where
  withJSON = OpenAI.openAIJSON

-- ZAI provider capabilities (same as OpenAI-compatible)
instance HasTools (Model GLM45 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45 ZAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM45 ZAI) where
  withJSON = OpenAI.openAIJSON

-- Composable providers for each backend
openAIGLM45 :: ComposableProvider (Model GLM45 OpenAI) ((), ((), ((), ())))
openAIGLM45 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45 OpenAI)

llamaCppGLM45 :: ComposableProvider (Model GLM45 LlamaCpp) ((), ((), ((), ())))
llamaCppGLM45 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45 LlamaCpp)

openRouterGLM45 :: ComposableProvider (Model GLM45 OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ((), ())))
openRouterGLM45 = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45 OpenRouter)

zaiGLM45 :: ComposableProvider (Model GLM45 ZAI) ((), ((), ((), ())))
zaiGLM45 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45 ZAI)

-- ============================================================================
-- Qwen Models
-- ============================================================================

-- Qwen 3 Coder - Code-specialized model supporting tools
-- Available via llama.cpp
data Qwen3Coder = Qwen3Coder deriving (Show, Eq)

-- Model name varies based on GGUF filename
instance ModelName (Model Qwen3Coder LlamaCpp) where
  modelName (Model _ _) = "Qwen3-Coder-30B-Instruct"  -- Common canonicalized name

-- Capability instances
instance HasTools (Model Qwen3Coder LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen3Coder LlamaCpp) where
  withJSON = OpenAI.openAIJSON

-- Composable provider
llamaCppQwen3Coder :: ComposableProvider (Model Qwen3Coder LlamaCpp) ((), ((), ()))
llamaCppQwen3Coder = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3Coder LlamaCpp)

-- ============================================================================
-- OpenRouter-specific Models
-- ============================================================================

-- Amazon Nova 2 Lite - supports tools and reasoning with reasoning_details
data Nova2Lite = Nova2Lite deriving (Show, Eq)

instance ModelName (Model Nova2Lite OpenRouter) where
  modelName (Model _ _) = "amazon/nova-2-lite-v1"

instance HasTools (Model Nova2Lite OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model Nova2Lite OpenRouter) where
  type ReasoningState (Model Nova2Lite OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

openRouterNova2Lite :: ComposableProvider (Model Nova2Lite OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ((), ())))
openRouterNova2Lite = withReasoning `chainProviders` withTools `chainProviders` OpenAI.normalizeEmptyContent `chainProviders` OpenAI.baseComposableProvider @(Model Nova2Lite OpenRouter)

-- Google Gemini 3 Pro Preview via OpenRouter
data Gemini3ProPreview = Gemini3ProPreview deriving (Show, Eq)

instance ModelName (Model Gemini3ProPreview OpenRouter) where
  modelName (Model _ _) = "google/gemini-3-pro-preview"

instance HasTools (Model Gemini3ProPreview OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model Gemini3ProPreview OpenRouter) where
  type ReasoningState (Model Gemini3ProPreview OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

openRouterGemini3ProPreview :: ComposableProvider (Model Gemini3ProPreview OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
openRouterGemini3ProPreview = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Gemini3ProPreview OpenRouter)

-- Google Gemini 3 Flash Preview via OpenRouter
data Gemini3FlashPreview = Gemini3FlashPreview deriving (Show, Eq)

instance ModelName (Model Gemini3FlashPreview OpenRouter) where
  modelName (Model _ _) = "google/gemini-3-flash-preview"

instance HasTools (Model Gemini3FlashPreview OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model Gemini3FlashPreview OpenRouter) where
  type ReasoningState (Model Gemini3FlashPreview OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

openRouterGemini3FlashPreview :: ComposableProvider (Model Gemini3FlashPreview OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
openRouterGemini3FlashPreview = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Gemini3FlashPreview OpenRouter)

-- Basic text-only model (for compile-time safety tests)
data BasicTextModel = BasicTextModel deriving (Show, Eq)

instance ModelName (Model BasicTextModel OpenAI) where
  modelName (Model _ _) = "basic-text-model"

-- BasicTextModel uses the default overlappable instance (just baseComposableProvider)

-- Tools-only model (no JSON, no reasoning)
data ToolsOnlyModel = ToolsOnlyModel deriving (Show, Eq)

instance ModelName (Model ToolsOnlyModel OpenAI) where
  modelName (Model _ _) = "tools-only-model"

instance HasTools (Model ToolsOnlyModel OpenAI) where
  withTools = OpenAI.openAITools

openAIToolsonly :: ComposableProvider (Model ToolsOnlyModel OpenAI) ((), ())
openAIToolsonly = withTools `chainProviders` OpenAI.baseComposableProvider @(Model ToolsOnlyModel OpenAI)

-- JSON-capable model (no tools, no reasoning)
data JSONModel = JSONModel deriving (Show, Eq)

instance ModelName (Model JSONModel OpenAI) where
  modelName (Model _ _) = "json-model"

instance HasJSON (Model JSONModel OpenAI) where
  withJSON = OpenAI.openAIJSON

openaiJSON :: ComposableProvider (Model JSONModel OpenAI) ((), ())
openaiJSON = withJSON `chainProviders` OpenAI.baseComposableProvider @(Model JSONModel OpenAI)
