{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PatternSynonyms #-}

{- |
Module: TestModels
Description: Test model definitions (re-exports production models + test-only models)

This module re-exports production models from UniversalLLM.Models and adds
test-specific model definitions that aren't suitable for production use.
-}

module TestModels
  ( -- * Re-exported Production Models
    ClaudeSonnet45(..)
  , ClaudeSonnet45NoReason(..)
  , ClaudeHaiku45(..)
  , ClaudeOpus46(..)
  , GLM45(..)
  , GLM45Air(..)
  , GLM46(..)
  , GLM47(..)
  , ZAI(..)
  , Qwen3CoderNext(..)
  , Qwen3Coder
  , Gemini3FlashPreview(..)
  , Gemini3ProPreview(..)
  , Nova2Lite(..)
    -- * Backward Compatibility Provider Aliases
  , anthropicSonnet45
  , anthropicSonnet45NoReason
  , anthropicSonnet45OAuth
  , anthropicSonnet45NoReasonOAuth
  , anthropicHaiku45
  , anthropicHaiku45OAuth
  , anthropicOpus46
  , anthropicOpus46OAuth
  , llamaCppGLM45
  , openRouterGLM45
  , openRouterGLM45Air
  , zaiGLM45
  , llamaCppQwen3Coder
  , openRouterNova2Lite
  , openRouterGemini3ProPreview
  , openRouterGemini3FlashPreview
    -- * Test-Only Models
  , TestPlaceholderModel(..)
  , openAITestPlaceholder
  , BasicTextModel(..)
  , ToolsOnlyModel(..)
  , JSONModel(..)
  , openAIToolsonly
  , openaiJSON
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.Anthropic (Anthropic(..), AnthropicOAuth(..), OAuthToolsState)
import UniversalLLM.Providers.OpenAI (OpenAI(..), LlamaCpp(..), OpenRouter(..), OpenAICompatible(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse, OpenAICompletionRequest, OpenAICompletionResponse)

-- Re-export production models
import UniversalLLM.Models.Anthropic.Claude
  ( ClaudeSonnet45(..)
  , ClaudeSonnet45NoReason(..)
  , ClaudeHaiku45(..)
  , ClaudeOpus46(..)
  )
import UniversalLLM.Models.ZhipuAI.GLM
  ( GLM45(..)
  , GLM45Air(..)
  , GLM46(..)
  , GLM47(..)
  , ZAI(..)
  )
import UniversalLLM.Models.Alibaba.Qwen
  ( Qwen3CoderNext(..)
  , Qwen3Coder
  )
import UniversalLLM.Models.Google.Gemini
  ( Gemini3FlashPreview(..)
  , Gemini3ProPreview(..)
  )
import UniversalLLM.Models.Amazon.Nova
  ( Nova2Lite(..)
  )

-- ============================================================================
-- Backward Compatibility Aliases
-- ============================================================================

-- Provider aliases for backward compatibility with tests
-- All now use route instead of named provider functions
anthropicSonnet45 :: ComposableProvider (Model ClaudeSonnet45 Anthropic) (Anthropic.AnthropicReasoningState, ((), ()))
anthropicSonnet45 = route

anthropicSonnet45NoReason :: ComposableProvider (Model ClaudeSonnet45NoReason Anthropic) ((), ())
anthropicSonnet45NoReason = route

-- NOTE: OAuthToolsState removed as of 2025 - blacklist workaround no longer needed
anthropicSonnet45OAuth :: ComposableProvider (Model ClaudeSonnet45 AnthropicOAuth) (Anthropic.AnthropicReasoningState, ((), ((), ())))
anthropicSonnet45OAuth = route

anthropicSonnet45NoReasonOAuth :: ComposableProvider (Model ClaudeSonnet45NoReason AnthropicOAuth) ((), ((), ()))
anthropicSonnet45NoReasonOAuth = route

anthropicHaiku45 :: ComposableProvider (Model ClaudeHaiku45 Anthropic) (Anthropic.AnthropicReasoningState, ((), ()))
anthropicHaiku45 = route

anthropicHaiku45OAuth :: ComposableProvider (Model ClaudeHaiku45 AnthropicOAuth) (Anthropic.AnthropicReasoningState, ((), ((), ())))
anthropicHaiku45OAuth = route

anthropicOpus46 :: ComposableProvider (Model ClaudeOpus46 Anthropic) (Anthropic.AnthropicReasoningState, ((), ()))
anthropicOpus46 = route

anthropicOpus46OAuth :: ComposableProvider (Model ClaudeOpus46 AnthropicOAuth) (Anthropic.AnthropicReasoningState, ((), ((), ())))
anthropicOpus46OAuth = route

-- Backward compat aliases for GLM45Air providers
llamaCppGLM45 :: ComposableProvider (Model GLM45Air LlamaCpp) ((), ((), ((), ())))
llamaCppGLM45 = route

openRouterGLM45Air :: ComposableProvider (Model GLM45Air OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ((), ())))
openRouterGLM45Air = route

zaiGLM45 :: ComposableProvider (Model GLM45Air ZAI) ((), ((), ((), ())))
zaiGLM45 = route

llamaCppQwen3Coder :: ComposableProvider (Model Qwen3Coder LlamaCpp) ((), ((), ()))
llamaCppQwen3Coder = route

openRouterNova2Lite :: ComposableProvider (Model Nova2Lite OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ((), ())))
openRouterNova2Lite = route

openRouterGemini3ProPreview :: ComposableProvider (Model Gemini3ProPreview OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
openRouterGemini3ProPreview = route

openRouterGemini3FlashPreview :: ComposableProvider (Model Gemini3FlashPreview OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
openRouterGemini3FlashPreview = route

-- Test-specific instances for GLM45
-- Note: GLM45 via LlamaCpp should use GLM45Air (better for self-hosting)

-- Backward compat alias for tests that used GLM45 with OpenRouter
-- Now redirects to GLM45Air which is the correct model
openRouterGLM45 :: ComposableProvider (Model GLM45Air OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ((), ())))
openRouterGLM45 = route

-- ============================================================================
-- Test-Only Models (not suitable for production)
-- ============================================================================

-- These models are used only for compile-time safety tests and shouldn't be
-- used in production code.

-- Generic test model placeholder (used for completion interface tests)
data TestPlaceholderModel = TestPlaceholderModel deriving (Show, Eq)

instance ModelName (Model TestPlaceholderModel OpenAI) where
  modelName (Model _ _) = "glm-4-plus"

instance HasTools (Model TestPlaceholderModel OpenAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model TestPlaceholderModel OpenAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model TestPlaceholderModel OpenAI) where
  withJSON = OpenAI.openAIJSON

openAITestPlaceholder :: ComposableProvider (Model TestPlaceholderModel OpenAI) ((), ((), ((), ())))
openAITestPlaceholder = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model TestPlaceholderModel OpenAI)

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
