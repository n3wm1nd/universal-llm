{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module TestModels where

import UniversalLLM.Core.Types
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.Anthropic (Anthropic(..))
import UniversalLLM.Providers.OpenAI (OpenAI(..))

-- ============================================================================
-- Anthropic Models
-- ============================================================================

-- Test model for Claude Sonnet 4.5 (full featured: tools)
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance ModelName Anthropic ClaudeSonnet45 where
  modelName _ = "claude-sonnet-4-5-20250929"

instance HasTools ClaudeSonnet45 Anthropic where
  toolsComposableProvider = Anthropic.toolsComposableProvider

instance ProviderImplementation Anthropic ClaudeSonnet45 where
  getComposableProvider = Anthropic.baseComposableProvider <> Anthropic.toolsComposableProvider

-- ============================================================================
-- OpenAI-Compatible Models (for testing with llama.cpp/GLM4.5)
-- ============================================================================

-- GLM4.5 via llama.cpp - supports tools, reasoning, and JSON
data GLM45 = GLM45 deriving (Show, Eq)

instance ModelName OpenAI GLM45 where
  modelName _ = "glm-4-plus"

instance HasTools GLM45 OpenAI where
  toolsComposableProvider = OpenAI.toolsComposableProvider

instance HasReasoning GLM45 OpenAI where
  reasoningComposableProvider = OpenAI.reasoningComposableProvider

instance HasJSON GLM45 OpenAI where
  jsonComposableProvider = OpenAI.jsonComposableProvider

instance ProviderImplementation OpenAI GLM45 where
  getComposableProvider = OpenAI.baseComposableProvider <> OpenAI.toolsComposableProvider <> OpenAI.reasoningComposableProvider <> OpenAI.jsonComposableProvider

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
  toolsComposableProvider = OpenAI.toolsComposableProvider

instance ProviderImplementation OpenAI ToolsOnlyModel where
  getComposableProvider = OpenAI.baseComposableProvider <> OpenAI.toolsComposableProvider

-- JSON-capable model (no tools, no reasoning)
data JSONModel = JSONModel deriving (Show, Eq)

instance ModelName OpenAI JSONModel where
  modelName _ = "json-model"

instance HasJSON JSONModel OpenAI where
  jsonComposableProvider = OpenAI.jsonComposableProvider

instance ProviderImplementation OpenAI JSONModel where
  getComposableProvider = OpenAI.baseComposableProvider <> OpenAI.jsonComposableProvider
