{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module TestModels where

import UniversalLLM.Core.Types
import UniversalLLM.Providers.Anthropic
import UniversalLLM.Providers.OpenAI

-- ============================================================================
-- Anthropic Models
-- ============================================================================

-- Test model for Claude Sonnet 4.5 (full featured: tools)
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance HasTools ClaudeSonnet45 Anthropic

instance ModelName Anthropic ClaudeSonnet45 where
  modelName _ = "claude-sonnet-4-5-20250929"

-- ============================================================================
-- OpenAI-Compatible Models (for testing with llama.cpp/GLM4.5)
-- ============================================================================

-- GLM4.5 via llama.cpp - supports tools, reasoning, and JSON
data GLM45 = GLM45 deriving (Show, Eq)

instance HasTools GLM45 OpenAI
instance HasReasoning GLM45 OpenAI
instance HasJSON GLM45 OpenAI

instance ModelName OpenAI GLM45 where
  modelName _ = "glm-4-plus"

-- Basic text-only model (for compile-time safety tests)
data BasicTextModel = BasicTextModel deriving (Show, Eq)

instance ModelName OpenAI BasicTextModel where
  modelName _ = "basic-text-model"

-- Tools-only model (no JSON, no reasoning)
data ToolsOnlyModel = ToolsOnlyModel deriving (Show, Eq)

instance HasTools ToolsOnlyModel OpenAI

instance ModelName OpenAI ToolsOnlyModel where
  modelName _ = "tools-only-model"

-- JSON-capable model (no tools, no reasoning)
data JSONModel = JSONModel deriving (Show, Eq)

instance HasJSON JSONModel OpenAI

instance ModelName OpenAI JSONModel where
  modelName _ = "json-model"
