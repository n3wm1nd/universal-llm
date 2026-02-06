{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{- |
Module: UniversalLLM.Models.Qwen
Description: Production-ready Qwen model definitions

This module provides tested, production-ready definitions for Alibaba's Qwen models.

= Available Models

* 'Qwen3Coder' - Code-specialized model with tool support

= Provider Support

Qwen models are currently available through:

- __llama.cpp__: Local inference with native tool support

Unlike GLM models, Qwen's chat template properly converts XML tool calls to OpenAI format,
so no special XML parsing is needed.

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Qwen

let model = Model Qwen3Coder LlamaCpp
let provider = qwen3Coder
@

= Authentication

- llama.cpp: No auth needed (local)
-}

module UniversalLLM.Models.Qwen
  ( -- * Model Types
    Qwen3Coder(..)
    -- * Composable Providers
  , qwen3Coder
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..))

--------------------------------------------------------------------------------
-- Qwen 3 Coder
--------------------------------------------------------------------------------

-- | Qwen 3 Coder - Code-specialized model with tool support
--
-- Capabilities:
-- - Tool calling (native OpenAI format via chat template)
-- - JSON mode
-- - High-quality code generation
-- - Streaming responses
--
-- Note: No extended reasoning support (unlike GLM)
data Qwen3Coder = Qwen3Coder deriving (Show, Eq)

instance ModelName (Model Qwen3Coder LlamaCpp) where
  modelName (Model _ _) = "Qwen3-Coder-30B-Instruct"

instance HasTools (Model Qwen3Coder LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasJSON (Model Qwen3Coder LlamaCpp) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for Qwen 3 Coder
--
-- Includes JSON mode and tool support. Unlike GLM, no special workarounds needed -
-- the model's chat template handles tool calls properly.
qwen3Coder :: ComposableProvider (Model Qwen3Coder LlamaCpp) ((), ((), ()))
qwen3Coder = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3Coder LlamaCpp)
