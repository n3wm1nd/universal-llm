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

* 'Qwen3CoderNext' - Latest Qwen 3 Coder variant (aliased as 'Qwen3Coder')
* 'Qwen3Coder30bInstruct' - Qwen 3 Coder 30B Instruct

= Provider Support

Qwen models are currently available through:

- __llama.cpp__: Local inference with native tool support

Unlike GLM models, Qwen's chat template properly converts XML tool calls to OpenAI format,
so no special XML parsing is needed.

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Alibaba.Qwen

-- Latest Qwen 3 Coder (currently Qwen3CoderNext)
let model = Model Qwen3CoderNext LlamaCpp
let provider = qwen3CoderNext

-- Or via the alias
let model = Model Qwen3Coder LlamaCpp
let provider = qwen3Coder
@

= Authentication

- llama.cpp: No auth needed (local)
-}

module UniversalLLM.Models.Alibaba.Qwen
  ( -- * Model Types
    Qwen3CoderNext(..)
  , Qwen3Coder30bInstruct(..)
    -- * Aliases
  , Qwen3Coder
  , qwen3Coder
    -- * Composable Providers
  , qwen3CoderNext
  , qwen3Coder30bInstruct
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..))

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

-- | Composable provider for Qwen 3 Coder Next via llama.cpp
qwen3CoderNext :: ComposableProvider (Model Qwen3CoderNext LlamaCpp) ((), ((), ()))
qwen3CoderNext = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3CoderNext LlamaCpp)

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

-- | Composable provider for Qwen 3 Coder 30B Instruct via llama.cpp
qwen3Coder30bInstruct :: ComposableProvider (Model Qwen3Coder30bInstruct LlamaCpp) ((), ((), ()))
qwen3Coder30bInstruct = withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Qwen3Coder30bInstruct LlamaCpp)

--------------------------------------------------------------------------------
-- Aliases (Qwen3Coder = latest = Qwen3CoderNext)
--------------------------------------------------------------------------------

-- | Alias for the latest Qwen 3 Coder variant ('Qwen3CoderNext')
type Qwen3Coder = Qwen3CoderNext

-- | Alias for 'qwen3CoderNext'
qwen3Coder :: ComposableProvider (Model Qwen3CoderNext LlamaCpp) ((), ((), ()))
qwen3Coder = qwen3CoderNext
