{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.OpenRouter
Description: Generic parametric model types for OpenRouter

This module provides parametric catch-all model types for use with OpenRouter
when you don't want to (or can't) define a dedicated model type.

For concrete, vendor-specific models accessed via OpenRouter, see:

* "UniversalLLM.Models.Google.Gemini"  - Google Gemini models
* "UniversalLLM.Models.Amazon.Nova"    - Amazon Nova models
* "UniversalLLM.Models.Moonshot.Kimi"  - Moonshot Kimi models
* "UniversalLLM.Models.Minimax.M"      - MiniMax models
* "UniversalLLM.Models.ZhipuAI.GLM"   - ZhipuAI GLM models (also available via ZAI / llama.cpp)

= Available Model Types

* 'Universal' - Text-only catch-all (specify model name via constructor)
* 'UniversalXMLTools' - XML-based tool support for any OpenRouter model

= Usage

@
import UniversalLLM
import UniversalLLM.Models.OpenRouter

-- Any OpenRouter model, text-only
let model = Model (Universal "some/new-model") OpenRouter
let provider = OpenAI.baseComposableProvider

-- Any OpenRouter model with XML tool injection
let model = Model (UniversalXMLTools "some/model-without-native-tools") OpenRouter
let provider = universalXMLTools
@

= Authentication

Set @OPENROUTER_API_KEY@ environment variable.
-}

module UniversalLLM.Models.OpenRouter
  ( -- * Model Types
    Universal(..)
  , UniversalXMLTools(..)
    -- * Composable Providers
  , universalXMLTools
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Providers.XMLToolCalls (withFullXMLToolSupport)
import Data.Text (Text)

--------------------------------------------------------------------------------
-- Universal Models
--------------------------------------------------------------------------------

-- | Universal model for OpenRouter - text-only
--
-- Allows specifying any OpenRouter-compatible model by name.
-- Only claims text support (the least common denominator).
--
-- This is the safe default for unknown models. If you know the model supports
-- specific capabilities, define a proper model type instead.
--
-- Example:
-- @
-- let model = Model (Universal "some/new-model") OpenRouter
-- let provider = OpenAI.baseComposableProvider
-- @
data Universal = Universal Text deriving (Show, Eq)

instance ModelName (Model Universal OpenRouter) where
  modelName (Model (Universal name) _) = name

-- Note: No capability instances - text-only is the least common denominator

-- | Universal model with XML tool support
--
-- Uses XML-based tool injection and parsing to add tool support to any OpenRouter model.
-- From OpenRouter's perspective, this is still text-only, but we inject tool definitions
-- as XML in the prompt and parse XML responses.
--
-- This is less reliable than native tool support but works with any model that can
-- follow XML formatting instructions.
--
-- Example:
-- @
-- let model = Model (UniversalXMLTools "some/model-without-native-tools") OpenRouter
-- let provider = universalXMLTools
-- @
data UniversalXMLTools = UniversalXMLTools Text deriving (Show, Eq)

instance ModelName (Model UniversalXMLTools OpenRouter) where
  modelName (Model (UniversalXMLTools name) _) = name

instance HasTools (Model UniversalXMLTools OpenRouter) where
  type ToolState (Model UniversalXMLTools OpenRouter) = ((), ())
  withTools = withFullXMLToolSupport OpenAI.baseComposableProvider

-- | Composable provider for UniversalXMLTools
--
-- Adds XML-based tool support to any OpenRouter model.
universalXMLTools :: ComposableProvider (Model UniversalXMLTools OpenRouter) ((), ())
universalXMLTools = withTools
