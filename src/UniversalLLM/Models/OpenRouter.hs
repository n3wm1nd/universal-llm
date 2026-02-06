{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.OpenRouter
Description: Production-ready OpenRouter model definitions

This module provides tested, production-ready definitions for models available via OpenRouter.
OpenRouter provides access to various models from different providers through a unified API.

= Available Models

* 'Gemini3FlashPreview' - Google's Gemini 3 Flash Preview
* 'Gemini3ProPreview' - Google's Gemini 3 Pro Preview
* 'Nova2Lite' - Amazon's Nova 2 Lite
* 'Universal' - Generic model (specify name via constructor)

= Provider-Specific Quirks

__Gemini__:
- Requires @thought_signature@ in function calls (cannot use fabricated tool history)
- Uses @reasoning_details@ instead of @reasoning_content@
- Reasoning details must be preserved in conversation history

__Nova__:
- Requires @toolConfig@ field when tool calls/results are in history
- Has reasoning but doesn't expose it via API
- Requires empty content fields (cannot be null)

= Usage

@
import UniversalLLM
import UniversalLLM.Models.OpenRouter

-- Use Gemini 3 Flash
let model = Model Gemini3FlashPreview OpenRouter
let provider = gemini3FlashPreview

-- Use Amazon Nova 2 Lite
let model = Model Nova2Lite OpenRouter
let provider = nova2Lite

-- Use any OpenRouter model
let model = Model (Universal "anthropic/claude-3-opus") OpenRouter
let provider = universal
@

= Authentication

Set @OPENROUTER_API_KEY@ environment variable.
-}

module UniversalLLM.Models.OpenRouter
  ( -- * Model Types
    Gemini3FlashPreview(..)
  , Gemini3ProPreview(..)
  , Nova2Lite(..)
  , Universal(..)
  , UniversalXMLTools(..)
    -- * Composable Providers
  , gemini3FlashPreview
  , gemini3ProPreview
  , nova2Lite
  , universalXMLTools
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..))
import UniversalLLM.Providers.XMLToolCalls (withFullXMLToolSupport)
import Data.Text (Text)

--------------------------------------------------------------------------------
-- Google Gemini 3 Flash Preview
--------------------------------------------------------------------------------

-- | Google Gemini 3 Flash Preview - Fast reasoning model
--
-- Capabilities:
-- - Tool calling
-- - Extended thinking via reasoning_details
-- - High-quality text generation
--
-- Quirks:
-- - Requires thought_signature in tool calls
-- - Cannot use fabricated tool history
data Gemini3FlashPreview = Gemini3FlashPreview deriving (Show, Eq)

instance ModelName (Model Gemini3FlashPreview OpenRouter) where
  modelName (Model _ _) = "google/gemini-3-flash-preview"

instance HasTools (Model Gemini3FlashPreview OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model Gemini3FlashPreview OpenRouter) where
  type ReasoningState (Model Gemini3FlashPreview OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- | Composable provider for Gemini 3 Flash Preview
gemini3FlashPreview :: ComposableProvider (Model Gemini3FlashPreview OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
gemini3FlashPreview = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Gemini3FlashPreview OpenRouter)

--------------------------------------------------------------------------------
-- Google Gemini 3 Pro Preview
--------------------------------------------------------------------------------

-- | Google Gemini 3 Pro Preview - More capable reasoning model
--
-- Similar to Flash but with enhanced capabilities.
data Gemini3ProPreview = Gemini3ProPreview deriving (Show, Eq)

instance ModelName (Model Gemini3ProPreview OpenRouter) where
  modelName (Model _ _) = "google/gemini-3-pro-preview"

instance HasTools (Model Gemini3ProPreview OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model Gemini3ProPreview OpenRouter) where
  type ReasoningState (Model Gemini3ProPreview OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- | Composable provider for Gemini 3 Pro Preview
gemini3ProPreview :: ComposableProvider (Model Gemini3ProPreview OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ()))
gemini3ProPreview = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model Gemini3ProPreview OpenRouter)

--------------------------------------------------------------------------------
-- Amazon Nova 2 Lite
--------------------------------------------------------------------------------

-- | Amazon Nova 2 Lite - Fast and efficient model
--
-- Capabilities:
-- - Tool calling
-- - Text generation
--
-- Quirks:
-- - Requires toolConfig field when tools are in history
-- - Has reasoning but doesn't expose it via API
-- - Requires empty content normalization
data Nova2Lite = Nova2Lite deriving (Show, Eq)

instance ModelName (Model Nova2Lite OpenRouter) where
  modelName (Model _ _) = "amazon/nova-2-lite-v1"

instance HasTools (Model Nova2Lite OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model Nova2Lite OpenRouter) where
  type ReasoningState (Model Nova2Lite OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- | Composable provider for Amazon Nova 2 Lite
--
-- Includes normalizeEmptyContent to handle Nova's requirement for non-null content.
nova2Lite :: ComposableProvider (Model Nova2Lite OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ((), ())))
nova2Lite = withReasoning `chainProviders` withTools `chainProviders` OpenAI.normalizeEmptyContent `chainProviders` OpenAI.baseComposableProvider @(Model Nova2Lite OpenRouter)

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
