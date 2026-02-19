{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.Google.Gemini
Description: Production-ready Google Gemini model definitions

This module provides tested, production-ready definitions for Google's Gemini models.
These models are currently accessed through OpenRouter.

= Available Models

* 'Gemini3FlashPreview' - Google's Gemini 3 Flash Preview (fast reasoning)
* 'Gemini3ProPreview' - Google's Gemini 3 Pro Preview (more capable)

= Provider-Specific Quirks

__Gemini (via OpenRouter)__:
- Requires @thought_signature@ in function calls (cannot use fabricated tool history)
- Uses @reasoning_details@ instead of @reasoning_content@
- Reasoning details must be preserved in conversation history

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Google.Gemini

-- Use Gemini 3 Flash
let model = Model Gemini3FlashPreview OpenRouter
let provider = gemini3FlashPreview

-- Use Gemini 3 Pro
let model = Model Gemini3ProPreview OpenRouter
let provider = gemini3ProPreview
@

= Authentication

Set @OPENROUTER_API_KEY@ environment variable.
-}

module UniversalLLM.Models.Google.Gemini
  ( -- * Model Types
    Gemini3FlashPreview(..)
  , Gemini3ProPreview(..)
    -- * Composable Providers
  , gemini3FlashPreview
  , gemini3ProPreview
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..))

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
