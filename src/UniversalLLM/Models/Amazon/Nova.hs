{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.Amazon.Nova
Description: Production-ready Amazon Nova model definitions

This module provides tested, production-ready definitions for Amazon's Nova models.
These models are currently accessed through OpenRouter.

= Available Models

* 'Nova2Lite' - Amazon's Nova 2 Lite (fast and efficient)

= Provider-Specific Quirks

__Nova (via OpenRouter / Amazon Bedrock)__:
- Requires @toolConfig@ field when tool calls/results are in history
- Has reasoning but doesn't expose it via API
- Requires empty content fields (cannot be null)

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Amazon.Nova

let model = Model Nova2Lite OpenRouter
let provider = nova2Lite
@

= Authentication

Set @OPENROUTER_API_KEY@ environment variable.
-}

module UniversalLLM.Models.Amazon.Nova
  ( -- * Model Types
    Nova2Lite(..)
    -- * Composable Providers
  , nova2Lite
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (OpenRouter(..))

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
