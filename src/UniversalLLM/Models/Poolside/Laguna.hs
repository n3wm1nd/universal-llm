{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: UniversalLLM.Models.Poolside.Laguna
Description: Production-ready Laguna model definitions

This module provides tested, production-ready definitions for Poolside's Laguna
model family, a mostly agent-focused (tool-calling) model line.

= Available Models

* 'LagunaS21' - Laguna S 2.1, via OpenRouter and llama.cpp

= Provider Support

- __OpenRouter__: Cloud inference (poolside/laguna-s-2.1)
- __llama.cpp__: Local inference (any Laguna-S-2.1 GGUF quant, e.g. UD-Q4_K_XL)

= Laguna-Specific Quirks

__Reasoning field differs by provider__: Laguna S 2.1 reasons on both providers,
but the wire field differs - OpenRouter exposes it via @reasoning_details@ (its
own encrypted/opaque format), while llama.cpp exposes the standard
@reasoning_content@ field. This is a property of the provider's API shape, not
the model itself - see 'UniversalLLM.Models.ZhipuAI.GLM' for the same pattern.

__Not multimodal__: no 'HasVision' instance on either provider - Laguna S 2.1
does not process image content.

__JSON schema adherence unconfirmed__: no 'HasJSON' instance on either provider
yet - claiming it requires a protocol probe (see
'Protocol.OpenAITests.ignoresJSONSchemaShape') confirming @response_format@/
@json_schema@ is actually consulted, not just accepted. Add the instance once
that's been verified against the live API.

= Usage

@
import UniversalLLM
import UniversalLLM.Models.Poolside.Laguna

let model = LagunaS21 \`via\` OpenRouter
let provider = route

-- Or via llama.cpp
let model = LagunaS21 \`via\` LlamaCpp
@

= Authentication

- OpenRouter: Set @OPENROUTER_API_KEY@
- llama.cpp: No auth needed (local)
-}

module UniversalLLM.Models.Poolside.Laguna
  ( -- * Model Types
    LagunaS21(..)
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))

--------------------------------------------------------------------------------
-- Laguna S 2.1
--------------------------------------------------------------------------------

-- | Laguna S 2.1 - Poolside's agent-focused model
--
-- Capabilities:
-- - Tool calling
-- - Extended thinking/reasoning
-- - Not multimodal (no vision support)
data LagunaS21 = LagunaS21 deriving (Show, Eq)

-- OpenRouter provider
instance ModelName (Model LagunaS21 OpenRouter) where
  modelName (Model _ _) = "poolside/laguna-s-2.1"

instance HasTools (Model LagunaS21 OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model LagunaS21 OpenRouter) where
  type ReasoningState (Model LagunaS21 OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- Note: no HasJSON instance - not yet confirmed via ignoresJSONSchemaShape probe.
-- Note: no HasVision instance - Laguna S 2.1 is not multimodal.
instance Routing (Model LagunaS21 OpenRouter) where
  type RoutingState (Model LagunaS21 OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model LagunaS21 OpenRouter)

-- LlamaCpp provider
--
-- Base model name only (no quant suffix) - canonicalizeGGUFNames strips
-- quant/version tags (e.g. "-UD-Q4_K_XL") from the loaded GGUF filename down to
-- this form, so this must match the truncated name, not a specific quant.
instance ModelName (Model LagunaS21 LlamaCpp) where
  modelName (Model _ _) = "Laguna-S-2.1"

instance HasTools (Model LagunaS21 LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model LagunaS21 LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - not yet confirmed via ignoresJSONSchemaShape probe.
-- Note: no HasVision instance - Laguna S 2.1 is not multimodal.
instance Routing (Model LagunaS21 LlamaCpp) where
  type RoutingState (Model LagunaS21 LlamaCpp) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model LagunaS21 LlamaCpp)
