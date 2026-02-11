{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

{- |
Module: UniversalLLM.Models.GLM
Description: Production-ready GLM model definitions

This module provides tested, production-ready definitions for Zhipu AI's GLM models.
These models can be accessed through multiple providers (llama.cpp, OpenRouter, ZAI).

= Available Models

* 'GLM45Air' - GLM-4.5-Air, a fast and capable model with tool support
* 'GLM46' - GLM-4.6, improved version via ZAI API
* 'GLM47' - GLM-4.7, previous version via ZAI API
* 'GLM5' - GLM-5, latest version via ZAI API

= Provider Support

GLM models are available through multiple providers:

- __llama.cpp__: Local inference with hybrid tool handling (OpenAI request, XML response)
- __OpenRouter__: Cloud inference via z-ai/glm-4.5-air:free
- __ZAI__: Official Zhipu AI API (api.z.ai)

= GLM-Specific Quirks

GLM models have some unique requirements:

1. __Null content handling__: GLM's Jinja2 template crashes on null content in messages
2. __Minimum tokens__: Need sufficient max_tokens when reasoning to avoid mid-block cutoff
3. __XML tool calls__ (llama.cpp only): Model outputs XML, requires special parsing

These quirks are handled automatically by the composable providers.

= Usage

@
import UniversalLLM
import UniversalLLM.Models.GLM

-- Use via llama.cpp
let model = Model GLM45Air LlamaCpp
let provider = glm45AirLlamaCpp

-- Use via OpenRouter
let model = Model GLM45Air OpenRouter
let provider = glm45AirOpenRouter

-- Use via ZAI
let model = Model GLM45Air ZAI
let provider = glm45AirZAI
@

= Authentication

- llama.cpp: No auth needed (local)
- OpenRouter: Set @OPENROUTER_API_KEY@
- ZAI: Set @ZAI_API_KEY@
-}

module UniversalLLM.Models.GLM
  ( -- * Model Types
    GLM45(..)
  , GLM45Air(..)
  , GLM46(..)
  , GLM47(..)
  , GLM5(..)
  , ZAI(..)
    -- * Composable Providers
    -- ** GLM-4.5
  , glm45
    -- ** GLM-4.5-Air
  , glm45AirLlamaCpp
  , glm45AirOpenRouter
  , glm45AirZAI
    -- ** GLM-4.6
  , glm46
    -- ** GLM-4.7
  , glm47
    -- ** GLM-5
  , glm5
    -- * Workaround Combinators
  , glmEnsureMinTokens
  , glmFixNullContent
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..), OpenAICompatible(..))
import UniversalLLM.Providers.XMLToolCalls (xmlResponseParser)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest(..), OpenAIMessage(..), OpenAIResponse)
import Data.Text (Text)
import qualified Data.Text as T

--------------------------------------------------------------------------------
-- GLM-Specific Workarounds
--------------------------------------------------------------------------------

-- | GLM-specific minimum token limit enforcer
--
-- GLM's Jinja2 template crashes if it runs out of tokens while generating a <think> block,
-- because it tries to parse incomplete reasoning content and hits null values.
--
-- This combinator ensures a minimum max_tokens is set to reduce the chance of mid-block cutoff,
-- but only if reasoning is enabled (checked via Reasoning config).
glmEnsureMinTokens :: forall model s.
                      (ProviderRequest model ~ OpenAIRequest)
                   => Int  -- ^ Minimum max_tokens
                   -> ComposableProvider model s
glmEnsureMinTokens minTokens _m configs _s =
    noopHandler
    { cpConfigHandler = \req ->
        if reasoningDisabled configs
          then req
          else case max_tokens req of
            Nothing -> req { max_tokens = Just minTokens }
            Just current | current < minTokens -> req { max_tokens = Just minTokens }
            Just _ -> req
    }
  where
    -- Check if reasoning is explicitly disabled in config
    reasoningDisabled :: [ModelConfig model] -> Bool
    reasoningDisabled cfg = any isReasoningFalse cfg
      where
        isReasoningFalse (Reasoning False) = True
        isReasoningFalse _ = False

-- | GLM-specific null content fixer
--
-- GLM's Jinja2 template can't handle null content in messages. Two issues:
-- 1. Assistant messages with tool calls have null content (per OpenAI spec)
-- 2. Tool results that return Aeson.Null get JSON-encoded to the string \"null\"
--
-- GLM's template tries to do string operations on content without checking if it's null,
-- causing \"Value is not callable: null\" errors.
--
-- This combinator fixes both by replacing null/problematic content with empty strings.
--
-- Apply this AFTER openAIWithTools in the provider chain.
glmFixNullContent :: forall model s.
                     (ProviderRequest model ~ OpenAIRequest)
                  => ComposableProvider model s
glmFixNullContent _model _configs _s =
    noopHandler
    { cpToRequest = \_msg req -> (fixAllNullContent req)
    }
  where
    -- Fix all messages in the request that have null content
    fixAllNullContent :: OpenAIRequest -> OpenAIRequest
    fixAllNullContent req = req { messages = map fixNullContent (messages req) }

    -- Replace null/problematic content with empty string and strip malformed think tags
    fixNullContent :: OpenAIMessage -> OpenAIMessage
    -- Assistant messages with null content (when they have tool calls)
    fixNullContent msg@OpenAIMessage{ role = "assistant", content = Nothing } =
      msg { content = Just "" }
    -- Tool result messages with the string "null"
    fixNullContent msg@OpenAIMessage{ role = "tool", content = Just "null" } =
      msg { content = Just "" }
    -- Assistant messages with content - strip any <think> tags that might be malformed
    fixNullContent msg@OpenAIMessage{ role = "assistant", content = Just contentTxt } =
      msg { content = Just (stripThinkTags contentTxt) }
    -- Everything else passes through unchanged
    fixNullContent msg = msg

    -- Strip all <think>...</think> blocks and any orphaned tags
    stripThinkTags :: Text -> Text
    stripThinkTags txt =
      let withoutBlocks = T.replace "</think>" "" $ T.replace "<think>" "" txt
      in withoutBlocks

--------------------------------------------------------------------------------
-- GLM-4.5
--------------------------------------------------------------------------------

-- | GLM-4.5 - Full model (ZAI only)
data GLM45 = GLM45 deriving (Show, Eq)

instance Provider (Model GLM45 ZAI) where
  type ProviderRequest (Model GLM45 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM45 ZAI) = OpenAIResponse

instance ModelName (Model GLM45 ZAI) where
  modelName (Model _ _) = "glm-4.5"

instance HasTools (Model GLM45 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45 ZAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM45 ZAI) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for GLM-4.5
glm45 :: ComposableProvider (Model GLM45 ZAI) ((), ((), ((), ())))
glm45 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45 ZAI)

--------------------------------------------------------------------------------
-- GLM-4.5-Air
--------------------------------------------------------------------------------

-- | GLM-4.5-Air - Fast and capable model with tools and reasoning
--
-- Capabilities:
-- - Tool calling (native or via XML depending on provider)
-- - Extended thinking/reasoning
-- - High-quality code generation
-- - Streaming responses
data GLM45Air = GLM45Air deriving (Show, Eq)

-- LlamaCpp provider
instance ModelName (Model GLM45Air LlamaCpp) where
  modelName (Model _ _) = "GLM-4.5-Air"

instance HasTools (Model GLM45Air LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45Air LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM45Air LlamaCpp) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for GLM-4.5-Air via llama.cpp
glm45AirLlamaCpp :: ComposableProvider (Model GLM45Air LlamaCpp) ((), ((), ((), ())))
glm45AirLlamaCpp = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45Air LlamaCpp)

-- OpenRouter provider (native OpenAI format)
instance ModelName (Model GLM45Air OpenRouter) where
  modelName (Model _ _) = "z-ai/glm-4.5-air:free"

instance HasTools (Model GLM45Air OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45Air OpenRouter) where
  type ReasoningState (Model GLM45Air OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

instance HasJSON (Model GLM45Air OpenRouter) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for GLM-4.5-Air via OpenRouter
glm45AirOpenRouter :: ComposableProvider (Model GLM45Air OpenRouter) (OpenAI.OpenRouterReasoningState, ((), ((), ())))
glm45AirOpenRouter = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45Air OpenRouter)

-- ZAI provider (official API)
-- Define ZAI as a provider
data ZAI = ZAI deriving (Show, Eq)

instance SupportsTemperature ZAI
instance SupportsMaxTokens ZAI
instance SupportsSeed ZAI
instance SupportsSystemPrompt ZAI
instance SupportsStop ZAI
instance SupportsStreaming ZAI

instance Provider (Model GLM45Air ZAI) where
  type ProviderRequest (Model GLM45Air ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM45Air ZAI) = OpenAIResponse

instance ModelName (Model GLM45Air ZAI) where
  modelName (Model _ _) = "GLM-4.5-Air"

instance HasTools (Model GLM45Air ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45Air ZAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM45Air ZAI) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for GLM-4.5-Air via ZAI
glm45AirZAI :: ComposableProvider (Model GLM45Air ZAI) ((), ((), ((), ())))
glm45AirZAI = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45Air ZAI)

--------------------------------------------------------------------------------
-- GLM-4.6
--------------------------------------------------------------------------------

-- | GLM-4.6 - Improved version of GLM-4.5 (ZAI only)
data GLM46 = GLM46 deriving (Show, Eq)

instance Provider (Model GLM46 ZAI) where
  type ProviderRequest (Model GLM46 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM46 ZAI) = OpenAIResponse

instance ModelName (Model GLM46 ZAI) where
  modelName (Model _ _) = "glm-4.6"

instance HasTools (Model GLM46 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM46 ZAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM46 ZAI) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for GLM-4.6
glm46 :: ComposableProvider (Model GLM46 ZAI) ((), ((), ((), ())))
glm46 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM46 ZAI)

--------------------------------------------------------------------------------
-- GLM-4.7
--------------------------------------------------------------------------------

-- | GLM-4.7 - Latest GLM model (ZAI only)
data GLM47 = GLM47 deriving (Show, Eq)

instance Provider (Model GLM47 ZAI) where
  type ProviderRequest (Model GLM47 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM47 ZAI) = OpenAIResponse

instance ModelName (Model GLM47 ZAI) where
  modelName (Model _ _) = "glm-4.7"

instance HasTools (Model GLM47 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM47 ZAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM47 ZAI) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for GLM-4.7
glm47 :: ComposableProvider (Model GLM47 ZAI) ((), ((), ((), ())))
glm47 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM47 ZAI)

--------------------------------------------------------------------------------
-- GLM-5
--------------------------------------------------------------------------------

-- | GLM-5 - Latest GLM model (ZAI only)
data GLM5 = GLM5 deriving (Show, Eq)

instance Provider (Model GLM5 ZAI) where
  type ProviderRequest (Model GLM5 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM5 ZAI) = OpenAIResponse

instance ModelName (Model GLM5 ZAI) where
  modelName (Model _ _) = "glm-5"

instance HasTools (Model GLM5 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM5 ZAI) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM5 ZAI) where
  withJSON = OpenAI.openAIJSON

-- | Composable provider for GLM-5
glm5 :: ComposableProvider (Model GLM5 ZAI) ((), ((), ((), ())))
glm5 = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM5 ZAI)
