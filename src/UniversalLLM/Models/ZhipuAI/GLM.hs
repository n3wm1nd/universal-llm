{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

{- |
Module: UniversalLLM.Models.ZhipuAI.GLM
Description: Production-ready GLM model definitions

This module provides tested, production-ready definitions for Zhipu AI's GLM models.
These models can be accessed through multiple providers (llama.cpp, OpenRouter, ZAI).

= Available Models

* 'GLM45Air' - GLM-4.5-Air, fast capable model with tool support
* 'GLM46' - GLM-4.6, improved version via ZAI API
* 'GLM47' - GLM-4.7, via ZAI, OpenRouter, and llama.cpp
* 'GLM47Flash' - GLM-4.7-Flash, fast low-cost variant via ZAI, OpenRouter, and llama.cpp

* 'GLM5' - GLM-5, via ZAI API
* 'GLM51' - GLM-5.1, via ZAI API
* 'GLM52' - GLM-5.2, via ZAI API and AlibabaCloudTokenPlan
* 'GLM5Turbo' - GLM-5-Turbo, fast variant via ZAI API

= Provider Support

GLM models are available through multiple providers:

- __llama.cpp__: Local inference (llama.cpp handles tool call extraction)
- __OpenRouter__: Cloud inference (z-ai/glm-4.5-air, z-ai/glm-4.7, z-ai/glm-4.7-flash)
- __ZAI__: Official Zhipu AI API (api.z.ai)
- __AlibabaCloudTokenPlan__: Alibaba's Token Plan model marketplace (GLM-5.2 only)

= GLM-Specific Quirks

GLM models have some unique requirements:

1. __Null content handling__: GLM's Jinja2 template crashes on null content in messages
2. __Minimum tokens__: Need sufficient max_tokens when reasoning to avoid mid-block cutoff
3. __No real JSON schema support via the ZAI backend__: ZAI's @response_format@/
   @json_schema@ handling accepts @strict: true@ and a schema, but the schema is never
   actually consulted by the model - confirmed by probing with a deliberately unusual,
   nested schema (@{"result": {"hue_list": [...]}}@ instead of the obvious
   @{"colors": [...]}@ for a "list 3 colors" prompt): the model ignores it and returns
   whatever shape it naturally guesses (a bare array, or its own object shape), wrapped
   in a markdown fence or not depending on whether the prompt happens to mention "json".
   See 'Protocol.OpenAITests.jsonSchemaShapeCompliance' for the probe and
   'Protocol.OpenAITests.wrapsJSONInFence' / 'Protocol.OpenAITests.jsonStaysInReasoningDetails'
   for the surface symptoms (fenced content on GLM-4.5-Air, content swallowed into
   reasoning_details on GLM-4.7/GLM-4.7-Flash). Because the schema itself is never
   honored, none of the GLM models routed through ZAI (directly or via OpenRouter) claim
   'HasJSON' - claiming it would mean "you get JSON," which is true, but callers of
   'HasJSON' expect "you get JSON matching your schema," which is false here. JSON mode
   with real, grammar-enforced schema-following works via llama.cpp and AlibabaCloudTokenPlan.

These quirks are handled automatically by the composable providers when needed.

= Usage

@
import UniversalLLM
import UniversalLLM.Models.ZhipuAI.GLM

-- All models use the route function via the Routing typeclass
let model = GLM45Air \`via\` LlamaCpp
let provider = route

-- Or with Model constructor
let model = Model GLM45Air OpenRouter
let provider = route
@

= Authentication

- llama.cpp: No auth needed (local)
- OpenRouter: Set @OPENROUTER_API_KEY@
- ZAI: Set @ZAI_API_KEY@
-}

module UniversalLLM.Models.ZhipuAI.GLM
  ( -- * Model Types
    GLM45(..)
  , GLM45Air(..)
  , GLM46(..)
  , GLM47(..)
  , GLM47Flash(..)

  , GLM5(..)
  , GLM51(..)
  , GLM52(..)
  , GLM5Turbo(..)
  , ZAI(..)
    -- * Workaround Combinators
  , glmEnsureMinTokens
  , glmFixNullContent
  ) where

import UniversalLLM
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..), AlibabaCloudTokenPlan(..))
import UniversalLLM.Protocols.OpenAI (OpenAIRequest(..), OpenAIMessage(..), OpenAIMessageContent(..), OpenAIResponse, enableOpenAIStreaming, isStreamingOpenAIRequest)
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
      msg { content = Just (TextContent "") }
    -- Tool result messages with the string "null"
    fixNullContent msg@OpenAIMessage{ role = "tool", content = Just (TextContent "null") } =
      msg { content = Just (TextContent "") }
    -- Assistant messages with content - strip any <think> tags that might be malformed
    fixNullContent msg@OpenAIMessage{ role = "assistant", content = Just (TextContent contentTxt) } =
      msg { content = Just (TextContent (stripThinkTags contentTxt)) }
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

instance EnableStreaming (Model GLM45 ZAI) where enableStreamingForProtocol = enableOpenAIStreaming; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM45 ZAI) where
  modelName (Model _ _) = "glm-4.5"

instance HasTools (Model GLM45 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45 ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM45 ZAI) where
  type RoutingState (Model GLM45 ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45 ZAI)

--------------------------------------------------------------------------------
-- GLM-4.5-Air
--------------------------------------------------------------------------------

-- | GLM-4.5-Air - Fast and capable model with tools and reasoning
--
-- Capabilities:
-- - Tool calling (standard OpenAI format)
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

instance Routing (Model GLM45Air LlamaCpp) where
  type RoutingState (Model GLM45Air LlamaCpp) = ((), ((), ((), ())))
  route = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45Air LlamaCpp)

-- OpenRouter provider (native OpenAI format)
--
-- Note: the free-tier slug (z-ai/glm-4.5-air:free) rejects response_format/json_schema
-- requests ("This model is unavailable for free"). Using the paid slug instead so
-- JSON mode actually works; this is a superset of the free tier's behaviour.
instance ModelName (Model GLM45Air OpenRouter) where
  modelName (Model _ _) = "z-ai/glm-4.5-air"

instance HasTools (Model GLM45Air OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45Air OpenRouter) where
  type ReasoningState (Model GLM45Air OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM45Air OpenRouter) where
  type RoutingState (Model GLM45Air OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45Air OpenRouter)

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

instance EnableStreaming (Model GLM45Air ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM45Air ZAI) where
  modelName (Model _ _) = "GLM-4.5-Air"

instance HasTools (Model GLM45Air ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM45Air ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM45Air ZAI) where
  type RoutingState (Model GLM45Air ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM45Air ZAI)

--------------------------------------------------------------------------------
-- GLM-4.6
--------------------------------------------------------------------------------

-- | GLM-4.6 - Improved version of GLM-4.5 (ZAI only)
data GLM46 = GLM46 deriving (Show, Eq)

instance Provider (Model GLM46 ZAI) where
  type ProviderRequest (Model GLM46 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM46 ZAI) = OpenAIResponse

instance EnableStreaming (Model GLM46 ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM46 ZAI) where
  modelName (Model _ _) = "glm-4.6"

instance HasTools (Model GLM46 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM46 ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM46 ZAI) where
  type RoutingState (Model GLM46 ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM46 ZAI)

--------------------------------------------------------------------------------
-- GLM-4.7
--------------------------------------------------------------------------------

-- | GLM-4.7 - Latest GLM model (ZAI only)
data GLM47 = GLM47 deriving (Show, Eq)

instance Provider (Model GLM47 ZAI) where
  type ProviderRequest (Model GLM47 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM47 ZAI) = OpenAIResponse

instance EnableStreaming (Model GLM47 ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM47 ZAI) where
  modelName (Model _ _) = "glm-4.7"

instance HasTools (Model GLM47 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM47 ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM47 ZAI) where
  type RoutingState (Model GLM47 ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM47 ZAI)

--------------------------------------------------------------------------------
-- GLM-5
--------------------------------------------------------------------------------

-- | GLM-5 - Latest GLM model (ZAI only)
data GLM5 = GLM5 deriving (Show, Eq)

instance Provider (Model GLM5 ZAI) where
  type ProviderRequest (Model GLM5 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM5 ZAI) = OpenAIResponse

instance EnableStreaming (Model GLM5 ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM5 ZAI) where
  modelName (Model _ _) = "glm-5"

instance HasTools (Model GLM5 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM5 ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM5 ZAI) where
  type RoutingState (Model GLM5 ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM5 ZAI)

--------------------------------------------------------------------------------
-- GLM-5.2
--------------------------------------------------------------------------------

-- | GLM-5.2 - Latest GLM model (ZAI, AlibabaCloudTokenPlan)
data GLM52 = GLM52 deriving (Show, Eq)

instance Provider (Model GLM52 ZAI) where
  type ProviderRequest (Model GLM52 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM52 ZAI) = OpenAIResponse

instance EnableStreaming (Model GLM52 ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM52 ZAI) where
  modelName (Model _ _) = "glm-5.2"

instance HasTools (Model GLM52 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM52 ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM52 ZAI) where
  type RoutingState (Model GLM52 ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM52 ZAI)

--------------------------------------------------------------------------------
-- GLM-5.2 via AlibabaCloudTokenPlan
--------------------------------------------------------------------------------

instance ModelName (Model GLM52 AlibabaCloudTokenPlan) where
  modelName (Model _ _) = "glm-5.2"

instance HasTools (Model GLM52 AlibabaCloudTokenPlan) where
  withTools = OpenAI.openAITools

instance HasJSON (Model GLM52 AlibabaCloudTokenPlan) where
  withJSON = OpenAI.openAIJSON

instance HasReasoning (Model GLM52 AlibabaCloudTokenPlan) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model GLM52 AlibabaCloudTokenPlan) where
  type RoutingState (Model GLM52 AlibabaCloudTokenPlan) = ((), ((), ((), ())))
  route = withReasoning `chainProviders` withJSON `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM52 AlibabaCloudTokenPlan)

--------------------------------------------------------------------------------
-- GLM-5.1
--------------------------------------------------------------------------------

-- | GLM-5.1 - Latest GLM model (ZAI only)
data GLM51 = GLM51 deriving (Show, Eq)

instance Provider (Model GLM51 ZAI) where
  type ProviderRequest (Model GLM51 ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM51 ZAI) = OpenAIResponse

instance EnableStreaming (Model GLM51 ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM51 ZAI) where
  modelName (Model _ _) = "glm-5.1"

instance HasTools (Model GLM51 ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM51 ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM51 ZAI) where
  type RoutingState (Model GLM51 ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM51 ZAI)

--------------------------------------------------------------------------------
-- GLM-5-Turbo
--------------------------------------------------------------------------------

-- | GLM-5-Turbo - Fast variant of GLM-5 (ZAI only)
data GLM5Turbo = GLM5Turbo deriving (Show, Eq)

instance Provider (Model GLM5Turbo ZAI) where
  type ProviderRequest (Model GLM5Turbo ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM5Turbo ZAI) = OpenAIResponse

instance EnableStreaming (Model GLM5Turbo ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM5Turbo ZAI) where
  modelName (Model _ _) = "glm-5-turbo"

instance HasTools (Model GLM5Turbo ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM5Turbo ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM5Turbo ZAI) where
  type RoutingState (Model GLM5Turbo ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM5Turbo ZAI)

--------------------------------------------------------------------------------
-- GLM-4.7 via OpenRouter
--------------------------------------------------------------------------------

instance ModelName (Model GLM47 OpenRouter) where
  modelName (Model _ _) = "z-ai/glm-4.7"

instance HasTools (Model GLM47 OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM47 OpenRouter) where
  type ReasoningState (Model GLM47 OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- Note: no HasJSON instance. Unlike GLM-4.5-Air (fenced content), GLM-4.7 via
-- OpenRouter puts the entire JSON answer in reasoning_details and never populates
-- content - confirmed at finish_reason: "stop", so it isn't token-budget truncation
-- either. There's no AssistantText to unwrap, so openAIJSONWithFencedFallback can't
-- recover it. See GLM-Specific Quirks at the top of this module and
-- Protocol.OpenAITests.jsonStaysInReasoningDetails for the probe.

instance Routing (Model GLM47 OpenRouter) where
  type RoutingState (Model GLM47 OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM47 OpenRouter)

--------------------------------------------------------------------------------
-- GLM-4.7-Flash
--------------------------------------------------------------------------------

-- | GLM-4.7-Flash - Fast, low-cost variant of GLM-4.7
--
-- Available via ZAI, OpenRouter, and llama.cpp.
data GLM47Flash = GLM47Flash deriving (Show, Eq)

-- ZAI provider
instance Provider (Model GLM47Flash ZAI) where
  type ProviderRequest (Model GLM47Flash ZAI) = OpenAIRequest
  type ProviderResponse (Model GLM47Flash ZAI) = OpenAIResponse

instance EnableStreaming (Model GLM47Flash ZAI) where enableStreamingForProtocol = enableOpenAIStreaming ; isStreamingRequest = isStreamingOpenAIRequest

instance ModelName (Model GLM47Flash ZAI) where
  modelName (Model _ _) = "glm-4.7-flash"

instance HasTools (Model GLM47Flash ZAI) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM47Flash ZAI) where
  withReasoning = OpenAI.openAIReasoning

-- Note: no HasJSON instance - see GLM-Specific Quirks at the top of this module.
instance Routing (Model GLM47Flash ZAI) where
  type RoutingState (Model GLM47Flash ZAI) = ((), ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM47Flash ZAI)

-- OpenRouter provider
instance ModelName (Model GLM47Flash OpenRouter) where
  modelName (Model _ _) = "z-ai/glm-4.7-flash"

instance HasTools (Model GLM47Flash OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM47Flash OpenRouter) where
  type ReasoningState (Model GLM47Flash OpenRouter) = OpenAI.OpenRouterReasoningState
  withReasoning = OpenAI.openRouterReasoning

-- Note: no HasJSON instance - same "JSON stays in reasoning_details, content: null at
-- finish_reason: stop" quirk as GLM-4.7 via OpenRouter (see GLM-Specific Quirks at the
-- top of this module and Protocol.OpenAITests.jsonStaysInReasoningDetails).

instance Routing (Model GLM47Flash OpenRouter) where
  type RoutingState (Model GLM47Flash OpenRouter) = (OpenAI.OpenRouterReasoningState, ((), ()))
  route = withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM47Flash OpenRouter)

-- LlamaCpp provider
instance ModelName (Model GLM47Flash LlamaCpp) where
  modelName (Model _ _) = "GLM-4.7-Flash"

instance HasTools (Model GLM47Flash LlamaCpp) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model GLM47Flash LlamaCpp) where
  withReasoning = OpenAI.openAIReasoning

instance HasJSON (Model GLM47Flash LlamaCpp) where
  withJSON = OpenAI.openAIJSON

instance Routing (Model GLM47Flash LlamaCpp) where
  type RoutingState (Model GLM47Flash LlamaCpp) = ((), ((), ((), ())))
  route = withJSON `chainProviders` withReasoning `chainProviders` withTools `chainProviders` OpenAI.baseComposableProvider @(Model GLM47Flash LlamaCpp)
