{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}

-- | Wire-level cache coherency primitives for testing ComposableProvider stacks.
--
-- Defines four typeclasses:
--
-- * 'CacheNormalized' — strip fields that are irrelevant to provider-side
--   caching (e.g. generation parameters, cache control annotations).  Two
--   requests that are equal after normalization should hit the same cache.
--
-- * 'HasWireMessages' — extract the message list from a provider request in a
--   protocol-agnostic way, for verbatim comparison in tests.
--
-- * 'SimulateResponse' — construct a plausible provider response from a list
--   of messages, as a stand-in for what an LLM API would return.
--
-- * 'AppendResponse' — merge a completed response's messages back into the
--   originating request, producing the combined history ready for the next
--   user turn.
--
-- = Cache coherency invariant
--
-- For a ComposableProvider that correctly handles conversation history:
--
-- @
-- let req1   = toProviderRequest cp model configs s  reqMsgs
-- let req2   = toProviderRequest cp model configs s' (reqMsgs ++ respMsgs ++ nextMsgs)
-- let merged = appendResponse req1 (simulateResponse respMsgs)
-- cacheNormalize merged `isPrefixOf` cacheNormalize req2
-- @
--
-- i.e. whatever the caching system saw for request 1 plus the response should
-- appear verbatim at the front of request 2, so the provider can serve from
-- cache for that prefix.
module Protocol.CacheCoherency
  ( CacheNormalized(..)
  , ModelOf
  , HasWireMessages(..)
  , SimulateResponse(..)
  , AppendResponse(..)
  -- * Helpers exported for model-specific overrides
  , stripCacheControlBlock
  , normalizeOpenAIRequestWith
  , normalizeNovaMessage
  ) where

import qualified Data.Aeson as Aeson
import UniversalLLM
import qualified UniversalLLM.Protocols.OpenAI as OP
import qualified UniversalLLM.Protocols.Anthropic as AP

-- | Extract the AI model type from a 'Model' application.
type family ModelOf m where
  ModelOf (Model a _p) = a

-- | Strip fields that are irrelevant to provider-side caching, for a specific
-- AI model @m@ and provider @p@.
--
-- The typeclass is split on model and provider so that normalization can be
-- tuned at any granularity: protocol-wide (the defaults here), per-provider,
-- or per-model.  Provider-level defaults are @OVERLAPPABLE@; model-specific
-- overrides defined alongside the model's tests take priority.
--
-- Known cache-irrelevant fields:
--
-- * __All Anthropic providers__: @cache_control@ annotations are hints /to/
--   the caching infrastructure, recomputed on every request.  Generation
--   parameters (@temperature@, @max_tokens@, @stream@) do not affect the key.
--
-- * __All OpenAI-protocol providers__: Generation parameters
--   (@temperature@, @max_tokens@, @seed@, @stream@) do not affect the key.
--   Messages are kept verbatim by default; models that normalize differently
--   (e.g. Nova's empty-content conversion) override the message instance.
class CacheNormalized m p a where
  cacheNormalize :: a -> a

-- ============================================================================
-- OpenAI shared helpers (exported for model-specific overrides)
-- ============================================================================

-- | Normalize an OpenAI request: strip generation parameters, apply
-- per-message normalization.
normalizeOpenAIRequestWith
  :: (OP.OpenAIMessage -> OP.OpenAIMessage)
  -> OP.OpenAIRequest
  -> OP.OpenAIRequest
normalizeOpenAIRequestWith f req = OP.OpenAIRequest
  { OP.model           = OP.model req
  , OP.messages        = map f (OP.messages req)
  , OP.temperature     = Nothing
  , OP.max_tokens      = Nothing
  , OP.seed            = Nothing
  , OP.tools           = OP.tools req
  , OP.response_format = OP.response_format req
  , OP.stream          = Nothing
  , OP.reasoning       = OP.reasoning req
  }

-- | Nova-specific wire message normalization: Amazon Bedrock converts
-- @content: ""@ to @content: null@ via @normalizeEmptyContent@ in its
-- provider chain.  Both mean "no content"; we equate them for comparison.
normalizeNovaMessage :: OP.OpenAIMessage -> OP.OpenAIMessage
normalizeNovaMessage msg = msg { OP.content = norm (OP.content msg) }
  where
    norm (Just "") = Nothing
    norm c         = c

-- ============================================================================
-- OpenAI protocol defaults (OVERLAPPABLE — one instance per wire type)
-- ============================================================================

instance {-# OVERLAPPABLE #-} CacheNormalized m p OP.OpenAIRequest where
  cacheNormalize = normalizeOpenAIRequestWith id

instance {-# OVERLAPPABLE #-} CacheNormalized m p OP.OpenAIMessage where
  cacheNormalize = id

-- ============================================================================
-- Anthropic protocol defaults (OVERLAPPABLE — one instance per wire type)
-- ============================================================================

-- | Strip cache_control from a single content block.
stripCacheControlBlock :: AP.AnthropicContentBlock -> AP.AnthropicContentBlock
stripCacheControlBlock (AP.AnthropicTextBlock txt _)       = AP.AnthropicTextBlock txt Nothing
stripCacheControlBlock (AP.AnthropicToolUseBlock i n v _)  = AP.AnthropicToolUseBlock i n v Nothing
stripCacheControlBlock (AP.AnthropicToolResultBlock i c _) = AP.AnthropicToolResultBlock i c Nothing
stripCacheControlBlock b@AP.AnthropicThinkingBlock{}       = b { AP.thinkingCacheControl = Nothing }

instance {-# OVERLAPPABLE #-} CacheNormalized m p AP.AnthropicMessage where
  cacheNormalize msg = msg { AP.content = map stripCacheControlBlock (AP.content msg) }

instance {-# OVERLAPPABLE #-} CacheNormalized m p AP.AnthropicRequest where
  cacheNormalize req = AP.AnthropicRequest
    { AP.model         = AP.model req
    , AP.messages      = map stripMsg (AP.messages req)
    , AP.max_tokens    = 0  -- generation parameter, not cache-relevant
    , AP.temperature   = Nothing
    , AP.system        = fmap (map (\b -> b { AP.systemCacheControl = Nothing })) (AP.system req)
    , AP.tools         = fmap (map (\t -> t { AP.anthropicToolCacheControl = Nothing })) (AP.tools req)
    , AP.stream        = Nothing
    , AP.thinking      = AP.thinking req
    , AP.output_config = AP.output_config req
    }
    where stripMsg msg = msg { AP.content = map stripCacheControlBlock (AP.content msg) }

-- ============================================================================
-- HasWireMessages
-- ============================================================================

-- | Extract the wire-level message list from a provider request, for verbatim
-- comparison in tests.
class HasWireMessages req where
  type WireMessage req
  wireMessages :: req -> [WireMessage req]

instance HasWireMessages OP.OpenAIRequest where
  type WireMessage OP.OpenAIRequest = OP.OpenAIMessage
  wireMessages = OP.messages

instance HasWireMessages AP.AnthropicRequest where
  type WireMessage AP.AnthropicRequest = AP.AnthropicMessage
  wireMessages = AP.messages

-- ============================================================================
-- SimulateResponse
-- ============================================================================

-- | Construct a plausible provider response from a list of messages.
--
-- This is a test-only stand-in for what an LLM API would return.  The
-- response is a single assistant turn: all 'AssistantText' and 'AssistantTool'
-- blocks are folded into one message, exactly as a real LLM API would return
-- them.  Other message types (user, system, tool results) are ignored since
-- they cannot appear in a response.
class SimulateResponse resp where
  simulateResponse :: [Message m] -> resp

instance SimulateResponse OP.OpenAIResponse where
  simulateResponse msgs =
    let textContent  = [ txt | AssistantText txt <- msgs ]
        toolCalls    = [ OP.convertFromToolCall tc | AssistantTool tc <- msgs ]
        -- OpenAI requires content = Just "" on assistant messages that carry
        -- tool_calls (for verbatim round-trip preservation); pure text messages
        -- use the actual text.  This matches what the provider produces.
        assistantMsg = OP.defaultOpenAIMessage
          { OP.role       = "assistant"
          , OP.content    = case (textContent, toolCalls) of
              (_,   _:_) -> Just ""           -- tool call: empty content as per provider
              (t:_, [])  -> Just t            -- text-only: use the text
              ([],  [])  -> Nothing
          , OP.tool_calls = case toolCalls of { [] -> Nothing; tcs -> Just tcs }
          }
    in OP.OpenAISuccess (OP.OpenAISuccessResponse [OP.OpenAIChoice assistantMsg])

instance SimulateResponse AP.AnthropicResponse where
  simulateResponse msgs =
    let toBlock (AssistantText txt)  = Just (AP.AnthropicTextBlock txt Nothing)
        toBlock (AssistantTool (ToolCall tcId tcName tcArgs)) =
          Just (AP.AnthropicToolUseBlock tcId tcName tcArgs Nothing)
        toBlock (AssistantTool (InvalidToolCall tcId tcName _ _)) =
          Just (AP.AnthropicToolUseBlock tcId tcName Aeson.Null Nothing)
        toBlock _                    = Nothing
        blocks = [ b | Just b <- map toBlock msgs ]
    in AP.AnthropicSuccess AP.defaultAnthropicSuccessResponse
        { AP.responseRole    = "assistant"
        , AP.responseContent = blocks
        }

-- ============================================================================
-- AppendResponse
-- ============================================================================

-- | Append the messages from a successful response to the originating request,
-- producing the combined history ready for the next user turn.
--
-- Returns 'Nothing' for error responses.
class AppendResponse req resp | req -> resp where
  appendResponse :: req -> resp -> Maybe req

instance AppendResponse OP.OpenAIRequest OP.OpenAIResponse where
  appendResponse req (OP.OpenAISuccess (OP.OpenAISuccessResponse choices)) =
    Just req { OP.messages = OP.messages req ++ map OP.message choices }
  appendResponse _ _ = Nothing

instance AppendResponse AP.AnthropicRequest AP.AnthropicResponse where
  appendResponse req (AP.AnthropicSuccess resp) =
    let assistantMsg = AP.AnthropicMessage
          { AP.role    = "assistant"
          , AP.content = AP.responseContent resp
          }
    in Just req { AP.messages = AP.messages req ++ [assistantMsg] }
  appendResponse _ _ = Nothing
