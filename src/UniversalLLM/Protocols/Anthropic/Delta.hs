{-# LANGUAGE OverloadedStrings #-}

-- | Anthropic SSE delta parsing and streaming content extraction.
--
-- = Delta semantics
--
-- Anthropic streaming delivers a sequence of typed SSE events. The full response
-- is accumulated via 'applyDelta', which delegates to 'mergeAnthropicDelta' for
-- event-type-aware merging (message_start, content_block_start, content_block_delta,
-- message_delta, etc.).
--
-- For real-time display, 'streamingContent' extracts human-visible chunks
-- (text or reasoning tokens) from a delta without needing to inspect the
-- accumulated state.
module UniversalLLM.Protocols.Anthropic.Delta
  ( Delta(..)
  , parseDelta
  , streamingContent
  , StreamingContent(..)
  ) where

import qualified Data.Aeson as Aeson
import           Data.Aeson (Value)
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import           Data.Text (Text)
import           Control.Applicative (asum)

-- | An opaque wrapper around a raw Anthropic SSE event JSON value.
--
-- The newtype prevents accidentally passing a full accumulated response where
-- a delta is expected, and makes function signatures self-documenting.
newtype Delta = Delta { deltaValue :: Value }
  deriving (Show, Eq)

-- | Visible streaming content contained in a single delta
data StreamingContent
  = StreamingText Text
  | StreamingReasoning Text
  deriving (Show, Eq)

-- | Parse an Anthropic SSE event from raw bytes.
parseDelta :: BS.ByteString -> Maybe Delta
parseDelta bs = Delta <$> Aeson.decodeStrict bs

-- | Extract human-visible content chunks from a delta for real-time display.
--
-- Only @content_block_delta@ events carry visible content; all other event
-- types (message_start, message_delta, etc.) return an empty list.
streamingContent :: Delta -> [StreamingContent]
streamingContent (Delta (Aeson.Object obj)) =
    case KM.lookup "type" obj of
        Just (Aeson.String "content_block_delta") ->
            case KM.lookup "delta" obj of
                Just (Aeson.Object delta) ->
                    asum
                        [ case KM.lookup "text" delta of
                            Just (Aeson.String t) | not (t == "") -> [StreamingText t]
                            _ -> []
                        , case KM.lookup "thinking" delta of
                            Just (Aeson.String t) | not (t == "") -> [StreamingReasoning t]
                            _ -> []
                        ]
                _ -> []
        _ -> []
streamingContent _ = []
