{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Generic JSON delta merging.
--
-- = Delta semantics
--
-- OpenAI-compatible streaming APIs send incremental JSON objects (\"deltas\")
-- that accumulate into the final response. The merge rules are driven purely
-- by JSON value types — no field names are special-cased here:
--
-- * @null@ in delta              → noop (accumulator field left unchanged)
-- * @String s@ in delta          → append to existing @String@, or initialize
-- * @Object@ in delta            → recurse field-by-field into accumulator
-- * @Array@ of @{\"index\":n, …}@  → merge each element into the accumulator
--   slot identified by @n@, recursing into that slot
-- * Any other scalar             → replace (last-write-wins)
--
-- The @choices[].delta@ wrapper present in every OpenAI chunk is just an
-- indexed array and falls out of the generic rules naturally.
module UniversalLLM.Protocols.OpenAI.Delta
  ( Delta(..)
  , parseDelta
  , applyDelta
  ) where

import qualified Data.Aeson as Aeson
import           Data.Aeson (Value)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import qualified Data.Vector as V
import           Data.Maybe (fromMaybe)

-- | An opaque wrapper around a raw delta JSON value.
--
-- The newtype prevents accidentally passing a full accumulated response where
-- a delta is expected, and makes function signatures self-documenting.
newtype Delta = Delta { deltaValue :: Value }
  deriving (Show, Eq)

-- | Wrap a JSON value as a 'Delta'.
parseDelta :: BS.ByteString -> Maybe Delta
parseDelta bs = Delta <$> Aeson.decodeStrict bs

-- | Merge a 'Delta' into a JSON accumulator.
--
-- Start with 'Aeson.Null' (or @Aeson.object []@) as the initial accumulator
-- and fold over deltas with this function. The result after the final delta
-- is the fully reconstructed response object.
applyDelta :: Value -> Delta -> Value
applyDelta acc (Delta delta) = merge acc delta
  where
    merge :: Value -> Value -> Value
    merge a                Aeson.Null       = a
    merge _                (Aeson.Bool b)   = Aeson.Bool b
    merge _                (Aeson.Number n) = Aeson.Number n
    merge (Aeson.String a) (Aeson.String d) = Aeson.String (a <> d)
    merge _                (Aeson.String d) = Aeson.String d
    merge (Aeson.Object a) (Aeson.Object d) = Aeson.Object (mergeObjects a d)
    merge _                (Aeson.Object d) = Aeson.Object (mergeObjects KM.empty d)
    merge (Aeson.Array  a) (Aeson.Array  d)
      | isIndexedArray d                    = Aeson.Array (mergeIndexed a d)
    merge _                v                = v

    mergeObjects :: KM.KeyMap Value -> KM.KeyMap Value -> KM.KeyMap Value
    mergeObjects a d =
      foldr (\(k, dv) m ->
               case dv of
                 Aeson.Null -> m  -- null is always noop, even for absent keys
                 _          -> KM.insert k (merge (fromMaybe Aeson.Null (KM.lookup k m)) dv) m)
            a (KM.toList d)

    -- An array is merged by index when every element is an object that
    -- carries an "index" field (as in OpenAI tool_calls / choices arrays).
    isIndexedArray :: V.Vector Value -> Bool
    isIndexedArray v = not (V.null v) && V.all hasIndex v
      where
        hasIndex (Aeson.Object o) = KM.member indexKey o
        hasIndex _                = False

    mergeIndexed :: V.Vector Value -> V.Vector Value -> V.Vector Value
    mergeIndexed a d = V.foldl' applyOne a d
      where
        applyOne vec dElem = case indexOf dElem of
          Nothing  -> vec
          Just idx ->
            let padded = padTo (idx + 1) vec
                aElem  = padded V.! idx
            in padded V.// [(idx, merge aElem dElem)]

    padTo :: Int -> V.Vector Value -> V.Vector Value
    padTo n v
      | V.length v >= n = v
      | otherwise        = v V.++ V.replicate (n - V.length v) Aeson.Null

    indexOf :: Value -> Maybe Int
    indexOf (Aeson.Object o) =
      case KM.lookup indexKey o of
        Just (Aeson.Number n) -> Just (round n)
        _                     -> Nothing
    indexOf _ = Nothing

    indexKey :: Key.Key
    indexKey = Key.fromString "index"
