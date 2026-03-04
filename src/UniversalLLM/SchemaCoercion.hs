{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}
-- | Schema-guided value coercion for broken LLM providers
--
-- Some LLM providers (notably z.ai) don't respect JSON Schema types and return
-- everything as strings in tool calls. This module provides coercion utilities
-- to fix up these malformed responses before parsing.
--
-- The coercion is schema-guided: we use the expected schema (from HasCodec)
-- to know what types fields should be, then coerce string values to the
-- correct JSON types.
module UniversalLLM.SchemaCoercion
  ( coerceWithCodec
  , coerceValue
  ) where

import Data.Aeson (Value(..))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Read as T
import qualified Data.Vector as V
import Autodocodec (HasCodec)
import Autodocodec.Schema (jsonSchemaViaCodec)

-- | Coerce a value to match the schema defined by its HasCodec instance
--
-- This fixes broken LLM responses where integers/booleans are returned as strings.
-- Uses the schema to guide coercion, so "123" becomes 123 only if the schema says integer.
coerceWithCodec :: forall a. HasCodec a => Value -> Either Text Value
coerceWithCodec val = coerceValue (Aeson.toJSON $ jsonSchemaViaCodec @a) val

-- | Recursively coerce a value to match a JSON Schema
--
-- Strategy:
-- - If schema says "integer" and value is a string, try to parse as integer
-- - If schema says "number" and value is a string, try to parse as number
-- - If schema says "boolean" and value is a string, parse "true"/"false"
-- - If schema says "object", recurse into properties
-- - If schema says "array", recurse into items
-- - For anyOf/oneOf (optional fields), try each option
-- - Otherwise, return value unchanged
coerceValue :: Value -> Value -> Either Text Value
coerceValue schema val = case (schema, val) of
  -- Simple type coercions
  (Object schemaObj, String s) -> case KM.lookup (Key.fromText "type") schemaObj of
    Just (String "integer") -> coerceInteger s
    Just (String "number") -> coerceNumber s
    Just (String "boolean") -> coerceBoolean s
    _ -> Right val  -- Not a simple type or already correct

  -- Object: recurse into properties
  (Object schemaObj, Object valObj) -> case KM.lookup (Key.fromText "properties") schemaObj of
    Just (Object props) -> do
      coercedProps <- traverse coerceProp (KM.toList valObj)
      return $ Object (KM.fromList coercedProps)
      where
        coerceProp (key, v) = case KM.lookup key props of
          Just propSchema -> (,) key <$> coerceValue propSchema v
          Nothing -> Right (key, v)  -- No schema for this property, keep as-is
    _ -> Right val  -- No properties defined, keep as-is

  -- Array: recurse into items
  (Object schemaObj, Array valArr) -> case KM.lookup (Key.fromText "items") schemaObj of
    Just itemSchema -> Array <$> traverse (coerceValue itemSchema) valArr
    _ -> Right val  -- No items schema, keep as-is

  -- anyOf/oneOf (for optional fields): try to match one branch
  (Object schemaObj, _) -> case KM.lookup (Key.fromText "anyOf") schemaObj of
    Just (Array options) -> tryAnyOf (V.toList options) val
    Nothing -> case KM.lookup (Key.fromText "oneOf") schemaObj of
      Just (Array options) -> tryAnyOf (V.toList options) val
      Nothing -> Right val  -- No anyOf/oneOf, keep as-is

  -- Default: value is already the right type or we don't know how to coerce
  _ -> Right val

-- | Try coercion with each option in anyOf/oneOf, return first success
tryAnyOf :: [Value] -> Value -> Either Text Value
tryAnyOf [] val = Right val  -- No options worked, keep original
tryAnyOf (opt:opts) val = case coerceValue opt val of
  Right coerced -> Right coerced
  Left _ -> tryAnyOf opts val  -- Try next option

-- | Coerce string to integer
coerceInteger :: Text -> Either Text Value
coerceInteger s = case T.decimal s of
  Right (n, rest) | T.null rest -> Right (Number (fromIntegral (n :: Integer)))
  _ -> Left $ "Cannot coerce to integer: " <> s

-- | Coerce string to number (float)
coerceNumber :: Text -> Either Text Value
coerceNumber s = case T.rational s of
  Right (n, rest) | T.null rest -> Right (Number (realToFrac (n :: Double)))
  _ -> Left $ "Cannot coerce to number: " <> s

-- | Coerce string to boolean
coerceBoolean :: Text -> Either Text Value
coerceBoolean "true" = Right (Bool True)
coerceBoolean "false" = Right (Bool False)
coerceBoolean s = Left $ "Cannot coerce to boolean: " <> s
