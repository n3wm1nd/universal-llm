{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE OverloadedStrings #-}

module UniversalLLM.Core.Tools
  ( -- * Core type classes
    ToolParameter(..)
  , Callable(..)
  , Tool(..)
  , TupleSchema(..)
  , TupleParser(..)
  , DefaultParamMeta(..)
  , BuildParamMeta(..)

    -- * Tool wrapper with metadata
  , ToolWrapped(..)
  , mkTool
  , mkToolWithMeta
  , mkToolUnsafe
  , ParamMetaBuilder

    -- * Helper functions
  , toToolDefinition
  , executeToolCall
  , tupleToSchema
  , tupleToDefaultSchema
  , parseJsonToTuple
  , parseJsonToDefaultTuple
  , checkMetaLength
  , defaultParamMeta

    -- * Re-exports from Types for convenience
  , ToolDefinition(..)
  , ToolCall(..)
  , ToolResult(..)
  , getToolCallId
  , getToolCallName
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Proxy (Proxy(..))
import Autodocodec (HasCodec, toJSONViaCodec, parseJSONViaCodec)
import Autodocodec.Schema (jsonSchemaViaCodec)
import Data.Kind (Type)
import Data.Aeson (Value, Object)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Aeson.Types (Parser, parseEither)

-- Re-export from Core.Types
import UniversalLLM.Core.Types
  ( ToolDefinition(..)
  , ToolCall(..)
  , ToolResult(..)
  , getToolCallId
  , getToolCallName
  )

-- | Type class for tool parameters
-- Provides parameter naming and description for JSON schema generation
-- All tool parameters must have HasCodec (for serialization) and ToolParameter (for metadata)
class HasCodec a => ToolParameter a where
  -- | Get parameter name, given its position (0-indexed)
  -- Default implementation generates "param_N" names
  paramName :: Proxy a -> Int -> Text
  paramName _ n = "param_" <> T.pack (show n)

  -- | Get parameter description for schema
  -- Default implementation provides a generic description
  paramDescription :: Proxy a -> Text
  paramDescription _ = "a parameter"

-- Default instances for common types
instance ToolParameter Text where
  paramName _ n = "text_" <> T.pack (show n)
  paramDescription _ = "a text string"

instance ToolParameter Int where
  paramName _ n = "number_" <> T.pack (show n)
  paramDescription _ = "an integer number"

instance ToolParameter Integer where
  paramName _ n = "number_" <> T.pack (show n)
  paramDescription _ = "an integer number"

instance ToolParameter Bool where
  paramName _ n = "bool_" <> T.pack (show n)
  paramDescription _ = "a boolean value (true or false)"

instance ToolParameter a => ToolParameter [a] where
  paramName _ n = "list_" <> T.pack (show n)
  paramDescription p = "a list of " <> paramDescription (Proxy @a)

instance ToolParameter a => ToolParameter (Maybe a) where
  paramName _ n = "optional_" <> T.pack (show n)
  paramDescription p = "an optional " <> paramDescription (Proxy @a)

-- | Type class to extract parameter schemas from nested tuple types
-- Takes a list of (name, description) metadata and uses it for schema generation
class TupleSchema params where
  -- Takes metadata list and returns (name, description, schema) triples
  tupleToSchemaList :: Proxy params -> [(Text, Text)] -> [(Text, Text, Value)]

-- Base case: () means no more parameters
instance TupleSchema () where
  tupleToSchemaList _ _ = []

-- Recursive case: (a, rest) - use metadata from list
instance (ToolParameter a, TupleSchema rest) => TupleSchema (a, rest) where
  tupleToSchemaList _ metas =
    let restSchemas = tupleToSchemaList (Proxy @rest) (drop 1 metas)
        (name, desc) = case metas of
                        ((n, d):_) -> (n, d)
                        [] -> error "tupleToSchemaList: metadata list too short"
        schema = Aeson.toJSON $ jsonSchemaViaCodec @a
    in (name, desc, schema) : restSchemas

-- | Convert tuple schema to JSON object schema using provided metadata
tupleToSchema :: TupleSchema params => Proxy params -> [(Text, Text)] -> Value
tupleToSchema p metas =
  let params = tupleToSchemaList p metas
      properties = Aeson.object [ (Key.fromText name, schema) | (name, _desc, schema) <- params ]
      required = [ name | (name, _, _) <- params ]
  in Aeson.object
       [ "type" Aeson..= ("object" :: Text)
       , "properties" Aeson..= properties
       , "required" Aeson..= required
       ]

-- | Convert tuple schema to JSON object schema using default parameter names
tupleToDefaultSchema :: (TupleSchema params, DefaultParamMeta params) => Proxy params -> Value
tupleToDefaultSchema p = tupleToSchema p (defaultParamMeta p)

-- ============================================================================
-- Tuple Parsing (JSON -> Tuple)
-- ============================================================================

-- | Type class to parse JSON objects into nested tuple types
-- Uses metadata list to look up parameters by their custom names
class TupleParser params where
  -- | Parse a JSON object into the tuple parameter structure using metadata
  -- Returns Either error message or parsed tuple
  parseJsonToTuple :: Proxy params -> [(Text, Text)] -> Object -> Either Text params

-- | Parse using default parameter names (text_0, number_1, etc.)
parseJsonToDefaultTuple :: (TupleParser params, DefaultParamMeta params)
                        => Proxy params -> Object -> Either Text params
parseJsonToDefaultTuple p obj = parseJsonToTuple p (defaultParamMeta p) obj

-- Base case: () means no parameters
instance TupleParser () where
  parseJsonToTuple _ _ _ = Right ()

-- Recursive case: (a, rest) - parse 'a' using name from metadata
instance (ToolParameter a, TupleParser rest) => TupleParser (a, rest) where
  parseJsonToTuple _ metas obj = do
    -- Get parameter name from metadata
    let paramName' = case metas of
                      ((n, _):_) -> n
                      [] -> error "parseJsonToTuple: metadata list too short"
        paramKey = Key.fromText paramName'

    paramValue <- case KM.lookup paramKey obj of
      Nothing -> Left $ "Missing parameter: " <> paramName'
      Just val -> Right val

    -- Parse the value using HasCodec
    parsedParam <- case parseEither parseJSONViaCodec paramValue of
      Left err -> Left $ "Failed to parse parameter " <> paramName' <> ": " <> T.pack err
      Right val -> Right val

    -- Recursively parse the rest
    restParams <- parseJsonToTuple (Proxy @rest) (drop 1 metas) obj

    return (parsedParam, restParams)

-- | Core class for callable tools (without metadata requirements)
-- Tracks: effect monad r, parameter type params, result type result
-- The functional dependencies ensure type inference works correctly
class Callable tool r params result | tool -> r, tool -> params, tool -> result where
  call :: tool -> params -> r result

-- | Tool class extends Callable with metadata (name and description)
-- This is what gets exposed to the LLM
class Callable tool r params result => Tool tool r params result where
  toolName :: tool -> Text
  toolDescription :: tool -> Text

-- | Wrapper to attach name, description, and parameter metadata to any callable function
data ToolWrapped f params = ToolWrapped
  { toolWrapName :: Text
  , toolWrapDescription :: Text
  , toolWrapParamMetas :: [(Text, Text)]  -- (name, description) for each parameter
  , wrappedFunction :: f
  }

-- | Helper to generate default parameter metadata from ToolParameter instances
class DefaultParamMeta params where
  -- Build metadata list, threading position forward
  defaultParamMetaFrom :: Proxy params -> Int -> [(Text, Text)]

  -- Count the number of parameters
  paramCount :: Proxy params -> Int

-- | Get default metadata starting from position 0
defaultParamMeta :: DefaultParamMeta params => Proxy params -> [(Text, Text)]
defaultParamMeta p = defaultParamMetaFrom p 0

instance DefaultParamMeta () where
  defaultParamMetaFrom _ _ = []
  paramCount _ = 0

instance (ToolParameter a, DefaultParamMeta rest) => DefaultParamMeta (a, rest) where
  defaultParamMetaFrom _ pos =
    let name = paramName (Proxy @a) pos
        desc = paramDescription (Proxy @a)
        restMetas = defaultParamMetaFrom (Proxy @rest) (pos + 1)
    in (name, desc) : restMetas

  paramCount _ = 1 + paramCount (Proxy @rest)

-- | Check that metadata list length matches parameter count
checkMetaLength :: forall params. DefaultParamMeta params
                => Proxy params -> [(Text, Text)] -> Either Text ()
checkMetaLength p metas =
  let expected = paramCount p
      provided = length metas
  in if expected == provided
     then Right ()
     else Left $ "Parameter metadata length mismatch: expected "
              <> T.pack (show expected) <> " parameters, got "
              <> T.pack (show provided) <> " metadata entries"

-- | Simple constructor for tool with automatic parameter names
-- Uses default names (text_0, number_1, etc.) from ToolParameter instances
mkTool :: forall tool r params result.
          (Callable tool r params result, DefaultParamMeta params)
       => Text -> Text -> tool -> ToolWrapped tool params
mkTool name desc fn = ToolWrapped name desc (defaultParamMeta (Proxy @params)) fn

-- | Unsafe constructor that doesn't check metadata length
-- Use with caution - only when you're certain the length matches
mkToolUnsafe :: Text -> Text -> [(Text, Text)] -> tool -> ToolWrapped tool params
mkToolUnsafe = ToolWrapped

-- ============================================================================
-- Vary-adic parameter metadata builder
-- ============================================================================

-- | Type family for vary-adic metadata builder
-- Produces a function that takes (name, description) pairs for each parameter
type family ParamMetaBuilder params result where
  ParamMetaBuilder () result = result
  ParamMetaBuilder (a, rest) result = Text -> Text -> ParamMetaBuilder rest result

-- | Build parameter metadata vary-adically
class BuildParamMeta params where
  -- Accumulates (name, description) pairs and applies them to result function at the end
  buildParamMeta :: Proxy params -> Int -> [(Text, Text)] -> ([(Text, Text)] -> result) -> ParamMetaBuilder params result

instance BuildParamMeta () where
  buildParamMeta _ _ acc finalize = finalize (reverse acc)

instance BuildParamMeta rest => BuildParamMeta (a, rest) where
  buildParamMeta _ pos acc finalize = \name desc ->
    buildParamMeta (Proxy @rest) (pos + 1) ((name, desc) : acc) finalize

-- | Create tool with vary-adic parameter metadata
-- The vary-adic function ensures correct parameter count at compile time
-- Usage:
--   mkToolWithMeta "add" "adds numbers" add
--     "x" "first number"
--     "y" "second number"
mkToolWithMeta :: forall tool r params result.
                  (Callable tool r params result, BuildParamMeta params, DefaultParamMeta params)
               => Text  -- ^ Tool name
               -> Text  -- ^ Tool description
               -> tool  -- ^ The function
               -> ParamMetaBuilder params (ToolWrapped tool params)
mkToolWithMeta name desc fn =
  buildParamMeta (Proxy @params) 0 [] $ \(metas :: [(Text, Text)]) ->
    -- The vary-adic builder ensures length is correct by construction
    -- But we add a runtime check as a sanity check
    case checkMetaLength (Proxy @params) metas of
      Left err -> error $ "mkToolWithMeta: " <> T.unpack err
      Right () -> ToolWrapped name desc metas fn :: ToolWrapped tool params

-- | ToolWrapped instances: delegate call to wrapped function, provide metadata
instance Callable f r params result => Callable (ToolWrapped f params) r params result where
  call (ToolWrapped _ _ _ f) params = call f params

instance Callable f r params result => Tool (ToolWrapped f params) r params result where
  toolName (ToolWrapped name _ _ _) = name
  toolDescription (ToolWrapped _ desc _ _) = desc

-- ============================================================================
-- Instances for function-based tools
-- ============================================================================

-- | Recursive instance: (a -> rest) where rest is Callable
-- If rest takes params_rest and produces result, then (a -> rest) takes (a, params_rest) and produces result
-- This handles n-ary curried functions by encoding parameters as nested tuples
--
-- Note: This works for all function arities. The base case (0-arity monadic actions)
-- requires specific instances for each monad type to avoid fundep conflicts.
instance (ToolParameter a, Callable rest m params_rest result)
      => Callable (a -> rest) m (a, params_rest) result where
  call f (param, restParams) = call (f param) restParams

-- | Base case instance for IO monad
-- Allows 0-arity IO actions to be used as tools
instance ToolParameter a => Callable (IO a) IO () a where
  call action () = action

-- Note: Users can add Callable instances for their own monads following the same pattern.
-- The key is to use a concrete monad type (not a type variable) to avoid conflicts.
--
-- For Polysemy Sem:
--   instance ToolParameter a => Callable (Sem r a) (Sem r) () a where
--     call action () = action
--
-- For custom monads:
--   instance ToolParameter a => Callable (MyMonad a) MyMonad () a where
--     call action () = action

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- | Convert a Tool to a ToolDefinition (metadata for LLM)
-- For ToolWrapped, uses the stored parameter metadata
toToolDefinition :: forall tool params m result.
                    (Tool (ToolWrapped tool params) m params result, TupleSchema params)
                 => ToolWrapped tool params -> ToolDefinition
toToolDefinition tool = ToolDefinition
  { toolDefName = toolName tool
  , toolDefDescription = toolDescription tool
  , toolDefParameters = tupleToSchema (Proxy @params) (toolWrapParamMetas tool)
  }

-- | Execute a tool call by parsing JSON parameters and calling the tool
-- Uses the parameter metadata stored in ToolWrapped for parsing
executeToolCall :: forall tool params m result.
                   (Tool (ToolWrapped tool params) m params result, TupleParser params, HasCodec result, Monad m)
                => ToolWrapped tool params
                -> ToolCall
                -> m ToolResult
executeToolCall _ invalid@(InvalidToolCall _ _ _ err) =
  return $ ToolResult invalid (Left err)

executeToolCall tool tc@(ToolCall _ _ params) = do
  case params of
    Aeson.Object obj -> do
      case parseJsonToTuple (Proxy @params) (toolWrapParamMetas tool) obj of
        Left err -> return $ ToolResult tc (Left $ "Parameter parsing failed: " <> err)
        Right parsedParams -> do
          result <- call tool parsedParams
          return $ ToolResult tc (Right $ toJSONViaCodec result)
    _ -> return $ ToolResult tc (Left "Expected parameters to be a JSON object")

