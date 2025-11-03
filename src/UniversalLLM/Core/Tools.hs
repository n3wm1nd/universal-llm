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

    -- * Tool wrapper with metadata
  , ToolWrapped(..)
  , mkTool

    -- * Helper functions
  , toToolDefinition
  , executeToolCall
  , tupleToSchema
  , parseJsonToTuple

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
-- Recursively builds list of parameter information
class TupleSchema params where
  -- Returns list of (name, description, schema) in reverse order (rightmost first)
  tupleToSchemaList :: Proxy params -> Int -> [(Text, Text, Value)]

-- Base case: () means no more parameters
instance TupleSchema () where
  tupleToSchemaList _ _ = []

-- Recursive case: (a, rest) - recurse first to get position, then add this param
instance (ToolParameter a, TupleSchema rest) => TupleSchema (a, rest) where
  tupleToSchemaList _ startPos =
    let restSchemas = tupleToSchemaList (Proxy @rest) (startPos + 1)
        position = startPos
        name = paramName (Proxy @a) position
        desc = paramDescription (Proxy @a)
        schema = Aeson.toJSON $ jsonSchemaViaCodec @a
    in (name, desc, schema) : restSchemas

-- | Convert tuple schema to JSON object schema
tupleToSchema :: TupleSchema params => Proxy params -> Value
tupleToSchema p =
  let params = reverse $ tupleToSchemaList p 0  -- reverse to get correct order
      properties = Aeson.object [ (Key.fromText name, schema) | (name, _desc, schema) <- params ]
      required = [ name | (name, _, _) <- params ]
  in Aeson.object
       [ "type" Aeson..= ("object" :: Text)
       , "properties" Aeson..= properties
       , "required" Aeson..= required
       ]

-- ============================================================================
-- Tuple Parsing (JSON -> Tuple)
-- ============================================================================

-- | Type class to parse JSON objects into nested tuple types
-- Looks up parameters by name in the JSON object
class TupleParser params where
  -- | Parse a JSON object into the tuple parameter structure
  -- Returns Either error message or parsed tuple
  parseJsonToTuple :: Proxy params -> Int -> Object -> Either Text params

-- Base case: () means no parameters
instance TupleParser () where
  parseJsonToTuple _ _ _ = Right ()

-- Recursive case: (a, rest) - parse 'a' from the object and recurse on rest
instance (ToolParameter a, TupleParser rest) => TupleParser (a, rest) where
  parseJsonToTuple _ startPos obj = do
    -- Parse the first parameter
    let position = startPos
        paramKey = Key.fromText $ paramName (Proxy @a) position

    paramValue <- case KM.lookup paramKey obj of
      Nothing -> Left $ "Missing parameter: " <> paramName (Proxy @a) position
      Just val -> Right val

    -- Parse the value using HasCodec
    parsedParam <- case parseEither parseJSONViaCodec paramValue of
      Left err -> Left $ "Failed to parse parameter " <> paramName (Proxy @a) position <> ": " <> T.pack err
      Right val -> Right val

    -- Recursively parse the rest
    restParams <- parseJsonToTuple (Proxy @rest) (startPos + 1) obj

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

-- | Wrapper to attach name and description to any callable function
-- This allows you to provide metadata for functions without needing special return types
data ToolWrapped f = ToolWrapped
  { toolWrapName :: Text
  , toolWrapDescription :: Text
  , wrappedFunction :: f
  }

-- | Smart constructor for ToolWrapped
mkTool :: Text -> Text -> f -> ToolWrapped f
mkTool = ToolWrapped

-- | ToolWrapped instances: delegate call to wrapped function, provide metadata
instance Callable f r params result => Callable (ToolWrapped f) r params result where
  call (ToolWrapped _ _ f) params = call f params

instance Callable f r params result => Tool (ToolWrapped f) r params result where
  toolName (ToolWrapped name _ _) = name
  toolDescription (ToolWrapped _ desc _) = desc

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
-- Generates JSON schema from the tuple parameter structure
toToolDefinition :: forall tool m params result.
                    (Tool tool m params result, TupleSchema params)
                 => tool -> ToolDefinition
toToolDefinition tool = ToolDefinition
  { toolDefName = toolName tool
  , toolDefDescription = toolDescription tool
  , toolDefParameters = tupleToSchema (Proxy @params)
  }

-- | Execute a tool call by parsing JSON parameters and calling the tool
executeToolCall :: forall tool m params result.
                   (Tool tool m params result, TupleParser params, HasCodec result, Monad m)
                => tool
                -> ToolCall
                -> m ToolResult
executeToolCall _ invalid@(InvalidToolCall _ _ _ err) =
  return $ ToolResult invalid (Left err)

executeToolCall tool tc@(ToolCall _ _ params) = do
  case params of
    Aeson.Object obj -> do
      case parseJsonToTuple (Proxy @params) 0 obj of
        Left err -> return $ ToolResult tc (Left $ "Parameter parsing failed: " <> err)
        Right parsedParams -> do
          result <- call tool parsedParams
          return $ ToolResult tc (Right $ toJSONViaCodec result)
    _ -> return $ ToolResult tc (Left "Expected parameters to be a JSON object")

