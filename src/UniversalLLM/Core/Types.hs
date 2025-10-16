{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module UniversalLLM.Core.Types where

import Data.Text (Text)
import Data.Aeson (Value, encode)
import qualified Data.Aeson as Aeson
import Autodocodec (HasCodec, toJSONViaCodec, eitherDecodeJSONViaCodec)
import Autodocodec.Schema (jsonSchemaViaCodec)
import Data.Kind (Type)

-- Core capabilities that models can have
class HasVision model
class HasJSON model

-- Model-level tool capability
class ModelHasTools model where
  getToolDefinitions :: model -> [ToolDefinition]
  setToolDefinitions :: [ToolDefinition] -> model -> model

-- Provider-specific model names
class ModelName provider model where
  modelName :: Text

-- Parameter extraction classes
class Temperature model provider where
  getTemperature :: model -> Maybe Double

class MaxTokens model provider where
  getMaxTokens :: model -> Maybe Int

class SystemPrompt model provider where
  getSystemPrompt :: model -> Maybe Text

class Seed model provider where
  getSeed :: model -> Maybe Int

-- Tool use types
-- Provider-agnostic tool definition (just metadata, no execution)
data ToolDefinition = ToolDefinition
  { toolDefName :: Text
  , toolDefDescription :: Text
  , toolDefParameters :: Value  -- JSON schema for parameters
  } deriving (Show, Eq)

-- The Tool class connects a tool type (which can hold config) to its parameters type
class (HasCodec (ToolParams tool), HasCodec (ToolOutput tool), Eq tool) => Tool tool m where
  type ToolParams tool :: Type
  type ToolOutput tool :: Type
  toolName :: tool -> Text
  toolDescription :: tool -> Text
  call :: tool -> ToolParams tool -> m (ToolOutput tool)

-- Existential wrapper for heterogeneous tool lists
data LLMTool m where
  LLMTool :: Tool tool m => tool -> LLMTool m

-- Extract tool definition from any Tool instance
toToolDefinition :: forall tool m. Tool tool m => tool -> ToolDefinition
toToolDefinition tool = ToolDefinition
  { toolDefName = toolName @tool @m tool
  , toolDefDescription = toolDescription @tool @m tool
  , toolDefParameters = Aeson.toJSON $ jsonSchemaViaCodec @(ToolParams tool)
  }

-- Extract tool definition from existential wrapper
llmToolToDefinition :: forall m. LLMTool m -> ToolDefinition
llmToolToDefinition (LLMTool (tool :: t)) = toToolDefinition @t @m tool

data ToolCall = ToolCall
  { toolCallId :: Text
  , toolCallName :: Text
  , toolCallParameters :: Value
  } deriving (Show, Eq)

data ToolResult = ToolResult
  { toolResultCallId :: Text
  , toolResultOutput :: Value
  } deriving (Show, Eq)

-- | Match a ToolCall to a LLMTool from the available tools list
matchToolCall :: forall m. [LLMTool m] -> ToolCall -> Maybe (LLMTool m)
matchToolCall tools tc =
  let name = toolCallName tc
  in find (\(LLMTool tool) -> toolName @_ @m tool == name) tools
  where
    find :: (a -> Bool) -> [a] -> Maybe a
    find _ [] = Nothing
    find p (x:xs) = if p x then Just x else find p xs

-- | Execute a tool call by matching and dispatching
-- Takes available tools and the tool call from the LLM
executeToolCall :: Monad m
                => [LLMTool m]  -- available tools
                -> ToolCall      -- call from LLM
                -> m (Maybe Value)  -- result (Nothing if tool not found or params invalid)
executeToolCall tools toolCall =
  case matchToolCall tools toolCall of
    Nothing -> return Nothing
    Just (LLMTool tool) -> executeWithTool tool toolCall

-- | Execute tool with typed params - all inside existential scope
executeWithTool :: forall tool m. (Tool tool m, Monad m)
                => tool
                -> ToolCall
                -> m (Maybe Value)
executeWithTool tool toolCall = do
  -- Decode JSON params to typed ToolParams
  case eitherDecodeJSONViaCodec (encode $ toolCallParameters toolCall) of
    Left _err -> return Nothing
    Right (params :: ToolParams tool) -> do
      -- Call tool's call method with typed params
      result <- call tool params
      -- Encode result back to JSON
      return $ Just $ toJSONViaCodec result

-- | LLM operation errors
data LLMError
  = NetworkError Text          -- Server unreachable, timeouts
  | ParseError Text            -- JSON malformed, unexpected structure
  | AuthError Text             -- Invalid API key, permissions
  | QuotaError Text            -- Rate limits, credit exhausted
  | ModelError Text            -- Model-specific errors, content policy
  | ProviderError Int Text     -- HTTP status + raw response for debugging
  deriving (Show, Eq)

-- Provider typeclass with associated types
class Provider provider model where
  type ProviderRequest provider
  type ProviderResponse provider

  toRequest :: provider
            -> model
            -> [Message model provider]
            -> ProviderRequest provider

  fromResponse :: ProviderResponse provider
               -> Either LLMError [Message model provider]

-- Provider-level capability markers
class ProviderSupportsTools provider

-- Combined constraint: BOTH model AND provider must support tools
class (ModelHasTools model, ProviderSupportsTools provider) => HasTools model provider
instance (ModelHasTools model, ProviderSupportsTools provider) => HasTools model provider

-- Protocol-level embedding classes
-- These define how to embed capabilities into protocol wire formats
-- Can be reused by any provider using the same protocol

-- Generic tool embedding for any protocol
class ProtocolEmbedTools protocol model where
  embedTools :: model -> protocol -> protocol
  embedTools _ req = req  -- Default: no-op

-- Generic tool call handling for any protocol/provider combination
class ProtocolHandleTools protocolToolCall model provider where
  handleToolCalls :: [protocolToolCall] -> [Message model provider]
  handleToolCalls _ = []  -- Default: ignore (shouldn't happen for non-tool models)

-- GADT Messages with capability constraints
data Message model provider where
  UserText :: Text -> Message model provider
  UserImage :: HasVision model => Text -> Text -> Message model provider
  AssistantText :: Text -> Message model provider
  AssistantTool :: HasTools model provider => [ToolCall] -> Message model provider
  SystemText :: Text -> Message model provider
  ToolResultMsg :: HasTools model provider => ToolResult -> Message model provider