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
import qualified Data.Text as Text
import Data.Aeson (Value, Result (..))
import Data.Aeson.Types (parse)
import qualified Data.Aeson as Aeson
import Autodocodec (HasCodec, toJSONViaCodec, parseJSONViaCodec)
import Autodocodec.Schema (jsonSchemaViaCodec)
import Data.Kind (Type)
import Data.List (find)

-- Provider capability markers - indicate what parameters a provider supports
class ProviderSupportsTemperature provider
class ProviderSupportsMaxTokens provider
class ProviderSupportsSeed provider
class ProviderSupportsSystemPrompt provider

-- ModelConfig GADT - configuration values with provider and model constraints
-- Only constructible if the provider supports the parameter and/or model
data ModelConfig provider model where
  Temperature :: ProviderSupportsTemperature provider => Double -> ModelConfig provider model
  MaxTokens :: ProviderSupportsMaxTokens provider => Int -> ModelConfig provider model
  Seed :: ProviderSupportsSeed provider => Int -> ModelConfig provider model
  SystemPrompt :: ProviderSupportsSystemPrompt provider => Text -> ModelConfig provider model
  Tools :: ModelHasTools model => [ToolDefinition] -> ModelConfig provider model

-- Core capabilities that models can have
class HasVision model
class HasJSON model

-- Model-level tool capability (just a marker, tools go in config)
class ModelHasTools model

-- Provider-specific model names
class ModelName provider model where
  modelName :: model -> Text

-- Tool use types
-- Provider-agnostic tool definition (just metadata, no execution)
data ToolDefinition = ToolDefinition
  { toolDefName :: Text
  , toolDefDescription :: Text
  , toolDefParameters :: Value  -- JSON schema for parameters
  } deriving (Show, Eq)

-- The Tool class connects a tool type (which can hold config) to its parameters type
class (HasCodec (ToolParams tool), HasCodec (ToolOutput tool)) => Tool tool m where
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

-- Type aliases for ToolCall parameters (for documentation)
type ToolCallId = Text
type ToolCallName = Text
type ToolCallArguments = Value
type RawArguments = Text
type ErrorMessage = Text

data ToolCall
  = ToolCall ToolCallId ToolCallName ToolCallArguments
  | InvalidToolCall ToolCallId ToolCallName RawArguments ErrorMessage
  deriving (Show, Eq)

-- | Get the ID from either variant of ToolCall
getToolCallId :: ToolCall -> Text
getToolCallId (ToolCall tcId _ _) = tcId
getToolCallId (InvalidToolCall tcId _ _ _) = tcId

-- | Get the name from either variant of ToolCall
getToolCallName :: ToolCall -> Text
getToolCallName (ToolCall _ tcName _) = tcName
getToolCallName (InvalidToolCall _ tcName _ _) = tcName

-- Tool execution result - either success or error
data ToolResult = ToolResult
  { toolResultCall :: ToolCall  -- The call we're responding to
  , toolResultOutput :: Either Text Value  -- Left = error message, Right = success value
  } deriving (Show, Eq)

-- | Match a ToolCall to a LLMTool from the available tools list
matchToolCall :: forall m. [LLMTool m] -> ToolCall -> Maybe (LLMTool m)
matchToolCall tools (ToolCall _ name _) = find matches tools
  where
    matches (LLMTool tool) = toolName @_ @m tool == name
matchToolCall _ (InvalidToolCall _ _ _ _) = Nothing

-- | Execute a tool call by matching and dispatching
-- Takes available tools and the tool call from the LLM
-- Returns a ToolResult with either success value or error message
executeToolCall :: Monad m
                => [LLMTool m]  -- available tools
                -> ToolCall      -- call from LLM
                -> m ToolResult
executeToolCall _ invalid@(InvalidToolCall _ _ _ err) =
  return $ ToolResult
    { toolResultCall = invalid
    , toolResultOutput = Left err
    }
executeToolCall tools toolCall@(ToolCall _ name _) =
  case matchToolCall tools toolCall of
    Nothing -> return $ ToolResult
      { toolResultCall = toolCall
      , toolResultOutput = Left $ "Tool not found: " <> name
      }
    Just (LLMTool tool) -> executeWithTool tool toolCall

-- | Execute tool with typed params - all inside existential scope
executeWithTool :: forall tool m. (Tool tool m, Monad m)
                => tool
                -> ToolCall
                -> m ToolResult
executeWithTool tool toolCall@(ToolCall _ _ params) =
  -- Decode JSON params to typed ToolParams
  case parse parseJSONViaCodec $ params of
    Error err -> return $ ToolResult toolCall
      (Left $ "Invalid parameters: " <> Text.pack err)
    Success (typedParams :: ToolParams tool) -> do
      -- Call tool's call method with typed params
      result <- call tool typedParams
      -- Encode result back to JSON
      return $ ToolResult toolCall (Right $ toJSONViaCodec result)
executeWithTool _ (InvalidToolCall _ _ _ _) =
  error "executeWithTool called with InvalidToolCall (should never happen)"

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
            -> [ModelConfig provider model]
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

-- Protocol configuration application
-- Protocols define how to apply ModelConfig values to their request format
class ApplyConfig protocol provider model where
  applyConfig :: [ModelConfig provider model] -> protocol -> protocol

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