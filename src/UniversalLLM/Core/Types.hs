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
{-# LANGUAGE TypeOperators #-}

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

-- Unified capability classes
-- SupportsX for parameters (things providers accept)
class SupportsTemperature a
class SupportsMaxTokens a
class SupportsSeed a
class SupportsSystemPrompt a

-- HasX for capabilities (features models/providers have)
class HasTools a
class HasVision a
class HasJSON a
class HasReasoning a

-- ModelConfig GADT - configuration values with provider and model constraints
-- Only constructible if the provider supports the parameter and/or model
data ModelConfig provider model where
  Temperature :: SupportsTemperature provider => Double -> ModelConfig provider model
  MaxTokens :: SupportsMaxTokens provider => Int -> ModelConfig provider model
  Seed :: SupportsSeed provider => Int -> ModelConfig provider model
  SystemPrompt :: SupportsSystemPrompt provider => Text -> ModelConfig provider model
  Tools :: (HasTools model, HasTools provider) => [ToolDefinition] -> ModelConfig provider model
  Reasoning :: (HasReasoning model, HasReasoning provider) => Bool -> ModelConfig provider model

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

-- | Execute a tool call by matching and dispatching
-- Takes available tools and the tool call from the LLM
-- Returns a ToolResult with either success value or error message
executeToolCall :: forall m. Monad m => [LLMTool m] -> ToolCall -> m ToolResult
executeToolCall _ invalid@(InvalidToolCall _ _ _ err) =
  return $ ToolResult invalid (Left err)

executeToolCall tools tc@(ToolCall _ name params) =
  case find matchesTool tools of
    Nothing ->
      return $ ToolResult tc (Left $ "Tool not found: " <> name)
    Just (LLMTool tool) ->
      case parse parseJSONViaCodec params of
        Error err ->
          return $ ToolResult tc (Left $ "Invalid parameters: " <> Text.pack err)
        Success typedParams -> do
          result <- call tool typedParams
          return $ ToolResult tc (Right $ toJSONViaCodec result)
  where
    matchesTool :: LLMTool m -> Bool
    matchesTool (LLMTool tool) = toolName @_ @m tool == name

-- | LLM operation errors
data LLMError
  = NetworkError Text          -- Server unreachable, timeouts
  | ParseError Text            -- JSON malformed, unexpected structure
  | AuthError Text             -- Invalid API key, permissions
  | QuotaError Text            -- Rate limits, credit exhausted
  | ModelError Text            -- Model-specific errors, content policy
  | ProviderError Int Text     -- HTTP status + raw response for debugging
  deriving (Show, Eq)

class Provider provider model where
  type ProviderRequest provider
  type ProviderResponse provider


-- Composable request building via message folding
-- A handler processes one message at a time, accumulating into the request
newtype MessageHandler provider model = MessageHandler
  { runHandler :: provider
               -> model
               -> [ModelConfig provider model]
               -> Message model provider
               -> ProviderRequest provider
               -> ProviderRequest provider
  }

-- Handlers compose via Semigroup/Monoid
instance Semigroup (MessageHandler provider model) where
  MessageHandler h1 <> MessageHandler h2 = MessageHandler $ \provider model configs msg req ->
    let req' = h1 provider model configs msg req
    in h2 provider model configs msg req'

instance Monoid (MessageHandler provider model) where
  mempty = MessageHandler $ \_ _ _ _ req -> req  -- Identity handler: pass through

-- A parser extracts messages from a response and can transform previously parsed messages
-- Takes: provider, model, configs, message history, accumulated messages, response -> produces final messages
-- Having full context allows intelligent interpretation (e.g., checking if JSON was requested)
type ResponseParser provider model =
  provider
  -> model
  -> [ModelConfig provider model]
  -> [Message model provider]  -- history (input messages)
  -> [Message provider model]  -- accumulator (parsed messages so far)
  -> ProviderResponse provider
  -> [Message provider model]

-- Composable bidirectional provider (couples toRequest and fromResponse)
data ComposableProvider provider model = ComposableProvider
  { cpToRequest :: MessageHandler provider model
  , cpFromResponse :: ResponseParser provider model
  }

-- ComposableProviders compose via Semigroup/Monoid
instance Semigroup (ComposableProvider provider model) where
  cp1 <> cp2 = ComposableProvider
    { cpToRequest = cpToRequest cp1 <> cpToRequest cp2
    , cpFromResponse = \provider model configs history acc resp ->
        let acc' = cpFromResponse cp1 provider model configs history acc resp
        in cpFromResponse cp2 provider model configs history acc' resp
    }

instance Monoid (ComposableProvider provider model) where
  mempty = ComposableProvider
    { cpToRequest = mempty
    , cpFromResponse = \_provider _model _configs _history acc _resp -> acc
    }

-- GADT Messages with capability constraints
data Message model provider where
  UserText :: Text -> Message model provider
  UserImage :: HasVision model => Text -> Text -> Message model provider
  UserRequestJSON :: (HasJSON model, HasJSON provider) => Text -> Value -> Message model provider  -- Text query + JSON schema
  AssistantText :: Text -> Message model provider
  AssistantReasoning :: Text -> Message model provider
  AssistantTool :: (HasTools model, HasTools provider) => ToolCall -> Message model provider
  AssistantJSON :: (HasJSON model, HasJSON provider) => Value -> Message model provider
  SystemText :: Text -> Message model provider
  ToolResultMsg :: (HasTools model, HasTools provider) => ToolResult -> Message model provider

instance Show (Message model provider) where
  show (UserText _) = "UserText"
  show (UserImage _ _) = "UserImage"
  show (UserRequestJSON _ _) = "UserRequestJSON"
  show (AssistantText _) = "AssistantText"
  show (AssistantReasoning _) = "AssistantReasoning"
  show (AssistantTool _) = "AssistantTool"
  show (AssistantJSON _) = "AssistantJSON"
  show (SystemText _) = "SystemText"
  show (ToolResultMsg _) = "ToolResultMsg"

-- Message direction (User vs Assistant) - useful for providers that need alternating roles
data MessageDirection = User | Assistant
  deriving (Eq, Show)

messageDirection :: Message model provider -> MessageDirection
messageDirection (UserText _) = User
messageDirection (UserImage _ _) = User
messageDirection (UserRequestJSON _ _) = User
messageDirection (ToolResultMsg _) = User  -- Tool results are user direction
messageDirection (SystemText _) = User  -- System messages are treated as user direction
messageDirection (AssistantText _) = Assistant
messageDirection (AssistantReasoning _) = Assistant
messageDirection (AssistantTool _) = Assistant
messageDirection (AssistantJSON _) = Assistant