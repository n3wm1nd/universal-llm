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

-- HasX for capabilities (features that require both model AND provider support)
-- These take two parameters: model and provider
-- Instances must be declared explicitly for each model/provider combination
-- Each instance must provide a combinator to augment a base provider with the capability
class Provider provider model => HasTools model provider where
  withTools :: ComposableProvider provider model -> ComposableProvider provider model

class Provider provider model => HasVision model provider where
  withVision :: ComposableProvider provider model -> ComposableProvider provider model

class Provider provider model => HasJSON model provider where
  withJSON :: ComposableProvider provider model -> ComposableProvider provider model

class Provider provider model => HasReasoning model provider where
  withReasoning :: ComposableProvider provider model -> ComposableProvider provider model

-- ModelConfig GADT - configuration values with provider and model constraints
-- Only constructible if the provider supports the parameter and/or model
data ModelConfig provider model where
  Temperature :: SupportsTemperature provider => Double -> ModelConfig provider model
  MaxTokens :: SupportsMaxTokens provider => Int -> ModelConfig provider model
  Seed :: SupportsSeed provider => Int -> ModelConfig provider model
  SystemPrompt :: SupportsSystemPrompt provider => Text -> ModelConfig provider model
  Tools :: HasTools model provider => [ToolDefinition] -> ModelConfig provider model
  Reasoning :: HasReasoning model provider => Bool -> ModelConfig provider model

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

-- ProviderImplementation: Encapsulates the complete provider composition for a model
-- Each model declares how to build requests and parse responses for each provider it supports
class Provider provider model => ProviderImplementation provider model where
  getComposableProvider :: ComposableProvider provider model


-- ============================================================================
-- Composition Operator for Parameter Threading
-- ============================================================================

-- Threads common parameters through function composition
-- The first 4 parameters (p, m, cfg, msg) are automatically passed to both functions
infixl 1 >>>
(>>>) :: (p -> m -> cfg -> msg -> a -> b)
      -> (p -> m -> cfg -> msg -> b -> c)
      -> (p -> m -> cfg -> msg -> a -> c)
f >>> g = \p m cfg msg a -> g p m cfg msg (f p m cfg msg a)

-- ============================================================================
-- Handler Types (now just type aliases, no newtypes)
-- ============================================================================

-- A handler processes one message at a time, transforming the request
type MessageHandler provider model =
  provider
  -> model
  -> [ModelConfig provider model]
  -> Message model provider
  -> ProviderRequest provider
  -> ProviderRequest provider

-- A parser extracts messages from a response and can transform previously parsed messages
-- Takes: provider, model, configs, message history, accumulated messages, response -> produces final messages
-- Having full context allows intelligent interpretation (e.g., checking if JSON was requested)
type ResponseParser provider model =
  provider
  -> model
  -> [ModelConfig provider model]
  -> [Message model provider]  -- history (input messages)
  -> [Message model provider]  -- accumulator (parsed messages so far)
  -> ProviderResponse provider
  -> [Message model provider]

-- Config handler: processes configs after messages (e.g., for system prompts, tools)
type ConfigHandler provider model =
  provider
  -> model
  -> [ModelConfig provider model]
  -> ProviderRequest provider
  -> ProviderRequest provider

-- Composable bidirectional provider (couples toRequest and fromResponse)
data ComposableProvider provider model = ComposableProvider
  { cpToRequest :: MessageHandler provider model
  , cpConfigHandler :: ConfigHandler provider model  -- Applied after message processing
  , cpFromResponse :: ResponseParser provider model
  }

-- Chain two providers together, applying cp1 first then cp2
-- This replaces the previous Semigroup (<>) instance with explicit naming
infixl 6 `chainProviders`
chainProviders :: ComposableProvider provider model
               -> ComposableProvider provider model
               -> ComposableProvider provider model
chainProviders cp1 cp2 = ComposableProvider
  { cpToRequest = \provider model configs msg req ->
      let req' = cpToRequest cp1 provider model configs msg req
      in cpToRequest cp2 provider model configs msg req'
  , cpConfigHandler = \provider model configs req ->
      let req' = cpConfigHandler cp1 provider model configs req
      in cpConfigHandler cp2 provider model configs req'
  , cpFromResponse = \provider model configs history acc resp ->
      let acc' = cpFromResponse cp1 provider model configs history acc resp
      in cpFromResponse cp2 provider model configs history acc' resp
  }

-- Helper functions for building requests and parsing responses

-- Build a provider request from messages using the model's provider implementation
-- The base provider is responsible for creating the initial empty request structure
toProviderRequest :: (ProviderImplementation provider model, Monoid (ProviderRequest provider))
                  => provider
                  -> model
                  -> [ModelConfig provider model]
                  -> [Message model provider]
                  -> ProviderRequest provider
toProviderRequest provider model configs msgs =
  let composableProvider = getComposableProvider
      messageHandler = cpToRequest composableProvider
      configHandler = cpConfigHandler composableProvider
      -- Start with mempty and fold over messages
      fold acc msg = messageHandler provider model configs msg acc
      reqAfterMessages = foldl fold mempty msgs
      -- Second: apply config handlers
  in configHandler provider model configs reqAfterMessages

-- Parse a provider response using the model's provider implementation
fromProviderResponse :: ProviderImplementation provider model
                     => provider
                     -> model
                     -> [ModelConfig provider model]
                     -> [Message model provider]  -- history
                     -> ProviderResponse provider
                     -> [Message model provider]
fromProviderResponse provider model configs history resp =
  let composableProvider = getComposableProvider
      parser = cpFromResponse composableProvider
  in parser provider model configs history [] resp

-- GADT Messages with capability constraints
data Message model provider where
  UserText :: Text -> Message model provider
  UserImage :: HasVision model provider => Text -> Text -> Message model provider
  UserRequestJSON :: HasJSON model provider => Text -> Value -> Message model provider  -- Text query + JSON schema
  AssistantText :: Text -> Message model provider
  AssistantReasoning :: HasReasoning model provider => Text -> Message model provider
  AssistantTool :: HasTools model provider => ToolCall -> Message model provider
  AssistantJSON :: HasJSON model provider => Value -> Message model provider
  SystemText :: Text -> Message model provider
  ToolResultMsg :: HasTools model provider => ToolResult -> Message model provider

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