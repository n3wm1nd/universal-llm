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
import Data.Aeson (Value)
import Control.Applicative ((<|>))


-- Unified capability classes
-- SupportsX for parameters (things providers accept)
class SupportsTemperature a
class SupportsMaxTokens a
class SupportsSeed a
class SupportsSystemPrompt a
class SupportsStop a
class SupportsStreaming a

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
  Stop :: SupportsStop provider => [Text] -> ModelConfig provider model
  Tools :: HasTools model provider => [ToolDefinition] -> ModelConfig provider model
  Reasoning :: HasReasoning model provider => Bool -> ModelConfig provider model
  Streaming :: SupportsStreaming provider => Bool -> ModelConfig provider model

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

-- CompletionProvider: Type families for completion requests/responses
-- Separate from chat to allow different wire formats if needed
class Provider provider model => CompletionProvider provider model where
  type CompletionRequest provider
  type CompletionResponse provider


-- ============================================================================
-- Handler Types
-- ============================================================================

-- Handler for encoding a single message into a request patch
type MessageEncoder provider model =
  Message model provider -> ProviderRequest provider -> ProviderRequest provider

-- Handler for modifying request based on configs
type ConfigEncoder provider model =
  ProviderRequest provider -> ProviderRequest provider

-- Handler for unfolding response into individual provider messages
-- Returns one message at a time, plus remaining response
type ResponseUnfolder provider model =
  ProviderResponse provider -> Maybe (Message model provider, ProviderResponse provider)

-- Collection of handlers for a specific provider/model/config combination
data ComposableProviderHandlers provider model = ComposableProviderHandlers
  { -- Request building phase
    cpPureMessageRequest :: [Message model provider] -> [Message model provider]
  , cpToRequest :: MessageEncoder provider model
  , cpConfigHandler :: ConfigEncoder provider model

  -- Response parsing phase
  , cpFromResponse :: ResponseUnfolder provider model
  , cpUpdateState :: ProviderResponse provider -> (provider, model) -> (provider, model)
  , cpPureMessageResponse :: [Message model provider] -> [Message model provider]

  -- Serialization
  , cpSerializeMessage :: Message model provider -> Maybe Value
  , cpDeserializeMessage :: Value -> Maybe (Message model provider)
  }

-- ComposableProvider is now a function that builds handlers given context
type ComposableProvider provider model =
  provider -> model -> [ModelConfig provider model] -> ComposableProviderHandlers provider model

-- No-op handler collection (all functions are identity or return Nothing)
noopHandler :: ComposableProviderHandlers provider model
noopHandler = ComposableProviderHandlers
  { cpPureMessageRequest = id
  , cpToRequest = \_ req -> req
  , cpConfigHandler = id
  , cpFromResponse = \_ -> Nothing
  , cpUpdateState = \_ (p, m) -> (p, m)
  , cpPureMessageResponse = id
  , cpSerializeMessage = \_ -> Nothing
  , cpDeserializeMessage = \_ -> Nothing
  }

-- Chain two handler collections together
chainProvidersAt :: ComposableProviderHandlers provider model -> ComposableProviderHandlers provider model -> ComposableProviderHandlers provider model
chainProvidersAt h1 h2 = ComposableProviderHandlers
  { cpPureMessageRequest = cpPureMessageRequest h2 . cpPureMessageRequest h1
  , cpToRequest = \msg req -> cpToRequest h2 msg (cpToRequest h1 msg req)
  , cpConfigHandler = cpConfigHandler h2 . cpConfigHandler h1
  , cpFromResponse = \resp -> case cpFromResponse h1 resp of
      Just (msg, resp') -> Just (msg, resp')
      Nothing -> cpFromResponse h2 resp
  , cpUpdateState = \resp (p', m') -> cpUpdateState h2 resp (cpUpdateState h1 resp (p', m'))
  , cpPureMessageResponse = cpPureMessageResponse h2 . cpPureMessageResponse h1
  , cpSerializeMessage = \msg -> cpSerializeMessage h1 msg <|> cpSerializeMessage h2 msg
  , cpDeserializeMessage = \val -> cpDeserializeMessage h1 val <|> cpDeserializeMessage h2 val
  }

-- Chain two providers together, applying cp1 first then cp2
infixl 6 `chainProviders`
chainProviders :: ComposableProvider provider model
               -> ComposableProvider provider model
               -> ComposableProvider provider model
chainProviders cp1 cp2 p m configs =
  (cp1 p m configs) `chainProvidersAt` (cp2 p m configs)

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
      handlers = composableProvider provider model configs
      -- Apply pure message transformations first
      pureMessages = cpPureMessageRequest handlers msgs
      -- Fold over messages, encoding each one
      fold acc msg = cpToRequest handlers msg acc
      reqAfterMessages = foldl fold mempty pureMessages
      -- Apply config handler
  in cpConfigHandler handlers reqAfterMessages

-- Parse a provider response using the model's provider implementation
fromProviderResponse :: ProviderImplementation provider model
                     => provider
                     -> model
                     -> [ModelConfig provider model]
                     -> [Message model provider]  -- history
                     -> ProviderResponse provider
                     -> (provider, model, [Message model provider])
fromProviderResponse provider model configs _history resp =
  let composableProvider = getComposableProvider
      handlers = composableProvider provider model configs
      -- Unfold response into messages
      messages = unfoldMessages (cpFromResponse handlers) resp
      -- Apply pure message transformations
      pureMessages = cpPureMessageResponse handlers messages
      -- Update state
      (provider', model') = cpUpdateState handlers resp (provider, model)
  in (provider', model', pureMessages)
  where
    unfoldMessages unfolder response =
      case unfolder response of
        Just (msg, resp') -> msg : unfoldMessages unfolder resp'
        Nothing -> []

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

-- ============================================================================
-- Completion Interface (Simpler than Chat - Just Prompt In, Text Out)
-- ============================================================================

-- Completion-specific handler types (no messages, just prompts)
-- Note: CompletionProvider constraint needed for access to CompletionRequest/CompletionResponse
type PromptHandler provider model =
  provider
  -> model
  -> [ModelConfig provider model]
  -> Text  -- The prompt
  -> CompletionRequest provider
  -> CompletionRequest provider

-- Config handler for completions (reuse pattern, but different request type)
type CompletionConfigHandler provider model =
  provider
  -> model
  -> [ModelConfig provider model]
  -> CompletionRequest provider
  -> CompletionRequest provider

-- Completion parser extracts text from response
type CompletionParser provider model =
  provider
  -> model
  -> [ModelConfig provider model]
  -> Text  -- Original prompt (for context)
  -> CompletionResponse provider
  -> Text  -- Completed text

-- Composable completion provider (simpler than chat)
data ComposableCompletionProvider provider model = ComposableCompletionProvider
  { ccpToRequest :: PromptHandler provider model
  , ccpConfigHandler :: CompletionConfigHandler provider model
  , ccpFromResponse :: CompletionParser provider model
  }

-- Provider implementation for completions
class Provider provider model => CompletionProviderImplementation provider model where
  getComposableCompletionProvider :: ComposableCompletionProvider provider model

-- Chain two completion providers together
infixl 6 `chainCompletionProviders`
chainCompletionProviders :: ComposableCompletionProvider provider model
                         -> ComposableCompletionProvider provider model
                         -> ComposableCompletionProvider provider model
chainCompletionProviders cp1 cp2 = ComposableCompletionProvider
  { ccpToRequest = \provider model configs prompt req ->
      let req' = ccpToRequest cp1 provider model configs prompt req
      in ccpToRequest cp2 provider model configs prompt req'
  , ccpConfigHandler = \provider model configs req ->
      let req' = ccpConfigHandler cp1 provider model configs req
      in ccpConfigHandler cp2 provider model configs req'
  , ccpFromResponse = \provider model configs prompt resp ->
      ccpFromResponse cp2 provider model configs prompt resp  -- Note: only cp2 sees the response (cp1 ignored)
  }

-- Build a completion request from a prompt
toCompletionRequest :: (CompletionProviderImplementation provider model, Monoid (CompletionRequest provider))
                    => provider
                    -> model
                    -> [ModelConfig provider model]
                    -> Text  -- The prompt
                    -> CompletionRequest provider
toCompletionRequest provider model configs prompt =
  let composableProvider = getComposableCompletionProvider
      promptHandler = ccpToRequest composableProvider
      configHandler = ccpConfigHandler composableProvider
      -- Apply prompt handler
      reqAfterPrompt = promptHandler provider model configs prompt mempty
      -- Apply config handlers
  in configHandler provider model configs reqAfterPrompt

-- Parse a completion response
fromCompletionResponse :: CompletionProviderImplementation provider model
                       => provider
                       -> model
                       -> [ModelConfig provider model]
                       -> Text  -- Original prompt
                       -> CompletionResponse provider
                       -> Text  -- Completed text
fromCompletionResponse provider model configs prompt resp =
  let composableProvider = getComposableCompletionProvider
      parser = ccpFromResponse composableProvider
  in parser provider model configs prompt resp