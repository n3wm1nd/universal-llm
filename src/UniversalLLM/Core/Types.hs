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
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ConstraintKinds #-}

module UniversalLLM.Core.Types where

import Data.Text (Text)
import Data.Aeson (Value)
import Control.Applicative ((<|>))

-- ============================================================================
-- Core Model Type
-- ============================================================================

-- | A unified type that bundles an AI model with its provider
-- Replaces separate (provider, model) parameters with a single type
data Model aiModel provider = Model aiModel provider

-- | Type operator for nice syntax: GPT4 `Via` OpenRouter
type a `Via` b = Model a b

-- | Extract the AI model type from a Model
type family AIModelOf m where
  AIModelOf (Model a p) = a

-- | Extract the provider type from a Model
type family ProviderOf m where
  ProviderOf (Model a p) = p

-- ============================================================================
-- Provider Classes
-- ============================================================================

class BaseComposableProvider m where
  type BaseState m
  type BaseState m = ()
  baseProvider :: ComposableProvider m (BaseState m)

-- Unified capability classes
-- SupportsX for parameters (things providers accept)
class SupportsTemperature a
class SupportsMaxTokens a
class SupportsSeed a
class SupportsSystemPrompt a
class SupportsStop a
class SupportsStreaming a

-- HasX for capabilities (features that require both model AND provider support)
-- These take a single unified model parameter
-- Instances must be declared explicitly for each model/provider combination
-- Each instance must provide a combinator to augment a base provider with the capability
class HasTools m where
  type ToolState m
  type ToolState m = ()
  withTools :: ComposableProvider m (ToolState m)

class HasVision m where
  type VisionState m
  type VisionState m = ()
  withVision :: ComposableProvider m (VisionState m)

class HasJSON m where
  type JSONState m
  type JSONState m = ()
  withJSON :: ComposableProvider m (JSONState m)

class HasReasoning m where
  type ReasoningState m
  type ReasoningState m = ()
  withReasoning :: ComposableProvider m (ReasoningState m)

-- ModelConfig GADT - configuration values with model constraints
-- Only constructible if the provider supports the parameter and/or model
data ModelConfig m where
  Temperature :: SupportsTemperature (ProviderOf m) => Double -> ModelConfig m
  MaxTokens :: SupportsMaxTokens (ProviderOf m) => Int -> ModelConfig m
  Seed :: SupportsSeed (ProviderOf m) => Int -> ModelConfig m
  SystemPrompt :: SupportsSystemPrompt (ProviderOf m) => Text -> ModelConfig m
  Stop :: SupportsStop (ProviderOf m) => [Text] -> ModelConfig m
  Tools :: HasTools m => [ToolDefinition] -> ModelConfig m
  Reasoning :: HasReasoning m => Bool -> ModelConfig m
  Streaming :: SupportsStreaming (ProviderOf m) => Bool -> ModelConfig m

-- Provider-specific model names
class ModelName m where
  modelName :: m -> Text


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

class Provider m where
  type ProviderRequest m
  type ProviderResponse m


-- CompletionProvider: Type families for completion requests/responses
-- Separate from chat to allow different wire formats if needed
class Provider m => CompletionProvider m where
  type CompletionRequest m
  type CompletionResponse m


-- ============================================================================
-- Handler Types
-- ============================================================================

-- Handler for encoding a single message into a request patch
type MessageEncoder m =
  Message m -> ProviderRequest m -> ProviderRequest m

-- Handler for modifying request based on configs
type ConfigEncoder m =
  ProviderRequest m -> ProviderRequest m

-- Handler for unfolding response into individual provider messages
-- Returns one message at a time, plus remaining response
-- Left err: API error or parse failure
-- Right Nothing: Successfully processed, no more messages to extract
-- Right (Just (msg, rest)): Successfully extracted message, continue with rest
type ResponseUnfolder m =
  ProviderResponse m -> Either LLMError (Maybe (Message m, ProviderResponse m))

-- Collection of handlers for a specific model/provider/config combination
data ComposableProviderHandlers m state = ComposableProviderHandlers
  { -- Request building phase
    cpPureMessageRequest :: [Message m] -> [Message m]
  , cpToRequest :: MessageEncoder m
  , cpConfigHandler :: ConfigEncoder m
  , cpPreRequest :: ProviderRequest m -> state -> state

  -- Response parsing phase
  , cpPostResponse :: ProviderResponse m -> state -> state
  , cpFromResponse :: ResponseUnfolder m
  , cpPureMessageResponse :: [Message m] -> [Message m]

  -- Serialization
  , cpSerializeMessage :: Message m -> Maybe Value
  , cpDeserializeMessage :: Value -> Maybe (Message m)
  }

-- ComposableProvider is now a function that builds handlers given context
type ComposableProvider m state =
  m -> [ModelConfig m] -> state -> ComposableProviderHandlers m state

-- No-op handler collection (all functions are identity or return Nothing)
noopHandler :: ComposableProviderHandlers m state
noopHandler = ComposableProviderHandlers
  { cpPureMessageRequest = id
  , cpToRequest = \_ req -> req
  , cpConfigHandler = id
  , cpPreRequest = \_ -> id
  , cpPostResponse = \_ -> id
  , cpFromResponse = \_ -> Right Nothing
  , cpPureMessageResponse = id
  , cpSerializeMessage = \_ -> Nothing
  , cpDeserializeMessage = \_ -> Nothing
  }

-- Chain two handler collections together
chainProvidersAt :: ComposableProviderHandlers m s -> ComposableProviderHandlers m s' -> ComposableProviderHandlers m (s,s')
chainProvidersAt h1 h2 = ComposableProviderHandlers
  { cpPureMessageRequest = cpPureMessageRequest h1 . cpPureMessageRequest h2
  , cpToRequest = \msg req -> cpToRequest h2 msg (cpToRequest h1 msg req)
  , cpConfigHandler = cpConfigHandler h1 . cpConfigHandler h2
  , cpPreRequest = \req (s, s') -> (cpPreRequest h1 req s, cpPreRequest h2 req s')
  , cpPostResponse = \resp (s, s') -> (cpPostResponse h1 resp s, cpPostResponse h2 resp s')
  , cpFromResponse = \resp -> case cpFromResponse h2 resp of
      Right (Just (msg, resp')) -> Right (Just (msg, resp'))
      Right Nothing -> cpFromResponse h1 resp
      Left err -> Left err  -- Propagate errors immediately
  , cpPureMessageResponse = cpPureMessageResponse h1 . cpPureMessageResponse h2
  , cpSerializeMessage = \msg -> cpSerializeMessage h1 msg <|> cpSerializeMessage h2 msg
  , cpDeserializeMessage = \val -> cpDeserializeMessage h1 val <|> cpDeserializeMessage h2 val
  }

-- Chain two providers together, applying cp1 first then cp2
infixr 6 `chainProviders`
chainProviders :: ComposableProvider m s
               -> ComposableProvider m s'
               -> ComposableProvider m (s,s')
chainProviders cp1 cp2 m configs (s,s') =
  (cp1 m configs s) `chainProvidersAt` (cp2 m configs s')

-- Helper functions for building requests and parsing responses

-- Build a provider request from messages using the model's provider implementation
-- The base provider is responsible for creating the initial empty request structure
toProviderRequest :: (Monoid (ProviderRequest m))
                  => ComposableProvider m providerstackstate
                  -> m
                  -> [ModelConfig m]
                  -> providerstackstate
                  -> [Message m]
                  -> (providerstackstate, ProviderRequest m)
toProviderRequest composableProvider model configs s msgs =
  let
      handlers = composableProvider model configs s
      -- Apply pure message transformations first
      pureMessages = cpPureMessageRequest handlers msgs
      -- Fold over messages, encoding each one
      fold acc msg = cpToRequest handlers msg acc
      reqAfterMessages = foldl fold mempty pureMessages
      s' = cpPreRequest handlers reqAfterMessages s
      -- Apply config handler
  in (s', cpConfigHandler handlers reqAfterMessages)

-- Parse a provider response using the model's provider implementation
fromProviderResponse :: ()
                     => ComposableProvider m providerstackstate
                     -> m
                     -> [ModelConfig m]
                     -> providerstackstate
                     -> ProviderResponse m
                     -> Either LLMError (providerstackstate, [Message m])
fromProviderResponse composableProvider model configs s resp =
  let handlers = composableProvider model configs s
  in case unfoldMessages (cpFromResponse handlers) resp of
      Left err -> Left err
      Right messages ->
        let -- Apply pure message transformations
            pureMessages = cpPureMessageResponse handlers messages
            -- Update state
            s' = cpPostResponse handlers resp s
        in Right (s', pureMessages)
  where
    unfoldMessages :: ResponseUnfolder m -> ProviderResponse m -> Either LLMError [Message m]
    unfoldMessages unfolder response =
      case unfolder response of
        Left err -> Left err
        Right Nothing -> Right []
        Right (Just (msg, resp')) -> do
          rest <- unfoldMessages unfolder resp'
          return (msg : rest)

-- GADT Messages with capability constraints
data Message m where
  UserText :: Text -> Message m
  UserImage :: HasVision m => Text -> Text -> Message m
  UserRequestJSON :: HasJSON m => Text -> Value -> Message m  -- Text query + JSON schema
  AssistantText :: Text -> Message m
  AssistantReasoning :: HasReasoning m => Text -> Message m
  AssistantTool :: HasTools m => ToolCall -> Message m
  AssistantJSON :: HasJSON m => Value -> Message m
  SystemText :: Text -> Message m
  ToolResultMsg :: HasTools m => ToolResult -> Message m

instance Show (Message m) where
  show (UserText _) = "UserText"
  show (UserImage _ _) = "UserImage"
  show (UserRequestJSON _ _) = "UserRequestJSON"
  show (AssistantText _) = "AssistantText"
  show (AssistantReasoning _) = "AssistantReasoning"
  show (AssistantTool _) = "AssistantTool"
  show (AssistantJSON _) = "AssistantJSON"
  show (SystemText _) = "SystemText"
  show (ToolResultMsg _) = "ToolResultMsg"

instance Eq (Message m) where
  UserText t1 == UserText t2 = t1 == t2
  UserImage url1 desc1 == UserImage url2 desc2 = url1 == url2 && desc1 == desc2
  UserRequestJSON q1 s1 == UserRequestJSON q2 s2 = q1 == q2 && s1 == s2
  AssistantText t1 == AssistantText t2 = t1 == t2
  AssistantReasoning t1 == AssistantReasoning t2 = t1 == t2
  AssistantTool tc1 == AssistantTool tc2 = tc1 == tc2
  AssistantJSON v1 == AssistantJSON v2 = v1 == v2
  SystemText t1 == SystemText t2 = t1 == t2
  ToolResultMsg tr1 == ToolResultMsg tr2 = tr1 == tr2
  _ == _ = False

-- Message direction (User vs Assistant) - useful for providers that need alternating roles
data MessageDirection = User | Assistant
  deriving (Eq, Show)

messageDirection :: Message m -> MessageDirection
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
type PromptHandler m =
  m
  -> [ModelConfig m]
  -> Text  -- The prompt
  -> CompletionRequest m
  -> CompletionRequest m

-- Config handler for completions (reuse pattern, but different request type)
type CompletionConfigHandler m =
  m
  -> [ModelConfig m]
  -> CompletionRequest m
  -> CompletionRequest m

-- Completion parser extracts text from response
type CompletionParser m =
  m
  -> [ModelConfig m]
  -> Text  -- Original prompt (for context)
  -> CompletionResponse m
  -> Text  -- Completed text

-- Composable completion provider (simpler than chat)
data ComposableCompletionProvider m = ComposableCompletionProvider
  { ccpToRequest :: PromptHandler m
  , ccpConfigHandler :: CompletionConfigHandler m
  , ccpFromResponse :: CompletionParser m
  }

-- Provider implementation for completions
class Provider m => CompletionProviderImplementation m where
  getComposableCompletionProvider :: ComposableCompletionProvider m

-- Chain two completion providers together
infixl 6 `chainCompletionProviders`
chainCompletionProviders :: ComposableCompletionProvider m
                         -> ComposableCompletionProvider m
                         -> ComposableCompletionProvider m
chainCompletionProviders cp1 cp2 = ComposableCompletionProvider
  { ccpToRequest = \model configs prompt req ->
      let req' = ccpToRequest cp1 model configs prompt req
      in ccpToRequest cp2 model configs prompt req'
  , ccpConfigHandler = \model configs req ->
      let req' = ccpConfigHandler cp1 model configs req
      in ccpConfigHandler cp2 model configs req'
  , ccpFromResponse = \model configs prompt resp ->
      ccpFromResponse cp2 model configs prompt resp  -- Note: only cp2 sees the response (cp1 ignored)
  }

-- Build a completion request from a prompt
toCompletionRequest :: (CompletionProviderImplementation m, Monoid (CompletionRequest m))
                    => m
                    -> [ModelConfig m]
                    -> Text  -- The prompt
                    -> CompletionRequest m
toCompletionRequest model configs prompt =
  let composableProvider = getComposableCompletionProvider
      promptHandler = ccpToRequest composableProvider
      configHandler = ccpConfigHandler composableProvider
      -- Apply prompt handler
      reqAfterPrompt = promptHandler model configs prompt mempty
      -- Apply config handlers
  in configHandler model configs reqAfterPrompt

-- Parse a completion response
fromCompletionResponse :: CompletionProviderImplementation m
                       => m
                       -> [ModelConfig m]
                       -> Text  -- Original prompt
                       -> CompletionResponse m
                       -> Text  -- Completed text
fromCompletionResponse model configs prompt resp =
  let composableProvider = getComposableCompletionProvider
      parser = ccpFromResponse composableProvider
  in parser model configs prompt resp

