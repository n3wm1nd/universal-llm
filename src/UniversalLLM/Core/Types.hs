{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}

module UniversalLLM.Core.Types where

import Data.Text (Text)

-- Core capabilities that models can have
class HasVision model
class HasJSON model
class HasTools model

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
data ToolCall = ToolCall
  { toolId :: Text
  , toolName :: Text
  , toolParameters :: Text -- JSON string
  } deriving (Show, Eq)

data ToolResult = ToolResult
  { toolCallId :: Text
  , toolOutput :: Text
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

-- GADT Messages with capability constraints
data Message model provider where
  UserText :: Text -> Message model provider
  UserImage :: HasVision model => Text -> Text -> Message model provider
  AssistantText :: Text -> Message model provider
  AssistantTool :: HasTools model => [ToolCall] -> Message model provider
  SystemText :: Text -> Message model provider
  ToolResultMsg :: HasTools model => ToolResult -> Message model provider