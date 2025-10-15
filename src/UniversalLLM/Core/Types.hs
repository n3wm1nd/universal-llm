{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
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
import Autodocodec (HasCodec, toJSONViaCodec, eitherDecodeJSONViaCodec)

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
-- The Tool class connects a tool type (which can hold config) to its parameters type
class (HasCodec (ToolParams tool), HasCodec (ToolOutput tool), Eq tool) => Tool tool m where
  type ToolParams tool :: *
  type ToolOutput tool :: *
  toolName :: tool -> Text
  toolDescription :: tool -> Text
  call :: tool -> ToolParams tool -> m (ToolOutput tool)

-- Existential wrapper for heterogeneous tool lists
data SomeTool m where
  SomeTool :: Tool tool m => tool -> SomeTool m

data ToolCall = ToolCall
  { toolCallId :: Text
  , toolCallName :: Text
  , toolCallParameters :: Value
  } deriving (Show, Eq)

data ToolResult = ToolResult
  { toolResultCallId :: Text
  , toolResultOutput :: Value
  } deriving (Show, Eq)

-- | Match a ToolCall to a SomeTool from the available tools list
matchToolCall :: forall m. [SomeTool m] -> ToolCall -> Maybe (SomeTool m)
matchToolCall tools tc =
  let name = toolCallName tc
  in find (\(SomeTool tool) -> toolName @_ @m tool == name) tools
  where
    find :: (a -> Bool) -> [a] -> Maybe a
    find _ [] = Nothing
    find p (x:xs) = if p x then Just x else find p xs

-- | Execute a tool call by matching and dispatching
-- Takes available tools and the tool call from the LLM
executeToolCall :: Monad m
                => [SomeTool m]  -- available tools
                -> ToolCall      -- call from LLM
                -> m (Maybe Value)  -- result (Nothing if tool not found or params invalid)
executeToolCall tools toolCall =
  case matchToolCall tools toolCall of
    Nothing -> return Nothing
    Just (SomeTool tool) -> executeWithTool tool toolCall

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

-- GADT Messages with capability constraints
data Message model provider where
  UserText :: Text -> Message model provider
  UserImage :: HasVision model => Text -> Text -> Message model provider
  AssistantText :: Text -> Message model provider
  AssistantTool :: HasTools model => [ToolCall] -> Message model provider
  SystemText :: Text -> Message model provider
  ToolResultMsg :: HasTools model => ToolResult -> Message model provider