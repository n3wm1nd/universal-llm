{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module UniversalLLM.Protocols.Anthropic where

import Autodocodec
import Data.Text (Text)
import Data.Aeson (Value)
import GHC.Generics (Generic)
import UniversalLLM.Core.Types

-- Request structure
data AnthropicRequest = AnthropicRequest
  { model :: Text
  , messages :: [AnthropicMessage]
  , max_tokens :: Int
  , temperature :: Maybe Double
  , system :: Maybe Text
  , tools :: Maybe [AnthropicToolDefinition]
  } deriving (Generic, Show, Eq)

data AnthropicMessage = AnthropicMessage
  { role :: Text
  , content :: [AnthropicContentBlock]
  } deriving (Generic, Show, Eq)

data AnthropicContentBlock
  = AnthropicTextBlock { textContent :: Text }
  | AnthropicToolUseBlock
      { toolUseId :: Text
      , toolUseName :: Text
      , toolUseInput :: Value
      }
  | AnthropicToolResultBlock
      { toolResultId :: Text
      , toolResultContent :: Text
      }
  deriving (Generic, Show, Eq)

data AnthropicToolDefinition = AnthropicToolDefinition
  { anthropicToolName :: Text
  , anthropicToolDescription :: Text
  , anthropicToolInputSchema :: Value
  } deriving (Generic, Show, Eq)

-- Response structure
data AnthropicResponse
  = AnthropicSuccess AnthropicSuccessResponse
  | AnthropicError AnthropicErrorResponse
  deriving (Show, Eq)

data AnthropicSuccessResponse = AnthropicSuccessResponse
  { responseId :: Text
  , responseModel :: Text
  , responseRole :: Text
  , responseContent :: [AnthropicContentBlock]
  , responseStopReason :: Maybe Text
  , responseUsage :: AnthropicUsage
  } deriving (Generic, Show, Eq)

data AnthropicUsage = AnthropicUsage
  { usageInputTokens :: Int
  , usageOutputTokens :: Int
  } deriving (Generic, Show, Eq)

data AnthropicErrorResponse = AnthropicErrorResponse
  { errorType :: Text
  , errorMessage :: Text
  } deriving (Generic, Show, Eq)

data AnthropicErrorDetail = AnthropicErrorDetail
  { errorDetailType :: Text
  , errorDetailMessage :: Text
  } deriving (Generic, Show, Eq)

-- Codec instances
instance HasCodec AnthropicRequest where
  codec = object "AnthropicRequest" $
    AnthropicRequest
      <$> requiredField "model" "Model name" .= model
      <*> requiredField "messages" "Messages" .= messages
      <*> requiredField "max_tokens" "Max tokens" .= max_tokens
      <*> optionalFieldWith "temperature" (dimapCodec realToFrac realToFrac scientificCodec) "Temperature" .= temperature
      <*> optionalField "system" "System prompt" .= system
      <*> optionalField "tools" "Tool definitions" .= tools

instance HasCodec AnthropicContentBlock where
  codec = object "AnthropicContentBlock" $
    dimapCodec fromEither toEither $
      possiblyJointEitherCodec textCodec (possiblyJointEitherCodec toolUseCodec toolResultCodec)
    where
      fromEither (Left text) = text
      fromEither (Right (Left toolUse)) = toolUse
      fromEither (Right (Right toolResult)) = toolResult

      toEither (AnthropicTextBlock txt) = Left (AnthropicTextBlock txt)
      toEither (AnthropicToolUseBlock a b c) = Right (Left (AnthropicToolUseBlock a b c))
      toEither (AnthropicToolResultBlock a b) = Right (Right (AnthropicToolResultBlock a b))

      textCodec =
        requiredField "type" "Block type" .= const ("text" :: Text)
        *> (AnthropicTextBlock <$> requiredField "text" "Text content" .= textContent)

      toolUseCodec =
        requiredField "type" "Block type" .= const ("tool_use" :: Text)
        *> (AnthropicToolUseBlock
          <$> requiredField "id" "Tool use ID" .= toolUseId
          <*> requiredField "name" "Tool name" .= toolUseName
          <*> requiredField "input" "Tool input" .= toolUseInput)

      toolResultCodec =
        requiredField "type" "Block type" .= const ("tool_result" :: Text)
        *> (AnthropicToolResultBlock
          <$> requiredField "tool_use_id" "Tool use ID" .= toolResultId
          <*> requiredField "content" "Result content" .= toolResultContent)

instance HasCodec AnthropicMessage where
  codec = object "AnthropicMessage" $
    AnthropicMessage
      <$> requiredField "role" "Role" .= role
      <*> requiredField "content" "Content" .= content

instance HasCodec AnthropicToolDefinition where
  codec = object "AnthropicToolDefinition" $
    AnthropicToolDefinition
      <$> requiredField "name" "Tool name" .= anthropicToolName
      <*> requiredField "description" "Tool description" .= anthropicToolDescription
      <*> requiredField "input_schema" "Input schema" .= anthropicToolInputSchema

instance HasCodec AnthropicResponse where
  codec = dimapCodec fromEither toEither $ eitherCodec (codec @AnthropicSuccessResponse) (codec @AnthropicErrorResponse)
    where
      fromEither (Left success) = AnthropicSuccess success
      fromEither (Right err) = AnthropicError err

      toEither (AnthropicSuccess success) = Left success
      toEither (AnthropicError err) = Right err

instance HasCodec AnthropicSuccessResponse where
  codec = object "AnthropicSuccessResponse" $
    AnthropicSuccessResponse
      <$> requiredField "id" "Response ID" .= responseId
      <*> requiredField "model" "Model" .= responseModel
      <*> requiredField "role" "Role" .= responseRole
      <*> requiredField "content" "Content" .= responseContent
      <*> optionalField "stop_reason" "Stop reason" .= responseStopReason
      <*> requiredField "usage" "Usage statistics" .= responseUsage

instance HasCodec AnthropicUsage where
  codec = object "AnthropicUsage" $
    AnthropicUsage
      <$> requiredField "input_tokens" "Input tokens" .= usageInputTokens
      <*> requiredField "output_tokens" "Output tokens" .= usageOutputTokens

instance HasCodec AnthropicErrorDetail where
  codec = object "AnthropicErrorDetail" $
    AnthropicErrorDetail
      <$> requiredField "type" "Error type" .= errorDetailType
      <*> requiredField "message" "Error message" .= errorDetailMessage

instance HasCodec AnthropicErrorResponse where
  codec = object "AnthropicErrorResponse" $
    dimapCodec fromDetail toDetail $
      requiredField "error" "Error details" .= id
    where
      fromDetail detail = AnthropicErrorResponse
        { errorType = errorDetailType detail
        , errorMessage = errorDetailMessage detail
        }
      toDetail response = AnthropicErrorDetail
        { errorDetailType = errorType response
        , errorDetailMessage = errorMessage response
        }

-- Helper: Convert ToolDefinition to Anthropic wire format
toAnthropicToolDef :: ToolDefinition -> AnthropicToolDefinition
toAnthropicToolDef toolDef = AnthropicToolDefinition
  { anthropicToolName = toolDefName toolDef
  , anthropicToolDescription = toolDefDescription toolDef
  , anthropicToolInputSchema = toolDefParameters toolDef
  }

-- Tool call conversion - handled via ProtocolHandleTools typeclass
-- This allows models without tool support to still parse text responses

-- Instance for tool-capable models: converts tool_use blocks to AssistantTool
instance (ModelHasTools model, ProviderSupportsTools provider) => ProtocolHandleTools AnthropicContentBlock model provider where
  handleToolCalls blocks =
    let toolCalls = [ ToolCall tid tname tinput
                    | AnthropicToolUseBlock tid tname tinput <- blocks
                    ]
    in if null toolCalls
       then []
       else [AssistantTool toolCalls]
