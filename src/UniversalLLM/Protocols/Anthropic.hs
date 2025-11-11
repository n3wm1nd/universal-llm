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
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import GHC.Generics (Generic)
import Control.Applicative ((<|>))
import Data.Maybe (listToMaybe)
import UniversalLLM.Core.Types

-- Request structure
-- Extended thinking configuration
data AnthropicThinkingConfig = AnthropicThinkingConfig
  { thinkingType :: Text  -- "enabled" to enable extended thinking
  , thinkingBudgetTokens :: Maybe Int  -- Optional token budget for thinking
  } deriving (Generic, Show, Eq)

data AnthropicRequest = AnthropicRequest
  { model :: Text
  , messages :: [AnthropicMessage]
  , max_tokens :: Int
  , temperature :: Maybe Double
  , system :: Maybe [AnthropicSystemBlock]
  , tools :: Maybe [AnthropicToolDefinition]
  , stream :: Maybe Bool
  , thinking :: Maybe AnthropicThinkingConfig
  } deriving (Generic, Show, Eq)

instance Semigroup AnthropicRequest where
  r1 <> r2 = AnthropicRequest
    { model = model r2  -- Right-biased for scalar fields
    , messages = messages r1 <> messages r2
    , max_tokens = max_tokens r2  -- Right-biased
    , temperature = temperature r2 <|> temperature r1  -- Right-biased with fallback
    , system = system r2 <|> system r1
    , tools = tools r2 <|> tools r1
    , stream = stream r2 <|> stream r1
    , thinking = thinking r2 <|> thinking r1
    }

instance Monoid AnthropicRequest where
  mempty = AnthropicRequest
    { model = ""
    , messages = []
    , max_tokens = 1000  -- Anthropic default
    , temperature = Nothing
    , system = Nothing
    , tools = Nothing
    , stream = Nothing
    , thinking = Nothing
    }

data AnthropicMessage = AnthropicMessage
  { role :: Text
  , content :: [AnthropicContentBlock]
  } deriving (Generic, Show, Eq)

-- System prompt blocks
data AnthropicSystemBlock = AnthropicSystemBlock
  { systemText :: Text
  , systemType :: Text  -- Always "text"
  } deriving (Generic, Show, Eq)

-- Type aliases for AnthropicContentBlock parameters (for documentation)
type TextContent = Text
type ToolUseId = Text
type ToolUseName = Text
type ToolUseInput = Value
type ToolResultId = Text
type ToolResultContent = Text

data AnthropicContentBlock
  = AnthropicTextBlock TextContent
  | AnthropicToolUseBlock ToolUseId ToolUseName ToolUseInput
  | AnthropicToolResultBlock ToolResultId ToolResultContent
  | AnthropicThinkingBlock Text
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
instance HasCodec AnthropicSystemBlock where
  codec = object "AnthropicSystemBlock" $
    AnthropicSystemBlock
      <$> requiredField "text" "System prompt text" .= systemText
      <*> requiredField "type" "Block type (always 'text')" .= systemType

instance HasCodec AnthropicThinkingConfig where
  codec = object "AnthropicThinkingConfig" $
    AnthropicThinkingConfig
      <$> requiredField "type" "Thinking type (enabled/disabled)" .= thinkingType
      <*> optionalField "budget_tokens" "Token budget for thinking" .= thinkingBudgetTokens

instance HasCodec AnthropicRequest where
  codec = object "AnthropicRequest" $
    AnthropicRequest
      <$> requiredField "model" "Model name" .= model
      <*> requiredField "messages" "Messages" .= messages
      <*> requiredField "max_tokens" "Max tokens" .= max_tokens
      <*> optionalFieldWith "temperature" (dimapCodec realToFrac realToFrac scientificCodec) "Temperature" .= temperature
      <*> optionalField "system" "System prompt" .= system
      <*> optionalField "tools" "Tool definitions" .= tools
      <*> optionalField "stream" "Enable streaming" .= stream
      <*> optionalField "thinking" "Extended thinking configuration" .= thinking

instance HasCodec AnthropicContentBlock where
  codec = object "AnthropicContentBlock" $
    dimapCodec fromEither toEither $
      possiblyJointEitherCodec textBlockCodec
        (possiblyJointEitherCodec toolUseBlockCodec
          (possiblyJointEitherCodec toolResultBlockCodec thinkingBlockCodec))
    where
      fromEither (Left txt) = AnthropicTextBlock txt
      fromEither (Right (Left (tid, tname, tinput))) = AnthropicToolUseBlock tid tname tinput
      fromEither (Right (Right (Left (rid, rcontent)))) = AnthropicToolResultBlock rid rcontent
      fromEither (Right (Right (Right thinking))) = AnthropicThinkingBlock thinking

      toEither (AnthropicTextBlock txt) = Left txt
      toEither (AnthropicToolUseBlock tid tname tinput) = Right (Left (tid, tname, tinput))
      toEither (AnthropicToolResultBlock rid rcontent) = Right (Right (Left (rid, rcontent)))
      toEither (AnthropicThinkingBlock thinking) = Right (Right (Right thinking))

      textBlockCodec =
        requiredField "type" "Block type" .= const ("text" :: Text)
        *> requiredField "text" "Text content" .= id

      toolUseBlockCodec =
        requiredField "type" "Block type" .= const ("tool_use" :: Text)
        *> ((,,)
          <$> requiredField "id" "Tool use ID" .= (\(tid, _, _) -> tid)
          <*> requiredField "name" "Tool name" .= (\(_, tname, _) -> tname)
          <*> requiredField "input" "Tool input" .= (\(_, _, tinput) -> tinput))

      toolResultBlockCodec =
        requiredField "type" "Block type" .= const ("tool_result" :: Text)
        *> ((,)
          <$> requiredField "tool_use_id" "Tool use ID" .= fst
          <*> requiredField "content" "Result content" .= snd)

      thinkingBlockCodec =
        requiredField "type" "Block type" .= const ("thinking" :: Text)
        *> requiredField "thinking" "Thinking content" .= id

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

-- NOTE: ProtocolHandleTools was removed - tool handling is now done inline in the provider

-- ============================================================================
-- Streaming Support - Delta Merger for SSE
-- ============================================================================

-- | Merge Anthropic streaming delta into accumulated response
-- Anthropic streaming sends events like:
-- - message_start: initial metadata
-- - content_block_delta: incremental text in delta.text
-- - message_delta: final stop_reason
mergeAnthropicDelta :: AnthropicResponse -> Value -> AnthropicResponse
mergeAnthropicDelta acc chunk =
    case (acc, extractEventType chunk) of
        (_, Just "message_start") -> initializeFromMessageStart chunk
        (AnthropicSuccess resp, Just "content_block_delta") ->
            case extractTextDelta chunk of
                Just text -> AnthropicSuccess $ appendText resp text
                Nothing -> acc
        (AnthropicSuccess resp, Just "message_delta") ->
            case extractStopReason chunk of
                Just reason -> AnthropicSuccess $ setStopReason resp reason
                Nothing -> acc
        _ -> acc
  where
    extractEventType :: Value -> Maybe Text
    extractEventType (Aeson.Object obj) = do
        Aeson.String eventType <- KM.lookup "type" obj
        return eventType
    extractEventType _ = Nothing

    extractTextDelta :: Value -> Maybe Text
    extractTextDelta (Aeson.Object obj) = do
        Aeson.Object delta <- KM.lookup "delta" obj
        Aeson.String text <- KM.lookup "text" delta
        return text
    extractTextDelta _ = Nothing

    extractStopReason :: Value -> Maybe Text
    extractStopReason (Aeson.Object obj) = do
        Aeson.Object delta <- KM.lookup "delta" obj
        Aeson.String reason <- KM.lookup "stop_reason" delta
        return reason
    extractStopReason _ = Nothing

    initializeFromMessageStart :: Value -> AnthropicResponse
    initializeFromMessageStart (Aeson.Object obj) =
        -- Extract initial message metadata if present
        let emptyResp = AnthropicSuccessResponse "" "" "assistant" [] Nothing (AnthropicUsage 0 0)
        in AnthropicSuccess emptyResp
    initializeFromMessageStart _ = AnthropicError (AnthropicErrorResponse "unknown" "Failed to initialize")

    appendText :: AnthropicSuccessResponse -> Text -> AnthropicSuccessResponse
    appendText resp text =
        case responseContent resp of
            [] -> resp { responseContent = [AnthropicTextBlock text] }
            [AnthropicTextBlock existing] -> resp { responseContent = [AnthropicTextBlock (existing <> text)] }
            xs -> resp  -- Preserve if more complex content

    setStopReason :: AnthropicSuccessResponse -> Text -> AnthropicSuccessResponse
    setStopReason resp reason = resp { responseStopReason = Just reason }
