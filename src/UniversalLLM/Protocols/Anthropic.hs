{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}


module UniversalLLM.Protocols.Anthropic where

import Autodocodec
import Data.Text (Text)
import Data.Text.Encoding (encodeUtf8)
import Data.Aeson (Value)
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import GHC.Generics (Generic)
import Control.Applicative ((<|>), asum)
import UniversalLLM

-- Request structure
-- Extended thinking configuration
data AnthropicThinkingConfig = AnthropicThinkingConfig
  { thinkingType :: Text  -- "enabled" for budget-based (Sonnet/Opus 4.5), "adaptive" for effort-based (Opus 4.6+)
  , thinkingBudgetTokens :: Maybe Int  -- Token budget for thinking (used with type="enabled")
  } deriving (Generic, Show, Eq)

-- Output configuration (Opus 4.6+)
data AnthropicOutputConfig = AnthropicOutputConfig
  { outputEffort :: Text  -- "low", "medium", "high", or "max" (max only on Opus 4.6)
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
  , output_config :: Maybe AnthropicOutputConfig
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
    , output_config = output_config r2 <|> output_config r1
    }

instance Monoid AnthropicRequest where
  mempty = AnthropicRequest
    { model = ""
    , messages = []
    , max_tokens = 2048  -- Must be > thinking.budget_tokens when reasoning is enabled
    , temperature = Nothing
    , system = Nothing
    , tools = Nothing
    , stream = Nothing
    , thinking = Nothing
    , output_config = Nothing
    }

data AnthropicMessage = AnthropicMessage
  { role :: Text
  , content :: [AnthropicContentBlock]
  } deriving (Generic, Show, Eq)

-- Cache control configuration
data CacheControl = CacheControl
  { cacheType :: Text  -- "ephemeral"
  , cacheTTL :: Text   -- "5m" or "1h"
  } deriving (Generic, Show, Eq)

-- System prompt blocks
data AnthropicSystemBlock = AnthropicSystemBlock
  { systemText :: Text
  , systemType :: Text  -- Always "text"
  , systemCacheControl :: Maybe CacheControl
  } deriving (Generic, Show, Eq)

-- Type aliases for AnthropicContentBlock parameters (for documentation)
type TextContent = Text
type ToolUseId = Text
type ToolUseName = Text
type ToolUseInput = Value
type ToolResultId = Text
type ToolResultContent = Text

data AnthropicContentBlock
  = AnthropicTextBlock TextContent (Maybe CacheControl)
  | AnthropicToolUseBlock ToolUseId ToolUseName ToolUseInput (Maybe CacheControl)
  | AnthropicToolResultBlock ToolResultId ToolResultContent (Maybe CacheControl)
  | AnthropicThinkingBlock
      { thinkingText :: Text
      , thinkingSignature :: Value  -- Required signature metadata from API
      , thinkingCacheControl :: Maybe CacheControl
      }
  deriving (Generic, Show, Eq)

data AnthropicToolDefinition = AnthropicToolDefinition
  { anthropicToolName :: Text
  , anthropicToolDescription :: Text
  , anthropicToolInputSchema :: Value
  , anthropicToolCacheControl :: Maybe CacheControl
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
instance HasCodec CacheControl where
  codec = object "CacheControl" $
    CacheControl
      <$> requiredField "type" "Cache type (ephemeral)" .= cacheType
      <*> requiredField "ttl" "Cache TTL (5m or 1h)" .= cacheTTL

instance HasCodec AnthropicSystemBlock where
  codec = object "AnthropicSystemBlock" $
    AnthropicSystemBlock
      <$> requiredField "text" "System prompt text" .= systemText
      <*> requiredField "type" "Block type (always 'text')" .= systemType
      <*> optionalField "cache_control" "Cache control configuration" .= systemCacheControl

instance HasCodec AnthropicThinkingConfig where
  codec = object "AnthropicThinkingConfig" $
    AnthropicThinkingConfig
      <$> requiredField "type" "Thinking type (enabled/adaptive)" .= thinkingType
      <*> optionalField "budget_tokens" "Token budget for thinking" .= thinkingBudgetTokens

instance HasCodec AnthropicOutputConfig where
  codec = object "AnthropicOutputConfig" $
    AnthropicOutputConfig
      <$> requiredField "effort" "Effort level (low/medium/high/max)" .= outputEffort

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
      <*> optionalField "output_config" "Output configuration (effort level)" .= output_config

instance HasCodec AnthropicContentBlock where
  codec = object "AnthropicContentBlock" $
    dimapCodec fromEither toEither $
      possiblyJointEitherCodec textBlockCodec
        (possiblyJointEitherCodec toolUseBlockCodec
          (possiblyJointEitherCodec toolResultBlockCodec thinkingBlockCodec))
    where
      fromEither (Left (txt, cc)) = AnthropicTextBlock txt cc
      fromEither (Right (Left (tid, tname, tinput, cc))) = AnthropicToolUseBlock tid tname tinput cc
      fromEither (Right (Right (Left (rid, rcontent, cc)))) = AnthropicToolResultBlock rid rcontent cc
      fromEither (Right (Right (Right (thinking, sig, cc)))) = AnthropicThinkingBlock thinking sig cc

      toEither (AnthropicTextBlock txt cc) = Left (txt, cc)
      toEither (AnthropicToolUseBlock tid tname tinput cc) = Right (Left (tid, tname, tinput, cc))
      toEither (AnthropicToolResultBlock rid rcontent cc) = Right (Right (Left (rid, rcontent, cc)))
      toEither (AnthropicThinkingBlock{..}) = Right (Right (Right (thinkingText, thinkingSignature, thinkingCacheControl)))

      textBlockCodec =
        requiredField "type" "Block type" .= const ("text" :: Text)
        *> ((,)
          <$> requiredField "text" "Text content" .= fst
          <*> optionalField "cache_control" "Cache control configuration" .= snd)

      toolUseBlockCodec =
        requiredField "type" "Block type" .= const ("tool_use" :: Text)
        *> ((,,,)
          <$> requiredField "id" "Tool use ID" .= (\(tid, _, _, _) -> tid)
          <*> requiredField "name" "Tool name" .= (\(_, tname, _, _) -> tname)
          <*> requiredField "input" "Tool input" .= (\(_, _, tinput, _) -> tinput)
          <*> optionalField "cache_control" "Cache control configuration" .= (\(_, _, _, cc) -> cc))

      toolResultBlockCodec =
        requiredField "type" "Block type" .= const ("tool_result" :: Text)
        *> ((,,)
          <$> requiredField "tool_use_id" "Tool use ID" .= (\(a, _, _) -> a)
          <*> requiredField "content" "Result content" .= (\(_, b, _) -> b)
          <*> optionalField "cache_control" "Cache control configuration" .= (\(_, _, c) -> c))

      thinkingBlockCodec =
        requiredField "type" "Block type" .= const ("thinking" :: Text)
        *> ((,,)
          <$> requiredField "thinking" "Thinking content" .= (\(t, _, _) -> t)
          <*> requiredField "signature" "Thinking signature metadata" .= (\(_, s, _) -> s)
          <*> optionalField "cache_control" "Cache control configuration" .= (\(_, _, cc) -> cc))

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
      <*> optionalField "cache_control" "Cache control configuration" .= anthropicToolCacheControl

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

-- ============================================================================
-- Default Values for Record Updates
-- ============================================================================

-- | Default Anthropic message (use record update syntax to set fields)
defaultAnthropicMessage :: AnthropicMessage
defaultAnthropicMessage = AnthropicMessage
  { role = ""
  , content = []
  }

-- | Default Anthropic system block (use record update syntax to set fields)
defaultAnthropicSystemBlock :: AnthropicSystemBlock
defaultAnthropicSystemBlock = AnthropicSystemBlock
  { systemText = ""
  , systemType = ""
  , systemCacheControl = Nothing
  }

-- | Default cache control (use record update syntax to set fields)
defaultCacheControl :: CacheControl
defaultCacheControl = CacheControl
  { cacheType = ""
  , cacheTTL = ""
  }

-- | Default Anthropic thinking config (use record update syntax to set fields)
defaultAnthropicThinkingConfig :: AnthropicThinkingConfig
defaultAnthropicThinkingConfig = AnthropicThinkingConfig
  { thinkingType = ""
  , thinkingBudgetTokens = Nothing
  }

-- | Default Anthropic tool definition (use record update syntax to set fields)
defaultAnthropicToolDefinition :: AnthropicToolDefinition
defaultAnthropicToolDefinition = AnthropicToolDefinition
  { anthropicToolName = ""
  , anthropicToolDescription = ""
  , anthropicToolInputSchema = Aeson.Object KM.empty
  , anthropicToolCacheControl = Nothing
  }

-- | Default Anthropic success response (use record update syntax to set fields)
defaultAnthropicSuccessResponse :: AnthropicSuccessResponse
defaultAnthropicSuccessResponse = AnthropicSuccessResponse
  { responseId = ""
  , responseModel = ""
  , responseRole = ""
  , responseContent = []
  , responseStopReason = Nothing
  , responseUsage = AnthropicUsage 0 0
  }

-- | Default Anthropic usage (use record update syntax to set fields)
defaultAnthropicUsage :: AnthropicUsage
defaultAnthropicUsage = AnthropicUsage
  { usageInputTokens = 0
  , usageOutputTokens = 0
  }

-- Helper: Convert ToolDefinition to Anthropic wire format
-- Cache control is set to Nothing by default, will be added by provider layer
toAnthropicToolDef :: ToolDefinition -> AnthropicToolDefinition
toAnthropicToolDef toolDef = AnthropicToolDefinition
  { anthropicToolName = toolDefName toolDef
  , anthropicToolDescription = toolDefDescription toolDef
  , anthropicToolInputSchema = toolDefParameters toolDef
  , anthropicToolCacheControl = Nothing
  }

-- Helper: Add cache control to the last tool definition (for caching tool list)
withToolCacheControl :: [AnthropicToolDefinition] -> [AnthropicToolDefinition]
withToolCacheControl [] = []
withToolCacheControl tools = init tools ++ [lastTool { anthropicToolCacheControl = Just (CacheControl "ephemeral" "5m") }]
  where lastTool = last tools

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
        (AnthropicSuccess resp, Just "content_block_start") ->
            case extractContentBlock chunk of
                Just block -> AnthropicSuccess $ addContentBlock resp block
                Nothing -> acc
        (AnthropicSuccess resp, Just "content_block_delta") ->
            case extractDeltaIndex chunk of
                Nothing -> acc
                Just idx ->
                    case asum [fmap (appendTextAt idx resp) (extractTextDelta chunk), fmap (appendToolInputAt idx resp) (extractInputJsonDelta chunk), fmap (appendThinkingAt idx resp) (extractThinkingDelta chunk), fmap (appendSignatureAt idx resp) (extractSignatureDelta chunk)] of
                        Just updated -> AnthropicSuccess updated
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

    extractDeltaIndex :: Value -> Maybe Int
    extractDeltaIndex (Aeson.Object obj) = do
        Aeson.Number idx <- KM.lookup "index" obj
        return $ round idx
    extractDeltaIndex _ = Nothing

    appendTextAt :: Int -> AnthropicSuccessResponse -> Text -> AnthropicSuccessResponse
    appendTextAt idx resp text =
        resp { responseContent = updateBlockAt idx (appendToText text) (responseContent resp) }
      where
        appendToText t (AnthropicTextBlock existing cc) = AnthropicTextBlock (existing <> t) cc
        appendToText _ block = block

    appendThinkingAt :: Int -> AnthropicSuccessResponse -> Text -> AnthropicSuccessResponse
    appendThinkingAt idx resp thinking =
        resp { responseContent = updateBlockAt idx (appendToThinking thinking) (responseContent resp) }
      where
        appendToThinking t block@AnthropicThinkingBlock{..} =
            block { thinkingText = thinkingText <> t }
        appendToThinking _ block = block

    appendSignatureAt :: Int -> AnthropicSuccessResponse -> Text -> AnthropicSuccessResponse
    appendSignatureAt idx resp signatureDelta =
        resp { responseContent = updateBlockAt idx (appendToSignature signatureDelta) (responseContent resp) }
      where
        appendToSignature sd block@AnthropicThinkingBlock{..} =
            -- Accumulate signature as string, similar to tool input
            let accumulatedSig = case thinkingSignature of
                    Aeson.String s -> s <> sd
                    _ -> sd
                parsedSig = case Aeson.decode (BSL.fromStrict (encodeUtf8 accumulatedSig)) of
                    Just val -> val
                    -- If not valid JSON yet, keep as string for now
                    Nothing -> Aeson.String accumulatedSig
            in block { thinkingSignature = parsedSig }
        appendToSignature _ block = block

    appendToolInputAt :: Int -> AnthropicSuccessResponse -> Text -> AnthropicSuccessResponse
    appendToolInputAt idx resp jsonDelta =
        resp { responseContent = updateBlockAt idx (appendToTool jsonDelta) (responseContent resp) }
      where
        appendToTool jd (AnthropicToolUseBlock toolId toolName currentInput cc) =
            let accumulatedJson = case currentInput of
                    Aeson.String s -> s <> jd
                    _ -> jd
                parsedInput = case Aeson.decode (BSL.fromStrict (encodeUtf8 accumulatedJson)) of
                    Just val -> val
                    Nothing ->
                        -- If parsing failed, check if it's empty string (no params) or incomplete JSON
                        if accumulatedJson == ""
                        then Aeson.Object KM.empty  -- Empty string = no parameters
                        else Aeson.String accumulatedJson  -- Incomplete JSON, keep accumulating
            in AnthropicToolUseBlock toolId toolName parsedInput cc
        appendToTool _ block = block

    updateBlockAt :: Int -> (AnthropicContentBlock -> AnthropicContentBlock) -> [AnthropicContentBlock] -> [AnthropicContentBlock]
    updateBlockAt idx f blocks = take idx blocks ++ [f (blocks !! idx)] ++ drop (idx + 1) blocks

    initializeFromMessageStart :: Value -> AnthropicResponse
    initializeFromMessageStart (Aeson.Object _obj) =
        -- Extract initial message metadata if present
        let emptyResp = AnthropicSuccessResponse "" "" "assistant" [] Nothing (AnthropicUsage 0 0)
        in AnthropicSuccess emptyResp
    initializeFromMessageStart _ = AnthropicError (AnthropicErrorResponse "unknown" "Failed to initialize")

    setStopReason :: AnthropicSuccessResponse -> Text -> AnthropicSuccessResponse
    setStopReason resp reason = resp { responseStopReason = Just reason }

    extractContentBlock :: Value -> Maybe AnthropicContentBlock
    extractContentBlock (Aeson.Object obj) = do
        Aeson.Object contentBlock <- KM.lookup "content_block" obj
        Aeson.String blockType <- KM.lookup "type" contentBlock
        case blockType of
            "tool_use" -> do
                Aeson.String toolId <- KM.lookup "id" contentBlock
                Aeson.String toolName <- KM.lookup "name" contentBlock
                -- Initial tool input is empty object at block start
                return $ AnthropicToolUseBlock toolId toolName (Aeson.Object KM.empty) Nothing
            "text" -> do
                Aeson.String text <- KM.lookup "text" contentBlock
                return $ AnthropicTextBlock text Nothing
            "thinking" -> do
                -- Thinking blocks may have "thinking" field but start empty
                let thinking = case KM.lookup "thinking" contentBlock of
                      Just (Aeson.String t) -> t
                      _ -> ""
                -- Signature may start as empty string and be filled via signature_delta events
                let signature = case KM.lookup "signature" contentBlock of
                      Just (Aeson.String "") -> Aeson.String ""  -- Empty string, will be filled by deltas
                      Just sig -> sig
                      Nothing -> Aeson.String ""  -- Default to empty string if not present
                return $ AnthropicThinkingBlock
                    { thinkingText = thinking
                    , thinkingSignature = signature
                    , thinkingCacheControl = Nothing
                    }
            _ -> Nothing
    extractContentBlock _ = Nothing

    extractInputJsonDelta :: Value -> Maybe Text
    extractInputJsonDelta (Aeson.Object obj) = do
        Aeson.Object delta <- KM.lookup "delta" obj
        Aeson.String partialJson <- KM.lookup "partial_json" delta
        return partialJson
    extractInputJsonDelta _ = Nothing

    addContentBlock :: AnthropicSuccessResponse -> AnthropicContentBlock -> AnthropicSuccessResponse
    addContentBlock resp block =
        resp { responseContent = responseContent resp ++ [block] }

    extractThinkingDelta :: Value -> Maybe Text
    extractThinkingDelta (Aeson.Object obj) = do
        Aeson.Object delta <- KM.lookup "delta" obj
        Aeson.String thinking <- KM.lookup "thinking" delta
        return thinking
    extractThinkingDelta _ = Nothing

    extractSignatureDelta :: Value -> Maybe Text
    extractSignatureDelta (Aeson.Object obj) = do
        Aeson.Object delta <- KM.lookup "delta" obj
        Aeson.String signature <- KM.lookup "signature" delta
        return signature
    extractSignatureDelta _ = Nothing
