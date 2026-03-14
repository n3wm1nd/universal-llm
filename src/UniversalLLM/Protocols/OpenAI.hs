{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module UniversalLLM.Protocols.OpenAI where

import Autodocodec
import Data.Aeson.Types (parseEither)
import Data.Text (Text)
import qualified Data.Text as Text
import Data.Aeson (Value)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Control.Applicative ((<|>), asum)
import Data.Maybe (listToMaybe)
import qualified Data.List
import UniversalLLM.Protocols.OpenAI.Delta (Delta(..), applyDelta)
import UniversalLLM

-- | Reasoning configuration for OpenRouter
data OpenAIReasoningConfig = OpenAIReasoningConfig
  { reasoning_enabled :: Maybe Bool
  , reasoning_max_tokens :: Maybe Int
  , reasoning_effort :: Maybe Text
  , reasoning_exclude :: Maybe Bool
  } deriving (Generic, Show, Eq)

instance HasCodec OpenAIReasoningConfig where
  codec = object "OpenAIReasoningConfig" $
    OpenAIReasoningConfig
      <$> optionalField "enabled" "Enable reasoning" .= reasoning_enabled
      <*> optionalField "max_tokens" "Max reasoning tokens" .= reasoning_max_tokens
      <*> optionalField "effort" "Reasoning effort level" .= reasoning_effort
      <*> optionalField "exclude" "Exclude reasoning from response" .= reasoning_exclude

data OpenAIRequest = OpenAIRequest
  { model :: Text
  , messages :: [OpenAIMessage]
  , temperature :: Maybe Double
  , max_tokens :: Maybe Int
  , seed :: Maybe Int
  , tools :: Maybe [OpenAIToolDefinition]
  , response_format :: Maybe OpenAIResponseFormat
  , stream :: Maybe Bool
  , reasoning :: Maybe OpenAIReasoningConfig
  } deriving (Generic, Show, Eq)

instance Semigroup OpenAIRequest where
  r1 <> r2 = OpenAIRequest
    { model = model r2  -- Right-biased for scalar fields
    , messages = messages r1 <> messages r2
    , temperature = temperature r2 <|> temperature r1  -- Right-biased with fallback
    , max_tokens = max_tokens r2 <|> max_tokens r1
    , seed = seed r2 <|> seed r1
    , tools = tools r1 <> tools r2  -- Concatenate tool lists
    , response_format = response_format r2 <|> response_format r1
    , stream = stream r2 <|> stream r1
    , reasoning = reasoning r2 <|> reasoning r1
    }

instance Monoid OpenAIRequest where
  mempty = OpenAIRequest
    { model = ""
    , messages = []
    , temperature = Nothing
    , max_tokens = Nothing
    , seed = Nothing
    , tools = Nothing
    , response_format = Nothing
    , stream = Nothing
    , reasoning = Nothing
    }

data OpenAIResponseFormat = OpenAIResponseFormat
  { responseType :: Text
  , json_schema :: Maybe Value
  } deriving (Generic, Show, Eq)

data OpenAIMessage = OpenAIMessage
  { role :: Text
  , content :: Maybe Text
  , reasoning_content :: Maybe Text
  , reasoning_details :: Maybe Value  -- OpenRouter reasoning metadata (must be preserved)
  , tool_calls :: Maybe [OpenAIToolCall]
  , tool_call_id :: Maybe Text
  } deriving (Generic, Show, Eq)

data OpenAIToolDefinition = OpenAIToolDefinition
  { tool_type :: Text
  , function :: OpenAIFunction
  } deriving (Generic, Show, Eq)

data OpenAIFunction = OpenAIFunction
  { name :: Text
  , description :: Text
  , parameters :: Value
  } deriving (Generic, Show, Eq)

data OpenAIToolCall = OpenAIToolCall
  { callId :: Text
  , toolCallType :: Text
  , toolFunction :: OpenAIToolFunction
  } deriving (Generic, Show, Eq)

data OpenAIToolFunction = OpenAIToolFunction
  { toolFunctionName :: Text
  , toolFunctionArguments :: Text
  } deriving (Generic, Show, Eq)

data OpenAIResponse
  = OpenAISuccess OpenAISuccessResponse
  | OpenAIError OpenAIErrorResponse
  deriving (Show, Eq)

data OpenAISuccessResponse = OpenAISuccessResponse
  { choices :: [OpenAIChoice]
  } deriving (Generic, Show, Eq)

data OpenAIChoice = OpenAIChoice
  { message :: OpenAIMessage
  } deriving (Generic, Show, Eq)

data OpenAIErrorResponse = OpenAIErrorResponse
  { errorDetail :: OpenAIErrorDetail
  } deriving (Generic, Show, Eq)

data OpenAIErrorDetail = OpenAIErrorDetail
  { code :: Int
  , errorMessage :: Text
  , errorType :: Maybe Text  -- Optional: OpenRouter doesn't include this field
  } deriving (Generic, Show, Eq)

instance HasCodec OpenAIRequest where
  codec = object "OpenAIRequest" $
    OpenAIRequest
      <$> requiredField "model" "Model name" .= model
      <*> requiredField "messages" "Messages" .= messages
      <*> optionalFieldWith "temperature" (dimapCodec realToFrac realToFrac scientificCodec) "Temperature" .= temperature
      <*> optionalFieldWith "max_tokens" (dimapCodec fromIntegral fromIntegral integerCodec) "Max tokens" .= max_tokens
      <*> optionalFieldWith "seed" (dimapCodec fromIntegral fromIntegral integerCodec) "Seed for deterministic output" .= seed
      <*> optionalField "tools" "Tool definitions" .= tools
      <*> optionalField "response_format" "Response format specification" .= response_format
      <*> optionalField "stream" "Enable streaming" .= stream
      <*> optionalField "reasoning" "Reasoning configuration (OpenRouter)" .= reasoning

instance HasCodec OpenAIResponseFormat where
  codec = object "OpenAIResponseFormat" $
    OpenAIResponseFormat
      <$> requiredField "type" "Response type (text or json_schema)" .= responseType
      <*> optionalField "json_schema" "JSON schema for structured output" .= json_schema

instance HasCodec OpenAIMessage where
  codec = object "OpenAIMessage" $
    OpenAIMessage
      <$> requiredField "role" "Role" .= role
      <*> optionalFieldOrNull "content" "Content" .= content
      <*> optionalFieldOrNull "reasoning_content" "Reasoning Content" .= reasoning_content
      <*> optionalField "reasoning_details" "OpenRouter reasoning metadata" .= reasoning_details
      <*> optionalField "tool_calls" "Tool calls" .= tool_calls
      <*> optionalField "tool_call_id" "Tool call ID for tool results" .= tool_call_id

instance HasCodec OpenAIToolDefinition where
  codec = object "OpenAIToolDefinition" $
    OpenAIToolDefinition
      <$> requiredField "type" "Tool type" .= tool_type
      <*> requiredField "function" "Function definition" .= function

instance HasCodec OpenAIFunction where
  codec = object "OpenAIFunction" $
    OpenAIFunction
      <$> requiredField "name" "Function name" .= name
      <*> requiredField "description" "Function description" .= description
      <*> requiredField "parameters" "Function parameters schema" .= parameters

instance HasCodec OpenAIToolCall where
  codec = object "OpenAIToolCall" $
    OpenAIToolCall
      <$> requiredField "id" "Tool call ID" .= callId
      <*> requiredField "type" "Tool call type" .= toolCallType
      <*> requiredField "function" "Function call" .= toolFunction

instance HasCodec OpenAIToolFunction where
  codec = object "OpenAIToolFunction" $
    OpenAIToolFunction
      <$> requiredField "name" "Function name" .= toolFunctionName
      <*> requiredField "arguments" "Function arguments" .= toolFunctionArguments

instance HasCodec OpenAIResponse where
  codec = dimapCodec fromEither toEither $ eitherCodec (codec @OpenAISuccessResponse) (codec @OpenAIErrorResponse)
    where
      fromEither (Left success) = OpenAISuccess success
      fromEither (Right err) = OpenAIError err

      toEither (OpenAISuccess success) = Left success
      toEither (OpenAIError err) = Right err

instance HasCodec OpenAISuccessResponse where
  codec = object "OpenAISuccessResponse" $
    OpenAISuccessResponse
      <$> requiredField "choices" "Choices" .= choices

instance HasCodec OpenAIChoice where
  codec = object "OpenAIChoice" $
    OpenAIChoice
      <$> requiredField "message" "Message" .= message

instance HasCodec OpenAIErrorResponse where
  codec = object "OpenAIErrorResponse" $
    OpenAIErrorResponse
      <$> requiredField "error" "Error" .= errorDetail

instance HasCodec OpenAIErrorDetail where
  codec = object "OpenAIErrorDetail" $
    OpenAIErrorDetail
      <$> requiredFieldWith "code" errorCodeCodec "Error code (int or string)" .= code
      <*> requiredField "message" "Error message" .= errorMessage
      <*> optionalField "type" "Error type (not included by all providers)" .= errorType
    where
      -- Some providers (like ZAI) return error code as string, others as int
      errorCodeCodec = dimapCodec parseCode showCode (eitherCodec codec codec)
      parseCode (Left i) = i  -- Int
      parseCode (Right s) = read (Text.unpack s)  -- String -> Int
      showCode i = Left i  -- Always encode as Int

-- Helper: Convert ToolDefinition to OpenAI wire format
toOpenAIToolDef :: ToolDefinition -> OpenAIToolDefinition
toOpenAIToolDef toolDef = OpenAIToolDefinition
  { tool_type = "function"
  , function = OpenAIFunction
      { name = toolDefName toolDef
      , description = toolDefDescription toolDef
      , parameters = toolDefParameters toolDef
      }
  }

-- ============================================================================
-- Default Values for Record Updates
-- ============================================================================

-- | Default OpenAI request (use record update syntax to set fields)
defaultOpenAIRequest :: OpenAIRequest
defaultOpenAIRequest = OpenAIRequest
  { model = ""
  , messages = []
  , temperature = Nothing
  , max_tokens = Nothing
  , seed = Nothing
  , tools = Nothing
  , response_format = Nothing
  , stream = Nothing
  , reasoning = Nothing
  }

-- | Default OpenAI message (use record update syntax to set fields)
defaultOpenAIMessage :: OpenAIMessage
defaultOpenAIMessage = OpenAIMessage
  { role = ""
  , content = Nothing
  , reasoning_content = Nothing
  , reasoning_details = Nothing
  , tool_calls = Nothing
  , tool_call_id = Nothing
  }

-- | Default OpenAI response format (use record update syntax to set fields)
defaultOpenAIResponseFormat :: OpenAIResponseFormat
defaultOpenAIResponseFormat = OpenAIResponseFormat
  { responseType = ""
  , json_schema = Nothing
  }

-- | Default OpenAI tool call (use record update syntax to set fields)
defaultOpenAIToolCall :: OpenAIToolCall
defaultOpenAIToolCall = OpenAIToolCall
  { callId = ""
  , toolCallType = ""
  , toolFunction = OpenAIToolFunction
      { toolFunctionName = ""
      , toolFunctionArguments = ""
      }
  }

-- | Default OpenAI tool function (use record update syntax to set fields)
defaultOpenAIToolFunction :: OpenAIToolFunction
defaultOpenAIToolFunction = OpenAIToolFunction
  { toolFunctionName = ""
  , toolFunctionArguments = ""
  }

-- | Default OpenAI success response (use record update syntax to set fields)
defaultOpenAISuccessResponse :: OpenAISuccessResponse
defaultOpenAISuccessResponse = OpenAISuccessResponse
  { choices = []
  }

-- | Default OpenAI choice (use record update syntax to set fields)
defaultOpenAIChoice :: OpenAIChoice
defaultOpenAIChoice = OpenAIChoice
  { message = defaultOpenAIMessage
  }

-- Helper: Convert OpenAI tool call to generic ToolCall
convertToolCall :: OpenAIToolCall -> ToolCall
convertToolCall tc =
  case Aeson.eitherDecodeStrict (TE.encodeUtf8 argsText) of
    Right params -> ToolCall
      (callId tc)
      (toolFunctionName (toolFunction tc))
      params
    Left err -> InvalidToolCall
      (callId tc)
      (toolFunctionName (toolFunction tc))
      argsText  -- Preserve original invalid string
      ("Malformed JSON in tool call arguments: " <> Text.pack err)
  where
    argsText = toolFunctionArguments (toolFunction tc)

convertFromToolCall :: ToolCall -> OpenAIToolCall
convertFromToolCall (ToolCall tcId tcName tcParams) = OpenAIToolCall
  { callId = tcId
  , toolCallType = "function"
  , toolFunction = OpenAIToolFunction
      { toolFunctionName = tcName
      , toolFunctionArguments = TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ tcParams
      }
  }
convertFromToolCall (InvalidToolCall tcId tcName _tcArgs _err) = OpenAIToolCall
  { callId = tcId
  , toolCallType = "function"
  , toolFunction = OpenAIToolFunction
      { toolFunctionName = tcName
      -- Use empty JSON object instead of malformed string to avoid breaking chat templates
      , toolFunctionArguments = "{}"
      }
  }

-- ============================================================================
-- Streaming Helper
-- ============================================================================

-- | Enable streaming for OpenAI requests
-- Only callable when the model has streaming support (constrained by HasStreaming)
enableOpenAIStreaming :: OpenAIRequest -> OpenAIRequest
enableOpenAIStreaming req = req { stream = Just True }

-- ============================================================================
-- OpenAI Completion API (Legacy /v1/completions endpoint)
-- ============================================================================

data OpenAICompletionRequest = OpenAICompletionRequest
  { completionModel :: Text
  , prompt :: Text
  , completionTemperature :: Maybe Double
  , completionMaxTokens :: Maybe Int
  , stop :: Maybe [Text]
  } deriving (Generic, Show, Eq)

instance Semigroup OpenAICompletionRequest where
  r1 <> r2 = OpenAICompletionRequest
    { completionModel = completionModel r2  -- Right-biased for scalar fields
    , prompt = prompt r2  -- Right-biased (last prompt wins)
    , completionTemperature = completionTemperature r2 <|> completionTemperature r1
    , completionMaxTokens = completionMaxTokens r2 <|> completionMaxTokens r1
    , stop = stop r1 <> stop r2  -- Concatenate stop sequences
    }

instance Monoid OpenAICompletionRequest where
  mempty = OpenAICompletionRequest
    { completionModel = ""
    , prompt = ""
    , completionTemperature = Nothing
    , completionMaxTokens = Nothing
    , stop = Nothing
    }

data OpenAICompletionChoice = OpenAICompletionChoice
  { completionText :: Text
  , completionIndex :: Int
  , completionFinishReason :: Maybe Text
  } deriving (Generic, Show, Eq)

data OpenAICompletionResponse
  = OpenAICompletionSuccess OpenAICompletionSuccessResponse
  | OpenAICompletionError OpenAIErrorResponse  -- Reuse error type from chat
  deriving (Show, Eq)

data OpenAICompletionSuccessResponse = OpenAICompletionSuccessResponse
  { completionChoices :: [OpenAICompletionChoice]
  } deriving (Generic, Show, Eq)

instance HasCodec OpenAICompletionRequest where
  codec = object "OpenAICompletionRequest" $
    OpenAICompletionRequest
      <$> requiredField "model" "Model name" .= completionModel
      <*> requiredField "prompt" "Prompt text" .= prompt
      <*> optionalFieldWith "temperature" (dimapCodec realToFrac realToFrac scientificCodec) "Temperature" .= completionTemperature
      <*> optionalFieldWith "max_tokens" (dimapCodec fromIntegral fromIntegral integerCodec) "Max tokens" .= completionMaxTokens
      <*> optionalField "stop" "Stop sequences" .= stop

instance HasCodec OpenAICompletionChoice where
  codec = object "OpenAICompletionChoice" $
    OpenAICompletionChoice
      <$> requiredField "text" "Completion text" .= completionText
      <*> requiredField "index" "Choice index" .= completionIndex
      <*> optionalField "finish_reason" "Finish reason" .= completionFinishReason

instance HasCodec OpenAICompletionResponse where
  codec = dimapCodec fromEither toEither $ eitherCodec (codec @OpenAICompletionSuccessResponse) (codec @OpenAIErrorResponse)
    where
      fromEither (Left success) = OpenAICompletionSuccess success
      fromEither (Right err) = OpenAICompletionError err

      toEither (OpenAICompletionSuccess success) = Left success
      toEither (OpenAICompletionError err) = Right err

instance HasCodec OpenAICompletionSuccessResponse where
  codec = object "OpenAICompletionSuccessResponse" $
    OpenAICompletionSuccessResponse
      <$> requiredField "choices" "Completion choices" .= completionChoices
-- ============================================================================
-- Streaming Support
-- ============================================================================

-- | Streaming content for real-time display during an OpenAI stream.
data OpenAIStreamingContent
  = OpenAIStreamingText      Text  -- ^ Increment to @content@
  | OpenAIStreamingReasoning Text  -- ^ Increment to @reasoning_content@ / @reasoning@
  deriving (Show, Eq)

-- | Extract display chunks from a delta by inspecting OpenAI-specific field names.
--
-- Descends into @choices[].delta@ and collects non-empty, non-null values from
-- @reasoning_content@ (or @reasoning@) and @content@, in that order.
extractOpenAIStreamingContent :: Delta -> [OpenAIStreamingContent]
extractOpenAIStreamingContent (Delta val) =
  case val of
    Aeson.Object obj ->
      case KM.lookup "choices" obj of
        Just (Aeson.Array choices) -> concatMap fromChoice (V.toList choices)
        _                          -> []
    _ -> []
  where
    fromChoice (Aeson.Object choice) =
      case KM.lookup "delta" choice of
        Just (Aeson.Object delta) ->
          [ OpenAIStreamingReasoning t
          | Just t <- [nonEmpty "reasoning_content" delta
                       `orElse` nonEmpty "reasoning" delta]
          ] ++
          [ OpenAIStreamingText t
          | Just t <- [nonEmpty "content" delta]
          ]
        _ -> []
    fromChoice _ = []

    nonEmpty k m = case KM.lookup k m of
      Just (Aeson.String t) | not (Text.null t) -> Just t
      _                                          -> Nothing

    orElse (Just x) _ = Just x
    orElse Nothing  y = y

-- | Merge an OpenAI streaming chunk (as a raw 'Value') into the accumulated
-- response.
--
-- Streaming chunks use a @delta@ key inside @choices[]@ elements, while the
-- non-streaming codec expects a @message@ key.  We normalise each incoming
-- chunk by renaming @delta@ → @message@ so the accumulated 'Value' always
-- matches the codec's expected shape.
mergeOpenAIDelta :: OpenAIResponse -> Value -> OpenAIResponse
mergeOpenAIDelta acc chunk =
    let normalised = normaliseDeltaKey chunk
        accValue   = toJSONViaCodec acc
        merged     = applyDelta accValue (Delta normalised)
    in case parseEither parseJSONViaCodec merged of
        Right r -> normaliseResponse r
        Left _  -> acc  -- malformed accumulated state, keep as-is
  where
    -- Rename "delta" → "message" inside every element of choices[].
    normaliseDeltaKey :: Value -> Value
    normaliseDeltaKey (Aeson.Object obj) =
      Aeson.Object $ case KM.lookup "choices" obj of
        Just (Aeson.Array cs) ->
          KM.insert "choices" (Aeson.Array (V.map renameChoice cs)) obj
        _ -> obj
    normaliseDeltaKey v = v

    renameChoice :: Value -> Value
    renameChoice (Aeson.Object o) =
      case KM.lookup "delta" o of
        Just deltaVal ->
          Aeson.Object $ KM.insert "message" deltaVal $ KM.delete "delta" o
        Nothing -> Aeson.Object o
    renameChoice v = v

    -- OpenAI always includes a content field (even empty string) in the final
    -- non-streaming response. Streaming deltas never send content: "" so we
    -- normalise Nothing → Just "" after accumulation to match that invariant.
    normaliseResponse :: OpenAIResponse -> OpenAIResponse
    normaliseResponse (OpenAISuccess (OpenAISuccessResponse cs)) =
      OpenAISuccess (OpenAISuccessResponse (map normaliseChoice cs))
    normaliseResponse r = r

    normaliseChoice :: OpenAIChoice -> OpenAIChoice
    normaliseChoice (OpenAIChoice msg) =
      OpenAIChoice msg { content = content msg <|> Just "" }
