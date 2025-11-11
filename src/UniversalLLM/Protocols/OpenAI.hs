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
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module UniversalLLM.Protocols.OpenAI where

import Autodocodec
import Data.Text (Text)
import qualified Data.Text as Text
import Data.Aeson (Value)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Control.Applicative ((<|>))
import Data.Maybe (listToMaybe)
import qualified Data.List
import UniversalLLM.Core.Types

data OpenAIRequest = OpenAIRequest
  { model :: Text
  , messages :: [OpenAIMessage]
  , temperature :: Maybe Double
  , max_tokens :: Maybe Int
  , seed :: Maybe Int
  , tools :: Maybe [OpenAIToolDefinition]
  , response_format :: Maybe OpenAIResponseFormat
  , stream :: Maybe Bool
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
    }

data OpenAIResponseFormat = OpenAIResponseFormat
  { responseType :: Text
  , json_schema :: Maybe Value
  } deriving (Generic, Show, Eq)

data OpenAIMessage = OpenAIMessage
  { role :: Text
  , content :: Maybe Text
  , reasoning_content :: Maybe Text
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
  , errorType :: Text
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
      <$> requiredField "code" "Error code" .= code
      <*> requiredField "message" "Error message" .= errorMessage
      <*> requiredField "type" "Error type" .= errorType

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
-- Streaming Support - Delta Merger for SSE
-- ============================================================================

-- | Represents what's in a streaming delta
data DeltaContent
    = ContentDelta Text
    | ToolCallsDelta [(Int, OpenAIToolCall)]  -- List of (index, toolCall) pairs
    | EmptyDelta

-- | Merge OpenAI streaming delta into accumulated response
-- OpenAI streaming format sends incremental deltas in choices[].delta
-- We accumulate these deltas to reconstruct the final message
-- Handles both content deltas and tool_calls deltas
mergeOpenAIDelta :: OpenAIResponse -> Value -> OpenAIResponse
mergeOpenAIDelta acc chunk =
    case acc of
        OpenAISuccess (OpenAISuccessResponse choices) ->
            case extractDelta chunk of
                Just (ContentDelta txt) ->
                    OpenAISuccess $ OpenAISuccessResponse $ mergeContentDelta choices txt
                Just (ToolCallsDelta toolCallDeltas) ->
                    OpenAISuccess $ OpenAISuccessResponse $ mergeToolCallsDelta choices toolCallDeltas
                Just EmptyDelta ->
                    -- Empty delta (e.g., role only, or finish_reason), keep accumulator
                    acc
                Nothing ->
                    -- Can't parse delta, keep accumulator
                    acc
        _ -> acc  -- Error response, keep as-is
  where

    extractDelta :: Value -> Maybe DeltaContent
    extractDelta (Aeson.Object obj) = do
        Aeson.Array choicesArr <- KM.lookup "choices" obj
        Aeson.Object choice <- listToMaybe (V.toList choicesArr)
        Aeson.Object delta <- KM.lookup "delta" choice

        -- Try content first
        case KM.lookup "content" delta of
            Just (Aeson.String txt) -> return $ ContentDelta txt
            _ ->
                -- Try tool_calls
                case KM.lookup "tool_calls" delta of
                    Just (Aeson.Array toolCallsArr) -> do
                        -- Manually parse each tool call from the array
                        -- Each delta has an "index" field indicating which tool call it updates
                        let parsedCalls = [(idx, tc) | Aeson.Object tcObj <- V.toList toolCallsArr
                                                     , Just (idx, tc) <- [parseToolCallDelta tcObj]]
                        if null parsedCalls
                            then return EmptyDelta
                            else return $ ToolCallsDelta parsedCalls
                    _ -> return EmptyDelta  -- No content or tool_calls
    extractDelta _ = Nothing

    -- Parse a single tool call delta from JSON object
    -- Returns (index, toolCall) where index is used to match deltas together
    parseToolCallDelta :: KM.KeyMap Value -> Maybe (Int, OpenAIToolCall)
    parseToolCallDelta obj = do
        -- Get the index - this tells us which tool call in the list this delta belongs to
        Aeson.Number indexNum <- KM.lookup "index" obj
        let index = floor indexNum

        -- id might be present in first chunk, empty in subsequent chunks
        let tcId = case KM.lookup "id" obj of
                Just (Aeson.String s) -> s
                _ -> ""
        -- type is "function"
        let tcType = case KM.lookup "type" obj of
                Just (Aeson.String s) -> s
                _ -> "function"
        -- function object contains name and arguments (both might be partial)
        Aeson.Object funcObj <- KM.lookup "function" obj
        let funcName = case KM.lookup "name" funcObj of
                Just (Aeson.String s) -> s
                _ -> ""
        let funcArgs = case KM.lookup "arguments" funcObj of
                Just (Aeson.String s) -> s
                _ -> ""

        return (index, OpenAIToolCall tcId tcType (OpenAIToolFunction funcName funcArgs))

    mergeContentDelta :: [OpenAIChoice] -> Text -> [OpenAIChoice]
    mergeContentDelta [] txt =
        [OpenAIChoice (OpenAIMessage "assistant" (Just txt) Nothing Nothing Nothing)]
    mergeContentDelta [OpenAIChoice msg] txt =
        [OpenAIChoice (msg { content = Just $ maybe txt (<> txt) (content msg) })]
    mergeContentDelta xs _ = xs

    mergeToolCallsDelta :: [OpenAIChoice] -> [(Int, OpenAIToolCall)] -> [OpenAIChoice]
    mergeToolCallsDelta [] indexedCalls =
        -- Initialize with tool calls at their indices
        let initialCalls = buildToolCallList indexedCalls
        in [OpenAIChoice (OpenAIMessage "assistant" Nothing Nothing (Just initialCalls) Nothing)]
    mergeToolCallsDelta [OpenAIChoice msg] indexedCalls =
        [OpenAIChoice (msg { tool_calls = Just $ mergeToolCallsByIndex (tool_calls msg) indexedCalls })]
    mergeToolCallsDelta xs _ = xs

    -- Build initial tool call list from indexed deltas
    buildToolCallList :: [(Int, OpenAIToolCall)] -> [OpenAIToolCall]
    buildToolCallList indexedCalls =
        -- Sort by index and extract just the tool calls
        map snd $ Data.List.sortOn fst indexedCalls

    -- Merge tool calls by index
    -- OpenAI streaming sends tool calls with an "index" field to indicate which call to update
    mergeToolCallsByIndex :: Maybe [OpenAIToolCall] -> [(Int, OpenAIToolCall)] -> [OpenAIToolCall]
    mergeToolCallsByIndex Nothing indexedCalls = buildToolCallList indexedCalls
    mergeToolCallsByIndex (Just existing) indexedCalls =
        foldl (\acc (idx, newCall) -> mergeToolCallAtIndex acc idx newCall) existing indexedCalls

    -- Merge a single tool call at a specific index
    mergeToolCallAtIndex :: [OpenAIToolCall] -> Int -> OpenAIToolCall -> [OpenAIToolCall]
    mergeToolCallAtIndex existing idx newCall
        | idx >= length existing =
            -- Index beyond current list, append (with padding if necessary)
            existing ++ replicate (idx - length existing) emptyToolCall ++ [newCall]
        | otherwise =
            -- Update existing tool call at index
            let (before, at:after) = splitAt idx existing
                merged = mergeToolCallPair at newCall
            in before ++ [merged] ++ after

    -- Empty placeholder tool call
    emptyToolCall :: OpenAIToolCall
    emptyToolCall = OpenAIToolCall "" "function" (OpenAIToolFunction "" "")

    -- Merge two tool calls (accumulate fields)
    mergeToolCallPair :: OpenAIToolCall -> OpenAIToolCall -> OpenAIToolCall
    mergeToolCallPair existing new =
        OpenAIToolCall
            { callId = if Text.null (callId new) then callId existing else callId new
            , toolCallType = if Text.null (toolCallType new) then toolCallType existing else toolCallType new
            , toolFunction = OpenAIToolFunction
                { toolFunctionName =
                    if Text.null (toolFunctionName (toolFunction new))
                        then toolFunctionName (toolFunction existing)
                        else toolFunctionName (toolFunction new)
                , toolFunctionArguments =
                    -- Accumulate arguments
                    toolFunctionArguments (toolFunction existing) <>
                    toolFunctionArguments (toolFunction new)
                }
            }
