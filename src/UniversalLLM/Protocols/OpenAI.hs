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
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BSL
import GHC.Generics (Generic)
import Control.Applicative ((<|>))
import UniversalLLM.Core.Types

data OpenAIRequest = OpenAIRequest
  { model :: Text
  , messages :: [OpenAIMessage]
  , temperature :: Maybe Double
  , max_tokens :: Maybe Int
  , seed :: Maybe Int
  , tools :: Maybe [OpenAIToolDefinition]
  , response_format :: Maybe OpenAIResponseFormat
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
convertFromToolCall (InvalidToolCall tcId tcName tcArgs _err) = OpenAIToolCall
  { callId = tcId
  , toolCallType = "function"
  , toolFunction = OpenAIToolFunction
      { toolFunctionName = tcName
      , toolFunctionArguments = tcArgs  -- Return original invalid arguments unchanged
      }
  }

-- Instance for handling OpenAI tool calls
instance (HasTools model, HasTools provider) => ProtocolHandleTools OpenAIToolCall model provider where
  handleToolCalls calls = map (AssistantTool . convertToolCall) calls


instance (HasJSON model, HasJSON provider) => ProtocolHandleJSON' 'True OpenAIMessage model provider  where
  handleJSONResponse' :: Value -> Message model provider
  handleJSONResponse' = AssistantJSON
-- Instance for handling JSON responses (for JSON-capable models)
instance (ProtocolHandleJSON' flag OpenAIMessage model provider, HasJSONPred model provider flag) => ProtocolHandleJSON OpenAIMessage model provider where
  handleJSONResponse :: Value -> Message model provider
  handleJSONResponse = handleJSONResponse' @flag @OpenAIMessage

-- Instance for handling reasoning (default: no-op)
instance ProtocolHandleReasoning OpenAIMessage model provider where
  handleReasoning _ = []