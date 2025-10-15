{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}

module UniversalLLM.Providers.OpenAI where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI (OpenAIRequest(..), OpenAIResponse(..), OpenAISuccessResponse(..), OpenAIErrorResponse(..), OpenAIErrorDetail(..), OpenAIMessage(..), OpenAIChoice(..), OpenAIToolDefinition(..), OpenAIToolCall(..), OpenAIToolFunction(..), OpenAIFunction(..))
import qualified UniversalLLM.Protocols.OpenAI as OpenAIProtocol
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL
import Autodocodec.Schema (jsonSchemaViaCodec)
import Autodocodec (codec)

-- OpenAI provider (phantom type)
data OpenAI = OpenAI deriving (Show, Eq)

-- Pure functions: no IO, just transformations
toRequest :: forall model m. (ModelName OpenAI model, Temperature model OpenAI, MaxTokens model OpenAI, Seed model OpenAI)
          => OpenAI -> model -> [Message model OpenAI] -> [SomeTool m] -> OpenAIRequest
toRequest _provider model messages tools = OpenAIRequest
  { model = modelName @OpenAI @model
  , messages = map convertMessage messages
  , temperature = getTemperature @model @OpenAI model
  , max_tokens = getMaxTokens @model @OpenAI model
  , seed = getSeed @model @OpenAI model
  , tools = if null tools then Nothing else Just (map toOpenAIToolDef tools)
  }

toOpenAIToolDef :: forall m. SomeTool m -> OpenAIToolDefinition
toOpenAIToolDef (SomeTool tool) = case tool of
  (t :: t) -> OpenAIToolDefinition
    { tool_type = "function"
    , function = OpenAIFunction
        { name = toolName @t @m t
        , description = toolDescription @t @m t
        , parameters = Aeson.toJSON $ jsonSchemaViaCodec @(ToolParams t)
        }
    }

fromResponse :: HasTools model => OpenAIResponse -> Either LLMError [Message model OpenAI]
fromResponse (OpenAIError (OpenAIErrorResponse errDetail)) =
  Left $ ProviderError (code errDetail) $ errorMessage errDetail <> " (" <> errorType errDetail <> ")"
fromResponse (OpenAISuccess (OpenAISuccessResponse choices)) = case choices of
  (choice:_) ->
    let msg = message choice
    in case (content msg, tool_calls msg) of
      (Just txt, Nothing) -> Right [AssistantText txt]
      (_, Just calls) -> Right [AssistantTool (map convertToolCall calls)]
      (Nothing, Nothing) -> Left $ ParseError "No content or tool calls in response"
  [] -> Left $ ParseError "No choices returned in OpenAI response"

convertMessage :: Message model OpenAI -> OpenAIMessage
convertMessage (UserText text) = OpenAIMessage "user" (Just text) Nothing Nothing
convertMessage (UserImage text _imageData) = OpenAIMessage "user" (Just text) Nothing Nothing -- simplified
convertMessage (AssistantText text) = OpenAIMessage "assistant" (Just text) Nothing Nothing
convertMessage (AssistantTool calls) = OpenAIMessage "assistant" Nothing (Just $ map convertToToolCall calls) Nothing
convertMessage (SystemText text) = OpenAIMessage "system" (Just text) Nothing Nothing
convertMessage (ToolResultMsg result) = OpenAIMessage "tool" (Just $ encodeValue $ toolResultOutput result) Nothing (Just $ toolResultCallId result)
  where
    encodeValue :: Aeson.Value -> Text
    encodeValue = TE.decodeUtf8 . BSL.toStrict . Aeson.encode

convertToolCall :: OpenAIToolCall -> ToolCall
convertToolCall tc =
  let argsText = toolFunctionArguments (toolFunction tc)
      argsValue = case Aeson.eitherDecodeStrict (TE.encodeUtf8 argsText) of
        Left _ -> Aeson.object [] -- fallback to empty object on parse error
        Right v -> v
  in ToolCall
    { toolCallId = callId tc
    , toolCallName = toolFunctionName (toolFunction tc)
    , toolCallParameters = argsValue
    }

convertToToolCall :: ToolCall -> OpenAIToolCall
convertToToolCall tc = OpenAIToolCall
  { callId = toolCallId tc
  , toolCallType = "function"
  , toolFunction = OpenAIToolFunction
      { toolFunctionName = toolCallName tc
      , toolFunctionArguments = TE.decodeUtf8 $ BSL.toStrict $ Aeson.encode $ toolCallParameters tc
      }
  }