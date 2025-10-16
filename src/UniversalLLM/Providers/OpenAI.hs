{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}

module UniversalLLM.Providers.OpenAI where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI (OpenAIRequest(..), OpenAIResponse(..), OpenAISuccessResponse(..), OpenAIErrorResponse(..), OpenAIErrorDetail(..), OpenAIMessage(..), OpenAIChoice(..), OpenAIToolDefinition(..), OpenAIToolCall(..), OpenAIToolFunction(..), OpenAIFunction(..))
import Data.Text (Text)
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL

-- OpenAI provider (phantom type)
data OpenAI = OpenAI deriving (Show, Eq)

-- Typeclass to optionally embed tools in OpenAI requests
class OpenAIEmbedTools model where
  embedTools :: model -> OpenAIRequest -> OpenAIRequest
  -- Default: no tools
  embedTools _ req = req { tools = Nothing }

-- Specific instance: embed tools (for models with HasTools)
instance HasTools model => OpenAIEmbedTools model where
  embedTools model req =
    let toolDefs = getToolDefinitions model
    in req { tools = Just (map toOpenAIToolDef toolDefs) }

-- Pure functions: no IO, just transformations
toRequest :: forall model. (ModelName OpenAI model, Temperature model OpenAI, MaxTokens model OpenAI, Seed model OpenAI, OpenAIEmbedTools model)
          => OpenAI -> model -> [Message model OpenAI] -> OpenAIRequest
toRequest _provider model messages =
  let baseRequest = OpenAIRequest
        { model = modelName @OpenAI @model
        , messages = map convertMessage messages
        , temperature = getTemperature @model @OpenAI model
        , max_tokens = getMaxTokens @model @OpenAI model
        , seed = getSeed @model @OpenAI model
        , tools = Nothing
        }
  in embedTools model baseRequest

toOpenAIToolDef :: ToolDefinition -> OpenAIToolDefinition
toOpenAIToolDef toolDef = OpenAIToolDefinition
  { tool_type = "function"
  , function = OpenAIFunction
      { name = toolDefName toolDef
      , description = toolDefDescription toolDef
      , parameters = toolDefParameters toolDef
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