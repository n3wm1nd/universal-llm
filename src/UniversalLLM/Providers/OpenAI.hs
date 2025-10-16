{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}

module UniversalLLM.Providers.OpenAI where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import Data.Text (Text)
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL

-- OpenAI provider (phantom type)
data OpenAI = OpenAI deriving (Show, Eq)

-- Declare that OpenAI supports tools
instance ProviderSupportsTools OpenAI

-- Provider typeclass implementation
-- Uses protocol-level ProtocolEmbedTools for tool embedding
-- Uses generic ProtocolHandleTools for parsing tool calls from responses
instance (ModelName OpenAI model, Temperature model OpenAI, MaxTokens model OpenAI, Seed model OpenAI, ProtocolEmbedTools OpenAIRequest model, ProtocolHandleTools OpenAIToolCall model OpenAI)
         => Provider OpenAI model where
  type ProviderRequest OpenAI = OpenAIRequest
  type ProviderResponse OpenAI = OpenAIResponse

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

  fromResponse = fromResponse'

-- Generic tool call handling for any protocol/provider combination
class ProtocolHandleTools protocolToolCall model provider where
  handleToolCalls :: [protocolToolCall] -> [Message model provider]
  handleToolCalls _ = []  -- Default: ignore (shouldn't happen for non-tool models)

instance (ModelHasTools model, ProviderSupportsTools provider) => ProtocolHandleTools OpenAIToolCall model provider where
  handleToolCalls calls = [AssistantTool (map convertToolCall calls)]

fromResponse' :: forall model. ProtocolHandleTools OpenAIToolCall model OpenAI => OpenAIResponse -> Either LLMError [Message model OpenAI]
fromResponse' (OpenAIError (OpenAIErrorResponse errDetail)) =
  Left $ ProviderError (code errDetail) $ errorMessage errDetail <> " (" <> errorType errDetail <> ")"
fromResponse' (OpenAISuccess (OpenAISuccessResponse choices)) = case choices of
  (choice:_) ->
    let msg = message choice
    in case (content msg, tool_calls msg) of
      (Just txt, Nothing) -> Right [AssistantText txt]
      (_, Just calls) -> Right $ handleToolCalls @OpenAIToolCall @model @OpenAI calls
      (Nothing, Nothing) -> Left $ ParseError "No content or tool calls in response"
  [] -> Left $ ParseError "No choices returned in OpenAI response"

convertMessage :: Message model OpenAI -> OpenAIMessage
convertMessage (UserText text) = OpenAIMessage "user" (Just text) Nothing Nothing
convertMessage (UserImage text _imageData) = OpenAIMessage "user" (Just text) Nothing Nothing -- simplified
convertMessage (AssistantText text) = OpenAIMessage "assistant" (Just text) Nothing Nothing
convertMessage (AssistantTool calls) = OpenAIMessage "assistant" Nothing (Just $ map convertFromToolCall calls) Nothing
convertMessage (SystemText text) = OpenAIMessage "system" (Just text) Nothing Nothing
convertMessage (ToolResultMsg result) = OpenAIMessage "tool" (Just $ encodeValue $ toolResultOutput result) Nothing (Just $ toolResultCallId result)
  where
    encodeValue :: Aeson.Value -> Text
    encodeValue = TE.decodeUtf8 . BSL.toStrict . Aeson.encode