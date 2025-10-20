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
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL

-- OpenAI provider (phantom type)
data OpenAI = OpenAI deriving (Show, Eq)

-- Declare OpenAI capabilities
instance HasTools OpenAI
instance HasJSON OpenAI
instance SupportsTemperature OpenAI
instance SupportsMaxTokens OpenAI
instance SupportsSeed OpenAI
instance SupportsSystemPrompt OpenAI

-- Apply configuration to OpenAI request
instance ApplyConfig OpenAIRequest OpenAI model where
  applyConfig configs req = foldr applyOne req configs
    where
      applyOne (Temperature t) r = r { temperature = Just t }
      applyOne (MaxTokens mt) r = r { max_tokens = Just mt }
      applyOne (Seed s) r = r { seed = Just s }
      applyOne (SystemPrompt sp) r =
        let sysMsg = OpenAIMessage "system" (Just sp) Nothing Nothing
        in r { messages = sysMsg : messages r }
      applyOne (Tools toolDefs) r = r { tools = Just (map toOpenAIToolDef toolDefs) }

-- Provider typeclass implementation
instance (ModelName OpenAI model, ProtocolHandleTools OpenAIToolCall model OpenAI, ProtocolHandleJSON model OpenAI)
         => Provider OpenAI model where
  type ProviderRequest OpenAI = OpenAIRequest
  type ProviderResponse OpenAI = OpenAIResponse

  toRequest _provider mdl configs msgs =
    let (oaiMessages, responseFormat) = convertMessages msgs
        baseRequest = OpenAIRequest
          { model = modelName @OpenAI mdl
          , messages = oaiMessages
          , temperature = Nothing
          , max_tokens = Nothing
          , seed = Nothing
          , tools = Nothing
          , response_format = responseFormat
          }
    in applyConfig configs baseRequest

  fromResponse = fromResponse'

-- Convert messages and extract JSON schema from the LAST message if it's UserRequestJSON
convertMessages :: [Message model OpenAI] -> ([OpenAIMessage], Maybe OpenAIResponseFormat)
convertMessages msgs =
  let oaiMsgs = map convertMessageOrJSON msgs
      lastSchema = case reverse msgs of
        (UserRequestJSON _ schema : _) -> Just $ OpenAIResponseFormat "json_schema" (Just schema)
        _ -> Nothing
  in (oaiMsgs, lastSchema)
  where
    convertMessageOrJSON (UserRequestJSON text _schema) = OpenAIMessage "user" (Just text) Nothing Nothing
    convertMessageOrJSON msg = convertMessage msg

fromResponse' :: forall model. (ProtocolHandleTools OpenAIToolCall model OpenAI, ProtocolHandleJSON model OpenAI) => OpenAIResponse -> Either LLMError [Message model OpenAI]
fromResponse' (OpenAIError (OpenAIErrorResponse errDetail)) =
  Left $ ProviderError (code errDetail) $ errorMessage errDetail <> " (" <> errorType errDetail <> ")"
fromResponse' (OpenAISuccess (OpenAISuccessResponse chcs)) =
  case chcs of
    (choice:_) -> case (content msg, tool_calls msg) of
      (Just txt, Nothing) -> parseContentAsTextOrJSON txt
      (_, Just calls) -> Right $ handleToolCalls @OpenAIToolCall @model @OpenAI calls
      (Nothing, Nothing) -> Left $ ParseError "No content or tool calls in response"
      where msg = message choice
    [] -> Left $ ParseError "No choices returned in OpenAI response"
  where
    -- Try to parse content as JSON, fall back to text
    parseContentAsTextOrJSON txt =
      case Aeson.eitherDecodeStrict (TE.encodeUtf8 txt) of
        Right jsonVal -> Right [handleJSONResponse @model @OpenAI jsonVal]
        Left _ -> Right [AssistantText txt]

convertMessage :: Message model OpenAI -> OpenAIMessage
convertMessage (UserText text) = OpenAIMessage "user" (Just text) Nothing Nothing
convertMessage (AssistantText text) = OpenAIMessage "assistant" (Just text) Nothing Nothing
convertMessage (AssistantTool call) = OpenAIMessage "assistant" Nothing (Just [convertFromToolCall call]) Nothing
convertMessage (AssistantJSON jsonVal) = OpenAIMessage "assistant" (Just jsonText) Nothing Nothing
  where jsonText = TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ jsonVal
convertMessage (SystemText text) = OpenAIMessage "system" (Just text) Nothing Nothing
convertMessage (ToolResultMsg result) =
  OpenAIMessage "tool" (Just resultContent) Nothing (Just resultCallId)
  where
    resultCallId = getToolCallId (toolResultCall result)
    resultContent = either id (TE.decodeUtf8 . BSL.toStrict . Aeson.encode)
                  $ toolResultOutput result
-- Catch-all for unsupported message types
convertMessage msg = error $ "Unsupported message type for OpenAI provider: " ++ show msg