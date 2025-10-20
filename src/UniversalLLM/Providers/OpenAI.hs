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
instance (ModelName OpenAI model, ProtocolHandleTools OpenAIToolCall model OpenAI)
         => Provider OpenAI model where
  type ProviderRequest OpenAI = OpenAIRequest
  type ProviderResponse OpenAI = OpenAIResponse

  toRequest _provider mdl configs msgs =
    let baseRequest = OpenAIRequest
          { model = modelName @OpenAI mdl
          , messages = map convertMessage msgs
          , temperature = Nothing
          , max_tokens = Nothing
          , seed = Nothing
          , tools = Nothing
          }
    in applyConfig configs baseRequest

  fromResponse = fromResponse'

fromResponse' :: forall model. ProtocolHandleTools OpenAIToolCall model OpenAI => OpenAIResponse -> Either LLMError [Message model OpenAI]
fromResponse' (OpenAIError (OpenAIErrorResponse errDetail)) =
  Left $ ProviderError (code errDetail) $ errorMessage errDetail <> " (" <> errorType errDetail <> ")"
fromResponse' (OpenAISuccess (OpenAISuccessResponse chcs)) =
  case chcs of
    (choice:_) -> case (content msg, tool_calls msg) of
      (Just txt, Nothing) -> Right [AssistantText txt]
      (_, Just calls) -> Right $ handleToolCalls @OpenAIToolCall @model @OpenAI calls
      (Nothing, Nothing) -> Left $ ParseError "No content or tool calls in response"
      where msg = message choice
    [] -> Left $ ParseError "No choices returned in OpenAI response"

convertMessage :: Message model OpenAI -> OpenAIMessage
convertMessage (UserText text) = OpenAIMessage "user" (Just text) Nothing Nothing
convertMessage (UserImage text _imageData) = OpenAIMessage "user" (Just text) Nothing Nothing -- simplified
convertMessage (AssistantText text) = OpenAIMessage "assistant" (Just text) Nothing Nothing
convertMessage (AssistantTool call) = OpenAIMessage "assistant" Nothing (Just [convertFromToolCall call]) Nothing
convertMessage (SystemText text) = OpenAIMessage "system" (Just text) Nothing Nothing
convertMessage (ToolResultMsg result) =
  OpenAIMessage "tool" (Just resultContent) Nothing (Just resultCallId)
  where
    resultCallId = getToolCallId (toolResultCall result)
    resultContent = either id (TE.decodeUtf8 . BSL.toStrict . Aeson.encode)
                  $ toolResultOutput result