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

-- Declare OpenAI parameter support
instance ProviderSupportsTemperature OpenAI
instance ProviderSupportsMaxTokens OpenAI
instance ProviderSupportsSeed OpenAI
instance ProviderSupportsSystemPrompt OpenAI

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

  toRequest _provider model configs messages =
    let baseRequest = OpenAIRequest
          { model = modelName @OpenAI model
          , messages = map convertMessage messages
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
convertMessage (ToolResultMsg result) =
  let resultCallId = toolCallId (toolResultCall result)
      resultContent = case toolResultOutput result of
        Left errMsg -> errMsg  -- Error message as text
        Right value -> TE.decodeUtf8 $ BSL.toStrict $ Aeson.encode value  -- Success value as JSON
  in OpenAIMessage "tool" (Just resultContent) Nothing (Just resultCallId)