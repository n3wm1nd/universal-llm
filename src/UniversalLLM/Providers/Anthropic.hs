{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module UniversalLLM.Providers.Anthropic where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.Anthropic
import Data.Text (Text)

-- Anthropic provider (phantom type)
data Anthropic = Anthropic deriving (Show, Eq)

-- Pure functions: no IO, just transformations
toRequest :: forall model. (ModelName Anthropic model, Temperature model Anthropic, MaxTokens model Anthropic, SystemPrompt model Anthropic)
          => Anthropic -> model -> [Message model Anthropic] -> AnthropicRequest
toRequest _provider model messages =
  let (systemMsg, otherMsgs) = extractSystem messages
      systemFromModel = getSystemPrompt @model @Anthropic model
      finalSystem = systemMsg <> systemFromModel
  in AnthropicRequest
    { model = modelName @Anthropic @model
    , messages = map convertMessage otherMsgs
    , max_tokens = maybe 1000 id (getMaxTokens @model @Anthropic model)
    , temperature = getTemperature @model @Anthropic model
    , system = finalSystem
    }

fromResponse :: AnthropicResponse -> Either LLMError [Message model Anthropic]
fromResponse resp = case responseContent resp of
  (c:_) -> Right [AssistantText (contentText c)]
  [] -> Left $ ParseError "No content returned in Anthropic response"

extractSystem :: [Message model Anthropic] -> (Maybe Text, [Message model Anthropic])
extractSystem (SystemText txt : rest) = (Just txt, rest)
extractSystem msgs = (Nothing, msgs)

convertMessage :: Message model Anthropic -> AnthropicMessage
convertMessage (UserText text) = AnthropicMessage "user" text
convertMessage (UserImage text _imageData) = AnthropicMessage "user" text -- simplified
convertMessage (AssistantText text) = AnthropicMessage "assistant" text
convertMessage (SystemText _) = error "System messages should be extracted"