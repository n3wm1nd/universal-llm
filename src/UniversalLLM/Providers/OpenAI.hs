{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module UniversalLLM.Providers.OpenAI where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import Data.Text (Text)

-- OpenAI provider (phantom type)
data OpenAI = OpenAI deriving (Show, Eq)

-- Pure functions: no IO, just transformations
toRequest :: forall model. (ModelName OpenAI model, Temperature model OpenAI, MaxTokens model OpenAI, Seed model OpenAI)
          => OpenAI -> model -> [Message model OpenAI] -> OpenAIRequest
toRequest _provider model messages = OpenAIRequest
  { model = modelName @OpenAI @model
  , messages = map convertMessage messages
  , temperature = getTemperature @model @OpenAI model
  , max_tokens = getMaxTokens @model @OpenAI model
  , seed = getSeed @model @OpenAI model
  }

fromResponse :: OpenAIResponse -> Either LLMError [Message model OpenAI]
fromResponse resp = case choices resp of
  (choice:_) -> Right [AssistantText (content $ message choice)]
  [] -> Left $ ParseError "No choices returned in OpenAI response"

convertMessage :: Message model OpenAI -> OpenAIMessage
convertMessage (UserText text) = OpenAIMessage "user" text
convertMessage (UserImage text _imageData) = OpenAIMessage "user" text -- simplified
convertMessage (AssistantText text) = OpenAIMessage "assistant" text
convertMessage (SystemText text) = OpenAIMessage "system" text