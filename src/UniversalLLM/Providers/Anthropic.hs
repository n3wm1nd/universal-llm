{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Providers.Anthropic where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.Anthropic
import Data.Text (Text)

-- Anthropic provider (phantom type)
data Anthropic = Anthropic deriving (Show, Eq)

-- Declare Anthropic parameter support
instance ProviderSupportsTemperature Anthropic
instance ProviderSupportsMaxTokens Anthropic
instance ProviderSupportsSystemPrompt Anthropic
-- Note: Anthropic does NOT support Seed

-- Apply configuration to Anthropic request
instance ApplyConfig AnthropicRequest Anthropic model where
  applyConfig configs req = foldr applyOne req configs
    where
      applyOne (Temperature t) r = r { temperature = Just t }
      applyOne (MaxTokens mt) r = r { max_tokens = mt }
      applyOne (SystemPrompt sp) r = r { system = Just sp }
      -- Tools, Seed not supported by Anthropic, but we need a catch-all to avoid warnings
      -- This pattern should never be reached due to GADT constraints, but GHC doesn't know that
      applyOne _ r = r

-- Pure functions: no IO, just transformations
toRequest :: forall model. ModelName Anthropic model
          => Anthropic -> model -> [ModelConfig Anthropic model] -> [Message model Anthropic] -> AnthropicRequest
toRequest _provider model configs messages =
  let (systemMsg, otherMsgs) = extractSystem messages
      baseRequest = AnthropicRequest
        { model = modelName @Anthropic model
        , messages = map convertMessage otherMsgs
        , max_tokens = 1000  -- default, will be overridden by config if present
        , temperature = Nothing
        , system = systemMsg
        }
  in applyConfig configs baseRequest

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