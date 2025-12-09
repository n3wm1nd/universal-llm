{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}

module Proxy.OpenAICompat
  ( parseOpenAIRequest
  , buildOpenAIResponse
  , ProxyConfig(..)
  ) where

import UniversalLLM
import UniversalLLM.Protocols.OpenAI
import Data.Text (Text)

-- | Configuration extracted from an OpenAI request
-- This is the intermediate universal representation
data ProxyConfig m = ProxyConfig
  { proxyMessages :: [Message m]
  , proxyConfigs :: [ModelConfig m]
  }

-- ============================================================================
-- OpenAI Request -> Universal Messages + Config
-- ============================================================================

-- | Parse an OpenAI request into universal message format
-- This is the key translation layer: OpenAI wire format -> Universal format
parseOpenAIRequest :: forall m.
                      (SupportsTemperature (ProviderOf m),
                       SupportsMaxTokens (ProviderOf m))
                   => OpenAIRequest
                   -> Either Text (ProxyConfig m)
parseOpenAIRequest req = do
  msgs <- mapM parseOpenAIMessage (messages req)
  let configs = extractConfigs req
  return $ ProxyConfig
    { proxyMessages = msgs
    , proxyConfigs = configs
    }

-- | Extract configuration from OpenAI request
extractConfigs :: forall m.
                  (SupportsTemperature (ProviderOf m), SupportsMaxTokens (ProviderOf m))
               => OpenAIRequest
               -> [ModelConfig m]
extractConfigs req = concat
  [ maybe [] (\t -> [Temperature t]) (temperature req)
  , maybe [] (\m -> [MaxTokens m]) (max_tokens req)
  -- TODO: Add tools, system prompt, etc. as we expand support
  ]

-- | Parse a single OpenAI message into universal format
parseOpenAIMessage :: forall m.
                      OpenAIMessage
                   -> Either Text (Message m)
parseOpenAIMessage oaiMsg =
  case role oaiMsg of
    "user" -> case content oaiMsg of
      Just txt -> Right $ UserText txt
      Nothing -> Left "User message missing content"

    "assistant" -> case content oaiMsg of
      Just txt -> Right $ AssistantText txt
      Nothing -> Left "Assistant message missing content"

    "system" -> case content oaiMsg of
      Just txt -> Right $ SystemText txt
      Nothing -> Left "System message missing content"

    r -> Left $ "Unknown role: " <> r

-- ============================================================================
-- Universal Messages -> OpenAI Response
-- ============================================================================

-- | Build an OpenAI response from universal messages
-- This is the reverse translation: Universal format -> OpenAI wire format
buildOpenAIResponse :: forall m.
                       [Message m]
                    -> Either Text OpenAIResponse
buildOpenAIResponse [] =
  Left "No messages to convert to response"

buildOpenAIResponse msgs = do
  -- Take the last assistant message as the response
  -- In a real implementation, we'd accumulate all response messages
  let lastMsg = last msgs
  oaiMsg <- messageToOpenAI lastMsg
  return $ OpenAISuccess $ OpenAISuccessResponse
    { choices = [OpenAIChoice { message = oaiMsg }]
    }

-- | Convert a universal message to OpenAI format
messageToOpenAI :: forall m.
                   Message m
                -> Either Text OpenAIMessage
messageToOpenAI (UserText txt) = Right $ OpenAIMessage
  { role = "user"
  , content = Just txt
  , reasoning_content = Nothing
  , tool_calls = Nothing
  , tool_call_id = Nothing
  }

messageToOpenAI (AssistantText txt) = Right $ OpenAIMessage
  { role = "assistant"
  , content = Just txt
  , reasoning_content = Nothing
  , tool_calls = Nothing
  , tool_call_id = Nothing
  }

messageToOpenAI (SystemText txt) = Right $ OpenAIMessage
  { role = "system"
  , content = Just txt
  , reasoning_content = Nothing
  , tool_calls = Nothing
  , tool_call_id = Nothing
  }

messageToOpenAI _ =
  Left "Unsupported message type for OpenAI response (tools/vision not yet implemented)"
