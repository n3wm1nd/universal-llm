{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module UniversalLLM.Protocols.Anthropic where

import Autodocodec
import Data.Text (Text)
import GHC.Generics (Generic)

data AnthropicRequest = AnthropicRequest
  { model :: Text
  , messages :: [AnthropicMessage]
  , max_tokens :: Int
  , temperature :: Maybe Double
  , system :: Maybe Text
  } deriving (Generic, Show, Eq)

data AnthropicMessage = AnthropicMessage
  { role :: Text
  , content :: Text
  } deriving (Generic, Show, Eq)

data AnthropicResponse = AnthropicResponse
  { responseContent :: [AnthropicContent]
  } deriving (Generic, Show, Eq)

data AnthropicContent = AnthropicContent
  { contentType :: Text
  , contentText :: Text
  } deriving (Generic, Show, Eq)

instance HasCodec AnthropicRequest where
  codec = object "AnthropicRequest" $
    AnthropicRequest
      <$> requiredField "model" "Model name" .= model
      <*> requiredField "messages" "Messages" .= messages
      <*> requiredField "max_tokens" "Max tokens" .= max_tokens
      <*> optionalFieldWith "temperature" (dimapCodec realToFrac realToFrac scientificCodec) "Temperature" .= temperature
      <*> optionalField "system" "System prompt" .= system

instance HasCodec AnthropicMessage where
  codec = object "AnthropicMessage" $
    AnthropicMessage
      <$> requiredField "role" "Role" .= role
      <*> requiredField "content" "Content" .= content

instance HasCodec AnthropicResponse where
  codec = object "AnthropicResponse" $
    AnthropicResponse
      <$> requiredField "content" "Content" .= responseContent

instance HasCodec AnthropicContent where
  codec = object "AnthropicContent" $
    AnthropicContent
      <$> requiredField "type" "Content type" .= contentType
      <*> requiredField "text" "Text content" .= contentText