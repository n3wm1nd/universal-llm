{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module UniversalLLM.Core.Serialization
  ( -- * Serialization helpers for ComposableProvider
    serializeBaseMessage
  , deserializeBaseMessage
  , serializeToolMessages
  , deserializeToolMessages
  , serializeVisionMessages
  , deserializeVisionMessages
  , serializeJSONMessages
  , deserializeJSONMessages
  , serializeReasoningMessages
  , deserializeReasoningMessages

  -- * Lower-level helpers
  , serializeToolCall
  , deserializeToolCall
  , serializeToolResult
  , deserializeToolResult
  ) where

import Data.Text (Text)
import Data.Aeson (Value, (.=), (.:), (.:?))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Types as Aeson

import UniversalLLM.Core.Types
  ( Message(..)
  , ToolCall(..)
  , ToolResult(..)
  , HasTools
  , HasVision
  , HasJSON
  , HasReasoning
  )

-- ============================================================================
-- Base Messages (no capability constraints)
-- ============================================================================

-- | Serialize base messages (UserText, AssistantText, SystemText)
serializeBaseMessage :: Message m -> Maybe Value
serializeBaseMessage (UserText txt) = Just $ Aeson.object
  [ "type" .= ("UserText" :: Text)
  , "text" .= txt
  ]
serializeBaseMessage (AssistantText txt) = Just $ Aeson.object
  [ "type" .= ("AssistantText" :: Text)
  , "text" .= txt
  ]
serializeBaseMessage (SystemText txt) = Just $ Aeson.object
  [ "type" .= ("SystemText" :: Text)
  , "text" .= txt
  ]
serializeBaseMessage _ = Nothing  -- Not a base message

-- | Deserialize base messages (UserText, AssistantText, SystemText)
deserializeBaseMessage :: Value -> Maybe (Message m)
deserializeBaseMessage val = Aeson.parseMaybe parseObj val
  where
    parseObj = Aeson.withObject "Message" $ \obj -> do
      msgType <- obj .: "type" :: Aeson.Parser Text
      case msgType of
        "UserText" -> UserText <$> obj .: "text"
        "AssistantText" -> AssistantText <$> obj .: "text"
        "SystemText" -> SystemText <$> obj .: "text"
        _ -> fail "Not a base message type"

-- ============================================================================
-- Tool Messages (requires HasTools constraint)
-- ============================================================================

-- | Serialize tool call
serializeToolCall :: ToolCall -> Value
serializeToolCall (ToolCall tcId name args) = Aeson.object
  [ "id" .= tcId
  , "name" .= name
  , "arguments" .= args
  , "valid" .= True
  ]
serializeToolCall (InvalidToolCall tcId name rawArgs err) = Aeson.object
  [ "id" .= tcId
  , "name" .= name
  , "raw_arguments" .= rawArgs
  , "error" .= err
  , "valid" .= False
  ]

-- | Deserialize tool call
deserializeToolCall :: Value -> Maybe ToolCall
deserializeToolCall val = Aeson.parseMaybe parseObj val
  where
    parseObj = Aeson.withObject "ToolCall" $ \obj -> do
      tcId <- obj .: "id"
      name <- obj .: "name"
      valid <- obj .: "valid"
      if valid
        then do
          args <- obj .: "arguments"
          return $ ToolCall tcId name args
        else do
          rawArgs <- obj .: "raw_arguments"
          err <- obj .: "error"
          return $ InvalidToolCall tcId name rawArgs err

-- | Serialize tool result
serializeToolResult :: ToolResult -> Value
serializeToolResult (ToolResult tc output) = Aeson.object
  [ "call" .= serializeToolCall tc
  , "output" .= case output of
      Left err -> Aeson.object ["error" .= err]
      Right val -> Aeson.object ["value" .= val]
  ]

-- | Deserialize tool result
deserializeToolResult :: Value -> Maybe ToolResult
deserializeToolResult val = Aeson.parseMaybe parseObj val
  where
    parseObj = Aeson.withObject "ToolResult" $ \obj -> do
      call <- obj .: "call" >>= \v -> case deserializeToolCall v of
        Just tc -> return tc
        Nothing -> fail "Invalid tool call"
      output <- obj .: "output" >>= Aeson.withObject "ToolOutput" (\o -> do
        -- Try to parse as error first (optional field)
        mErr <- o .:? "error" :: Aeson.Parser (Maybe Text)
        case mErr of
          Just err -> return $ Left err
          Nothing -> do
            -- Parse as value
            v <- o .: "value"
            return $ Right v)
      return $ ToolResult call output

-- | Serialize tool messages (requires HasTools constraint)
serializeToolMessages :: HasTools m => Message m -> Maybe Value
serializeToolMessages (AssistantTool tc) = Just $ Aeson.object
  [ "type" .= ("AssistantTool" :: Text)
  , "tool_call" .= serializeToolCall tc
  ]
serializeToolMessages (ToolResultMsg tr) = Just $ Aeson.object
  [ "type" .= ("ToolResultMsg" :: Text)
  , "tool_result" .= serializeToolResult tr
  ]
serializeToolMessages _ = Nothing  -- Not a tool message

-- | Deserialize tool messages (requires HasTools constraint)
deserializeToolMessages :: HasTools m => Value -> Maybe (Message m)
deserializeToolMessages val = Aeson.parseMaybe parseObj val
  where
    parseObj = Aeson.withObject "Message" $ \obj -> do
      msgType <- obj .: "type" :: Aeson.Parser Text
      case msgType of
        "AssistantTool" -> do
          tcVal <- obj .: "tool_call"
          case deserializeToolCall tcVal of
            Just tc -> return $ AssistantTool tc
            Nothing -> fail "Invalid tool call"
        "ToolResultMsg" -> do
          trVal <- obj .: "tool_result"
          case deserializeToolResult trVal of
            Just tr -> return $ ToolResultMsg tr
            Nothing -> fail "Invalid tool result"
        _ -> fail "Not a tool message type"

-- ============================================================================
-- Vision Messages (requires HasVision constraint)
-- ============================================================================

-- | Serialize vision messages (requires HasVision constraint)
serializeVisionMessages :: HasVision m => Message m -> Maybe Value
serializeVisionMessages (UserImage mediaType imageData) = Just $ Aeson.object
  [ "type" .= ("UserImage" :: Text)
  , "media_type" .= mediaType
  , "data" .= imageData
  ]
serializeVisionMessages _ = Nothing

-- | Deserialize vision messages (requires HasVision constraint)
deserializeVisionMessages :: HasVision m => Value -> Maybe (Message m)
deserializeVisionMessages val = Aeson.parseMaybe parseObj val
  where
    parseObj = Aeson.withObject "Message" $ \obj -> do
      msgType <- obj .: "type" :: Aeson.Parser Text
      case msgType of
        "UserImage" -> do
          mediaType <- obj .: "media_type"
          imageData <- obj .: "data"
          return $ UserImage mediaType imageData
        _ -> fail "Not a vision message type"

-- ============================================================================
-- JSON Messages (requires HasJSON constraint)
-- ============================================================================

-- | Serialize JSON messages (requires HasJSON constraint)
serializeJSONMessages :: HasJSON m => Message m -> Maybe Value
serializeJSONMessages (UserRequestJSON query schema) = Just $ Aeson.object
  [ "type" .= ("UserRequestJSON" :: Text)
  , "query" .= query
  , "schema" .= schema
  ]
serializeJSONMessages (AssistantJSON val) = Just $ Aeson.object
  [ "type" .= ("AssistantJSON" :: Text)
  , "value" .= val
  ]
serializeJSONMessages _ = Nothing

-- | Deserialize JSON messages (requires HasJSON constraint)
deserializeJSONMessages :: HasJSON m => Value -> Maybe (Message m)
deserializeJSONMessages val = Aeson.parseMaybe parseObj val
  where
    parseObj = Aeson.withObject "Message" $ \obj -> do
      msgType <- obj .: "type" :: Aeson.Parser Text
      case msgType of
        "UserRequestJSON" -> do
          query <- obj .: "query"
          schema <- obj .: "schema"
          return $ UserRequestJSON query schema
        "AssistantJSON" -> do
          v <- obj .: "value"
          return $ AssistantJSON v
        _ -> fail "Not a JSON message type"

-- ============================================================================
-- Reasoning Messages (requires HasReasoning constraint)
-- ============================================================================

-- | Serialize reasoning messages (requires HasReasoning constraint)
serializeReasoningMessages :: HasReasoning m => Message m -> Maybe Value
serializeReasoningMessages (AssistantReasoning txt) = Just $ Aeson.object
  [ "type" .= ("AssistantReasoning" :: Text)
  , "text" .= txt
  ]
serializeReasoningMessages _ = Nothing

-- | Deserialize reasoning messages (requires HasReasoning constraint)
deserializeReasoningMessages :: HasReasoning m => Value -> Maybe (Message m)
deserializeReasoningMessages val = Aeson.parseMaybe parseObj val
  where
    parseObj = Aeson.withObject "Message" $ \obj -> do
      msgType <- obj .: "type" :: Aeson.Parser Text
      case msgType of
        "AssistantReasoning" -> AssistantReasoning <$> obj .: "text"
        _ -> fail "Not a reasoning message type"
