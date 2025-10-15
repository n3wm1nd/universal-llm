{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

module Main where

import UniversalLLM
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import System.Environment (lookupEnv)
import System.Exit (exitFailure)
import Data.Text (Text)
import qualified Data.Text as T
import Autodocodec (toJSONViaCodec, eitherDecodeJSONViaCodec, HasCodec, codec, object, optionalField, (.=))
import qualified Autodocodec
import Autodocodec.Schema (jsonSchemaViaCodec)
import qualified Data.Aeson as Aeson
import Network.HTTP.Simple
import Control.Monad (unless)
import Control.Monad.Trans.Except (ExceptT(..), runExceptT, except, withExceptT)
import Control.Monad.IO.Class (liftIO, MonadIO)
import Data.Time (getCurrentTime, formatTime, defaultTimeLocale)

-- Simple model configuration
data MistralModel = MistralModel
  { mistralTemperature :: Maybe Double
  , mistralMaxTokens :: Maybe Int
  , mistralSeed :: Maybe Int
  } deriving (Show, Eq)

instance HasTools MistralModel
instance Temperature MistralModel provider where getTemperature = mistralTemperature
instance MaxTokens MistralModel provider where getMaxTokens = mistralMaxTokens
instance Seed MistralModel provider where getSeed = mistralSeed
instance ModelName OpenAI MistralModel where modelName = "mistral-7b-instruct"

-- Tool result type
data TimeResponse = TimeResponse
  { currentTime :: Text
  } deriving (Show, Eq)

instance Autodocodec.HasCodec TimeResponse where
  codec = object "TimeResponse" $
    TimeResponse <$> Autodocodec.requiredField "current_time" "Current time" .= currentTime

-- Tool parameters type
data GetTimeParams = GetTimeParams
  { timezone :: Maybe Text
  } deriving (Show, Eq)

instance Autodocodec.HasCodec GetTimeParams where
  codec = object "GetTimeParams" $
    GetTimeParams <$> optionalField "timezone" "Timezone" .= timezone

-- Tool type that carries its implementation
data GetTime m = GetTime
  { getTimeImpl :: GetTimeParams -> m TimeResponse
  }

-- Eq instance (tools are equal if they have the same name, ignore the function)
instance Eq (GetTime m) where
  _ == _ = True

instance Monad m => Tool (GetTime m) m where
  type ToolParams (GetTime m) = GetTimeParams
  type ToolOutput (GetTime m) = TimeResponse
  toolName _ = "get_time"
  toolDescription _ = "Get the current time"
  call (GetTime impl) params = impl params

-- Tool value with actual implementation
getTimeTool :: MonadIO m => GetTime m
getTimeTool = GetTime $ \params -> do
  now <- liftIO getCurrentTime
  let timeStr = T.pack $ formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S UTC" now
  liftIO $ putStrLn $ "    ⏰ " <> T.unpack timeStr
  return $ TimeResponse timeStr

-- HTTP call to llama.cpp server
callLLM :: String -> OpenAIRequest -> ExceptT LLMError IO OpenAIResponse
callLLM baseUrl request = do
  req <- liftIO $ parseRequest $ "POST " ++ baseUrl ++ "/v1/chat/completions"
  let req' = setRequestHeaders [("Content-Type", "application/json")]
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request) req

  response <- httpLBS req'
  withExceptT (ParseError . T.pack) $ except $ eitherDecodeJSONViaCodec (getResponseBody response)

-- Main agent loop - continues until text response (non-tool)
agentLoop :: String -> MistralModel -> [Message MistralModel OpenAI] -> ExceptT LLMError IO ()
agentLoop serverUrl model messages = do
  -- Available tools
  let tools :: [SomeTool IO]
      tools = [SomeTool (getTimeTool @IO)]

  -- Call LLM with tools
  let request = toRequest OpenAI model messages tools
  response <- callLLM serverUrl request
  responses <- except $ fromResponse response

  newMsgs <- liftIO $ concat <$> mapM (handleResponse tools) responses
  -- Continue loop only if there are tool results to send back
  unless (null newMsgs) $
    agentLoop serverUrl model (messages ++ responses ++ newMsgs)

-- Handle different response types
-- Returns empty list for text (stops loop), tool results for tool calls (continues loop)
handleResponse :: [SomeTool IO] -> Message MistralModel OpenAI -> IO [Message MistralModel OpenAI]
handleResponse _ (AssistantText text) = do
  putStrLn $ "🤖 " <> T.unpack text
  return [] -- Empty list stops the loop

handleResponse tools (AssistantTool calls) = do
  putStrLn $ "🔧 Calling " <> show (length calls) <> " tool(s)"
  results <- mapM (executeCall tools) calls
  return $ zipWith (\c r -> ToolResultMsg $ ToolResult (toolCallId c) r) calls results -- Tool results continue loop

handleResponse _ _ = return []

-- Execute a single tool call with logging
executeCall :: [SomeTool IO] -> ToolCall -> IO Aeson.Value
executeCall tools call = do
  putStrLn $ "  → " <> T.unpack (toolCallName call)
  result <- executeToolCall tools call
  case result of
    Nothing -> do
      putStrLn $ "    ❌ Tool not found or invalid parameters"
      return $ Aeson.object ["error" Aeson..= ("Tool execution failed" :: Text)]
    Just value -> return value

main :: IO ()
main = do
  serverUrl <- lookupEnv "LLAMA_SERVER_URL" >>= \case
    Nothing -> putStrLn "Set LLAMA_SERVER_URL environment variable" >> exitFailure
    Just url -> return url

  putStrLn $ "Using server: " <> serverUrl
  putStrLn "=== Tool Calling Demo ===\n"

  let model = MistralModel (Just 0.7) (Just 200) Nothing
  let initialMsg = [UserText "Use the get_time tool to tell me what time it is."]

  result <- runExceptT $ agentLoop serverUrl model initialMsg
  case result of
    Left err -> putStrLn $ "❌ Error: " <> show err
    Right () -> return ()
