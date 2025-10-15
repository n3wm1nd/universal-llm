{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

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
import Control.Monad.IO.Class (liftIO)
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

-- Tool parameters type
data GetTimeParams = GetTimeParams
  { timezone :: Maybe Text
  } deriving (Show, Eq)

instance Autodocodec.HasCodec GetTimeParams where
  codec = object "GetTimeParams" $
    GetTimeParams <$> optionalField "timezone" "Timezone" .= timezone

-- Tool type (zero-sized, no config needed)
data GetTime = GetTime deriving (Show, Eq)

instance Tool GetTime where
  type ToolParams GetTime = GetTimeParams
  toolName _ = "get_time"
  toolDescription _ = "Get the current time"

-- Tool value
getTimeTool :: GetTime
getTimeTool = GetTime

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
  -- Call LLM with tools
  let request = toRequest OpenAI model messages [SomeTool getTimeTool]
  response <- callLLM serverUrl request
  responses <- except $ fromResponse response

  newMsgs <- liftIO $ concat <$> mapM handleResponse responses
  -- Continue loop only if there are tool results to send back
  unless (null newMsgs) $
    agentLoop serverUrl model (messages ++ responses ++ newMsgs)

-- Handle different response types
-- Returns empty list for text (stops loop), tool results for tool calls (continues loop)
handleResponse :: Message MistralModel OpenAI -> IO [Message MistralModel OpenAI]
handleResponse (AssistantText text) = do
  putStrLn $ "🤖 " <> T.unpack text
  return [] -- Empty list stops the loop

handleResponse (AssistantTool calls) = do
  putStrLn $ "🔧 Calling " <> show (length calls) <> " tool(s)"
  results <- mapM executeToolCall calls
  return $ zipWith (\c r -> ToolResultMsg $ ToolResult (toolCallId c) r) calls results -- Tool results continue loop

handleResponse _ = return []

-- Execute a tool call (simple dispatch with JSON Values)
executeToolCall :: ToolCall -> IO Aeson.Value
executeToolCall call = do
  putStrLn $ "  → " <> T.unpack (toolCallName call)
  case toolCallName call of
    "get_time" -> do
      now <- getCurrentTime
      let timeStr = formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S UTC" now
      putStrLn $ "    ⏰ " <> timeStr
      return $ Aeson.object ["current_time" Aeson..= timeStr]
    _ -> return $ Aeson.object ["error" Aeson..= ("Unknown tool" :: Text)]

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
