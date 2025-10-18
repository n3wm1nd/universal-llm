{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

module Main (main) where

import UniversalLLM
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import System.Environment (lookupEnv)
import System.Exit (exitFailure)
import Data.Text (Text)
import qualified Data.Text as T
import Autodocodec (toJSONViaCodec, eitherDecodeJSONViaCodec, HasCodec, codec, object, optionalField, (.=))
import qualified Autodocodec
import qualified Data.Aeson as Aeson
import Network.HTTP.Simple
import Control.Monad (unless)
import Control.Monad.Trans.Except (ExceptT(..), runExceptT, except, withExceptT)
import Control.Monad.IO.Class (liftIO, MonadIO)
import Data.Time (getCurrentTime, formatTime, defaultTimeLocale)

-- ============================================================================
-- Model Definition
-- ============================================================================

-- Simple model (just model identity)
data MistralModel = MistralModel deriving (Show, Eq)

instance ModelHasTools MistralModel

instance ModelName OpenAI MistralModel where
  modelName = "mistral-7b-instruct"

-- ============================================================================
-- Tool Definition
-- ============================================================================

-- Tool parameters type
data GetTimeParams = GetTimeParams
  { timezone :: Maybe Text
  } deriving (Show, Eq)

instance Autodocodec.HasCodec GetTimeParams where
  codec = object "GetTimeParams" $
    GetTimeParams <$> optionalField "timezone" "Timezone" .= timezone

-- Tool result type
data TimeResponse = TimeResponse
  { currentTime :: Text
  } deriving (Show, Eq)

instance Autodocodec.HasCodec TimeResponse where
  codec = object "TimeResponse" $
    TimeResponse <$> Autodocodec.requiredField "current_time" "Current time" .= currentTime

-- Tool type that carries its implementation
data GetTime m = GetTime (GetTimeParams -> m TimeResponse)

instance Tool (GetTime m) m where
  type ToolParams (GetTime m) = GetTimeParams
  type ToolOutput (GetTime m) = TimeResponse
  toolName _ = "get_time"
  toolDescription _ = "Get the current time"
  call (GetTime impl) params = impl params

-- Tool value with actual implementation
getTimeTool :: MonadIO m => GetTime m
getTimeTool = GetTime $ \_params -> do
  now <- liftIO getCurrentTime
  let timeStr = T.pack $ formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S UTC" now
  liftIO $ putStrLn $ "    ⏰ " <> T.unpack timeStr
  return $ TimeResponse timeStr

-- ============================================================================
-- HTTP Transport
-- ============================================================================

-- HTTP call to llama.cpp server
callLLM :: String -> OpenAIRequest -> ExceptT LLMError IO OpenAIResponse
callLLM baseUrl request = do
  req <- liftIO $ parseRequest $ "POST " ++ baseUrl ++ "/v1/chat/completions"
  let req' = setRequestHeaders [("Content-Type", "application/json")]
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request) req

  response <- httpLBS req'
  withExceptT (ParseError . T.pack) $ except $ eitherDecodeJSONViaCodec (getResponseBody response)

-- ============================================================================
-- Agent Loop
-- ============================================================================

-- Main agent loop - continues until text response (non-tool)
agentLoop :: String -> [ModelConfig OpenAI MistralModel] -> [Message MistralModel OpenAI] -> ExceptT LLMError IO ()
agentLoop serverUrl configs messages = do
  -- Available tools (for execution)
  let tools :: [LLMTool IO]
      tools = [LLMTool (getTimeTool @IO)]

      -- Build config with tools added
      toolDefs = map llmToolToDefinition tools
      configs' = configs ++ [Tools toolDefs]

  -- Call LLM (model + config)
  let request = toRequest OpenAI MistralModel configs' messages
  response <- callLLM serverUrl request
  responses <- except $ fromResponse response

  newMsgs <- liftIO $ concat <$> mapM (handleResponse tools) responses
  -- Continue loop only if there are tool results to send back
  unless (null newMsgs) $
    agentLoop serverUrl configs' (messages ++ responses ++ newMsgs)

-- Handle different response types
-- Returns empty list for text (stops loop), tool results for tool calls (continues loop)
handleResponse :: [LLMTool IO] -> Message MistralModel OpenAI -> IO [Message MistralModel OpenAI]
handleResponse _ (AssistantText text) = do
  putStrLn $ "🤖 " <> T.unpack text
  return [] -- Empty list stops the loop

handleResponse tools (AssistantTool calls) = do
  putStrLn $ "🔧 Calling " <> show (length calls) <> " tool(s)"
  results <- mapM (executeCall tools) calls
  return $ map ToolResultMsg results  -- Tool results continue loop

handleResponse _ _ = return []

-- Execute a single tool call with logging
executeCall :: [LLMTool IO] -> ToolCall -> IO ToolResult
executeCall tools callTool = do
  putStrLn $ "  → " <> T.unpack (toolCallName callTool)
  result <- executeToolCall tools callTool
  case toolResultOutput result of
    Left errMsg -> putStrLn $ "    ❌ " <> T.unpack errMsg
    Right _ -> return ()
  return result

-- ============================================================================
-- Main Entry Point
-- ============================================================================

main :: IO ()
main = do
  serverUrl <- lookupEnv "LLAMA_SERVER_URL" >>= \case
    Nothing -> putStrLn "Set LLAMA_SERVER_URL environment variable" >> exitFailure
    Just url -> return url

  putStrLn $ "Using server: " <> serverUrl
  putStrLn "=== Tool Calling Demo ===\n"

  -- Build config for MistralModel with OpenAI provider
  let configs = [ Temperature 0.7
                , MaxTokens 200
                ]
  let initialMsg = [UserText "Use the get_time tool to tell me what time it is."]

  result <- runExceptT $ agentLoop serverUrl configs initialMsg
  case result of
    Left err -> putStrLn $ "❌ Error: " <> show err
    Right () -> return ()
