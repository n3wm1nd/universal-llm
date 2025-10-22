{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
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
import Common.HTTP (LLMCall, mkLLMCall)
import System.Environment (lookupEnv)
import System.Exit (exitFailure)
import Data.Text (Text)
import qualified Data.Text as T
import Autodocodec (HasCodec, codec, object, optionalField, (.=))
import qualified Autodocodec
import Control.Monad (unless)
import Control.Monad.Trans.Except (ExceptT(..), runExceptT, except)
import Control.Monad.IO.Class (liftIO, MonadIO)
import Data.Time (getCurrentTime, formatTime, defaultTimeLocale)

-- ============================================================================
-- Model Definition
-- ============================================================================

-- Simple model (just model identity)
data MistralModel = MistralModel deriving (Show, Eq)

instance ModelName OpenAI MistralModel where
  modelName _ = "mistral-7b-instruct"

instance HasTools MistralModel OpenAI where
  toolsComposableProvider = UniversalLLM.Providers.OpenAI.toolsComposableProvider

instance HasJSON MistralModel OpenAI where
  jsonComposableProvider = UniversalLLM.Providers.OpenAI.jsonComposableProvider

instance ProviderImplementation OpenAI MistralModel where
  getComposableProvider = UniversalLLM.Providers.OpenAI.baseComposableProvider <> UniversalLLM.Providers.OpenAI.toolsComposableProvider <> UniversalLLM.Providers.OpenAI.jsonComposableProvider

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
-- Request/Response Helpers
-- ============================================================================

-- Build OpenAI request using the model's provider implementation
buildRequest :: forall model. ProviderImplementation OpenAI model
             => model
             -> [ModelConfig OpenAI model]
             -> [Message model OpenAI]
             -> OpenAIRequest
buildRequest = toProviderRequest OpenAI

-- Parse OpenAI response using the model's provider implementation
parseResponse :: forall model. (ProviderImplementation OpenAI model, ModelName OpenAI model)
              => model
              -> [ModelConfig OpenAI model]
              -> [Message model OpenAI]  -- history
              -> OpenAIResponse
              -> Either LLMError [Message model OpenAI]
parseResponse model configs history resp =
  let msgs = fromProviderResponse OpenAI model configs history resp
  in if null msgs
     then Left $ ParseError "No messages parsed from response"
     else Right msgs

-- ============================================================================
-- Agent Loop
-- ============================================================================

-- Main agent loop - continues until text response (non-tool)
agentLoop :: LLMCall OpenAIRequest OpenAIResponse -> [ModelConfig OpenAI MistralModel] -> [Message MistralModel OpenAI] -> ExceptT LLMError IO ()
agentLoop callLLM configs messages = do
  -- Available tools (for execution)
  let tools :: [LLMTool IO]
      tools = [LLMTool (getTimeTool @IO)]

      -- Build config with tools added
      toolDefs = map llmToolToDefinition tools
      configs' = configs ++ [Tools toolDefs]

  -- Call LLM (model + config)
  let request = buildRequest @MistralModel MistralModel configs' messages
  response <- callLLM request
  responses <- except $ parseResponse @MistralModel MistralModel configs' messages response

  newMsgs <- liftIO $ concat <$> mapM (handleResponse tools) responses
  -- Continue loop only if there are tool results to send back
  unless (null newMsgs) $
    agentLoop callLLM configs' (messages ++ responses ++ newMsgs)

-- Handle different response types
-- Returns empty list for text (stops loop), tool results for tool calls (continues loop)
handleResponse :: [LLMTool IO] -> Message MistralModel OpenAI -> IO [Message MistralModel OpenAI]
handleResponse _ (AssistantText text) = do
  putStrLn $ "🤖 " <> T.unpack text
  return [] -- Empty list stops the loop

handleResponse tools (AssistantTool call) = do
  putStrLn $ "🔧 Calling tool"
  result <- executeCall tools call
  return [ToolResultMsg result]  -- Tool result continues loop

handleResponse _ _ = return []

-- Execute a single tool call with logging
executeCall :: [LLMTool IO] -> ToolCall -> IO ToolResult
executeCall tools callTool = do
  putStrLn $ "  → " <> T.unpack (getToolCallName callTool)
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

  -- Build LLM call function with endpoint and headers
  let callLLM = mkLLMCall (serverUrl ++ "/v1/chat/completions") [("Content-Type", "application/json")]

  -- Build config for MistralModel with OpenAI provider
  let configs = [ Temperature 0.7
                , MaxTokens 200
                ]
  let initialMsg = [UserText "Use the get_time tool to tell me what time it is."]

  result <- runExceptT $ agentLoop callLLM configs initialMsg
  case result of
    Left err -> putStrLn $ "❌ Error: " <> show err
    Right () -> return ()
