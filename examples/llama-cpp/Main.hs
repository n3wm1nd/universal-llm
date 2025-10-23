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
{-# LANGUAGE TypeOperators #-}

module Main (main) where

import UniversalLLM
import UniversalLLM.Providers.OpenAI
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
-- Agent Loop (polymorphic business logic - works with ANY provider/model supporting tools)
-- ============================================================================
--
-- KEY ARCHITECTURAL POINT:
-- The business logic below is fully polymorphic over provider and model types.
-- It works with ANY provider/model combination that supports the required capabilities
-- (in this case: ProviderImplementation and HasTools).
--
-- This demonstrates how user code should be written: decoupled from specific providers
-- and models, using only the capability constraints needed. The concrete provider/model
-- selection happens only at the entry point (main), making the code flexible and reusable.
--
-- ============================================================================

-- Main agent loop - continues until text response (non-tool)
-- This is polymorphic over BOTH provider and model
agentLoop :: forall provider model req resp.
             (ProviderImplementation provider model, HasTools model provider,
              req ~ ProviderRequest provider, resp ~ ProviderResponse provider,
              Monoid req)
          => provider
          -> model
          -> [LLMTool IO]
          -> LLMCall req resp
          -> [ModelConfig provider model]
          -> [Message model provider]
          -> ExceptT LLMError IO ()
agentLoop provider model tools callLLM configs messages = do
  -- Build config with tools added
  let toolDefs = map llmToolToDefinition tools
      configs' = configs ++ [Tools toolDefs]

  -- Build and send request (provider-agnostic)
  let request = toProviderRequest provider model configs' messages
  response <- callLLM request

  -- Parse response (provider-agnostic)
  let msgs = fromProviderResponse provider model configs' messages response
  responses <- if null msgs
               then except $ Left $ ParseError "No messages parsed from response"
               else return msgs

  newMsgs <- liftIO $ concat <$> mapM (handleResponse tools) responses
  -- Continue loop only if there are tool results to send back
  unless (null newMsgs) $
    agentLoop provider model tools callLLM configs' (messages ++ responses ++ newMsgs)

-- Handle different response types (polymorphic over provider and model)
-- Returns empty list for text (stops loop), tool results for tool calls (continues loop)
handleResponse :: forall provider model. HasTools model provider
               => [LLMTool IO]
               -> Message model provider
               -> IO [Message model provider]
handleResponse _ (AssistantText text) = do
  putStrLn $ "🤖 " <> T.unpack text
  return [] -- Empty list stops the loop

handleResponse tools (AssistantTool toolCall) = do
  putStrLn $ "🔧 Calling tool"
  result <- executeCall tools toolCall
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

  -- Available tools (for execution)
  let tools :: [LLMTool IO]
      tools = [LLMTool (getTimeTool @IO)]

  -- Concrete provider and model selection (only place where specific types matter)
  let provider = OpenAI
      model = MistralModel

  -- Build config
  let configs = [ Temperature 0.7
                , MaxTokens 200
                ]
  let initialMsg = [UserText "Use the get_time tool to tell me what time it is."]

  -- Business logic (agentLoop) is polymorphic - works with any provider/model
  result <- runExceptT $ agentLoop provider model tools callLLM configs initialMsg
  case result of
    Left err -> putStrLn $ "❌ Error: " <> show err
    Right () -> return ()
