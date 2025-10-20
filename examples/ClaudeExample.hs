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
import UniversalLLM.Providers.Anthropic (Anthropic(..), withMagicSystemPrompt, oauthHeaders)
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
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

-- Claude Sonnet 4.5 model
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance HasTools ClaudeSonnet45

instance ModelName Anthropic ClaudeSonnet45 where
  modelName _ = "claude-sonnet-4-5-20250929"

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

-- | Build an Anthropic-specific LLM call function with OAuth
-- Automatically applies magic system prompt to all requests
mkAnthropicCall :: Text -> LLMCall AnthropicRequest AnthropicResponse
mkAnthropicCall oauthToken =
  let baseLLMCall = mkLLMCall "https://api.anthropic.com/v1/messages" (oauthHeaders oauthToken)
  in \request -> baseLLMCall (withMagicSystemPrompt request)

-- ============================================================================
-- Agent Loop
-- ============================================================================

-- Main agent loop - continues until text response (non-tool)
agentLoop :: LLMCall AnthropicRequest AnthropicResponse -> [ModelConfig Anthropic ClaudeSonnet45] -> [Message ClaudeSonnet45 Anthropic] -> ExceptT LLMError IO ()
agentLoop callClaude configs messages = do
  -- Available tools (for execution)
  let tools :: [LLMTool IO]
      tools = [LLMTool (getTimeTool @IO)]

      -- Build config with tools added
      toolDefs = map llmToolToDefinition tools
      configs' = configs ++ [Tools toolDefs]

  -- Call LLM (model + config)
  let request = toRequest Anthropic ClaudeSonnet45 configs' messages
  response <- callClaude request
  responses <- except $ fromResponse response

  newMsgs <- liftIO $ concat <$> mapM (handleResponse tools) responses
  -- Continue loop only if there are tool results to send back
  unless (null newMsgs) $
    agentLoop callClaude configs' (messages ++ responses ++ newMsgs)

-- Handle different response types
-- Returns empty list for text (stops loop), tool results for tool calls (continues loop)
handleResponse :: [LLMTool IO] -> Message ClaudeSonnet45 Anthropic -> IO [Message ClaudeSonnet45 Anthropic]
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
  oauthToken <- lookupEnv "ANTHROPIC_OAUTH_TOKEN" >>= \case
    Nothing -> putStrLn "Set ANTHROPIC_OAUTH_TOKEN environment variable" >> exitFailure
    Just token -> return $ T.pack token

  putStrLn "Using Claude Sonnet 4.5 with OAuth"
  putStrLn "=== Tool Calling Demo ===\n"

  -- Build LLM call function for Anthropic (automatically applies magic system prompt)
  let callClaude = mkAnthropicCall oauthToken

  -- Build config for ClaudeSonnet45 with Anthropic provider
  let configs = [ Temperature 0.7
                , MaxTokens 200
                ]
  let initialMsg = [UserText "Use the get_time tool to tell me what time it is."]

  result <- runExceptT $ agentLoop callClaude configs initialMsg
  case result of
    Left err -> putStrLn $ "❌ Error: " <> show err
    Right () -> return ()
