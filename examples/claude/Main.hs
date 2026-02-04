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
import UniversalLLM.Tools
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import UniversalLLM.Providers.Anthropic (AnthropicOAuth(..), oauthHeaders)
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import qualified Data.Map.Strict as Map
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

-- Claude Sonnet 4.5 model with OAuth
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet45 AnthropicOAuth) where
  modelName _ = "claude-sonnet-4-5-20250929"

instance HasTools (Model ClaudeSonnet45 AnthropicOAuth) where
  type ToolState (Model ClaudeSonnet45 AnthropicOAuth) = AnthropicProvider.OAuthToolsState
  withTools = AnthropicProvider.anthropicOAuthBlacklistedTools

-- Composable provider for ClaudeSonnet45 with OAuth (includes magic system prompt automatically)
claudeSonnet45ComposableProvider :: ComposableProvider (Model ClaudeSonnet45 AnthropicOAuth) (AnthropicProvider.OAuthToolsState, ((), ()))
claudeSonnet45ComposableProvider = withTools `chainProviders` AnthropicProvider.anthropicOAuthMagicPrompt `chainProviders` AnthropicProvider.baseComposableProvider @(Model ClaudeSonnet45 AnthropicOAuth)

-- ============================================================================
-- Tool Definition
-- ============================================================================

-- Tool result type
data TimeResponse = TimeResponse
  { currentTime :: Text
  } deriving (Show, Eq)

instance Autodocodec.HasCodec TimeResponse where
  codec = object "TimeResponse" $
    TimeResponse <$> Autodocodec.requiredField "current_time" "Current time" .= currentTime

instance ToolParameter TimeResponse where
  paramName _ n = "time_response_" <> T.pack (show n)
  paramDescription _ = "time response"

-- Make TimeResponse a ToolFunction so functions returning it become tools automatically
instance ToolFunction TimeResponse where
  toolFunctionName _ = "get_time"
  toolFunctionDescription _ = "Get the current time"

-- Tool implementation as a bare 0-arity function - no wrapper needed!
getTimeTool :: IO TimeResponse
getTimeTool = do
  now <- liftIO getCurrentTime
  let timeStr = T.pack $ formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S UTC" now
  liftIO $ putStrLn $ "    ⏰ " <> T.unpack timeStr
  return $ TimeResponse timeStr

-- ============================================================================
-- HTTP Transport
-- ============================================================================

-- | Build an Anthropic-specific LLM call function with OAuth
-- (OAuth provider includes magic system prompt automatically)
mkAnthropicCall :: Text -> LLMCall AnthropicRequest AnthropicResponse
mkAnthropicCall oauthToken =
  mkLLMCall "https://api.anthropic.com/v1/messages" (oauthHeaders oauthToken)

-- ============================================================================
-- Agent Loop (polymorphic business logic - works with ANY provider/model supporting tools)
-- ============================================================================
--
-- KEY ARCHITECTURAL POINT:
-- The business logic below is fully polymorphic over provider and model types.
-- It works with ANY provider/model combination that supports the required capabilities
-- (in this case: HasTools).
--
-- This demonstrates how user code should be written: decoupled from specific providers
-- and models, using only the capability constraints needed. The concrete provider/model
-- selection happens only at the entry point (main), making the code flexible and reusable.
--
-- ============================================================================

-- Main agent loop - continues until text response (non-tool)
-- This is polymorphic over BOTH provider and model
agentLoop :: forall m req resp state.
             (HasTools m,
              req ~ ProviderRequest m, resp ~ ProviderResponse m,
              Monoid req)
          => ComposableProvider m state
          -> m
          -> state
          -> [LLMTool IO]
          -> LLMCall req resp
          -> [ModelConfig m]
          -> [Message m]
          -> ExceptT LLMError IO ()
agentLoop composableProvider model state tools callLLM configs messages = do
  -- Build and send request (provider-agnostic)
  let request = snd $ toProviderRequest composableProvider model configs state messages
  response <- callLLM request

  -- Parse response (provider-agnostic)
  responses <- case fromProviderResponse composableProvider model configs state response of
    Left err -> except $ Left err
    Right (_state, msgs) ->
      if null msgs
      then except $ Left $ ParseError "No messages parsed from response"
      else return msgs

  newMsgs <- liftIO $ concat <$> mapM (handleResponse tools) responses
  -- Continue loop only if there are tool results to send back
  unless (null newMsgs) $
    agentLoop composableProvider model state tools callLLM configs (messages ++ responses ++ newMsgs)

-- Handle different response types (polymorphic over provider and model)
-- Returns empty list for text (stops loop), tool results for tool calls (continues loop)
handleResponse :: forall m. HasTools m
               => [LLMTool IO]
               -> Message m
               -> IO [Message m]
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
  result <- executeToolCallFromList tools callTool
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

  -- Build LLM call function for Anthropic OAuth
  let callClaude = mkAnthropicCall oauthToken

  -- Available tools (for execution)
  let tools :: [LLMTool IO]
      tools = [LLMTool getTimeTool]

  -- Concrete provider and model selection (only place where specific types matter)
  let model = Model ClaudeSonnet45 AnthropicOAuth

  -- Build config
  let toolDefs = map llmToolToDefinition tools
      configs = [ Temperature 0.7
                , MaxTokens 200
                , Tools toolDefs
                ]
  let initialMsg = [UserText "Use the get_time tool to tell me what time it is."]

  -- Business logic (agentLoop) is polymorphic - works with any provider/model
  -- State: (OAuthToolsState, ((), ()))
  let initialState = (AnthropicProvider.OAuthToolsState Map.empty, ((), ()))
  result <- runExceptT $ agentLoop claudeSonnet45ComposableProvider model initialState tools callClaude configs initialMsg
  case result of
    Left err -> putStrLn $ "❌ Error: " <> show err
    Right () -> return ()
