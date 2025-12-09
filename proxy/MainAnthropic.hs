{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module Main (main) where

import UniversalLLM
import UniversalLLM.Providers.OpenAI (OpenAI(..))
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider
import UniversalLLM.Providers.Anthropic (Anthropic(..))
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import UniversalLLM.Protocols.Anthropic
import qualified UniversalLLM.Protocols.OpenAI as OpenAIProto
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import Proxy.OpenAICompat

import Network.Wai
import Network.Wai.Handler.Warp (run)
import Network.HTTP.Types
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Autodocodec (parseJSONViaCodec, toJSONViaCodec)
import qualified Data.Aeson as Aeson
import Data.Aeson.Types (parseEither)
import Control.Monad.Trans.Except (ExceptT, runExceptT, except, throwE)
import Control.Monad.IO.Class (liftIO)
import System.Environment (lookupEnv)
import Data.CaseInsensitive (mk)
import Network.HTTP.Simple (httpLBS, setRequestBodyLBS, setRequestHeaders, parseRequest_, getResponseBody, setRequestMethod)

-- ============================================================================
-- Model Definition
-- ============================================================================

-- Define the specific model we're using in this proxy
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance ModelName (Model ClaudeSonnet45 Anthropic) where
  modelName _ = "claude-sonnet-4.5-20250929"

instance HasTools (Model ClaudeSonnet45 Anthropic) where
  withTools = AnthropicProvider.anthropicTools

-- Composable provider for ClaudeSonnet45 with tools
claudeSonnet45ComposableProvider :: ComposableProvider (Model ClaudeSonnet45 Anthropic) ((), ())
claudeSonnet45ComposableProvider = withTools `chainProviders` AnthropicProvider.baseComposableProvider @(Model ClaudeSonnet45 Anthropic)

-- ============================================================================
-- Configuration
-- ============================================================================

type BackendProvider = Anthropic
type BackendModel = ClaudeSonnet45

backendModel :: Model ClaudeSonnet45 Anthropic
backendModel = Model ClaudeSonnet45 Anthropic

-- ============================================================================
-- HTTP Transport (copied from examples, will be shared eventually)
-- ============================================================================

-- | Make an HTTP call to the backend LLM provider
callBackend :: String  -- ^ API Key
            -> AnthropicRequest
            -> IO (Either LLMError AnthropicResponse)
callBackend apiKey request = do
  let url = "https://api.anthropic.com/v1/messages"
      headers = [ (mk "x-api-key", TE.encodeUtf8 $ T.pack apiKey)
                , (mk "Content-Type", "application/json")
                , (mk "anthropic-version", "2023-06-01")
                ]
      reqBody = Aeson.encode $ toJSONViaCodec request
      httpReq = parseRequest_ url
      httpReq' = setRequestMethod "POST"
               $ setRequestHeaders headers
               $ setRequestBodyLBS reqBody httpReq

  response <- httpLBS httpReq'
  let responseBody = getResponseBody response

  case Aeson.eitherDecode responseBody of
    Left err -> return $ Left $ ParseError $ T.pack err
    Right jsonValue -> case parseEither parseJSONViaCodec jsonValue of
      Left err2 -> return $ Left $ ParseError $ T.pack err2
      Right resp -> return $ Right resp

-- ============================================================================
-- Proxy Handler
-- ============================================================================

-- | Main proxy handler: OpenAI request -> Universal -> Backend -> Universal -> OpenAI response
handleProxy :: String  -- ^ API key for backend
            -> BSL.ByteString  -- ^ Request body
            -> ExceptT LLMError IO BSL.ByteString
handleProxy apiKey reqBody = do
  -- 1. Parse incoming OpenAI request
  jsonValue <- case Aeson.eitherDecode reqBody of
    Left err -> throwE $ ParseError $ "Failed to decode JSON: " <> T.pack err
    Right v -> return v

  oaiRequest <- case parseEither (parseJSONViaCodec @OpenAIRequest) jsonValue of
    Left err -> throwE $ ParseError $ "Failed to parse OpenAI request: " <> T.pack err
    Right (req :: OpenAIRequest) -> return req

  liftIO $ putStrLn $ "📥 Received request for model: " <> T.unpack (OpenAIProto.model oaiRequest)

  -- 2. Convert to universal format (Messages + Config)
  proxyConfig <- case parseOpenAIRequest oaiRequest of
    Left err -> throwE $ ParseError $ "Failed to parse request: " <> err
    Right cfg -> return cfg

  liftIO $ putStrLn $ "🔄 Converted to " <> show (length $ proxyMessages proxyConfig) <> " universal messages"

  -- 3. Convert universal format to backend provider request
  let backendRequest = snd $ toProviderRequest claudeSonnet45ComposableProvider
                                                 backendModel
                                                 (proxyConfigs proxyConfig)
                                                 ((), ())
                                                 (proxyMessages proxyConfig)

  liftIO $ putStrLn $ "📤 Sending to backend provider (Anthropic Claude-3.5-Sonnet)"

  -- 4. Call backend
  backendResponse <- liftIO $ callBackend apiKey backendRequest
  response <- except backendResponse

  liftIO $ putStrLn "📨 Received response from backend"

  -- 5. Parse backend response to universal messages
  universalMessages <- case fromProviderResponse claudeSonnet45ComposableProvider
                                                 backendModel
                                                 (proxyConfigs proxyConfig)
                                                 ((), ())
                                                 response of
    Left err -> throwE $ ParseError $ "Failed to parse backend response: " <> T.pack (show err)
    Right (_state, messages) -> return messages

  liftIO $ putStrLn $ "🔄 Converted to " <> show (length universalMessages) <> " universal messages"

  -- 6. Convert universal messages back to OpenAI format
  oaiResponse <- case buildOpenAIResponse universalMessages of
    Left err -> throwE $ ParseError $ "Failed to build OpenAI response: " <> err
    Right resp -> return resp

  liftIO $ putStrLn "✅ Sending OpenAI-compatible response\n"

  return $ Aeson.encode $ toJSONViaCodec oaiResponse

-- ============================================================================
-- WAI Application
-- ============================================================================

app :: String -> Application
app apiKey req respond = do
  if requestMethod req == "POST" && pathInfo req == ["v1", "chat", "completions"]
    then do
      body <- strictRequestBody req
      result <- runExceptT $ handleProxy apiKey body
      case result of
        Right responseBody ->
          respond $ responseLBS status200
            [(hContentType, "application/json")]
            responseBody
        Left err ->
          respond $ responseLBS status500
            [(hContentType, "application/json")]
            (Aeson.encode $ toJSONViaCodec $ OpenAIProto.OpenAIError $ OpenAIProto.OpenAIErrorResponse
              { OpenAIProto.errorDetail = OpenAIProto.OpenAIErrorDetail
                  { OpenAIProto.code = 500
                  , OpenAIProto.errorMessage = T.pack $ show err
                  , OpenAIProto.errorType = "proxy_error"
                  }
              })
    else
      respond $ responseLBS status404
        [(hContentType, "text/plain")]
        "Not Found\n\nSupported endpoint: POST /v1/chat/completions"

-- ============================================================================
-- Main Entry Point
-- ============================================================================

main :: IO ()
main = do
  apiKey <- lookupEnv "ANTHROPIC_API_KEY" >>= \case
    Nothing -> error "Set ANTHROPIC_API_KEY environment variable"
    Just key -> return key

  let port = 8081
  putStrLn $ "🚀 Universal LLM Proxy (Anthropic Backend) starting on port " <> show port
  putStrLn $ "📍 Backend: Anthropic Claude Sonnet 4.5"
  putStrLn $ "🔌 Endpoint: http://localhost:" <> show port <> "/v1/chat/completions"
  putStrLn $ "💡 Accepts OpenAI-format requests, translates to Anthropic"
  putStrLn ""

  run port (app apiKey)
