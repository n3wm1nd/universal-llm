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
import UniversalLLM.Providers.Anthropic (Anthropic(..), withMagicSystemPrompt, oauthHeaders)
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
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
data GPT4o = GPT4o deriving (Show, Eq)

instance ModelName OpenAI GPT4o where
  modelName _ = "gpt-4o"

instance HasTools GPT4o OpenAI where
  toolsComposableProvider = OpenAIProvider.toolsComposableProvider

instance HasJSON GPT4o OpenAI where
  jsonComposableProvider = OpenAIProvider.jsonComposableProvider

instance ProviderImplementation OpenAI GPT4o where
  getComposableProvider =
    OpenAIProvider.baseComposableProvider
    <> OpenAIProvider.toolsComposableProvider
    <> OpenAIProvider.jsonComposableProvider

-- ============================================================================
-- Configuration
-- ============================================================================

type BackendProvider = OpenAI
type BackendModel = GPT4o

backendProvider :: BackendProvider
backendProvider = OpenAI

backendModel :: BackendModel
backendModel = GPT4o

-- ============================================================================
-- HTTP Transport (copied from examples, will be shared eventually)
-- ============================================================================

-- | Make an HTTP call to the backend LLM provider
callBackend :: String  -- ^ API Key
            -> OpenAIRequest
            -> IO (Either LLMError OpenAIResponse)
callBackend apiKey request = do
  let url = "https://api.openai.com/v1/chat/completions"
      headers = [ (mk "Authorization", TE.encodeUtf8 $ "Bearer " <> T.pack apiKey)
                , (mk "Content-Type", "application/json")
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

  oaiRequest <- case parseEither parseJSONViaCodec jsonValue of
    Left err -> throwE $ ParseError $ "Failed to parse OpenAI request: " <> T.pack err
    Right req -> return req

  liftIO $ putStrLn $ "📥 Received request for model: " <> T.unpack (model oaiRequest)

  -- 2. Convert to universal format (Messages + Config)
  proxyConfig <- case parseOpenAIRequest @BackendProvider @BackendModel oaiRequest of
    Left err -> throwE $ ParseError $ "Failed to parse request: " <> err
    Right cfg -> return cfg

  liftIO $ putStrLn $ "🔄 Converted to " <> show (length $ proxyMessages proxyConfig) <> " universal messages"

  -- 3. Convert universal format to backend provider request
  let backendRequest = toProviderRequest backendProvider backendModel
                         (proxyConfigs proxyConfig)
                         (proxyMessages proxyConfig)

  liftIO $ putStrLn $ "📤 Sending to backend provider (OpenAI GPT-4o)"

  -- 4. Call backend
  backendResponse <- liftIO $ callBackend apiKey backendRequest
  response <- except backendResponse

  liftIO $ putStrLn "📨 Received response from backend"

  -- 5. Parse backend response to universal messages
  let universalMessages = fromProviderResponse backendProvider backendModel
                            (proxyConfigs proxyConfig)
                            (proxyMessages proxyConfig)
                            response

  liftIO $ putStrLn $ "🔄 Converted to " <> show (length universalMessages) <> " universal messages"

  -- 6. Convert universal messages back to OpenAI format
  oaiResponse <- case buildOpenAIResponse @BackendProvider @BackendModel universalMessages of
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
            (Aeson.encode $ toJSONViaCodec $ OpenAIError $ OpenAIErrorResponse
              { errorDetail = OpenAIErrorDetail
                  { code = 500
                  , errorMessage = T.pack $ show err
                  , errorType = "proxy_error"
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
  apiKey <- lookupEnv "OPENAI_API_KEY" >>= \case
    Nothing -> error "Set OPENAI_API_KEY environment variable"
    Just key -> return key

  let port = 8080
  putStrLn $ "🚀 Universal LLM Proxy starting on port " <> show port
  putStrLn $ "📍 Backend: OpenAI GPT-4o"
  putStrLn $ "🔌 Endpoint: http://localhost:" <> show port <> "/v1/chat/completions"
  putStrLn ""

  run port (app apiKey)
