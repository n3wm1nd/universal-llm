{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import UniversalLLM
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Protocols.OpenAI
import System.Environment (lookupEnv)
import System.Exit (exitFailure)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Autodocodec (toJSONViaCodec, eitherDecodeJSONViaCodec)
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy.Char8 as L8
import Network.HTTP.Simple
import Control.Exception (try, SomeException)

-- Define our actual model ad-hoc for this example
data MistralModel = MistralModel
  { mistralTemperature :: Maybe Double
  , mistralMaxTokens :: Maybe Int
  , mistralSeed :: Maybe Int
  } deriving (Show, Eq)

-- This model doesn't have vision or JSON mode capabilities
-- instance HasVision MistralModel  -- Commented out - no vision
-- instance HasJSON MistralModel    -- Commented out - no JSON mode

-- Parameter extraction
instance Temperature MistralModel provider where
  getTemperature = mistralTemperature

instance MaxTokens MistralModel provider where
  getMaxTokens = mistralMaxTokens

instance Seed MistralModel provider where
  getSeed = mistralSeed

-- We'll pretend it's "gpt-4o" for the OpenAI-compatible API
-- (since llama.cpp ignores the model field anyway)
instance ModelName OpenAI MistralModel where
  modelName = "mistral-7b-instruct"

-- HTTP transport function - part of the application, not the library
callLlamaCpp :: String -> OpenAIRequest -> IO (Either LLMError OpenAIResponse)
callLlamaCpp baseUrl request = do
  result <- try $ do
    let url = baseUrl ++ "/v1/chat/completions"

    -- Create HTTP request
    initialRequest <- parseRequest ("POST " ++ url)
    let headers = [ ("Content-Type", "application/json")
                  , ("Accept", "application/json")
                  ]

    let httpRequest = setRequestHeaders headers
                    $ setRequestBodyLBS (Aeson.encode (toJSONViaCodec request))
                    $ initialRequest

    -- Make the request
    response <- httpLBS httpRequest

    -- Parse response
    let responseBody = getResponseBody response
    case eitherDecodeJSONViaCodec responseBody of
      Left err -> return $ Left $ ParseError $ "Failed to parse response: " <> T.pack err
      Right openaiResp -> return $ Right openaiResp

  case result of
    Left (e :: SomeException) -> return $ Left $ NetworkError $ "HTTP request failed: " <> T.pack (show e)
    Right res -> return res

main :: IO ()
main = do
  -- Get server URL from environment
  maybeUrl <- lookupEnv "LLAMA_SERVER_URL"
  serverUrl <- case maybeUrl of
    Nothing -> do
      putStrLn "Error: LLAMA_SERVER_URL environment variable is required"
      putStrLn "Example: export LLAMA_SERVER_URL=http://localhost:8080"
      exitFailure
    Just url -> do
      putStrLn $ "Using server: " ++ url
      return url

  putStrLn $ "Demo: Preparing request for llama.cpp server at: " ++ serverUrl

  -- Create a model configuration that matches what's actually running
  let model = MistralModel
        { mistralTemperature = Just 1.5
        , mistralMaxTokens = Just 100
        , mistralSeed = Nothing  -- Let server handle randomness
        }

  -- Create messages using our type-safe interface
  let messages = [ UserText "Hello! Can you tell me a short knock-knock joke?"
                 ]

  -- Use our pure transformation functions (the core value of our library)
  let provider = OpenAI
  let request = toRequest provider model messages

  putStrLn "\n=== Pure Transformation Demo ==="
  putStrLn $ "Model name: " ++ T.unpack (UniversalLLM.Protocols.OpenAI.model request)
  putStrLn $ "Temperature: " ++ show (temperature request)
  putStrLn $ "Max tokens: " ++ show (max_tokens request)
  putStrLn $ "Number of messages: " ++ show (length (UniversalLLM.Protocols.OpenAI.messages request))

  -- Show the generated JSON (this would be sent via HTTP)
  putStrLn "\n=== Generated JSON Request ==="
  let jsonRequest = toJSONViaCodec request
  L8.putStrLn $ Aeson.encode jsonRequest

  putStrLn "\n=== Making Real API Call ==="
  putStrLn $ "Calling: " ++ serverUrl ++ "/v1/chat/completions"

  result <- callLlamaCpp serverUrl request

  case result of
    Left err -> do
      putStrLn $ "❌ API Error: " ++ show err
      exitFailure
    Right response -> do
      putStrLn "✅ Success! Response received:"
      case fromResponse response of
        Left parseErr -> do
          putStrLn $ "❌ Parse Error: " ++ show parseErr
          exitFailure
        Right responseMessages -> do
          putStrLn "\n=== LLM Response ==="
          mapM_ printMessage responseMessages

          putStrLn "\n=== Type Safety Demo ==="
          putStrLn "✓ MistralModel defined ad-hoc for this example"
          putStrLn "✓ No vision/JSON capabilities (compile-time enforced)"
          putStrLn "✓ Temperature/tokens automatically extracted from model"
          putStrLn "✓ Provider-specific formatting applied"
          putStrLn "✓ Real HTTP call made and response parsed!"

printMessage :: Message model provider -> IO ()
printMessage (AssistantText text) = T.putStrLn $ "🤖 " <> text
printMessage (UserText text) = T.putStrLn $ "👤 " <> text
printMessage (SystemText text) = T.putStrLn $ "⚙️  " <> text
printMessage _ = putStrLn "📄 Other message type"
