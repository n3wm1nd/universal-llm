{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Test.Hspec
import qualified CachedIntegrationSpec
import qualified ComposableHandlersSpec
import qualified OpenAIComposableSpec
import qualified OpenAIStreamingSpec
import qualified AnthropicComposableSpec
import qualified AnthropicStreamingSpec
import qualified PropertySpec
import qualified XMLPropertySpec
import qualified XMLProvidersSpec
import qualified CoreTypesSpec
import qualified ToolDefinitionIntegrationSpec
import qualified CompletionSpec
import qualified ToolsSpec
import qualified TestCache
import qualified TestHTTP
import qualified ModelRegistry
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse, OpenAICompletionRequest, OpenAICompletionResponse)
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import System.Environment (lookupEnv)
import qualified Data.Text as T
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KeyMap
import qualified Data.Vector as V
import Control.Exception (catch, SomeException)
import System.FilePath (takeBaseName, takeExtension)

-- | Canonicalize a GGUF filename to a model name
-- Extracts the model name from filenames like "/models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"
-- Returns "GLM-4.5-Air" by removing quantization info, part numbers, and extension
-- Works from the back step by step:
--   1. Extract basename and drop extension (using FilePath functions)
--   2. Drop "xxxx-of-yyyy" multi-part suffix (if present)
--   3. Drop quantization suffix like "Q4_K_M" (if present)
canonicalizeGGUFName :: String -> String
canonicalizeGGUFName filename =
  let -- Step 1: Get basename without path or extension
      base = takeBaseName filename
      -- Step 2: Drop multi-part suffix (0001-of-0002)
      step2 = dropMultiPartSuffix base
      -- Step 3: Drop quantization suffix (Q4_K_M, etc.)
      step3 = dropQuantizationSuffix step2
  in step3
  where
    -- Drop "xxxx-of-yyyy" from the end if present
    dropMultiPartSuffix :: String -> String
    dropMultiPartSuffix s =
      case reverse (splitOn '-' s) of
        (lastNum : "of" : prevNum : rest)
          | all isDigit lastNum && all isDigit prevNum -> intercalate "-" (reverse rest)
        _ -> s

    -- Drop quantization suffix like "Q4_K_M" if present
    -- Pattern: starts with Q, followed by digit, contains underscores
    dropQuantizationSuffix :: String -> String
    dropQuantizationSuffix s =
      case reverse (splitOn '-' s) of
        (quant : rest) | isQuantization quant -> intercalate "-" (reverse rest)
        _ -> s

    -- Check if string looks like quantization: Q<digit>_<letters>
    isQuantization :: String -> Bool
    isQuantization s = case s of
      ('Q':d:rest) | isDigit d && '_' `elem` rest -> True
      _ -> False

    isDigit :: Char -> Bool
    isDigit c = c `elem` ("0123456789" :: String)

    splitOn :: Char -> String -> [String]
    splitOn _ "" = []
    splitOn delim str =
      let (before, remainder) = break (== delim) str
      in before : case remainder of
                    [] -> []
                    (_:after) -> splitOn delim after

    intercalate :: String -> [String] -> String
    intercalate _ [] = ""
    intercalate _ [x] = x
    intercalate sep (x:xs) = x ++ sep ++ intercalate sep xs

-- | Query llama.cpp server for loaded model information
-- Returns the canonicalized model name if successful
queryLlamaCppModel :: String -> IO (Maybe String)
queryLlamaCppModel baseUrl = do
  let url = baseUrl ++ "/v1/models"
  catch (do
    response <- TestHTTP.httpGet url
    case Aeson.decode response of
      Just (Aeson.Object obj) -> do
        -- llama.cpp returns: {"data": [{"id": "model-name.gguf", ...}]}
        case KeyMap.lookup "data" obj of
          Just (Aeson.Array models) | not (V.null models) -> do
            case V.head models of
              Aeson.Object modelObj -> do
                case KeyMap.lookup "id" modelObj of
                  Just (Aeson.String modelId) ->
                    return $ Just (canonicalizeGGUFName $ T.unpack modelId)
                  _ -> return Nothing
              _ -> return Nothing
          _ -> return Nothing
      _ -> return Nothing
    ) (\(_ :: SomeException) -> return Nothing)

main :: IO ()
main = do
  -- Determine which mode to use based on environment variable
  mode <- lookupEnv "TEST_MODE"

  -- API credentials and server URLs
  anthropicToken <- fmap T.pack <$> lookupEnv "ANTHROPIC_OAUTH_TOKEN"
  openaiApiKey <- lookupEnv "OPENAI_API_KEY"
  openrouterApiKey <- lookupEnv "OPENROUTER_API_KEY"
  llamacppUrl <- lookupEnv "LLAMACPP_URL"
  openaiCompatUrl <- lookupEnv "OPENAI_COMPAT_URL"  -- Generic OpenAI-compatible endpoint
  _openaiCompatModel <- lookupEnv "OPENAI_COMPAT_MODEL"  -- For providers that need model name in env

  let cachePath = "test-cache"

  -- Build OpenAI response provider (official OpenAI API)
  let openaiProvider :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
      openaiProvider = case mode of
        Just "record" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.recordMode cachePath $
            TestHTTP.httpCall "https://api.openai.com/v1/chat/completions" headers
        Just "update" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.updateMode cachePath $
            TestHTTP.httpCall "https://api.openai.com/v1/chat/completions" headers
        Just "live" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.liveMode $
            TestHTTP.httpCall "https://api.openai.com/v1/chat/completions" headers
        _ -> TestCache.playbackMode cachePath

  -- Build OpenRouter response provider
  let openrouterProvider :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
      openrouterProvider = case mode of
        Just "record" | Just apiKey <- openrouterApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.recordMode cachePath $
            TestHTTP.httpCall "https://openrouter.ai/api/v1/chat/completions" headers
        Just "update" | Just apiKey <- openrouterApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.updateMode cachePath $
            TestHTTP.httpCall "https://openrouter.ai/api/v1/chat/completions" headers
        Just "live" | Just apiKey <- openrouterApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.liveMode $
            TestHTTP.httpCall "https://openrouter.ai/api/v1/chat/completions" headers
        _ -> TestCache.playbackMode cachePath

  -- Query llama.cpp server for loaded model (if URL is set)
  llamacppLoadedModel <- case llamacppUrl of
    Just url -> queryLlamaCppModel url
    Nothing -> return Nothing

  -- Print info about llama.cpp model
  case (llamacppUrl, llamacppLoadedModel) of
    (Just url, Just loadedModel) ->
      putStrLn $ "llama.cpp server at " ++ url ++ " has model loaded: " ++ loadedModel
    (Just url, Nothing) ->
      putStrLn $ "Warning: Could not query llama.cpp server at " ++ url
    _ -> return ()

  -- Build llama.cpp response provider
  -- Only makes live requests if model name matches the request's model field
  let llamacppProvider :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
      llamacppProvider = case mode of
        Just "record" | Just url <- llamacppUrl ->
          TestCache.recordMode cachePath $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        Just "update" | Just url <- llamacppUrl ->
          TestCache.updateMode cachePath $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        Just "live" | Just url <- llamacppUrl ->
          TestCache.liveMode $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        _ -> TestCache.playbackMode cachePath

  -- Build generic OpenAI-compatible response provider
  let openaiCompatProvider :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
      openaiCompatProvider = case mode of
        Just "record" | Just url <- openaiCompatUrl ->
          TestCache.recordMode cachePath $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        Just "update" | Just url <- openaiCompatUrl ->
          TestCache.updateMode cachePath $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        Just "live" | Just url <- openaiCompatUrl ->
          TestCache.liveMode $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        _ -> TestCache.playbackMode cachePath

  -- Build Anthropic response provider
  -- Note: withMagicSystemPrompt is applied BEFORE caching so it's part of the cache key
  let anthropicProvider :: TestCache.ResponseProvider AnthropicRequest AnthropicResponse
      anthropicProvider = case mode of
        Just "record" | Just token <- anthropicToken ->
          let headers = AnthropicProvider.oauthHeaders token
              baseCall req = TestHTTP.httpCall "https://api.anthropic.com/v1/messages" headers req
              wrappedCall req = TestCache.recordMode cachePath baseCall (AnthropicProvider.withMagicSystemPrompt req)
          in wrappedCall
        Just "update" | Just token <- anthropicToken ->
          let headers = AnthropicProvider.oauthHeaders token
              baseCall req = TestHTTP.httpCall "https://api.anthropic.com/v1/messages" headers req
              wrappedCall req = TestCache.updateMode cachePath baseCall (AnthropicProvider.withMagicSystemPrompt req)
          in wrappedCall
        Just "live" | Just token <- anthropicToken ->
          let headers = AnthropicProvider.oauthHeaders token
              baseCall req = TestHTTP.httpCall "https://api.anthropic.com/v1/messages" headers req
              wrappedCall req = TestCache.liveMode baseCall (AnthropicProvider.withMagicSystemPrompt req)
          in wrappedCall
        _ -> \req -> TestCache.playbackMode cachePath (AnthropicProvider.withMagicSystemPrompt req)

  -- Build Anthropic streaming response provider (for SSE responses)
  let anthropicStreamingProvider :: TestCache.ResponseProvider AnthropicRequest BSL.ByteString
      anthropicStreamingProvider = case mode of
        Just "record" | Just token <- anthropicToken ->
          let headers = AnthropicProvider.oauthHeaders token
              baseCall req = TestHTTP.httpCallStreaming "https://api.anthropic.com/v1/messages" headers req
              wrappedCall req = TestCache.recordModeRaw cachePath baseCall (AnthropicProvider.withMagicSystemPrompt req)
          in wrappedCall
        Just "update" | Just token <- anthropicToken ->
          let headers = AnthropicProvider.oauthHeaders token
              baseCall req = TestHTTP.httpCallStreaming "https://api.anthropic.com/v1/messages" headers req
              wrappedCall req = TestCache.updateModeRaw cachePath baseCall (AnthropicProvider.withMagicSystemPrompt req)
          in wrappedCall
        Just "live" | Just token <- anthropicToken ->
          let headers = AnthropicProvider.oauthHeaders token
              baseCall req = TestHTTP.httpCallStreaming "https://api.anthropic.com/v1/messages" headers req
              wrappedCall req = TestCache.liveMode baseCall (AnthropicProvider.withMagicSystemPrompt req)
          in wrappedCall
        _ -> \req -> TestCache.playbackModeRaw cachePath (AnthropicProvider.withMagicSystemPrompt req)

  -- Build OpenAI streaming response provider (for SSE responses)
  let openaiStreamingProvider :: TestCache.ResponseProvider OpenAIRequest BSL.ByteString
      openaiStreamingProvider = case mode of
        Just "record" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.recordModeRaw cachePath $
            TestHTTP.httpCallStreaming "https://api.openai.com/v1/chat/completions" headers
        Just "update" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.updateModeRaw cachePath $
            TestHTTP.httpCallStreaming "https://api.openai.com/v1/chat/completions" headers
        Just "live" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.liveMode $
            TestHTTP.httpCallStreaming "https://api.openai.com/v1/chat/completions" headers
        _ -> TestCache.playbackModeRaw cachePath

  -- Build Completion response provider (for /v1/completions endpoint)
  let completionProvider :: TestCache.ResponseProvider OpenAICompletionRequest OpenAICompletionResponse
      completionProvider = case mode of
        Just "record" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.recordMode cachePath $
            TestHTTP.httpCall "https://api.openai.com/v1/completions" headers
        Just "update" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.updateMode cachePath $
            TestHTTP.httpCall "https://api.openai.com/v1/completions" headers
        Just "live" | Just apiKey <- openaiApiKey ->
          let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
          in TestCache.liveMode $
            TestHTTP.httpCall "https://api.openai.com/v1/completions" headers
        _ -> TestCache.playbackMode cachePath

  hspec $ do
    describe "Composable Handlers" ComposableHandlersSpec.spec

    -- Property-based tests (QuickCheck)
    describe "Property Tests" PropertySpec.spec
    describe "XML Tool Call Properties" XMLPropertySpec.spec
    describe "Core Types Properties" CoreTypesSpec.spec

    -- Tools tests
    ToolsSpec.spec

    -- Tool definition integration tests
    ToolDefinitionIntegrationSpec.spec

    -- XML Providers integration tests
    XMLProvidersSpec.spec

    -- Composable provider integration tests
    describe "OpenAI Composable Provider (cached)" $ OpenAIComposableSpec.spec openaiProvider
    describe "OpenAI Streaming Provider (cached)" $ OpenAIStreamingSpec.spec openaiStreamingProvider
    describe "Anthropic Composable Provider (cached)" $ AnthropicComposableSpec.spec anthropicProvider
    describe "Anthropic Streaming Provider (cached)" $ AnthropicStreamingSpec.spec anthropicStreamingProvider

    -- Completion interface tests
    describe "OpenAI Completion Interface (cached)" $ CompletionSpec.spec completionProvider

    -- Model Registry Tests (standardized tests for all models)
    let providers = ModelRegistry.Providers
          { ModelRegistry.anthropicProvider = anthropicProvider
          , ModelRegistry.openaiProvider = openaiProvider
          , ModelRegistry.openrouterProvider = openrouterProvider
          , ModelRegistry.llamacppProvider = llamacppProvider
          , ModelRegistry.openaiCompatProvider = openaiCompatProvider
          }
    describe "Model Registry" $ ModelRegistry.modelTests providers

    -- Cache infrastructure tests
    describe "Test Cache" CachedIntegrationSpec.spec
