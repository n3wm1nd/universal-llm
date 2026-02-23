{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TestProviders
  ( Providers(..)
  , buildProviders
  ) where

import qualified Data.Text as T
import qualified Data.ByteString.Lazy as BSL
import System.Environment (lookupEnv)

import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse, OpenAICompletionRequest, OpenAICompletionResponse)
import qualified UniversalLLM.Protocols.OpenAI as OpenAI
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import qualified TestCache
import qualified TestHTTP
import GGUFNames (queryLlamaCppModel)

-- | All providers needed by the test suite
data Providers = Providers
  { openaiProvider             :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , openrouterProvider         :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , openrouterStreamingProvider :: TestCache.ResponseProvider OpenAIRequest BSL.ByteString
  , zaiProvider                :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , llamacppProvider           :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , openaiCompatProvider       :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , anthropicProvider          :: TestCache.ResponseProvider AnthropicRequest AnthropicResponse
  , anthropicStreamingProvider :: TestCache.ResponseProvider AnthropicRequest BSL.ByteString
  , openaiStreamingProvider    :: TestCache.ResponseProvider OpenAIRequest BSL.ByteString
  , completionProvider         :: TestCache.ResponseProvider OpenAICompletionRequest OpenAICompletionResponse
  , llamacppLoadedModel        :: Maybe [String]
  }

-- | Build all providers from environment variables
buildProviders :: IO Providers
buildProviders = do
  mode <- lookupEnv "TEST_MODE"

  -- API credentials and server URLs
  anthropicToken <- fmap T.pack <$> lookupEnv "ANTHROPIC_OAUTH_TOKEN"
  openaiApiKey <- lookupEnv "OPENAI_API_KEY"
  openrouterApiKey <- lookupEnv "OPENROUTER_API_KEY"
  zaiApiKey <- lookupEnv "ZAI_API_KEY"
  llamacppUrl <- lookupEnv "LLAMACPP_ENDPOINT"
  openaiCompatUrl <- lookupEnv "OPENAI_COMPAT_URL"
  _openaiCompatModel <- lookupEnv "OPENAI_COMPAT_MODEL"

  let cachePath = "test-cache"

  -- Query llama.cpp server for loaded model (if URL is set)
  loadedModel <- case llamacppUrl of
    Just url -> queryLlamaCppModel url
    Nothing -> return Nothing

  -- Print info about llama.cpp model
  case (llamacppUrl, loadedModel) of
    (Just url, Just loadedModels) ->
      putStrLn $ "llama.cpp server at " ++ url ++ " has model loaded. Variants: " ++ show loadedModels
    (Just url, Nothing) ->
      putStrLn $ "Warning: Could not query llama.cpp server at " ++ url
    _ -> return ()

  -- Helper: Check if requested model matches any of the loaded model variants
  let modelMatches :: OpenAIRequest -> (Bool, String)
      modelMatches req = case loadedModel of
        Nothing -> (True, "")
        Just loadedModels ->
          let requestedModel = T.unpack (OpenAI.model req)
              matches = requestedModel `elem` loadedModels
              loadedModelName = case loadedModels of
                (first:_) -> show first
                [] -> "unknown"
              errMsg = if matches then ""
                       else "Skipped: loaded model " ++ loadedModelName
                            ++ " does not match requested model " ++ show requestedModel
          in (matches, errMsg)

  return Providers
    { openaiProvider = buildOpenAI mode openaiApiKey cachePath
    , openrouterProvider = buildOpenRouter mode openrouterApiKey cachePath
    , openrouterStreamingProvider = buildOpenRouterStreaming mode openrouterApiKey cachePath
    , zaiProvider = buildZAI mode zaiApiKey cachePath
    , llamacppProvider = buildLlamaCpp mode llamacppUrl modelMatches cachePath
    , openaiCompatProvider = buildOpenAICompat mode openaiCompatUrl cachePath
    , anthropicProvider = buildAnthropic mode anthropicToken cachePath
    , anthropicStreamingProvider = buildAnthropicStreaming mode anthropicToken cachePath
    , openaiStreamingProvider = buildOpenAIStreaming mode openaiApiKey cachePath
    , completionProvider = buildCompletion mode openaiApiKey cachePath
    , llamacppLoadedModel = loadedModel
    }

-- Individual provider builders

buildOpenAI :: Maybe String -> Maybe String -> TestCache.CachePath
            -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildOpenAI mode openaiApiKey cachePath = case mode of
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

buildOpenRouter :: Maybe String -> Maybe String -> TestCache.CachePath
                -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildOpenRouter mode openrouterApiKey cachePath = case mode of
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

buildOpenRouterStreaming :: Maybe String -> Maybe String -> TestCache.CachePath
                        -> TestCache.ResponseProvider OpenAIRequest BSL.ByteString
buildOpenRouterStreaming mode openrouterApiKey cachePath = case mode of
  Just "record" | Just apiKey <- openrouterApiKey ->
    let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
    in TestCache.recordModeRaw cachePath $
      TestHTTP.httpCallStreaming "https://openrouter.ai/api/v1/chat/completions" headers
  Just "update" | Just apiKey <- openrouterApiKey ->
    let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
    in TestCache.updateModeRaw cachePath $
      TestHTTP.httpCallStreaming "https://openrouter.ai/api/v1/chat/completions" headers
  Just "live" | Just apiKey <- openrouterApiKey ->
    let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
    in TestCache.liveMode $
      TestHTTP.httpCallStreaming "https://openrouter.ai/api/v1/chat/completions" headers
  _ -> TestCache.playbackModeRaw cachePath

buildZAI :: Maybe String -> Maybe String -> TestCache.CachePath
         -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildZAI mode zaiApiKey cachePath = case mode of
  Just "record" | Just apiKey <- zaiApiKey ->
    let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
    in TestCache.recordMode cachePath $
      TestHTTP.httpCall "https://api.z.ai/api/coding/paas/v4/chat/completions" headers
  Just "update" | Just apiKey <- zaiApiKey ->
    let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
    in TestCache.updateMode cachePath $
      TestHTTP.httpCall "https://api.z.ai/api/coding/paas/v4/chat/completions" headers
  Just "live" | Just apiKey <- zaiApiKey ->
    let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
    in TestCache.liveMode $
      TestHTTP.httpCall "https://api.z.ai/api/coding/paas/v4/chat/completions" headers
  _ -> TestCache.playbackMode cachePath

buildLlamaCpp :: Maybe String -> Maybe String -> (OpenAIRequest -> (Bool, String)) -> TestCache.CachePath
              -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildLlamaCpp mode llamacppUrl modelMatches cachePath = case mode of
  Just "record" | Just url <- llamacppUrl ->
    TestCache.recordModeWithFilterMsg cachePath modelMatches $
      TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
  Just "update" | Just url <- llamacppUrl ->
    TestCache.updateModeWithFilterMsg cachePath modelMatches $
      TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
  Just "live" | Just url <- llamacppUrl ->
    TestCache.liveModeWithFilterMsg modelMatches $
      TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
  _ -> TestCache.playbackMode cachePath

buildOpenAICompat :: Maybe String -> Maybe String -> TestCache.CachePath
                  -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildOpenAICompat mode openaiCompatUrl cachePath = case mode of
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

buildAnthropic :: Maybe String -> Maybe T.Text -> TestCache.CachePath
               -> TestCache.ResponseProvider AnthropicRequest AnthropicResponse
buildAnthropic mode anthropicToken cachePath = case mode of
  Just "record" | Just token <- anthropicToken ->
    let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
        baseCall req = TestHTTP.httpCall "https://api.anthropic.com/v1/messages" headers req
    in TestCache.recordMode cachePath baseCall
  Just "update" | Just token <- anthropicToken ->
    let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
        baseCall req = TestHTTP.httpCall "https://api.anthropic.com/v1/messages" headers req
    in TestCache.updateMode cachePath baseCall
  Just "live" | Just token <- anthropicToken ->
    let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
        baseCall req = TestHTTP.httpCall "https://api.anthropic.com/v1/messages" headers req
    in TestCache.liveMode baseCall
  _ -> TestCache.playbackMode cachePath

buildAnthropicStreaming :: Maybe String -> Maybe T.Text -> TestCache.CachePath
                       -> TestCache.ResponseProvider AnthropicRequest BSL.ByteString
buildAnthropicStreaming mode anthropicToken cachePath = case mode of
  Just "record" | Just token <- anthropicToken ->
    let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
        baseCall req = TestHTTP.httpCallStreaming "https://api.anthropic.com/v1/messages" headers req
    in TestCache.recordModeRaw cachePath baseCall
  Just "update" | Just token <- anthropicToken ->
    let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
        baseCall req = TestHTTP.httpCallStreaming "https://api.anthropic.com/v1/messages" headers req
    in TestCache.updateModeRaw cachePath baseCall
  Just "live" | Just token <- anthropicToken ->
    let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
        baseCall req = TestHTTP.httpCallStreaming "https://api.anthropic.com/v1/messages" headers req
    in TestCache.liveMode baseCall
  _ -> TestCache.playbackModeRaw cachePath

buildOpenAIStreaming :: Maybe String -> Maybe String -> TestCache.CachePath
                    -> TestCache.ResponseProvider OpenAIRequest BSL.ByteString
buildOpenAIStreaming mode openaiApiKey cachePath = case mode of
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

buildCompletion :: Maybe String -> Maybe String -> TestCache.CachePath
               -> TestCache.ResponseProvider OpenAICompletionRequest OpenAICompletionResponse
buildCompletion mode openaiApiKey cachePath = case mode of
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
