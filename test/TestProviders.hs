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
import qualified ErrorClassifier
import GGUFNames (queryLlamaCppModel)

-- | All providers needed by the test suite
data Providers = Providers
  { openaiProvider             :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , openrouterProvider         :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , openrouterStreamingProvider :: TestCache.ResponseProvider OpenAIRequest BSL.ByteString
  , zaiProvider                :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , alibabaCloudProvider       :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , llamacppProvider           :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
  , llamacppStreamingProvider  :: TestCache.ResponseProvider OpenAIRequest BSL.ByteString
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
  alibabaCloudApiKey <- lookupEnv "ALIBABACLOUD_API_KEY"
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
    , alibabaCloudProvider = buildAlibabaCloud mode alibabaCloudApiKey cachePath
    , llamacppProvider = buildLlamaCpp mode llamacppUrl modelMatches cachePath
    , llamacppStreamingProvider = buildLlamaCppStreaming mode llamacppUrl modelMatches cachePath
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
buildOpenAI mode openaiApiKey cachePath =
  let endpoint = "https://api.openai.com/v1/chat/completions"
  in case mode of
    Just "record" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.recordMode cachePath endpoint $
        TestHTTP.httpCall endpoint headers
    Just "update" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.updateMode cachePath endpoint $
        TestHTTP.httpCall endpoint headers
    Just "live" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.liveMode $
        TestHTTP.httpCall endpoint headers
    _ -> TestCache.playbackMode cachePath endpoint

buildOpenRouter :: Maybe String -> Maybe String -> TestCache.CachePath
                -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildOpenRouter mode openrouterApiKey cachePath =
  let endpoint = "https://openrouter.ai/api/v1/chat/completions"
  in case mode of
    Just "record" | Just apiKey <- openrouterApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.recordModeWithErrorCheck cachePath endpoint ErrorClassifier.classifyOpenAIError $
        TestHTTP.httpCall endpoint headers
    Just "update" | Just apiKey <- openrouterApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.updateModeWithErrorCheck cachePath endpoint (Just ErrorClassifier.classifyOpenAIError) $
        TestHTTP.httpCall endpoint headers
    Just "live" | Just apiKey <- openrouterApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.liveMode $
        TestHTTP.httpCall endpoint headers
    _ -> TestCache.playbackMode cachePath endpoint

buildOpenRouterStreaming :: Maybe String -> Maybe String -> TestCache.CachePath
                        -> TestCache.ResponseProvider OpenAIRequest BSL.ByteString
buildOpenRouterStreaming mode openrouterApiKey cachePath =
  let endpoint = "https://openrouter.ai/api/v1/chat/completions"
  in case mode of
    Just "record" | Just apiKey <- openrouterApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.recordModeRaw cachePath endpoint $
        TestHTTP.httpCallStreaming endpoint headers
    Just "update" | Just apiKey <- openrouterApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.updateModeRaw cachePath endpoint $
        TestHTTP.httpCallStreaming endpoint headers
    Just "live" | Just apiKey <- openrouterApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.liveMode $
        TestHTTP.httpCallStreaming endpoint headers
    _ -> TestCache.playbackModeRaw cachePath endpoint

buildZAI :: Maybe String -> Maybe String -> TestCache.CachePath
         -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildZAI mode zaiApiKey cachePath =
  let endpoint = "https://api.z.ai/api/coding/paas/v4/chat/completions"
  in case mode of
    Just "record" | Just apiKey <- zaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.recordModeWithErrorCheck cachePath endpoint ErrorClassifier.classifyOpenAIError $
        TestHTTP.httpCall endpoint headers
    Just "update" | Just apiKey <- zaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.updateModeWithErrorCheck cachePath endpoint (Just ErrorClassifier.classifyOpenAIError) $
        TestHTTP.httpCall endpoint headers
    Just "live" | Just apiKey <- zaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.liveMode $
        TestHTTP.httpCall endpoint headers
    _ -> TestCache.playbackMode cachePath endpoint

buildAlibabaCloud :: Maybe String -> Maybe String -> TestCache.CachePath
                  -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildAlibabaCloud mode alibabaCloudApiKey cachePath =
  let endpoint = "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions"
  in case mode of
    Just "record" | Just apiKey <- alibabaCloudApiKey ->
      let headers = [ ("Content-Type", "application/json")
                    , ("Authorization", T.pack ("Bearer " ++ apiKey))
                    , ("User-Agent", "universal-llm/testharness")
                    ]
      in TestCache.recordMode cachePath endpoint $
        TestHTTP.httpCall endpoint headers
    Just "update" | Just apiKey <- alibabaCloudApiKey ->
      let headers = [ ("Content-Type", "application/json")
                    , ("Authorization", T.pack ("Bearer " ++ apiKey))
                    , ("User-Agent", "universal-llm/testharness")
                    ]
      in TestCache.updateMode cachePath endpoint $
        TestHTTP.httpCall endpoint headers
    Just "live" | Just apiKey <- alibabaCloudApiKey ->
      let headers = [ ("Content-Type", "application/json")
                    , ("Authorization", T.pack ("Bearer " ++ apiKey))
                    , ("User-Agent", "universal-llm/testharness")
                    ]
      in TestCache.liveMode $
        TestHTTP.httpCall endpoint headers
    _ -> TestCache.playbackMode cachePath endpoint

buildLlamaCpp :: Maybe String -> Maybe String -> (OpenAIRequest -> (Bool, String)) -> TestCache.CachePath
              -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildLlamaCpp mode llamacppUrl modelMatches cachePath =
  let canonicalEndpoint = "llamacpp://v1/chat/completions"  -- Fixed endpoint for cache keys
  in case mode of
    Just "record" | Just url <- llamacppUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.recordModeWithFilterMsg cachePath canonicalEndpoint modelMatches $
        TestHTTP.httpCall actualEndpoint [("Content-Type", "application/json")]
    Just "update" | Just url <- llamacppUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.updateModeWithFilterMsg cachePath canonicalEndpoint modelMatches $
        TestHTTP.httpCall actualEndpoint [("Content-Type", "application/json")]
    Just "live" | Just url <- llamacppUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.liveModeWithFilterMsg modelMatches $
        TestHTTP.httpCall actualEndpoint [("Content-Type", "application/json")]
    _ -> TestCache.playbackMode cachePath canonicalEndpoint

buildOpenAICompat :: Maybe String -> Maybe String -> TestCache.CachePath
                  -> TestCache.ResponseProvider OpenAIRequest OpenAIResponse
buildOpenAICompat mode openaiCompatUrl cachePath =
  let canonicalEndpoint = "openai-compat://v1/chat/completions"  -- Fixed endpoint for cache keys
  in case mode of
    Just "record" | Just url <- openaiCompatUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.recordMode cachePath canonicalEndpoint $
        TestHTTP.httpCall actualEndpoint [("Content-Type", "application/json")]
    Just "update" | Just url <- openaiCompatUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.updateMode cachePath canonicalEndpoint $
        TestHTTP.httpCall actualEndpoint [("Content-Type", "application/json")]
    Just "live" | Just url <- openaiCompatUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.liveMode $
        TestHTTP.httpCall actualEndpoint [("Content-Type", "application/json")]
    _ -> TestCache.playbackMode cachePath canonicalEndpoint

buildAnthropic :: Maybe String -> Maybe T.Text -> TestCache.CachePath
               -> TestCache.ResponseProvider AnthropicRequest AnthropicResponse
buildAnthropic mode anthropicToken cachePath =
  let endpoint = "https://api.anthropic.com/v1/messages"
  in case mode of
    Just "record" | Just token <- anthropicToken ->
      let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
          baseCall req = TestHTTP.httpCall endpoint headers req
      in TestCache.recordMode cachePath endpoint baseCall
    Just "update" | Just token <- anthropicToken ->
      let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
          baseCall req = TestHTTP.httpCall endpoint headers req
      in TestCache.updateMode cachePath endpoint baseCall
    Just "live" | Just token <- anthropicToken ->
      let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
          baseCall req = TestHTTP.httpCall endpoint headers req
      in TestCache.liveMode baseCall
    _ -> TestCache.playbackMode cachePath endpoint

buildAnthropicStreaming :: Maybe String -> Maybe T.Text -> TestCache.CachePath
                       -> TestCache.ResponseProvider AnthropicRequest BSL.ByteString
buildAnthropicStreaming mode anthropicToken cachePath =
  let endpoint = "https://api.anthropic.com/v1/messages"
  in case mode of
    Just "record" | Just token <- anthropicToken ->
      let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
          baseCall req = TestHTTP.httpCallStreaming endpoint headers req
      in TestCache.recordModeRaw cachePath endpoint baseCall
    Just "update" | Just token <- anthropicToken ->
      let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
          baseCall req = TestHTTP.httpCallStreaming endpoint headers req
      in TestCache.updateModeRaw cachePath endpoint baseCall
    Just "live" | Just token <- anthropicToken ->
      let headers = ("Content-Type", "application/json") : AnthropicProvider.oauthHeaders token
          baseCall req = TestHTTP.httpCallStreaming endpoint headers req
      in TestCache.liveMode baseCall
    _ -> TestCache.playbackModeRaw cachePath endpoint

buildOpenAIStreaming :: Maybe String -> Maybe String -> TestCache.CachePath
                    -> TestCache.ResponseProvider OpenAIRequest BSL.ByteString
buildOpenAIStreaming mode openaiApiKey cachePath =
  let endpoint = "https://api.openai.com/v1/chat/completions"
  in case mode of
    Just "record" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.recordModeRaw cachePath endpoint $
        TestHTTP.httpCallStreaming endpoint headers
    Just "update" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.updateModeRaw cachePath endpoint $
        TestHTTP.httpCallStreaming endpoint headers
    Just "live" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.liveMode $
        TestHTTP.httpCallStreaming endpoint headers
    _ -> TestCache.playbackModeRaw cachePath endpoint

buildLlamaCppStreaming :: Maybe String -> Maybe String -> (OpenAIRequest -> (Bool, String)) -> TestCache.CachePath
                      -> TestCache.ResponseProvider OpenAIRequest BSL.ByteString
buildLlamaCppStreaming mode llamacppUrl modelMatches cachePath =
  let canonicalEndpoint = "llamacpp://v1/chat/completions"  -- Fixed endpoint for cache keys
  in case mode of
    Just "record" | Just url <- llamacppUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.recordModeRawWithFilterMsg cachePath canonicalEndpoint modelMatches $
        TestHTTP.httpCallStreaming actualEndpoint [("Content-Type", "application/json")]
    Just "update" | Just url <- llamacppUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.updateModeRawWithFilterMsg cachePath canonicalEndpoint modelMatches $
        TestHTTP.httpCallStreaming actualEndpoint [("Content-Type", "application/json")]
    Just "live" | Just url <- llamacppUrl ->
      let actualEndpoint = url ++ "/v1/chat/completions"
      in TestCache.liveModeWithFilterMsg modelMatches $
        TestHTTP.httpCallStreaming actualEndpoint [("Content-Type", "application/json")]
    _ -> TestCache.playbackModeRaw cachePath canonicalEndpoint

buildCompletion :: Maybe String -> Maybe String -> TestCache.CachePath
               -> TestCache.ResponseProvider OpenAICompletionRequest OpenAICompletionResponse
buildCompletion mode openaiApiKey cachePath =
  let endpoint = "https://api.openai.com/v1/completions"
  in case mode of
    Just "record" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.recordMode cachePath endpoint $
        TestHTTP.httpCall endpoint headers
    Just "update" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.updateMode cachePath endpoint $
        TestHTTP.httpCall endpoint headers
    Just "live" | Just apiKey <- openaiApiKey ->
      let headers = [("Content-Type", "application/json"), ("Authorization", T.pack ("Bearer " ++ apiKey))]
      in TestCache.liveMode $
        TestHTTP.httpCall endpoint headers
    _ -> TestCache.playbackMode cachePath endpoint
