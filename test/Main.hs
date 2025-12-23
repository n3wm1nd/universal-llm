{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Test.Hspec
import Data.List (intercalate, nub, isInfixOf)
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
import qualified ReasoningConfigSpec
import qualified Protocol.OpenAITests
import qualified Models.GLM45Air
import qualified Models.Qwen3Coder
import qualified Models.Gemini3Flash
import qualified Models.Nova2Lite
import qualified Models.Claude
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse(..), OpenAIErrorResponse(..), OpenAIErrorDetail(..), OpenAICompletionRequest, OpenAICompletionResponse)
import qualified UniversalLLM.Protocols.OpenAI as OpenAI
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import System.Environment (lookupEnv)
import qualified Data.Text as T
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KeyMap
import qualified Data.Vector as V
import Control.Exception (catch, SomeException)
import System.FilePath (takeBaseName, takeExtension)
import Data.Char (toLower, isDigit, isUpper)

-- | Predicates to identify different types of model name parts
-- These are intentionally overlapping - context determines meaning

-- | Multi-part file indicators: "0001", "of", "0002"
isMultiPart :: String -> Bool
isMultiPart "of" = True
isMultiPart p = all isDigit p && length p >= 4

-- | Quantization methods: "Q8_K_XL", "Q4_K_M"
isQuantization :: String -> Bool
isQuantization ('Q':d:rest) | isDigit d && '_' `elem` rest = True
isQuantization _ = False

-- | Short version/variant tags: "UD", "A3B"
-- Must NOT be a size parameter (those end in B/M and should be kept)
isVersionTag :: String -> Bool
isVersionTag p = length p <= 3
              && all isUpperOrDigit p
              && any isUpper p
              && not (isSizeParam p)  -- Don't treat size params as version tags
  where isUpperOrDigit c = isUpper c || isDigit c

-- | Parameter count: "30B", "7B", "1.5B"
isSizeParam :: String -> Bool
isSizeParam p = case reverse p of
  ('B':rest) | not (null rest) && all isDigitOrDot rest -> True
  ('M':rest) | not (null rest) && all isDigitOrDot rest -> True
  _ -> False
  where isDigitOrDot c = isDigit c || c == '.'

-- | Known tuning/training indicators
isTuning :: String -> Bool
isTuning "Instruct" = True
isTuning "Chat" = True
isTuning "Code" = True
isTuning "Coder" = True
isTuning _ = False

-- | Split string on delimiter
splitOn :: Char -> String -> [String]
splitOn _ "" = []
splitOn delim str =
  let (before, remainder) = break (== delim) str
  in before : case remainder of
                [] -> []
                (_:after) -> splitOn delim after

-- | Canonicalize a GGUF filename to possible model names
-- Example: "Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf"
-- Returns: Multiple variants including:
--   - Progressive truncation: "Qwen3-Coder-30B-Instruct", "Qwen3-Coder-30B", "Qwen3-Coder"
--   - Without size params: "Qwen3-Coder-Instruct"
--   - Dash variants: "Qwen-3-Coder"
--   - Lowercase versions of all
canonicalizeGGUFNames :: String -> [String]
canonicalizeGGUFNames filename =
  let base = takeBaseName filename
      parts = splitOn '-' base
      -- Drop junk from the END (quantization, version tags, multi-part)
      cleaned = reverse $ dropWhile isJunk (reverse parts)
      -- Generate variants
      progressive = [intercalate "-" (take n cleaned) | n <- reverse [1..length cleaned], n > 0]
      -- Generate progressive variants on the cleaned list (with version tags removed)
      noVersions = filter (not . isVersionTag) cleaned
      progressiveNoVer = [intercalate "-" (take n noVersions) | n <- reverse [1..length noVersions], n > 0]
      -- Also generate variants with optional parts filtered out
      withoutOptional = [intercalate "-" (filter isCore cleaned)]  -- Drop size + version
      allVariants = nub (progressive ++ progressiveNoVer ++ withoutOptional)
      -- Add dash-in-name variants ("Qwen3" -> "Qwen-3")
      withDashes = concatMap addDashInName allVariants
      -- Add lowercase
      withCase = nub (withDashes ++ map (map toLower) withDashes)
      -- Filter out empty strings
      valid = filter (not . null) withCase
  in valid
  where
    -- Parts to drop from the end only
    isJunk p = isMultiPart p || isQuantization p || isVersionTag p

    -- Core parts to keep when filtering (not size params or version tags)
    isCore p = not (isSizeParam p || isVersionTag p)

    -- Add dash before first digit in parts: "Qwen3-Coder" -> "Qwen-3-Coder"
    addDashInName :: String -> [String]
    addDashInName s =
      let parts' = splitOn '-' s
          transformed = map dashBeforeDigit parts'
          result = intercalate "-" transformed
      in if result == s then [s] else [s, result]

    -- "Qwen3" -> "Qwen-3", but "4.5" stays "4.5"
    dashBeforeDigit :: String -> String
    dashBeforeDigit part = case break isDigit part of
      (prefix, suffix) | not (null prefix) && not (null suffix) && head suffix /= '.'
        -> prefix ++ "-" ++ suffix
      _ -> part

-- | Query llama.cpp server for loaded model information
-- Returns the list of possible canonicalized model names
queryLlamaCppModel :: String -> IO (Maybe [String])
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
                    return $ Just (canonicalizeGGUFNames $ T.unpack modelId)
                  _ -> return Nothing
              _ -> return Nothing
          _ -> return Nothing
      _ -> return Nothing
    ) (\(_ :: SomeException) -> return Nothing)

-- | Test suite for canonicalization
canonicalizationTests :: Spec
canonicalizationTests = describe "GGUF Name Canonicalization" $ do
  it "handles GLM-4.5-Air with quantization" $ do
    let result = canonicalizeGGUFNames "GLM-4.5-Air-Q4_K_M.gguf"
    result `shouldContain` ["GLM-4.5-Air"]
    result `shouldContain` ["glm-4.5-air"]
    result `shouldContain` ["GLM-4.5"]
    result `shouldContain` ["GLM"]

  it "handles Qwen with size param and variants" $ do
    let result = canonicalizeGGUFNames "Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf"
    result `shouldContain` ["Qwen3-Coder-30B-Instruct"]
    result `shouldContain` ["Qwen3-Coder-Instruct"]  -- Without size
    result `shouldContain` ["Qwen3-Coder"]
    result `shouldContain` ["Qwen-3-Coder"]  -- Dash variant
    result `shouldContain` ["qwen-3-coder"]  -- Lowercase

main :: IO ()
main = do
  -- Determine which mode to use based on environment variable
  mode <- lookupEnv "TEST_MODE"

  -- API credentials and server URLs
  anthropicToken <- fmap T.pack <$> lookupEnv "ANTHROPIC_OAUTH_TOKEN"
  openaiApiKey <- lookupEnv "OPENAI_API_KEY"
  openrouterApiKey <- lookupEnv "OPENROUTER_API_KEY"
  zaiApiKey <- lookupEnv "ZAI_API_KEY"
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

  -- Build OpenRouter streaming response provider (for SSE responses)
  let openrouterStreamingProvider :: TestCache.ResponseProvider OpenAIRequest BSL.ByteString
      openrouterStreamingProvider = case mode of
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

  -- Build ZAI response provider
  let zaiProvider :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
      zaiProvider = case mode of
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

  -- Query llama.cpp server for loaded model (if URL is set)
  llamacppLoadedModel <- case llamacppUrl of
    Just url -> queryLlamaCppModel url
    Nothing -> return Nothing

  -- Print info about llama.cpp model
  case (llamacppUrl, llamacppLoadedModel) of
    (Just url, Just loadedModels) ->
      putStrLn $ "llama.cpp server at " ++ url ++ " has model loaded. Variants: " ++ show loadedModels
    (Just url, Nothing) ->
      putStrLn $ "Warning: Could not query llama.cpp server at " ++ url
    _ -> return ()

  -- Helper: Check if requested model matches any of the loaded model variants
  -- Returns (matches, error message) for better pending messages
  let modelMatches :: OpenAIRequest -> (Bool, String)
      modelMatches req = case llamacppLoadedModel of
        Nothing -> (True, "")  -- No loaded model info, proceed anyway
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

  -- Build llama.cpp response provider
  -- Only makes live requests if model name matches the request's model field
  let llamacppProvider :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
      llamacppProvider = case mode of
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
    -- Model-specific tests (protocol probes + standard tests)
    describe "Models" $ do
      Models.GLM45Air.testsOpenRouter openrouterProvider
      Models.GLM45Air.testsLlamaCpp llamacppProvider "GLM-4.5-Air"
      Models.Qwen3Coder.testsLlamaCpp llamacppProvider "Qwen-3-Coder"
      Models.Gemini3Flash.testsOpenRouter openrouterProvider
      Models.Nova2Lite.testsOpenRouter openrouterProvider
      Models.Claude.testsSonnet45 anthropicProvider

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
    describe "OpenAI Composable Provider (cached)" $ OpenAIComposableSpec.spec openrouterProvider
    describe "OpenAI Streaming Provider (cached)" $ OpenAIStreamingSpec.spec openrouterStreamingProvider
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
          , ModelRegistry.zaiProvider = zaiProvider
          }
    describe "Model Registry" $ ModelRegistry.modelTests providers

    -- Reasoning config tests (unit tests)
    describe "Reasoning Config" ReasoningConfigSpec.spec

    -- Cache infrastructure tests
    describe "Test Cache" CachedIntegrationSpec.spec

    -- Canonicalization tests
    canonicalizationTests
