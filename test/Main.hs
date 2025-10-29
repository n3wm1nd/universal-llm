{-# LANGUAGE OverloadedStrings #-}

module Main where

import Test.Hspec
import qualified CachedIntegrationSpec
import qualified ComposableHandlersSpec
import qualified OpenAIComposableSpec
import qualified AnthropicComposableSpec
import qualified PropertySpec
import qualified XMLPropertySpec
import qualified XMLProvidersSpec
import qualified CoreTypesSpec
import qualified ToolDefinitionIntegrationSpec
import qualified TestCache
import qualified TestHTTP
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import System.Environment (lookupEnv)
import qualified Data.Text as T

main :: IO ()
main = do
  -- Determine which mode to use based on environment variable
  mode <- lookupEnv "TEST_MODE"
  anthropicToken <- fmap T.pack <$> lookupEnv "ANTHROPIC_OAUTH_TOKEN"
  openaiUrl <- lookupEnv "OPENAI_SERVER_URL"

  let cachePath = "test-cache"

  -- Build OpenAI response provider
  let openaiProvider :: TestCache.ResponseProvider OpenAIRequest OpenAIResponse
      openaiProvider = case mode of
        Just "record" | Just url <- openaiUrl ->
          TestCache.recordMode cachePath $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        Just "update" | Just url <- openaiUrl ->
          TestCache.updateMode cachePath $
            TestHTTP.httpCall (url ++ "/v1/chat/completions") [("Content-Type", "application/json")]
        Just "live" | Just url <- openaiUrl ->
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

  hspec $ do
    describe "Composable Handlers" ComposableHandlersSpec.spec

    -- Property-based tests (QuickCheck)
    describe "Property Tests" PropertySpec.spec
    describe "XML Tool Call Properties" XMLPropertySpec.spec
    describe "Core Types Properties" CoreTypesSpec.spec

    -- Tool definition integration tests
    ToolDefinitionIntegrationSpec.spec

    -- XML Providers integration tests
    XMLProvidersSpec.spec

    -- Composable provider integration tests
    describe "OpenAI Composable Provider (cached)" $ OpenAIComposableSpec.spec openaiProvider
    describe "Anthropic Composable Provider (cached)" $ AnthropicComposableSpec.spec anthropicProvider

    -- Cache infrastructure tests
    describe "Test Cache" CachedIntegrationSpec.spec
