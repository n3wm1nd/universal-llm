{-# LANGUAGE OverloadedStrings #-}

module Main where

import Test.Hspec
import qualified CachedIntegrationSpec
import qualified ComposableHandlersSpec
import qualified OpenAIComposableSpec
import qualified OpenAIStreamingSpec
import qualified AnthropicComposableSpec
import qualified AnthropicStreamingSpec
import qualified AnthropicStreamingThinkingSpec
import qualified PropertySpec
import qualified XMLPropertySpec
import qualified XMLProvidersSpec
import qualified CoreTypesSpec
import qualified ToolDefinitionIntegrationSpec
import qualified CompletionSpec
import qualified ToolsSpec
import qualified ReasoningConfigSpec
import qualified Protocol.OpenAITests
import qualified Models.ZhipuAI.GLM
import qualified Models.Alibaba.Qwen3Coder
import qualified Models.Google.Gemini
import qualified Models.Amazon.Nova2Lite
import qualified Models.Anthropic.Claude
import qualified Models.Moonshot.KimiK25
import qualified Models.Minimax.MinimaxM25
import qualified Models.OpenAI.GPTOSS
import GGUFNames (canonicalizationTests)
import TestProviders (Providers(..), buildProviders)

main :: IO ()
main = do
  p <- buildProviders

  hspec $ do
    -- Model-specific tests (protocol probes + standard tests)
    describe "Models" $ do
      Models.ZhipuAI.GLM.testsGLM45AirOpenRouter (openrouterProvider p)
      Models.ZhipuAI.GLM.testsGLM45AirLlamaCpp (llamacppProvider p) "GLM-4.5-Air"
      Models.ZhipuAI.GLM.testsGLM45AirZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM45ZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM46ZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM47ZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM5ZAI (zaiProvider p)
      Models.Alibaba.Qwen3Coder.testsLlamaCpp (llamacppProvider p) "Qwen-3-Coder"
      Models.Google.Gemini.testsGemini3FlashOpenRouter (openrouterProvider p)
      Models.Google.Gemini.testsGemini3ProOpenRouter (openrouterProvider p)
      Models.Amazon.Nova2Lite.testsOpenRouter (openrouterProvider p)
      Models.Anthropic.Claude.testsSonnet45 (anthropicProvider p)
      Models.Anthropic.Claude.testsSonnet46 (anthropicProvider p)
      Models.Anthropic.Claude.testsHaiku45 (anthropicProvider p)
      Models.Anthropic.Claude.testsOpus46 (anthropicProvider p)
      Models.Moonshot.KimiK25.testsOpenRouter (openrouterProvider p)
      Models.Minimax.MinimaxM25.testsLlamaCpp (llamacppProvider p) "MiniMax-M2.5"
      Models.OpenAI.GPTOSS.testsLlamaCpp (llamacppProvider p) "gpt-oss-120b"
      Models.OpenAI.GPTOSS.testsOpenRouter (openrouterProvider p)
      Models.Minimax.MinimaxM25.testsOpenRouter (openrouterProvider p)

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
    describe "OpenAI Composable Provider (cached)" $ OpenAIComposableSpec.spec (openrouterProvider p)
    describe "OpenAI Streaming Provider (cached)" $ OpenAIStreamingSpec.spec (openrouterStreamingProvider p)
    describe "Anthropic Composable Provider (cached)" $ AnthropicComposableSpec.spec (anthropicProvider p)
    describe "Anthropic Streaming Provider (cached)" $ AnthropicStreamingSpec.spec (anthropicStreamingProvider p)

    -- Anthropic streaming thinking signature test
    AnthropicStreamingThinkingSpec.spec

    -- Completion interface tests
    describe "OpenAI Completion Interface (cached)" $ CompletionSpec.spec (completionProvider p)

    -- Reasoning config tests (unit tests)
    describe "Reasoning Config" ReasoningConfigSpec.spec

    -- Cache infrastructure tests
    describe "Test Cache" CachedIntegrationSpec.spec

    -- Canonicalization tests
    canonicalizationTests
