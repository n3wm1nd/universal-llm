{-# LANGUAGE OverloadedStrings #-}

module Main where

import Test.Hspec
import qualified CachedIntegrationSpec
import qualified DeltaSpec
import qualified ComposableHandlersSpec
import qualified OpenAIComposableSpec
import qualified OpenAIStreamingSpec
import qualified StreamingReconstructionSpec
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
import qualified Models.Alibaba.Qwen
import qualified Models.Google.Gemini
import qualified Models.Amazon.Nova2Lite
import qualified Models.Anthropic.Claude
import qualified Models.Moonshot.KimiK25
import qualified Models.Minimax.MinimaxM25
import qualified Models.OpenAI.GPT
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
      Models.ZhipuAI.GLM.testsGLM47OpenRouter (openrouterProvider p)
      Models.ZhipuAI.GLM.testsGLM47AlibabaCloud (alibabaCloudCodingProvider p)
      Models.ZhipuAI.GLM.testsGLM47FlashZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM47FlashOpenRouter (openrouterProvider p)
      Models.ZhipuAI.GLM.testsGLM47FlashLlamaCpp (llamacppProvider p) "GLM-4.7-Flash"

      Models.ZhipuAI.GLM.testsGLM5ZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM5AlibabaCloud (alibabaCloudCodingProvider p)
      Models.ZhipuAI.GLM.testsGLM51ZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM52ZAI (zaiProvider p)
      Models.ZhipuAI.GLM.testsGLM5TurboZAI (zaiProvider p)
      Models.Alibaba.Qwen.testsQwen35_122BOpenRouter (openrouterProvider p)
      Models.Alibaba.Qwen.testsQwen35_122BLlamaCpp (llamacppProvider p) "Qwen3.5-122B"
      Models.Alibaba.Qwen.testsQwen35_40BLlamaCpp (llamacppProvider p) "Qwen3.5-40B"
      Models.Alibaba.Qwen.testsQwen35PlusAlibabaCloud (alibabaCloudCodingProvider p)
      Models.Alibaba.Qwen.testsQwen36PlusAlibabaCloud (alibabaCloudCodingProvider p)
      Models.Alibaba.Qwen.testsQwen36PlusOpenRouter (openrouterProvider p)
      Models.Alibaba.Qwen.testsQwen37MaxOpenRouter (openrouterProvider p)
      Models.Alibaba.Qwen.testsQwen3CoderNextLlamaCpp (llamacppProvider p) "Qwen3-Coder-Next"
      Models.Alibaba.Qwen.testsQwen3Coder30bInstructLlamaCpp (llamacppProvider p) "Qwen3-Coder-30B-Instruct"
      Models.Alibaba.Qwen.testsQwen3CoderNextAlibabaCloud (alibabaCloudCodingProvider p)
      Models.Alibaba.Qwen.testsQwen3CoderPlusAlibabaCloud (alibabaCloudCodingProvider p)
      Models.Google.Gemini.testsGemini3FlashOpenRouter (openrouterProvider p)
      Models.Google.Gemini.testsGemini3ProOpenRouter (openrouterProvider p)
      Models.Amazon.Nova2Lite.testsOpenRouter (openrouterProvider p)
      Models.Anthropic.Claude.testsSonnet45 (anthropicOAuthProvider p)
      Models.Anthropic.Claude.testsSonnet46 (anthropicOAuthProvider p)
      Models.Anthropic.Claude.testsHaiku45 (anthropicOAuthProvider p)
      Models.Anthropic.Claude.testsOpus46 (anthropicOAuthProvider p)
      Models.Anthropic.Claude.testsOpus48 (anthropicOAuthProvider p)
      Models.Anthropic.Claude.testsFable5 (anthropicOAuthProvider p)
      Models.Moonshot.KimiK25.testsOpenRouter (openrouterProvider p)
      Models.Moonshot.KimiK25.testsAlibabaCloud (alibabaCloudCodingProvider p)
      Models.Minimax.MinimaxM25.testsOpenRouter (openrouterProvider p)
      Models.Minimax.MinimaxM25.testsLlamaCpp (llamacppProvider p) "MiniMax-M2.5"
      Models.Minimax.MinimaxM25.testsAlibabaCloud (alibabaCloudCodingProvider p)
      Models.OpenAI.GPT.testsGPTOSSOpenRouter (openrouterProvider p)
      Models.OpenAI.GPT.testsGPTOSSLlamaCpp (llamacppProvider p) "gpt-oss-120b"
      Models.OpenAI.GPT.testsGPT53CodexOpenRouter (openrouterProvider p)
      Models.OpenAI.GPT.testsGPT53ChatOpenRouter (openrouterProvider p)
      Models.OpenAI.GPT.testsGPT54ProOpenRouter (openrouterProvider p)
      Models.OpenAI.GPT.testsGPT54OpenRouter (openrouterProvider p)

    describe "Delta" DeltaSpec.spec
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
    describe "Streaming Reconstruction Tests (llamacpp)" $ StreamingReconstructionSpec.spec (llamacppLoadedModel p) (llamacppProvider p) (llamacppStreamingProvider p)
    describe "Anthropic Composable Provider (cached)" $ AnthropicComposableSpec.spec (anthropicOAuthProvider p)
    describe "Anthropic Streaming Provider (cached)" $ AnthropicStreamingSpec.spec (anthropicOAuthStreamingProvider p)

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
