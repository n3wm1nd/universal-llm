{-# LANGUAGE OverloadedStrings #-}

-- | Model Test Registry
--
-- This is the SINGLE PLACE where models are registered for testing.
-- When you implement a new model, add it here with the test suites you want to run.
--
-- Usage:
--   describe "My New Model" $ testModel MyProvider MyModel getResponse [text, tools, reasoning]
--
module ModelRegistry (modelTests, Providers(..)) where

import Test.Hspec
import qualified TestModels
import qualified StandardTests as ST
import TestHelpers (testModel)
import UniversalLLM.Providers.Anthropic (Anthropic(..))
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import TestModels (ZAI(..))
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import TestCache (ResponseProvider)

-- ============================================================================
-- Model Test Registry
-- ============================================================================
--
-- Add your model here! Each is a single describe + testModel call.

data Providers = Providers
  { anthropicProvider :: ResponseProvider AnthropicRequest AnthropicResponse
  , openaiProvider :: ResponseProvider OpenAIRequest OpenAIResponse
  , openrouterProvider :: ResponseProvider OpenAIRequest OpenAIResponse
  , llamacppProvider :: ResponseProvider OpenAIRequest OpenAIResponse
  , openaiCompatProvider :: ResponseProvider OpenAIRequest OpenAIResponse
  , zaiProvider :: ResponseProvider OpenAIRequest OpenAIResponse
  }

modelTests :: Providers -> Spec
modelTests providers = do

  -- Anthropic Models
  describe "Claude Sonnet 4.5" $
    testModel TestModels.anthropicSonnet45 Anthropic TestModels.ClaudeSonnet45 (anthropicProvider providers)
      [ ST.text, ST.tools ]

  describe "Claude Sonnet 4.5 with Reasoning" $
    testModel TestModels.anthropicSonnet45Reasoning Anthropic TestModels.ClaudeSonnet45WithReasoning (anthropicProvider providers)
      [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]

  -- LlamaCpp Models
  -- GLM 4.5 via llama.cpp - Test against llama.cpp server when the model is loaded
  describe "GLM 4.5 (llama.cpp)" $
    testModel TestModels.llamaCppGLM45 LlamaCpp TestModels.GLM45 (llamacppProvider providers)
      [ ST.text, ST.tools, ST.reasoning ]

  -- OpenRouter Models
  -- GLM 4.5 via OpenRouter
  describe "GLM 4.5 (OpenRouter)" $
    testModel TestModels.openRouterGLM45 OpenRouter TestModels.GLM45 (openrouterProvider providers)
      [ ST.text, ST.tools, ST.reasoning ]

  -- Amazon Nova 2 Lite via OpenRouter - Note: Nova doesn't return AssistantReasoning via OpenRouter
  describe "Amazon Nova 2 Lite (OpenRouter)" $
    testModel TestModels.openRouterNova2Lite OpenRouter TestModels.Nova2Lite (openrouterProvider providers)
      [ ST.text, ST.tools, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

  -- Google Gemini 3 Pro Preview via OpenRouter - Test reasoning_details preservation with tool calls
  describe "Gemini 3 Pro Preview (OpenRouter)" $
    testModel TestModels.openRouterGemini3ProPreview OpenRouter TestModels.Gemini3ProPreview (openrouterProvider providers)
      [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

  -- ZAI Models
  -- GLM 4.5 via ZAI coding endpoint
  describe "GLM 4.5 (ZAI)" $
    testModel TestModels.zaiGLM45 ZAI TestModels.GLM45 (zaiProvider providers)
      [ ST.text, ST.tools, ST.reasoning ]
