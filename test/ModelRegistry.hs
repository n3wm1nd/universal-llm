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
import qualified StandardTests as ST
import TestHelpers (testModel)
import UniversalLLM (Model(..), Via(..))
import UniversalLLM.Providers.Anthropic (Anthropic(..), AnthropicOAuth(..))
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import UniversalLLM.Models.Anthropic
  ( ClaudeSonnet45(..)
  , ClaudeSonnet45NoReason(..)
  , claudeSonnet45OAuth
  , claudeSonnet45NoReasonOAuth
  )
import UniversalLLM.Models.GLM
  ( GLM45Air(..)
  , GLM5(..)
  , ZAI(..)
  , glm45AirLlamaCpp
  , glm45AirOpenRouter
  , glm45AirZAI
  , glm5
  )
import UniversalLLM.Models.OpenRouter
  ( Gemini3FlashPreview(..)
  , Gemini3ProPreview(..)
  , Nova2Lite(..)
  , gemini3FlashPreview
  , gemini3ProPreview
  , nova2Lite
  )
import UniversalLLM.Models.KimiK25
  ( KimiK25(..)
  , kimiK25
  )
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
  describe "Claude Sonnet 4.5 (no reasoning)" $
    testModel claudeSonnet45NoReasonOAuth (Model ClaudeSonnet45NoReason AnthropicOAuth) (anthropicProvider providers)
      [ ST.text, ST.tools ]

  describe "Claude Sonnet 4.5" $
    testModel claudeSonnet45OAuth (Model ClaudeSonnet45 AnthropicOAuth) (anthropicProvider providers)
      [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]

  -- LlamaCpp Models
  -- GLM 4.5 via llama.cpp - Test against llama.cpp server when the model is loaded
  describe "GLM 4.5 (llama.cpp)" $
    testModel glm45AirLlamaCpp (Model GLM45Air LlamaCpp) (llamacppProvider providers)
      [ ST.text, ST.tools, ST.reasoning ]

  -- OpenRouter Models
  -- GLM 4.5 Air via OpenRouter
  describe "GLM 4.5 (OpenRouter)" $
    testModel glm45AirOpenRouter (Model GLM45Air OpenRouter) (openrouterProvider providers)
      [ ST.text, ST.tools, ST.reasoning ]

  -- Amazon Nova 2 Lite via OpenRouter
  -- Note: Uses normalizeEmptyContent to work with Bedrock, so can't do verbatim preservation
  -- Nova doesn't return AssistantReasoning via OpenRouter
  describe "Amazon Nova 2 Lite (OpenRouter)" $
    testModel nova2Lite (Model Nova2Lite OpenRouter) (openrouterProvider providers)
      [ ST.text, ST.tools ]

  -- Google Gemini 3 Pro Preview via OpenRouter - Test reasoning_details preservation with tool calls
  describe "Gemini 3 Pro Preview (OpenRouter)" $
    testModel gemini3ProPreview (Model Gemini3ProPreview OpenRouter) (openrouterProvider providers)
      [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

  -- Google Gemini 3 Flash Preview via OpenRouter - High speed reasoning model with tool use
  describe "Gemini 3 Flash Preview (OpenRouter)" $
    testModel gemini3FlashPreview (Model Gemini3FlashPreview OpenRouter) (openrouterProvider providers)
      [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

  -- Moonshot AI Models
  describe "Kimi K2.5 (OpenRouter)" $
    testModel kimiK25 (Model KimiK25 OpenRouter) (openrouterProvider providers)
      [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools, ST.openAIReasoningDetailsPreservation ]

  -- ZAI Models
  -- GLM 4.5 via ZAI coding endpoint
  describe "GLM 4.5 (ZAI)" $
    testModel glm45AirZAI (Model GLM45Air ZAI) (zaiProvider providers)
      [ ST.text, ST.tools, ST.reasoning ]

  -- GLM 5 via ZAI coding endpoint (requires higher-tier subscription)
  xdescribe "GLM 5 (ZAI)" $
    testModel glm5 (Model GLM5 ZAI) (zaiProvider providers)
      [ ST.text, ST.tools, ST.reasoning ]
