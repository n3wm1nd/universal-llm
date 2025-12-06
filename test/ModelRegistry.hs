{-# LANGUAGE OverloadedStrings #-}

-- | Model Test Registry
--
-- This is the SINGLE PLACE where models are registered for testing.
-- When you implement a new model, add it here with the test suites you want to run.
--
-- Usage:
--   describe "My New Model" $ testModel MyProvider MyModel [text, tools, reasoning]
--
module ModelRegistry (modelTests) where

import Test.Hspec
import qualified TestModels
import qualified StandardTests as ST
import StandardTests (StandardTest(..))
import UniversalLLM.Core.Types
import UniversalLLM.Providers.Anthropic (Anthropic(..))
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider
import UniversalLLM.Protocols.Anthropic (AnthropicRequest, AnthropicResponse)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import TestCache (ResponseProvider)

-- ============================================================================
-- Model Test Registry
-- ============================================================================
--
-- Add your model here! Each is a single describe + testModel call.

modelTests :: ResponseProvider AnthropicRequest AnthropicResponse
           -> ResponseProvider OpenAIRequest OpenAIResponse
           -> Spec
modelTests anthropicProvider openaiProvider = do

  -- Anthropic Models
  describe "Claude Sonnet 4.5" $
    testModel TestModels.anthropicSonnet45 Anthropic TestModels.ClaudeSonnet45 ((), ()) anthropicProvider
      [ ST.text, ST.tools ]

  describe "Claude Sonnet 4.5 with Reasoning" $
    testModel TestModels.anthropicSonnet45Reasoning Anthropic TestModels.ClaudeSonnet45WithReasoning ((), ((), ())) anthropicProvider
      [ ST.text, ST.tools, ST.reasoning, ST.reasoningWithTools ]

  -- OpenAI-Compatible Models
  describe "GLM 4.5" $
    testModel TestModels.openAIGLM45 OpenAIProvider.OpenAI TestModels.GLM45 ((), ((), ((), ()))) openaiProvider
      [ ST.text, ST.tools ]

-- ============================================================================
-- Test Model Helper
-- ============================================================================

testModel :: ComposableProvider provider model state
          -> provider
          -> model
          -> state
          -> ResponseProvider (ProviderRequest provider) (ProviderResponse provider)
          -> [StandardTest provider model state]
          -> Spec
testModel cp provider model initialState getResponse testSuites =
  mapM_ (\test -> runStandardTest cp provider model initialState getResponse test) testSuites

runStandardTest :: ComposableProvider provider model state
                -> provider
                -> model
                -> state
                -> ResponseProvider (ProviderRequest provider) (ProviderResponse provider)
                -> StandardTest provider model state
                -> Spec
runStandardTest cp provider model initialState getResponse (StandardTest testFn) =
  testFn cp provider model initialState getResponse
