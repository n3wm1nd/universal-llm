{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module ReasoningConfigSpec (spec) where

import Test.Hspec
import UniversalLLM
import UniversalLLM.Providers.OpenAI (OpenRouter(..), OpenAI(..))
import UniversalLLM.Protocols.OpenAI
import TestModels
import Data.Default (def)

spec :: Spec
spec = do
  describe "Reasoning Config" $ do
    describe "OpenRouter Gemini" $ do
      it "sets reasoning field when Reasoning True in configs" $ do
        let configs :: [ModelConfig (Model Gemini3ProPreview OpenRouter)]
            configs = [Reasoning True, MaxTokens 1000]
            msgs = [UserText "Hello"]
            (_, req) = toProviderRequest openRouterGemini3ProPreview (Model Gemini3ProPreview OpenRouter) configs def msgs

        -- Check that reasoning field is set
        case reasoning req of
          Nothing -> expectationFailure "reasoning field should be set when Reasoning True is in configs"
          Just reasoningCfg -> do
            reasoning_enabled reasoningCfg `shouldBe` Just True
            reasoning_effort reasoningCfg `shouldBe` Just "low"

      it "does not set reasoning field when Reasoning False in configs" $ do
        let configs :: [ModelConfig (Model Gemini3ProPreview OpenRouter)]
            configs = [Reasoning False, MaxTokens 1000]
            msgs = [UserText "Hello"]
            (_, req) = toProviderRequest openRouterGemini3ProPreview (Model Gemini3ProPreview OpenRouter) configs def msgs

        reasoning req `shouldBe` Nothing

      it "does not set reasoning field when Reasoning not in configs" $ do
        let configs :: [ModelConfig (Model Gemini3ProPreview OpenRouter)]
            configs = [MaxTokens 1000]
            msgs = [UserText "Hello"]
            (_, req) = toProviderRequest openRouterGemini3ProPreview (Model Gemini3ProPreview OpenRouter) configs def msgs

        reasoning req `shouldBe` Nothing

    describe "OpenAI GLM45" $ do
      it "sets reasoning field when Reasoning True in configs" $ do
        let configs :: [ModelConfig (Model GLM45 OpenAI)]
            configs = [Reasoning True, MaxTokens 1000]
            msgs = [UserText "Hello"]
            (_, req) = toProviderRequest openAIGLM45 (Model GLM45 OpenAI) configs ((), ((), ((), ()))) msgs

        -- Check that reasoning field is set
        case reasoning req of
          Nothing -> expectationFailure "reasoning field should be set when Reasoning True is in configs"
          Just reasoningCfg -> do
            reasoning_enabled reasoningCfg `shouldBe` Just True
            reasoning_effort reasoningCfg `shouldBe` Just "low"
