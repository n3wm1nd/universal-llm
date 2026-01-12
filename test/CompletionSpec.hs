{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}

module CompletionSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.Providers.OpenAI

-- Helper to build completion request
buildCompletionRequest :: forall m. (CompletionProviderImplementation m, ModelName m)
                       => m
                       -> [ModelConfig m]
                       -> Text
                       -> CompletionRequest m
buildCompletionRequest = toCompletionRequest

-- Helper to parse completion response
parseCompletion :: forall m. (CompletionProviderImplementation m, ModelName m)
                => m
                -> [ModelConfig m]
                -> Text
                -> CompletionResponse m
                -> Either LLMError Text
parseCompletion model configs prompt (OpenAICompletionError (OpenAIErrorResponse errDetail)) =
  let typeInfo = case errorType errDetail of
        Just t -> " (" <> t <> ")"
        Nothing -> ""
  in Left $ ProviderError (code errDetail) $ errorMessage errDetail <> typeInfo
parseCompletion model configs prompt resp =
  let text = fromCompletionResponse model configs prompt resp
  in if T.null text
     then Left $ ParseError "No completion text returned"
     else Right text

spec :: ResponseProvider OpenAICompletionRequest OpenAICompletionResponse -> Spec
spec getResponse = do
  describe "OpenAI Completion Interface" $ do

    it "completes a simple prompt" $ do
      let model = Model GLM45 OpenAI  -- Using a test model with provider
          configs = [MaxTokens 50, Temperature 0.7]
          prompt = "The capital of France is"

          req = buildCompletionRequest model configs prompt

      resp <- getResponse req

      case parseCompletion model configs prompt resp of
        Right completedText -> do
          -- Should contain "Paris" or similar
          T.toLower completedText `shouldSatisfy` T.isInfixOf "paris"
          -- Should not be empty
          T.length completedText `shouldSatisfy` (> 0)

        Left err -> expectationFailure $ "parseCompletion failed: " ++ show err

    it "respects MaxTokens configuration" $ do
      let model = Model GLM45 OpenAI
          configs = [MaxTokens 10]  -- Very short
          prompt = "Once upon a time"

          req = buildCompletionRequest model configs prompt

      -- Verify request has max_tokens set
      completionMaxTokens req `shouldBe` Just 10

    it "respects Temperature configuration" $ do
      let model = Model GLM45 OpenAI
          configs = [Temperature 0.5]
          prompt = "Test"

          req = buildCompletionRequest model configs prompt

      -- Verify request has temperature set
      completionTemperature req `shouldBe` Just 0.5

    it "respects Stop sequences configuration" $ do
      let model = Model GLM45 OpenAI
          configs = [Stop ["\n\n", "###"]]
          prompt = "Test"

          req = buildCompletionRequest model configs prompt

      -- Verify request has stop sequences set
      stop req `shouldBe` Just ["\n\n", "###"]

    it "completes prompts with stop sequences" $ do
      let model = Model GLM45 OpenAI
          configs = [MaxTokens 100, Stop ["\n"]]  -- Stop at first newline
          prompt = "List three colors: 1."

          req = buildCompletionRequest model configs prompt

      resp <- getResponse req

      case parseCompletion model configs prompt resp of
        Right completedText -> do
          -- Should have stopped at first newline, so text should be relatively short
          -- and not contain multiple list items
          T.count "\n" completedText `shouldSatisfy` (<= 1)

        Left err -> expectationFailure $ "parseCompletion failed: " ++ show err

    it "sets the model name in the request" $ do
      let model = Model GLM45 OpenAI
          configs = []
          prompt = "Test"

          req = buildCompletionRequest model configs prompt

      -- Verify model name is set correctly
      completionModel req `shouldBe` modelName (Model GLM45 OpenAI)

    it "sets the prompt in the request" $ do
      let model = Model GLM45 OpenAI
          configs = []
          testPrompt = "This is a test prompt"

          req = buildCompletionRequest model configs testPrompt

      -- Verify prompt is set correctly
      prompt req `shouldBe` testPrompt
