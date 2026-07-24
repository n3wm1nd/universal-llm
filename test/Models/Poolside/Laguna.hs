{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

{- |
Module: Models.Poolside.Laguna

Test suite for Poolside's Laguna S 2.1 model.

= Models Covered

* Laguna S 2.1 (OpenRouter, llama.cpp)

= Provider-Specific Quirks

__OpenRouter:__
  Uses reasoning_details field instead of standard reasoning_content

__llama.cpp:__
  Uses standard reasoning_content field
  Model name is dynamically determined from the loaded GGUF file (any quant
  of Laguna-S-2.1, e.g. UD-Q4_K_XL)

Neither provider has HasJSON or HasVision - see 'UniversalLLM.Models.Poolside.Laguna'
for why.
-}

module Models.Poolside.Laguna
  ( testsLagunaS21OpenRouter
  , testsLagunaS21LlamaCpp
  ) where

import UniversalLLM (via, route)
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse)
import UniversalLLM.Providers.OpenAI (LlamaCpp(..), OpenRouter(..))
import UniversalLLM.Models.Poolside.Laguna (LagunaS21(..))
import Protocol.OpenAITests
import qualified StandardTests as ST
import qualified ComposableProviderTests as CPT
import TestCache (ResponseProvider)
import TestHelpers (testModel, testModelOffline)
import Test.Hspec (Spec, describe)
import qualified Data.Text as T
import Data.Text (Text)

-- | Test Laguna S 2.1 via OpenRouter
--
-- Includes both protocol probes (wire format) and standard tests (high-level API).
testsLagunaS21OpenRouter :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
testsLagunaS21OpenRouter provider = do
  describe "Laguna S 2.1 via OpenRouter" $ do
    describe "Protocol" $ do
      basicText provider "poolside/laguna-s-2.1"
      toolCalling provider "poolside/laguna-s-2.1"
      acceptsToolResults provider "poolside/laguna-s-2.1"
      acceptsToolResultNoTools provider "poolside/laguna-s-2.1"
      acceptsToolResultToolGone provider "poolside/laguna-s-2.1"
      acceptsStaleToolInHistory provider "poolside/laguna-s-2.1"
      acceptsOldToolCallStillAvailable provider "poolside/laguna-s-2.1"
      consecutiveUserMessages provider "poolside/laguna-s-2.1"
      startsWithAssistant provider "poolside/laguna-s-2.1"
      reasoningViaDetails provider "poolside/laguna-s-2.1"
      providerErrorResponse provider

    describe "Standard Tests" $
      testModel route (LagunaS21 `via` OpenRouter) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools ]

    describe "Composable Provider Tests" $
      testModelOffline route (LagunaS21 `via` OpenRouter)
        [ CPT.cacheCoherency, CPT.cacheCoherencyWithTools ]

-- | Test Laguna S 2.1 via llama.cpp
--
-- Takes the canonicalized model name as determined by querying the llama.cpp
-- server. The model name is extracted from the loaded GGUF file.
testsLagunaS21LlamaCpp :: ResponseProvider OpenAIRequest OpenAIResponse -> Text -> Spec
testsLagunaS21LlamaCpp provider modelName = do
  describe ("Laguna S 2.1 via llama.cpp with " <> T.unpack modelName) $ do
    describe "Protocol" $ do
      basicText provider modelName
      toolCalling provider modelName
      acceptsToolResults provider modelName
      acceptsToolResultNoTools provider modelName
      acceptsToolResultToolGone provider modelName
      acceptsStaleToolInHistory provider modelName
      acceptsOldToolCallStillAvailable provider modelName
      consecutiveUserMessages provider modelName
      startsWithAssistant provider modelName
      reasoning provider modelName

    describe "Standard Tests" $
      testModel route (LagunaS21 `via` LlamaCpp) provider
        [ ST.text, ST.systemMessage, ST.systemMessageMidConversation, ST.multipleSystemPrompts, ST.tools, ST.reasoning, ST.reasoningWithTools ]
