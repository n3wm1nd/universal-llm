{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleContexts #-}

-- | Standard Test Suites
--
-- This module defines reusable test suites that can be run against any model.
-- Each test suite is a function that takes a composable provider and runs
-- a series of tests against it.
module StandardTests
  ( StandardTest(..)
  , text
  , tools
  , reasoning
  , reasoningWithTools
  ) where

import Test.Hspec
import qualified Data.Text as T
import UniversalLLM.Core.Types
import TestCache (ResponseProvider)

-- | A standard test is just a function that runs hspec tests
newtype StandardTest provider model state = StandardTest
  ( ComposableProvider provider model state
    -> provider
    -> model
    -> state
    -> ResponseProvider (ProviderRequest provider) (ProviderResponse provider)
    -> Spec
  )

-- ============================================================================
-- Basic Text Tests
-- ============================================================================

text :: ( Monoid (ProviderRequest provider)
        , SupportsMaxTokens provider
        )
     => StandardTest provider model state
text = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Basic Text" $ do
    it "sends message and receives response" $ do
      -- Use larger token limit to accommodate reasoning models
      let configs = [MaxTokens 2048]
          msgs = [UserText "What is 2+2?"]
          (_, req) = toProviderRequest cp provider model configs initialState msgs

      resp <- getResponse req

      let parsedMsgs = either (error . show) snd $ fromProviderResponse cp provider model configs initialState resp

      -- Should get back at least one assistant message
      length parsedMsgs `shouldSatisfy` (> 0)

    it "maintains conversation history" $ do
      -- Use larger token limit to accommodate reasoning models
      let configs = [MaxTokens 2048]
          msgs1 = [UserText "What is 2+2?"]
          (state1, req1) = toProviderRequest cp provider model configs initialState msgs1

      resp1 <- getResponse req1
      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp1

      -- First response should contain messages
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- Continue conversation
      let msgs2 = msgs1 ++ parsedMsgs1 ++ [UserText "What about 3+3?"]
          (_, req2) = toProviderRequest cp provider model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

      -- Second response should also contain messages
      length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Tool Tests
-- ============================================================================

tools :: StandardTest provider model state
tools = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Tool Calling" $ do
    it "completes tool calling flow" $
      pending -- TODO: Implement generic tool calling test

-- ============================================================================
-- Reasoning Tests
-- ============================================================================

reasoning :: ( Monoid (ProviderRequest provider)
             , SupportsMaxTokens provider
             )
          => StandardTest provider model state
reasoning = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Reasoning" $ do
    it "handles reasoning messages" $ do
      let configs = [MaxTokens 4096]
          msgs = [UserText "Think step by step: What is 15 * 23?"]
          (state1, req) = toProviderRequest cp provider model configs initialState msgs

      resp <- getResponse req

      let (state2, parsedMsgs) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp

      -- Should get back at least one message
      length parsedMsgs `shouldSatisfy` (> 0)

      -- Should handle reasoning in round-trip
      let msgs2 = msgs ++ parsedMsgs ++ [UserText "Now what is 20 * 30?"]
          (_, req2) = toProviderRequest cp provider model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

      length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Combined Tests
-- ============================================================================

reasoningWithTools :: ( Monoid (ProviderRequest provider)
                      , SupportsMaxTokens provider
                      )
                   => StandardTest provider model state
reasoningWithTools = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Reasoning + Tools" $ do
    it "handles reasoning with tool calls" $ do
      -- First request: Ask a question that requires reasoning and tool use
      let configs = [MaxTokens 4096]
          msgs = [UserText "Think carefully: What files match the pattern '*.md'? Use the available tools."]
          (state1, req1) = toProviderRequest cp provider model configs initialState msgs

      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp1

      -- Should get back messages (reasoning and/or tool calls)
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- Continue conversation with tool results
      -- This is where reasoning_details preservation is critical
      let msgs2 = msgs ++ parsedMsgs1 ++ [UserText "Thank you, now tell me more about those files."]
          (_, req2) = toProviderRequest cp provider model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

      -- Should successfully handle the continuation
      length parsedMsgs2 `shouldSatisfy` (> 0)
