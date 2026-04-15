{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

-- | Offline tests for ComposableProvider stacks.
--
-- These tests verify invariants of the request/response translation layer
-- without hitting any live API.  Each 'ComposableProviderTest' receives the
-- composable provider and model under test, but no response provider.
module ComposableProviderTests
  ( ComposableProviderTest(..)
  , testModelOffline
  , cacheCoherency
  ) where

import Test.Hspec
import Data.Default (Default, def)
import Data.List (isPrefixOf)
import UniversalLLM
import Protocol.CacheCoherency
  ( CacheNormalized(..)
  , HasWireMessages(..)
  , AppendResponse(..)
  , SimulateResponse(..)
  )

-- | An offline test: has access to the composable provider and model, but
-- makes no network calls.
newtype ComposableProviderTest m state = ComposableProviderTest
  ( ComposableProvider m state
    -> m
    -> state
    -> Spec
  )

-- | Run a list of offline tests against a model.
testModelOffline :: (Default state)
                 => ComposableProvider m state
                 -> m
                 -> [ComposableProviderTest m state]
                 -> Spec
testModelOffline cp model tests =
  mapM_ (\(ComposableProviderTest f) -> f cp model def) tests

-- | Helper that ties the response type to the request type via the
-- 'AppendResponse' functional dependency, avoiding ambiguity from
-- non-injective type families.
simulateAndAppend :: ( SimulateResponse resp
                     , AppendResponse req resp
                     )
                  => req -> [Message m] -> Maybe req
simulateAndAppend req msgs = appendResponse req (simulateResponse msgs)

-- | Verify the cache coherency invariant.
--
-- LLM APIs expect: messages(req2) starts with messages(req1) ++ responseMessages,
-- because they assume callers blindly append the response JSON to their stored
-- request JSON.  After cache-normalization (stripping fields irrelevant to the
-- provider cache key), this prefix must hold exactly:
--
-- @
-- let req1     = toProviderRequest reqMsgs
--     req2     = toProviderRequest (reqMsgs ++ respMsgs ++ nextMsgs)
--     merged   = appendResponse req1 (simulateResponse respMsgs)
-- cacheNormalize (wireMessages merged)
--   `isPrefixOf` cacheNormalize (wireMessages req2)
-- @
--
-- Deviations anywhere in the prefix are a problem; deviations at the tail
-- (the new user turn being added) are expected and acceptable.
cacheCoherency :: ( Provider m
                  , SupportsMaxTokens (ProviderOf m)
                  , SimulateResponse (ProviderResponse m)
                  , AppendResponse (ProviderRequest m) (ProviderResponse m)
                  , HasWireMessages (ProviderRequest m)
                  , CacheNormalized (ProviderRequest m)
                  , CacheNormalized (WireMessage (ProviderRequest m))
                  , Eq (WireMessage (ProviderRequest m))
                  , Show (WireMessage (ProviderRequest m))
                  )
               => ComposableProviderTest m state
cacheCoherency = ComposableProviderTest $ \cp model initialState ->
  describe "Cache Coherency" $ do

    -- Basic case: system prompt + user message, assistant replies, then a
    -- follow-up user message.  The first request + simulated response must
    -- appear as an exact cache-normalized prefix of the second request.
    it "system prompt + user message followed by assistant reply" $ do
      let configs  = [MaxTokens 2048]
          reqMsgs  = [ SystemText "You are a helpful assistant."
                     , UserText  "What is 2+2?"
                     ]
          respMsgs = [ AssistantText "4." ]
          nextMsgs = [ UserText "And 3+3?" ]

          (state1, req1) = toProviderRequest cp model configs initialState reqMsgs
          (_, req2)      = toProviderRequest cp model configs state1
                             (reqMsgs ++ respMsgs ++ nextMsgs)

          merged = simulateAndAppend req1 respMsgs

      case merged of
        Nothing ->
          expectationFailure "appendResponse returned Nothing (error response?)"
        Just mergedReq ->
          let expectedPrefix = map cacheNormalize (wireMessages mergedReq)
              actualMessages = map cacheNormalize (wireMessages req2)
          in if expectedPrefix `isPrefixOf` actualMessages
               then pure ()
               else expectationFailure $ unlines
                      [ "Cache coherency violation: request history is not a prefix of the follow-up request."
                      , "Expected prefix (" <> show (length expectedPrefix) <> " messages):"
                      , show expectedPrefix
                      , "Actual messages (" <> show (length actualMessages) <> " messages):"
                      , show actualMessages
                      ]
