{-# LANGUAGE OverloadedStrings #-}

-- | Test Helper Functions
--
-- Common utilities for running standardized tests against models.
module TestHelpers
  ( testModel
  , runStandardTest
  ) where

import Test.Hspec
import Data.Default (Default, def)
import Control.Exception (catch)
import StandardTests (StandardTest(..))
import UniversalLLM.Core.Types
import TestCache (ResponseProvider, CacheMissException(..))

-- | Run a list of standard tests against a model
-- Uses Default instance to initialize state automatically
testModel :: (Default state, Provider m)
          => ComposableProvider m state
          -> m
          -> ResponseProvider (ProviderRequest m) (ProviderResponse m)
          -> [StandardTest m state]
          -> Spec
testModel cp model getResponse testSuites =
  mapM_ (runStandardTest cp model getResponse) testSuites

-- | Run a single standard test
-- Wraps the response provider to catch cache misses and mark tests as pending
runStandardTest :: (Default state)
                => ComposableProvider m state
                -> m
                -> ResponseProvider (ProviderRequest m) (ProviderResponse m)
                -> StandardTest m state
                -> Spec
runStandardTest cp model getResponse (StandardTest testFn) =
  -- Wrap the response provider to catch cache misses
  let wrappedGetResponse req =
        catch (getResponse req) $ \(CacheMissException msg) -> do
          pendingWith $ "Cache miss: " ++ msg
          error "unreachable"  -- pendingWith throws, but GHC doesn't know that
  in testFn cp model def wrappedGetResponse
