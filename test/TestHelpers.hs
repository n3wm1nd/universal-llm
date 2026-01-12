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
import StandardTests (StandardTest(..))
import UniversalLLM
import TestCache (ResponseProvider)

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
runStandardTest :: (Default state)
                => ComposableProvider m state
                -> m
                -> ResponseProvider (ProviderRequest m) (ProviderResponse m)
                -> StandardTest m state
                -> Spec
runStandardTest cp model getResponse (StandardTest testFn) =
  testFn cp model def getResponse
