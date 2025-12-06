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
import UniversalLLM.Core.Types
import TestCache (ResponseProvider)

-- | Run a list of standard tests against a model
-- Uses Default instance to initialize state automatically
testModel :: (Default state)
          => ComposableProvider provider model state
          -> provider
          -> model
          -> ResponseProvider (ProviderRequest provider) (ProviderResponse provider)
          -> [StandardTest provider model state]
          -> Spec
testModel cp provider model getResponse testSuites =
  mapM_ (runStandardTest cp provider model getResponse) testSuites

-- | Run a single standard test
runStandardTest :: (Default state)
                => ComposableProvider provider model state
                -> provider
                -> model
                -> ResponseProvider (ProviderRequest provider) (ProviderResponse provider)
                -> StandardTest provider model state
                -> Spec
runStandardTest cp provider model getResponse (StandardTest testFn) =
  testFn cp provider model def getResponse
