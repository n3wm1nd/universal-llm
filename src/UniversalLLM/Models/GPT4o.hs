{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.GPT4o where

import UniversalLLM.Core.Types
import UniversalLLM.Providers.OpenAI

-- GPT-4o model (just model identity)
data GPT4o = GPT4o deriving (Show, Eq)

-- Provider-specific model names and support
instance ModelName OpenAI GPT4o where
  modelName _ = "gpt-4o"

-- OpenAI-specific capabilities with implementations
instance HasTools GPT4o OpenAI where
  toolsComposableProvider = UniversalLLM.Providers.OpenAI.toolsComposableProvider

instance HasJSON GPT4o OpenAI where
  jsonComposableProvider = UniversalLLM.Providers.OpenAI.jsonComposableProvider

-- Complete provider implementation
instance ProviderImplementation OpenAI GPT4o where
  getComposableProvider = UniversalLLM.Providers.OpenAI.baseComposableProvider <> UniversalLLM.Providers.OpenAI.toolsComposableProvider <> UniversalLLM.Providers.OpenAI.jsonComposableProvider

-- Note: HasVision not implemented yet for OpenAI
-- instance HasVision GPT4o OpenAI where
--   visionComposableProvider = visionComposableProvider'