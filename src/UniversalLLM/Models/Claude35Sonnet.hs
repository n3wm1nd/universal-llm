{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.Claude35Sonnet where

import Data.Text (Text)
import UniversalLLM.Core.Types
import UniversalLLM.Providers.Anthropic

-- Claude 3.5 Sonnet model (just model identity)
data Claude35Sonnet = Claude35Sonnet deriving (Show, Eq)

-- Capabilities
instance HasVision Claude35Sonnet

-- Provider-specific model names and support
instance ModelName Anthropic Claude35Sonnet where
  modelName = "claude-3-5-sonnet-20241022"

instance ProviderSupportsModel Anthropic Claude35Sonnet