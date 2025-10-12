{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.Claude35Sonnet where

import Data.Text (Text)
import UniversalLLM.Core.Types
import UniversalLLM.Providers.Anthropic

-- Claude 3.5 Sonnet model with parameters
data Claude35Sonnet = Claude35Sonnet
  { claudeTemperature :: Maybe Double
  , claudeMaxTokens :: Maybe Int
  , claudeSystemPrompt :: Maybe Text
  } deriving (Show, Eq)

-- Capabilities
instance HasVision Claude35Sonnet

-- Parameter extraction
instance Temperature Claude35Sonnet provider where
  getTemperature = claudeTemperature

instance MaxTokens Claude35Sonnet provider where
  getMaxTokens = claudeMaxTokens

instance SystemPrompt Claude35Sonnet provider where
  getSystemPrompt = claudeSystemPrompt

-- Provider-specific model names
instance ModelName Anthropic Claude35Sonnet where
  modelName = "claude-3-5-sonnet-20241022"