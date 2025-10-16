{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.GPT4o where

import UniversalLLM.Core.Types
import UniversalLLM.Providers.OpenAI

-- GPT-4o model with parameters
data GPT4o = GPT4o
  { gpt4oTemperature :: Maybe Double
  , gpt4oMaxTokens :: Maybe Int
  , gpt4oSeed :: Maybe Int
  } deriving (Show, Eq)

-- Capabilities
instance HasVision GPT4o
instance HasJSON GPT4o
instance HasTools GPT4o

-- Parameter extraction for any provider
instance Temperature GPT4o provider where
  getTemperature = gpt4oTemperature

instance MaxTokens GPT4o provider where
  getMaxTokens = gpt4oMaxTokens

instance Seed GPT4o provider where
  getSeed = gpt4oSeed

-- Provider-specific model names
instance ModelName OpenAI GPT4o where
  modelName = "gpt-4o"