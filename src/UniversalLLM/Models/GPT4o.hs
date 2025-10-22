{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.GPT4o where

import UniversalLLM.Core.Types
import UniversalLLM.Providers.OpenAI

-- GPT-4o model (just model identity)
data GPT4o = GPT4o deriving (Show, Eq)

-- OpenAI-specific capabilities
instance HasVision GPT4o OpenAI
instance HasJSON GPT4o OpenAI
instance HasTools GPT4o OpenAI

-- Provider-specific model names and support
instance ModelName OpenAI GPT4o where
  modelName _ = "gpt-4o"