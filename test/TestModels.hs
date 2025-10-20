{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module TestModels where

import UniversalLLM.Core.Types
import UniversalLLM.Providers.Anthropic

-- Test model for Claude Sonnet 4.5
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

-- Model supports tools
instance HasTools ClaudeSonnet45

-- Provider-specific model names and support
instance ModelName Anthropic ClaudeSonnet45 where
  modelName _ = "claude-sonnet-4-5-20250929"
