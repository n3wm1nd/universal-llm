{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.SimpleModel where

import UniversalLLM
import Data.Text (Text)

-- SimpleModel - minimal text-only model for basic use cases
-- Parameterized by model name for universal provider support
data SimpleModel = SimpleModel Text deriving (Show, Eq)

-- Universal ModelName instance - works with any provider
instance ModelName (Model SimpleModel provider) where
  modelName (Model (SimpleModel name) _) = name

-- Note: SimpleModel intentionally has no capability instances
-- It represents the minimal baseline - text input and text output only
-- No tools, vision, JSON mode, or reasoning capabilities
