{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.BasicModel where

import UniversalLLM.Core.Types
import Data.Text (Text)

-- Basic model - parameterized by model name for universal provider support
data BasicModel = BasicModel Text deriving (Show, Eq)

-- Default basic model instances
instance ModelHasTools BasicModel

-- Universal ModelName instance - works with any provider
instance ModelName provider BasicModel where
  modelName (BasicModel name) = name
