{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.BasicModel where

import UniversalLLM.Core.Types

-- Basic model configuration with common parameters
data BasicModel = BasicModel
  { basicTemperature :: Maybe Double
  , basicMaxTokens :: Maybe Int
  , basicSeed :: Maybe Int
  , basicToolDefinitions :: [ToolDefinition]
  } deriving (Show, Eq)

-- Default basic model instances
instance ModelHasTools BasicModel where
  getToolDefinitions = basicToolDefinitions
  setToolDefinitions toolDefs model = model { basicToolDefinitions = toolDefs }

instance Temperature BasicModel provider where
  getTemperature = basicTemperature

instance MaxTokens BasicModel provider where
  getMaxTokens = basicMaxTokens

instance Seed BasicModel provider where
  getSeed = basicSeed
