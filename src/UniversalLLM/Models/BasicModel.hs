{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

module UniversalLLM.Models.BasicModel where

import UniversalLLM.Core.Types

-- Basic model (just model identity)
data BasicModel = BasicModel deriving (Show, Eq)

-- Default basic model instances
instance ModelHasTools BasicModel
