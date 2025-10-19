{-# LANGUAGE OverloadedStrings #-}

module Main where

import Test.Hspec
import qualified OpenAIProtocolSpec

main :: IO ()
main = hspec $ do
  describe "OpenAI Protocol" OpenAIProtocolSpec.spec
