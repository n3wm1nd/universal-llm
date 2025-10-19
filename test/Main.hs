{-# LANGUAGE OverloadedStrings #-}

module Main where

import Test.Hspec
import qualified OpenAIProtocolSpec
import qualified AnthropicTransportSpec

main :: IO ()
main = hspec $ do
  describe "OpenAI Protocol" OpenAIProtocolSpec.spec
  describe "Anthropic Transport" AnthropicTransportSpec.spec
