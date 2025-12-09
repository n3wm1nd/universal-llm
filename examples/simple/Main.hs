{-# LANGUAGE OverloadedStrings #-}
-- this is just an example, not everything defined here is also used
{-# OPTIONS_GHC -Wno-unused-local-binds #-}

module Main where

import UniversalLLM
import UniversalLLM.Models.SimpleModel
import UniversalLLM.Models.FullFeaturedModel
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Providers.Anthropic

main :: IO ()
main = do
  -- Models are now unified with their providers
  -- SimpleModel: text-only, no capabilities
  let simpleModel = Model SimpleModel OpenAI
      simpleConfig :: [ModelConfig (Model SimpleModel OpenAI)]
      simpleConfig = [Temperature 0.7, MaxTokens 100]

  -- FullFeaturedModel: all implemented capabilities
  let fullOpenAIModel = Model FullFeaturedModel OpenAI
      fullOpenAIConfig :: [ModelConfig (Model FullFeaturedModel OpenAI)]
      fullOpenAIConfig = [Temperature 0.7, MaxTokens 100, Reasoning True]

      fullAnthropicModel = Model FullFeaturedModel Anthropic
      fullAnthropicConfig :: [ModelConfig (Model FullFeaturedModel Anthropic)]
      fullAnthropicConfig = [Temperature 0.7, MaxTokens 100, SystemPrompt "You are a helpful assistant."]

  -- Type-annotated messages for different model/provider combinations
  let simpleMessages :: [Message (Model SimpleModel OpenAI)]
      simpleMessages = [UserText "Hello, how are you?"]

      fullMessages :: [Message (Model FullFeaturedModel Anthropic)]
      fullMessages = [UserText "Hello, how are you?"]

      -- Vision example (not yet implemented in any provider)
      -- visionMessages :: [Message (Model FullFeaturedModel OpenAI)]
      -- visionMessages = [UserImage "What's in this image?" "base64data..."]

  putStrLn "This demonstrates the clean functional design:"
  putStrLn "- Models are phantom types (just identity)"
  putStrLn "- Config passed separately with GADT constraints"
  putStrLn "- GADTs ensure type safety at compile time"
  putStrLn "- Pure transformation functions"
  putStrLn "- Autodocodec handles serialization"
  putStrLn "- Multiple providers/models simultaneously supported"
  putStrLn "- External packages define real models (gpt-4o, claude-3-5-sonnet, etc.)"