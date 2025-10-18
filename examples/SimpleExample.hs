{-# LANGUAGE OverloadedStrings #-}

module Main where

import UniversalLLM
import UniversalLLM.Core.Types (ModelConfig(..), Message(..))
import UniversalLLM.Models.GPT4o
import UniversalLLM.Models.Claude35Sonnet
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Providers.Anthropic

main :: IO ()
main = do
  -- Models are just identity markers, config is separate
  let gpt4oModel = GPT4o
      gpt4oConfig :: [ModelConfig OpenAI GPT4o]
      gpt4oConfig = [Temperature 0.7, MaxTokens 100]

      claudeModel = Claude35Sonnet
      claudeConfig :: [ModelConfig Anthropic Claude35Sonnet]
      claudeConfig = [Temperature 0.7, MaxTokens 100, SystemPrompt "You are a helpful assistant."]

  -- Type-annotated messages for different model/provider combinations
  let openaiMessages :: [Message GPT4o OpenAI]
      openaiMessages = [UserText "Hello, how are you?"]

      claudeMessages :: [Message Claude35Sonnet Anthropic]
      claudeMessages = [UserText "Hello, how are you?"]

      -- Vision example (only GPT4o supports vision)
      visionMessages :: [Message GPT4o OpenAI]
      visionMessages = [UserImage "What's in this image?" "base64data..."]

  putStrLn "This demonstrates the clean functional design:"
  putStrLn "- Models are phantom types (just identity)"
  putStrLn "- Config passed separately with GADT constraints"
  putStrLn "- GADTs ensure type safety - vision only works with GPT4o"
  putStrLn "- Pure transformation functions"
  putStrLn "- Autodocodec handles serialization"
  putStrLn "- Multiple providers/models simultaneously supported"