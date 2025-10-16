{-# LANGUAGE OverloadedStrings #-}

module Main where

import UniversalLLM
import UniversalLLM.Models.GPT4o
import UniversalLLM.Models.Claude35Sonnet
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Providers.Anthropic

main :: IO ()
main = do
  -- Configure providers and models separately (no transport concerns)
  let openaiProvider = OpenAI

      gpt4oModel = GPT4o
        { gpt4oTemperature = Just 0.7
        , gpt4oMaxTokens = Just 100
        , gpt4oSeed = Nothing
        , gpt4oToolDefinitions = []
        }

      anthropicProvider = Anthropic

      claudeModel = Claude35Sonnet
        { claudeTemperature = Just 0.7
        , claudeMaxTokens = Just 100
        , claudeSystemPrompt = Nothing
        }

  -- Type-annotated messages for different model/provider combinations
  let openaiMessages :: [Message GPT4o OpenAI]
      openaiMessages = [UserText "Hello, how are you?"]

      claudeMessages :: [Message Claude35Sonnet Anthropic]
      claudeMessages = [SystemText "You are a helpful assistant.", UserText "Hello, how are you?"]

      -- Vision example (only GPT4o supports vision)
      visionMessages :: [Message GPT4o OpenAI]
      visionMessages = [UserImage "What's in this image?" "base64data..."]

  putStrLn "This demonstrates the clean functional design:"
  putStrLn "- Model types carry parameters"
  putStrLn "- Provider types carry auth/config"
  putStrLn "- GADTs ensure type safety - vision only works with GPT4o"
  putStrLn "- Pure transformation functions"
  putStrLn "- Autodocodec handles serialization"
  putStrLn "- Multiple providers/models simultaneously supported"