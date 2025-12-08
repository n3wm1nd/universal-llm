{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Nova2DebugSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import qualified Data.Aeson as Aeson
import Data.Default (def)
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.Providers.OpenAI

-- Debug test to figure out what's wrong with Nova 2 Lite
spec :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
spec getResponse = do
  describe "Nova 2 Lite Debug" $ do

    it "sends a simple text message without tools or reasoning" $ do
      let model = Nova2Lite
          configs = [MaxTokens 500]
          msgs = [UserText "Hello, how are you?" :: Message Nova2Lite OpenRouter]
          (state1, req1) = toProviderRequest openRouterNova2Lite OpenRouter model configs def msgs

      resp1 <- getResponse req1

      case fromProviderResponse openRouterNova2Lite OpenRouter model configs state1 resp1 of
        Right (_, parsedMsgs) -> do
          length parsedMsgs `shouldSatisfy` (> 0)
        Left err -> expectationFailure $ "Failed to parse response: " ++ show err

    it "sends a message with reasoning enabled but no tools" $ do
      let model = Nova2Lite
          configs = [MaxTokens 500, Reasoning True]
          msgs = [UserText "Think step by step: what is 2+2?" :: Message Nova2Lite OpenRouter]
          (state1, req1) = toProviderRequest openRouterNova2Lite OpenRouter model configs def msgs

      resp1 <- getResponse req1

      case fromProviderResponse openRouterNova2Lite OpenRouter model configs state1 resp1 of
        Right (_, parsedMsgs) -> do
          length parsedMsgs `shouldSatisfy` (> 0)
        Left err -> expectationFailure $ "Failed to parse response: " ++ show err

    it "sends a message with tools but no reasoning" $ do
      let model = Nova2Lite
          toolDef = ToolDefinition "get_weather" "Get weather for a location"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 500, Tools [toolDef]]
          msgs = [UserText "What's the weather in Paris?" :: Message Nova2Lite OpenRouter]
          (state1, req1) = toProviderRequest openRouterNova2Lite OpenRouter model configs def msgs

      resp1 <- getResponse req1

      case fromProviderResponse openRouterNova2Lite OpenRouter model configs state1 resp1 of
        Right (_, parsedMsgs) -> do
          length parsedMsgs `shouldSatisfy` (> 0)
        Left err -> expectationFailure $ "Failed to parse response: " ++ show err

    it "sends a message with both tools and reasoning" $ do
      let model = Nova2Lite
          toolDef = ToolDefinition "list_files" "List files matching a pattern"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 500, Reasoning True, Tools [toolDef]]
          msgs = [UserText "Think carefully: list all markdown files" :: Message Nova2Lite OpenRouter]
          (state1, req1) = toProviderRequest openRouterNova2Lite OpenRouter model configs def msgs

      resp1 <- getResponse req1

      case fromProviderResponse openRouterNova2Lite OpenRouter model configs state1 resp1 of
        Right (_, parsedMsgs) -> do
          length parsedMsgs `shouldSatisfy` (> 0)
        Left err -> expectationFailure $ "Failed to parse response: " ++ show err

    it "multi-turn conversation with tools and reasoning" $ do
      let model = Nova2Lite
          toolDef = ToolDefinition "list_files" "List files matching a pattern"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 500, Reasoning True, Tools [toolDef]]

          -- First turn
          msgs1 = [UserText "Think carefully: list all markdown files" :: Message Nova2Lite OpenRouter]
          (state1, req1) = toProviderRequest openRouterNova2Lite OpenRouter model configs def msgs1

      resp1 <- getResponse req1

      case fromProviderResponse openRouterNova2Lite OpenRouter model configs state1 resp1 of
        Right (state2, parsedMsgs1) -> do
          putStrLn $ "Parsed messages 1: " ++ show (map (\m -> case m of
            AssistantText _ -> "AssistantText"
            AssistantReasoning _ -> "AssistantReasoning"
            AssistantTool _ -> "AssistantTool"
            ToolResultMsg _ -> "ToolResultMsg"
            _ -> "Other") parsedMsgs1)

          length parsedMsgs1 `shouldSatisfy` (> 0)

          -- Add mock tool results if needed
          let toolResults = [ ToolResultMsg (ToolResult tc (Right (object ["files" .= (["README.md", "GUIDE.md"] :: [Text])])))
                            | AssistantTool tc <- parsedMsgs1 ]

          -- Second turn
          let msgs2 = msgs1 ++ parsedMsgs1 ++ toolResults ++ [UserText "Thank you!"]
              (_, req2) = toProviderRequest openRouterNova2Lite OpenRouter model configs state2 msgs2

          resp2 <- getResponse req2

          case fromProviderResponse openRouterNova2Lite OpenRouter model configs state2 resp2 of
            Right (_, parsedMsgs2) -> do
              length parsedMsgs2 `shouldSatisfy` (> 0)
            Left err -> expectationFailure $ "Failed to parse response 2: " ++ show err

        Left err -> expectationFailure $ "Failed to parse response 1: " ++ show err
