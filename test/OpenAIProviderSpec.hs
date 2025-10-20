{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}

module OpenAIProviderSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import TestCache (ResponseProvider)
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Models.GPT4o

spec :: ResponseProvider OpenAIRequest OpenAIResponse -> Spec
spec getResponse = do
  describe "OpenAI Provider - Basic Text" $ do

    it "sends message and receives text response" $ do
      let msgs = [UserText "What is 2+2?" :: Message GPT4o OpenAI]
          configs = [MaxTokens 50]
          req = toRequest OpenAI GPT4o configs msgs

      (response :: OpenAIResponse) <- getResponse req

      case fromResponse @OpenAI @GPT4o response of
        Right [AssistantText txt] -> do
          -- Debug: print what we got
          putStrLn $ "DEBUG: Received text: " ++ show txt
          T.isInfixOf "4" txt `shouldBe` True
        Right other -> expectationFailure $ "Expected single AssistantText message, got: " ++ show other
        Left err -> expectationFailure $ "fromResponse failed: " ++ show err

  describe "OpenAI Provider - Tool Calling" $ do

    it "requests tool call and receives tool_use response" $ do
      let toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the current weather in a given location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("The city and state" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          msgs = [UserText "What's the weather in Paris?" :: Message GPT4o OpenAI]
          configs = [Tools [toolDef], MaxTokens 100]
          req = toRequest OpenAI GPT4o configs msgs

      (response :: OpenAIResponse) <- getResponse req

      case fromResponse @OpenAI @GPT4o response of
        Right [AssistantTool toolCall] -> do
          getToolCallName toolCall `shouldBe` "get_weather"
          case toolCall of
            ToolCall _ _ _ -> return ()  -- Valid tool call
            InvalidToolCall _ _ _ _ -> expectationFailure "Expected valid ToolCall"
        Right _ -> expectationFailure "Expected AssistantTool with one tool call"
        Left err -> expectationFailure $ "fromResponse failed: " ++ show err

    it "sends tool result and receives final text response" $ do
      let toolDef = ToolDefinition "get_weather" "Get weather" (object [])
          toolCall = ToolCall "call_123" "get_weather"
                       (object ["location" .= ("Paris" :: Text)])
          toolResult = ToolResult toolCall
                         (Right $ object ["temp" .= ("22°C" :: Text)])
          history = [ UserText "What's the weather in Paris?" :: Message GPT4o OpenAI
                    , AssistantTool toolCall
                    , ToolResultMsg toolResult
                    ]
          req = toRequest OpenAI GPT4o [MaxTokens 100, Tools [toolDef]] history

      (response :: OpenAIResponse) <- getResponse req

      case fromResponse @OpenAI @GPT4o response of
        Right [AssistantText txt] -> do
          T.isInfixOf "22" txt `shouldBe` True
        Right _ -> expectationFailure "Expected single AssistantText message"
        Left err -> expectationFailure $ "fromResponse failed: " ++ show err

  describe "OpenAI Provider - JSON Mode" $ do

    it "requests JSON response and receives structured data" $ do
      let schema = object
            [ "type" .= ("object" :: Text)
            , "properties" .= object
                [ "colors" .= object
                    [ "type" .= ("array" :: Text)
                    , "items" .= object ["type" .= ("string" :: Text)]
                    ]
                ]
            , "required" .= (["colors"] :: [Text])
            ]
          msgs = [UserRequestJSON "List 3 primary colors" schema :: Message GPT4o OpenAI]
          req = toRequest OpenAI GPT4o [MaxTokens 100] msgs

      (response :: OpenAIResponse) <- getResponse req

      case fromResponse @OpenAI @GPT4o response of
        Right [AssistantJSON jsonVal] -> do
          -- Should be valid JSON with colors array
          case jsonVal of
            Aeson.Object obj ->
              case KM.lookup "colors" obj of
                Just (Aeson.Array arr) -> length arr `shouldSatisfy` (>= 3)
                _ -> expectationFailure "JSON missing 'colors' array"
            _ -> expectationFailure "Response not a JSON object"
        Right other -> expectationFailure $ "Expected [AssistantJSON], got: " ++ show other
        Left err -> expectationFailure $ "fromResponse failed: " ++ show err
