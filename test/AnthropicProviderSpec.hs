{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module AnthropicProviderSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import TestCache (ResponseProvider)
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.Anthropic
import qualified UniversalLLM.Providers.Anthropic as Provider
import TestModels (ClaudeSonnet45(..))

spec :: ResponseProvider AnthropicRequest AnthropicResponse -> Spec
spec getResponse = do
  describe "Anthropic Provider - Basic Text" $ do

    it "sends message and receives text response" $ do
      let testModel = ClaudeSonnet45
          testMessages = [UserText "What is 2+2?"]
          testConfigs = [MaxTokens 50, SystemPrompt "You are a helpful assistant."]
          req = Provider.withMagicSystemPrompt $
                  Provider.toRequest' Provider.Anthropic testModel testConfigs testMessages

      response <- getResponse req

      case Provider.fromResponse' @ClaudeSonnet45 response of
        Right [AssistantText txt] -> do
          -- Should contain the answer
          T.isInfixOf "4" txt `shouldBe` True
        Right _ -> expectationFailure "Expected single AssistantText message"
        Left err -> expectationFailure $ "fromResponse failed: " <> show err

  describe "Anthropic Provider - Tool Calling" $ do

    it "requests tool call and receives tool_use response" $ do
      let testModel = ClaudeSonnet45
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the current weather in a given location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("The city and state, e.g. San Francisco, CA" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          initialMessages = [UserText "What is the weather like in San Francisco?"]
          configs = [MaxTokens 100, Tools [toolDef]]
          req = Provider.toRequest' Provider.Anthropic testModel configs initialMessages

      response <- getResponse req

      case Provider.fromResponse' @ClaudeSonnet45 response of
        Right [AssistantTool toolCall] -> do
          getToolCallName toolCall `shouldBe` "get_weather"
          -- Should have extracted location parameter
          case toolCall of
            ToolCall _ _ _ -> return ()  -- Valid tool call
            InvalidToolCall _ _ _ _ -> expectationFailure "Expected valid ToolCall, not InvalidToolCall"
        Right _ -> expectationFailure "Expected AssistantTool with one tool call"
        Left err -> expectationFailure $ "fromResponse failed: " <> show err

    it "sends tool result and receives final text response" $ do
      let testModel = ClaudeSonnet45
          toolDef = ToolDefinition "get_weather" "Get weather"
                      (object ["type" .= ("object" :: Text)])
          -- Simulate previous conversation
          toolCall = ToolCall "toolu_123" "get_weather"
                       (object ["location" .= ("San Francisco, CA" :: Text)])
          toolResult = ToolResult toolCall
                         (Right $ object
                           [ "temperature" .= ("72°F" :: Text)
                           , "conditions" .= ("sunny" :: Text)
                           ])
          history = [ UserText "What is the weather like in San Francisco?"
                    , AssistantTool toolCall
                    , ToolResultMsg toolResult
                    ]
          req = Provider.toRequest' Provider.Anthropic testModel [MaxTokens 100, Tools [toolDef]] history

      response <- getResponse req

      case Provider.fromResponse' @ClaudeSonnet45 response of
        Right [AssistantText txt] -> do
          -- Should mention the weather from the tool result
          (T.isInfixOf "72" txt || T.isInfixOf "sunny" txt) `shouldBe` True
        Right _ -> expectationFailure "Expected single AssistantText message"
        Left err -> expectationFailure $ "fromResponse failed: " <> show err
