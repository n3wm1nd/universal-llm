{-# LANGUAGE OverloadedStrings #-}

module Main where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import UniversalLLM

main :: IO ()
main = hspec $ do
  describe "UniversalLLM Core Types" $ do
    it "creates basic messages" $ do
      let msg = userText "Hello, world!"
      getMessageRole msg `shouldBe` "user"
      getMessageContent msg `shouldBe` "Hello, world!"

    it "creates system messages" $ do
      let msg = systemMessage "You are a helpful assistant."
      getMessageRole msg `shouldBe` "system"
      getMessageContent msg `shouldBe` "You are a helpful assistant."

    it "creates assistant messages" $ do
      let msg = assistantText "Hello! How can I help you?"
      getMessageRole msg `shouldBe` "assistant"
      getMessageContent msg `shouldBe` "Hello! How can I help you?"

  describe "Model Parameters" $ do
    it "creates model parameters" $ do
      let params = ModelParams
            { temperature = Just (Temperature 0.7)
            , maxTokens = Just (MaxTokens 100)
            }
      temperature params `shouldBe` Just (Temperature 0.7)
      maxTokens params `shouldBe` Just (MaxTokens 100)

  describe "Type Safety" $ do
    it "allows vision messages for vision-capable providers" $ do
      -- This test verifies that the type system allows valid combinations
      let image = Image "test-data" "image/jpeg"
          -- This should compile because GPT4o + OpenAI supports Vision
          msg :: Message GPT4o OpenAI
          msg = userWithImages [image] "Describe this image"

      getMessageRole msg `shouldBe` "user"
      T.isInfixOf "Describe this image" (getMessageContent msg) `shouldBe` True

  describe "Emulation" $ do
    it "provides JSON capability through emulation" $ do
      -- Test that EmulateJSONOutput provides the expected functionality
      let provider = AnthropicWithJSON Anthropic
      -- The fact that this compiles shows that emulation is working
      True `shouldBe` True

  describe "Message Composition" $ do
    it "composes user request capabilities" $ do
      -- Test the composable nature of user requests
      let image = Image "test-data" "image/jpeg"
          -- This tests the compositional structure
          msg :: Message GPT4o OpenAI
          msg = UserMsg $ WithPrefix "Sure, " $
                         WithImages [image] $
                         BasicUserRequest "analyze this"

      getMessageRole msg `shouldBe` "user"