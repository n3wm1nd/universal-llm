{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ComposableHandlersSpec (spec) where

import Test.Hspec
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI hiding (model)
import qualified UniversalLLM.Protocols.OpenAI as OAI
import UniversalLLM.Providers.OpenAI hiding (handleReasoning)
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider

-- Test models with explicit capabilities
data BasicModel = BasicModel deriving (Show, Eq)
data ToolsModel = ToolsModel deriving (Show, Eq)
data ReasoningModel = ReasoningModel deriving (Show, Eq)
data FullFeaturedModel = FullFeaturedModel deriving (Show, Eq)

-- BasicModel: just text
instance ModelName OpenAI BasicModel where
  modelName _ = "basic-model"

-- ToolsModel: text + tools
instance ModelName OpenAI ToolsModel where
  modelName _ = "tools-model"
instance HasTools ToolsModel OpenAI where
  withTools = chainProviders OpenAIProvider.openAITools

-- ReasoningModel: text + reasoning
instance ModelName OpenAI ReasoningModel where
  modelName _ = "reasoning-model"
instance HasReasoning ReasoningModel OpenAI where
  withReasoning = chainProviders OpenAIProvider.openAIReasoning

-- FullFeaturedModel: text + tools + reasoning + JSON
instance ModelName OpenAI FullFeaturedModel where
  modelName _ = "full-featured-model"
instance HasTools FullFeaturedModel OpenAI where
  withTools = chainProviders OpenAIProvider.openAITools
instance HasReasoning FullFeaturedModel OpenAI where
  withReasoning = chainProviders OpenAIProvider.openAIReasoning
instance HasJSON FullFeaturedModel OpenAI where
  withJSON = chainProviders OpenAIProvider.openAIJSON

spec :: Spec
spec = do
  describe "Composable Message Handlers" $ do

    it "base handler sets model name and config (BasicModel)" $ do
      let provider = OpenAI
          model = BasicModel
          configs = [Temperature 0.7, MaxTokens 100]
          msg = UserText "test"
          req = handleBase provider model configs msg mempty

      OAI.model req `shouldBe` "basic-model"
      temperature req `shouldBe` Just 0.7
      max_tokens req `shouldBe` Just 100

    it "text handler converts user messages (BasicModel)" $ do
      let provider = OpenAI
          model = BasicModel
          configs = []
          msg = UserText "Hello"
          req = handleTextMessages provider model configs msg mempty

      length (messages req) `shouldBe` 1
      case head (messages req) of
        OpenAIMessage role (Just content) _ _ _ -> do
          role `shouldBe` "user"
          content `shouldBe` "Hello"
        _ -> expectationFailure "Expected user message"

    it "text handler merges consecutive user messages (BasicModel)" $ do
      let provider = OpenAI
          model = BasicModel
          configs = []
          msg1 = UserText "Hello"
          msg2 = UserText "World"
          req1 = handleTextMessages provider model configs msg1 mempty
          req2 = handleTextMessages provider model configs msg2 req1

      length (messages req2) `shouldBe` 1
      case head (messages req2) of
        OpenAIMessage "user" (Just content) _ _ _ ->
          content `shouldBe` "Hello\nWorld"
        _ -> expectationFailure "Expected merged user message"

    it "handlers compose manually (BasicModel)" $ do
      let provider = OpenAI
          model = BasicModel
          configs = [Temperature 0.5]
          msg = UserText "test"

          -- Manual composition
          req0 = mempty
          req1 = handleBase provider model configs msg req0
          req2 = handleTextMessages provider model configs msg req1

      OAI.model req2 `shouldBe` "basic-model"
      temperature req2 `shouldBe` Just 0.5
      length (messages req2) `shouldBe` 1

    it "reasoning handler converts reasoning messages (ReasoningModel)" $ do
      let provider = OpenAI
          model = ReasoningModel
          configs = []
          msg = AssistantReasoning "thinking about the problem..."
          req = OpenAIProvider.handleReasoning provider model configs () msg mempty

      length (messages req) `shouldBe` 1
      case head (messages req) of
        OpenAIMessage "assistant" Nothing (Just reasoning) Nothing Nothing ->
          reasoning `shouldBe` "thinking about the problem..."
        _ -> expectationFailure "Expected assistant reasoning message"

    it "reasoning handler composes with text handler (ReasoningModel)" $ do
      let provider = OpenAI
          model = ReasoningModel
          configs = []
          reasoningMsg = AssistantReasoning "let me think..."
          textMsg = AssistantText "Here's my answer"

          -- Compose: reasoning then text
          req0 = mempty
          req1 = OpenAIProvider.handleReasoning provider model configs () reasoningMsg req0
          req2 = handleTextMessages provider model configs textMsg req1

      length (messages req2) `shouldBe` 2
      case messages req2 of
        [OpenAIMessage "assistant" Nothing (Just reasoning) Nothing Nothing,
         OpenAIMessage "assistant" (Just content) Nothing Nothing Nothing] -> do
          reasoning `shouldBe` "let me think..."
          content `shouldBe` "Here's my answer"
        _ -> expectationFailure "Expected reasoning message followed by text message"

    it "handlers chain sequentially (BasicModel)" $ do
      let provider = OpenAI
          model = BasicModel
          configs = [Temperature 0.5]
          msg = UserText "test"

          -- Chain handlers manually
          req = handleTextMessages provider model configs msg
                  (handleBase provider model configs msg mempty)

      OAI.model req `shouldBe` "basic-model"
      temperature req `shouldBe` Just 0.5
      length (messages req) `shouldBe` 1
      case head (messages req) of
        OpenAIMessage "user" (Just content) _ _ _ ->
          content `shouldBe` "test"
        _ -> expectationFailure "Expected user message"

    it "handlers chain with reasoning (ReasoningModel)" $ do
      let provider = OpenAI
          model = ReasoningModel
          configs = []
          reasoningMsg = AssistantReasoning "thinking..."
          textMsg = AssistantText "answer"

          -- Chain handlers manually
          req1 = handleTextMessages provider model configs reasoningMsg
                   (OpenAIProvider.handleReasoning provider model configs () reasoningMsg mempty)
          req2 = handleTextMessages provider model configs textMsg
                   (OpenAIProvider.handleReasoning provider model configs () textMsg req1)

      length (messages req2) `shouldBe` 2

  describe "Composable Providers (bidirectional)" $ do

    it "baseComposableProvider handles text messages" $ do
      let provider = OpenAI
          model = BasicModel
          configs = []
          msg = UserText "Hello"
          handlers = baseComposableProvider @OpenAI @BasicModel provider model configs ()
          req = cpToRequest handlers msg mempty

      length (messages req) `shouldBe` 1
      case head (messages req) of
        OpenAIMessage "user" (Just content) _ _ _ ->
          content `shouldBe` "Hello"
        _ -> expectationFailure "Expected user message"

    it "composable providers compose via <>" $ do
      let provider = OpenAI
          model = ReasoningModel
          configs = []
          msg = AssistantReasoning "thinking"

          -- Compose base + reasoning
          composed = (baseComposableProvider @OpenAI @ReasoningModel) `chainProviders` OpenAIProvider.openAIReasoning
          handlers = composed provider model configs ((), ())
          req = cpToRequest handlers msg mempty

      length (messages req) `shouldBe` 1
      case head (messages req) of
        OpenAIMessage "assistant" Nothing (Just reasoning) Nothing Nothing ->
          reasoning `shouldBe` "thinking"
        _ -> expectationFailure "Expected reasoning message"

    -- This test demonstrates that type safety prevents using reasoning with non-reasoning models
    -- UNCOMMENT to verify it fails to compile with: No instance for 'HasReasoning BasicModel'
    {-
    it "SHOULD NOT COMPILE: reasoning handler with BasicModel" $ do
      let provider = OpenAI
          model = BasicModel  -- BasicModel does NOT have HasReasoning!
          configs = []
          msg = AssistantReasoning "this should fail"
          req = OpenAIProvider.handleReasoning provider model configs msg mempty
      length (messages req) `shouldBe` 1
    -}
