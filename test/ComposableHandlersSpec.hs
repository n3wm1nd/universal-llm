{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}

module ComposableHandlersSpec (spec) where

import Test.Hspec
import UniversalLLM
import UniversalLLM.Providers.OpenAI as OpenAIProvider
import UniversalLLM.Protocols.OpenAI hiding (model)
import qualified UniversalLLM.Protocols.OpenAI as OAI

-- Test models with explicit capabilities
data BasicModel = BasicModel deriving (Show, Eq)
data ToolsModel = ToolsModel deriving (Show, Eq)
data ReasoningModel = ReasoningModel deriving (Show, Eq)
data FullFeaturedModel = FullFeaturedModel deriving (Show, Eq)

-- BasicModel: just text
instance ModelName (Model BasicModel OpenAIProvider.OpenAI) where
  modelName (Model _ _) = "basic-model"

-- ToolsModel: text + tools
instance ModelName (Model ToolsModel OpenAIProvider.OpenAI) where
  modelName (Model _ _) = "tools-model"
instance HasTools (Model ToolsModel OpenAIProvider.OpenAI) where
  withTools = OpenAIProvider.openAITools

-- ReasoningModel: text + reasoning
instance ModelName (Model ReasoningModel OpenAIProvider.OpenAI) where
  modelName (Model _ _) = "reasoning-model"
instance HasReasoning (Model ReasoningModel OpenAIProvider.OpenAI) where
  withReasoning = OpenAIProvider.openAIReasoning

-- FullFeaturedModel: text + tools + reasoning + JSON
instance ModelName (Model FullFeaturedModel OpenAIProvider.OpenAI) where
  modelName (Model _ _) = "full-featured-model"
instance HasTools (Model FullFeaturedModel OpenAIProvider.OpenAI) where
  withTools = OpenAIProvider.openAITools
instance HasReasoning (Model FullFeaturedModel OpenAIProvider.OpenAI) where
  withReasoning = OpenAIProvider.openAIReasoning
instance HasJSON (Model FullFeaturedModel OpenAIProvider.OpenAI) where
  withJSON = OpenAIProvider.openAIJSON

spec :: Spec
spec = do
  describe "Composable Message Handlers" $ do

    it "baseComposableProvider sets model name and config (BasicModel)" $ do
      let model = Model BasicModel OpenAIProvider.OpenAI
          configs = [Temperature 0.7, MaxTokens 100]
          msg = UserText "test"
          handlers = OpenAIProvider.baseComposableProvider model configs ()
          req = cpToRequest handlers msg mempty

      OAI.model req `shouldBe` "basic-model"
      temperature req `shouldBe` Just 0.7
      max_tokens req `shouldBe` Just 100

    it "baseComposableProvider converts user messages (BasicModel)" $ do
      let model = Model BasicModel OpenAIProvider.OpenAI
          configs = []
          msg = UserText "Hello"
          handlers = OpenAIProvider.baseComposableProvider model configs ()
          req = cpToRequest handlers msg mempty

      length (messages req) `shouldBe` 1
      let msg = head (messages req)
      role msg `shouldBe` "user"
      content msg `shouldBe` Just "Hello"

    it "baseComposableProvider merges consecutive user messages (BasicModel)" $ do
      let model = Model BasicModel OpenAIProvider.OpenAI
          configs = []
          msg1 = UserText "Hello"
          msg2 = UserText "World"
          handlers = OpenAIProvider.baseComposableProvider model configs ()
          req1 = cpToRequest handlers msg1 mempty
          req2 = cpToRequest handlers msg2 req1

      length (messages req2) `shouldBe` 1
      let msg = head (messages req2)
      role msg `shouldBe` "user"
      content msg `shouldBe` Just "Hello\nWorld"

    it "composable providers compose manually (BasicModel)" $ do
      let model = Model BasicModel OpenAIProvider.OpenAI
          configs = [Temperature 0.5]
          msg = UserText "test"
          handlers = OpenAIProvider.baseComposableProvider model configs ()
          req = cpConfigHandler handlers (cpToRequest handlers msg mempty)

      OAI.model req `shouldBe` "basic-model"
      temperature req `shouldBe` Just 0.5
      length (messages req) `shouldBe` 1

    it "reasoning handler converts reasoning messages (ReasoningModel)" $ do
      let model = Model ReasoningModel OpenAIProvider.OpenAI
          configs = []
          msg = AssistantReasoning "thinking about the problem..."
          handlers = OpenAIProvider.openAIReasoning model configs ()
          req = cpToRequest handlers msg mempty

      length (messages req) `shouldBe` 1
      let msg = head (messages req)
      role msg `shouldBe` "assistant"
      content msg `shouldBe` Just ""
      reasoning_content msg `shouldBe` Just "thinking about the problem..."

    it "reasoning handler composes with base handler (ReasoningModel)" $ do
      let model = Model ReasoningModel OpenAIProvider.OpenAI
          configs = []
          reasoningMsg = AssistantReasoning "let me think..."
          textMsg = AssistantText "Here's my answer"

          -- Compose: reasoning then base
          baseHandlers = OpenAIProvider.baseComposableProvider model configs ()
          reasoningHandlers = OpenAIProvider.openAIReasoning model configs ()
          composedHandlers = reasoningHandlers `chainProvidersAt` baseHandlers

          req1 = cpToRequest composedHandlers reasoningMsg mempty
          req2 = cpToRequest composedHandlers textMsg req1

      length (messages req2) `shouldBe` 2
      let [msg1, msg2] = messages req2
      role msg1 `shouldBe` "assistant"
      content msg1 `shouldBe` Just ""
      reasoning_content msg1 `shouldBe` Just "let me think..."
      role msg2 `shouldBe` "assistant"
      content msg2 `shouldBe` Just "Here's my answer"

    it "composable providers chain (BasicModel)" $ do
      let model = Model BasicModel OpenAIProvider.OpenAI
          configs = [Temperature 0.5]
          msg = UserText "test"

          handlers = OpenAIProvider.baseComposableProvider model configs ()
          req = cpConfigHandler handlers (cpToRequest handlers msg mempty)

      OAI.model req `shouldBe` "basic-model"
      temperature req `shouldBe` Just 0.5
      length (messages req) `shouldBe` 1
      let msg = head (messages req)
      role msg `shouldBe` "user"
      content msg `shouldBe` Just "test"

    it "composable providers chain with reasoning (ReasoningModel)" $ do
      let model = Model ReasoningModel OpenAIProvider.OpenAI
          configs = []
          reasoningMsg = AssistantReasoning "thinking..."
          textMsg = AssistantText "answer"

          -- Chain handlers manually
          reasoningHandlers = OpenAIProvider.openAIReasoning model configs ()
          baseHandlers = OpenAIProvider.baseComposableProvider model configs ()
          composedHandlers = reasoningHandlers `chainProvidersAt` baseHandlers

          req1 = cpToRequest composedHandlers reasoningMsg mempty
          req2 = cpToRequest composedHandlers textMsg req1

      length (messages req2) `shouldBe` 2

  describe "Composable Providers (bidirectional)" $ do

    it "baseComposableProvider handles text messages" $ do
      let model = Model BasicModel OpenAIProvider.OpenAI
          configs = []
          msg = UserText "Hello"
          handlers = OpenAIProvider.baseComposableProvider model configs ()
          req = cpToRequest handlers msg mempty

      length (messages req) `shouldBe` 1
      let msg = head (messages req)
      role msg `shouldBe` "user"
      content msg `shouldBe` Just "Hello"

    it "composable providers compose via chainProviders" $ do
      let model = Model ReasoningModel OpenAIProvider.OpenAI
          configs = []
          msg = AssistantReasoning "thinking"

          -- Compose base + reasoning
          composed = OpenAIProvider.openAIReasoning `chainProviders` OpenAIProvider.baseComposableProvider
          handlers = composed model configs ((), ())
          req = cpToRequest handlers msg mempty

      length (messages req) `shouldBe` 1
      let msg = head (messages req)
      role msg `shouldBe` "assistant"
      content msg `shouldBe` Just ""
      reasoning_content msg `shouldBe` Just "thinking"

    -- This test demonstrates that type safety prevents using reasoning with non-reasoning models
    -- UNCOMMENT to verify it fails to compile with: No instance for 'HasReasoning BasicModel'
    {-
    it "SHOULD NOT COMPILE: reasoning handler with BasicModel" $ do
      let model = Model BasicModel OpenAIProvider.OpenAI  -- BasicModel does NOT have HasReasoning!
          configs = []
          msg = AssistantReasoning "this should fail"
          handlers = OpenAIProvider.openAIReasoning model configs ()
          req = cpToRequest handlers msg mempty
      length (messages req) `shouldBe` 1
    -}
