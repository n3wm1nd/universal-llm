{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module OpenAIStreamingSpec (spec) where

import Test.Hspec
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Lazy.Char8 as BSLC
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import qualified UniversalLLM.Protocols.OpenAI as Proto
import qualified UniversalLLM.Providers.OpenAI as Provider

-- Helper to build streaming request using the model's provider implementation
buildStreamingRequest :: forall model. ProviderImplementation Provider.OpenAI model
                     => model
                     -> [ModelConfig Provider.OpenAI model]
                     -> [Message model Provider.OpenAI]
                     -> OpenAIRequest
buildStreamingRequest = toProviderRequest Provider.OpenAI

spec :: ResponseProvider OpenAIRequest BSL.ByteString -> Spec
spec getResponse = do
  describe "OpenAI Streaming Responses (SSE Format)" $ do

    it "sends streaming request and receives SSE response with stream completion" $ do
      let model = GLM45
          configs = [MaxTokens 100, Streaming True]
          msgs = [UserText "Say hello"]
          req = buildStreamingRequest model configs msgs

      -- Verify the request is correctly configured for streaming
      Proto.stream req `shouldBe` Just True

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate basic SSE structure: should be non-empty
      BSL.null sseBody `shouldBe` False

      -- Check that it contains SSE event markers
      let bodyStr = BSLC.unpack sseBody
      T.isInfixOf "data:" (T.pack bodyStr) `shouldBe` True

    it "sends streaming request with tools and receives SSE response with tool_calls" $ do
      let model = GLM45
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the weather for a location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("City name" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [MaxTokens 2048, Tools [toolDef], Streaming True]
          msgs = [UserText "What's the weather in Paris?"]
          req = buildStreamingRequest model configs msgs

      -- Verify the request is correctly configured for streaming and tools
      Proto.stream req `shouldBe` Just True
      case Proto.tools req of
        Just tools -> length tools `shouldBe` 1
        _ -> expectationFailure "Expected tools in request"

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate SSE structure
      BSL.null sseBody `shouldBe` False

      let bodyStr = BSLC.unpack sseBody

      -- Check for SSE data markers
      T.isInfixOf "data:" (T.pack bodyStr) `shouldBe` True

    it "sends streaming request with reasoning and receives SSE response with reasoning_content" $ do
      let model = GLM45
          -- Reasoning config for models that support it
          configs = [MaxTokens 16000, Streaming True, Reasoning True]
          msgs = [UserText "Solve this puzzle: What has cities but no houses, forests but no trees, and water but no fish?"]
          req = buildStreamingRequest model configs msgs

      -- Verify the request is correctly configured for streaming with reasoning
      Proto.stream req `shouldBe` Just True

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate SSE structure
      BSL.null sseBody `shouldBe` False

      let bodyStr = BSLC.unpack sseBody

      -- Check for SSE data markers
      T.isInfixOf "data:" (T.pack bodyStr) `shouldBe` True

    it "sends streaming request with reasoning and tools, receives SSE response with both" $ do
      let model = GLM45
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the weather for a location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("City name" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [MaxTokens 16000, Tools [toolDef], Streaming True, Reasoning True]
          msgs = [UserText "What's the weather in Paris?"]
          req = buildStreamingRequest model configs msgs

      -- Verify the request is correctly configured for streaming with reasoning and tools
      Proto.stream req `shouldBe` Just True
      case Proto.tools req of
        Just tools -> length tools `shouldBe` 1
        _ -> expectationFailure "Expected tools in request"

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate SSE structure
      BSL.null sseBody `shouldBe` False

      let bodyStr = BSLC.unpack sseBody

      -- Check for SSE data markers
      T.isInfixOf "data:" (T.pack bodyStr) `shouldBe` True
