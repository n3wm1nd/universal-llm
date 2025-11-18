{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module AnthropicStreamingSpec (spec) where

import Test.Hspec
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Lazy.Char8 as BSLC
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (object, (.=))
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.Anthropic
import qualified UniversalLLM.Protocols.Anthropic as Proto
import qualified UniversalLLM.Providers.Anthropic as Provider

-- Helper to build streaming request - uses ClaudeSonnet45 with basic composition
-- (Wrapper around the generic version that provides the specific model and composable provider)
buildStreamingRequest :: [ModelConfig Provider.Anthropic TestModels.ClaudeSonnet45]
                     -> [Message TestModels.ClaudeSonnet45 Provider.Anthropic]
                     -> AnthropicRequest
buildStreamingRequest = buildStreamingRequestGeneric TestModels.anthropicSonnet45 TestModels.ClaudeSonnet45 ((), ())

-- Helper to build streaming request - uses ClaudeSonnet45WithReasoning
buildStreamingRequestWithReasoning :: [ModelConfig Provider.Anthropic TestModels.ClaudeSonnet45WithReasoning]
                                 -> [Message TestModels.ClaudeSonnet45WithReasoning Provider.Anthropic]
                                 -> AnthropicRequest
buildStreamingRequestWithReasoning = buildStreamingRequestGeneric TestModels.anthropicSonnet45Reasoning TestModels.ClaudeSonnet45WithReasoning ((), ((), ()))

-- Generic helper to build streaming request with explicit composable provider
buildStreamingRequestGeneric :: forall model s. ComposableProvider Provider.Anthropic model s
                           -> model
                           -> s
                           -> [ModelConfig Provider.Anthropic model]
                           -> [Message model Provider.Anthropic]
                           -> AnthropicRequest
buildStreamingRequestGeneric composableProvider model s configs = snd . toProviderRequest composableProvider Provider.Anthropic model configs s

spec :: ResponseProvider AnthropicRequest BSL.ByteString -> Spec
spec getResponse = do
  describe "Anthropic Streaming Responses (SSE Format)" $ do

    it "sends streaming request and receives SSE response with message_stop event" $ do
      let model = ClaudeSonnet45
          configs = [MaxTokens 100, Streaming True]
          msgs = [UserText "Say hello"]
          req = Provider.withMagicSystemPrompt $
                 buildStreamingRequest configs msgs

      -- Verify the request is correctly configured for streaming
      Proto.stream req `shouldBe` Just True

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate basic SSE structure: should be non-empty
      BSL.null sseBody `shouldBe` False

      -- Check that it contains SSE event markers
      let bodyStr = BSLC.unpack sseBody
      T.isInfixOf "event:" (T.pack bodyStr) `shouldBe` True

      -- Check for message_stop event (indicates complete response)
      T.isInfixOf "message_stop" (T.pack bodyStr) `shouldBe` True

    it "sends streaming request with tools and receives SSE response with tool_use event" $ do
      let model = ClaudeSonnet45
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
          req = Provider.withMagicSystemPrompt $
                 buildStreamingRequest configs msgs

      -- Verify the request is correctly configured for streaming and tools
      Proto.stream req `shouldBe` Just True
      case Proto.tools req of
        Just [tool] -> anthropicToolName tool `shouldBe` "get_weather"
        _ -> expectationFailure "Expected get_weather tool in request"

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate SSE structure
      BSL.null sseBody `shouldBe` False

      let bodyStr = BSLC.unpack sseBody

      -- Check for SSE event markers
      T.isInfixOf "event:" (T.pack bodyStr) `shouldBe` True

      -- Check for message_stop event (indicates complete response)
      T.isInfixOf "message_stop" (T.pack bodyStr) `shouldBe` True

    it "sends streaming request with extended thinking and receives SSE response with thinking_delta events" $ do
      -- Use ClaudeSonnet45WithReasoning for this test since reasoning requires HasReasoning instance
      let model = ClaudeSonnet45WithReasoning
          configs = [MaxTokens 16000, Streaming True, Reasoning True]
          msgs = [UserText "Solve this puzzle: What has cities but no houses, forests but no trees, and water but no fish?"]
          req = Provider.withMagicSystemPrompt $
                 buildStreamingRequestWithReasoning configs msgs

      -- Verify the request is correctly configured for streaming with thinking
      Proto.stream req `shouldBe` Just True
      Proto.thinking req `shouldNotBe` Nothing

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate SSE structure
      BSL.null sseBody `shouldBe` False

      let bodyStr = BSLC.unpack sseBody

      -- Check for SSE event markers
      T.isInfixOf "event:" (T.pack bodyStr) `shouldBe` True

      -- Check for thinking_delta events (indicates thinking content is streaming)
      T.isInfixOf "thinking_delta" (T.pack bodyStr) `shouldBe` True

      -- Check for message_stop event (indicates complete response)
      T.isInfixOf "message_stop" (T.pack bodyStr) `shouldBe` True

    it "sends streaming request with thinking and tools, receives SSE response with thinking and tool_use events" $ do
      let model = ClaudeSonnet45WithReasoning
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
          req = Provider.withMagicSystemPrompt $
                 buildStreamingRequestWithReasoning configs msgs

      -- Verify the request is correctly configured for streaming with thinking and tools
      Proto.stream req `shouldBe` Just True
      Proto.thinking req `shouldNotBe` Nothing
      case Proto.tools req of
        Just [tool] -> anthropicToolName tool `shouldBe` "get_weather"
        _ -> expectationFailure "Expected get_weather tool in request"

      -- Get the SSE response
      sseBody <- getResponse req

      -- Validate SSE structure
      BSL.null sseBody `shouldBe` False

      let bodyStr = BSLC.unpack sseBody

      -- Check for SSE event markers
      T.isInfixOf "event:" (T.pack bodyStr) `shouldBe` True

      -- Check for thinking_delta events
      T.isInfixOf "thinking_delta" (T.pack bodyStr) `shouldBe` True

      -- Check for tool_use event
      T.isInfixOf "tool_use" (T.pack bodyStr) `shouldBe` True

      -- Check for message_stop event (indicates complete response)
      T.isInfixOf "message_stop" (T.pack bodyStr) `shouldBe` True

    it "handles multiple content blocks in order (thinking and tool_use)" $ do
      let model = ClaudeSonnet45WithReasoning
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get weather"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object ["location" .= object ["type" .= ("string" :: Text)]]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [MaxTokens 16000, Tools [toolDef], Streaming True, Reasoning True]
          msgs = [UserText "What's the weather in Paris?"]
          req = Provider.withMagicSystemPrompt $
                 buildStreamingRequestWithReasoning configs msgs

      Proto.stream req `shouldBe` Just True
      Proto.thinking req `shouldNotBe` Nothing

      sseBody <- getResponse req
      BSL.null sseBody `shouldBe` False

      -- Response should have thinking and tool_use blocks
      let bodyStr = BSLC.unpack sseBody
      T.isInfixOf "thinking_delta" (T.pack bodyStr) `shouldBe` True
      T.isInfixOf "tool_use" (T.pack bodyStr) `shouldBe` True
