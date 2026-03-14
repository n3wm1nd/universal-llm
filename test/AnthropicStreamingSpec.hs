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
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import Network.SSE (parseSSEComplete, sseEventType, sseEventData)
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM
import UniversalLLM.Protocols.Anthropic
import qualified UniversalLLM.Protocols.Anthropic as Proto
import qualified UniversalLLM.Providers.Anthropic as Provider
import Data.Default (Default(..))

-- Helper to build streaming request - uses ClaudeSonnet45NoReason with basic composition
-- (Wrapper around generic version that provides specific model and composable provider)
buildStreamingRequest :: [ModelConfig (Model TestModels.ClaudeSonnet45NoReason Provider.AnthropicOAuth)]
                     -> [Message (Model TestModels.ClaudeSonnet45NoReason Provider.AnthropicOAuth)]
                     -> AnthropicRequest
buildStreamingRequest configs msgs = buildStreamingRequestGeneric TestModels.anthropicSonnet45NoReasonOAuth (Model TestModels.ClaudeSonnet45NoReason Provider.AnthropicOAuth) ((), ((), ())) configs msgs

-- Helper to build streaming request - uses ClaudeSonnet45 (with reasoning)
buildStreamingRequestWithReasoning :: [ModelConfig (Model TestModels.ClaudeSonnet45 Provider.AnthropicOAuth)]
                                 -> [Message (Model TestModels.ClaudeSonnet45 Provider.AnthropicOAuth)]
                                 -> AnthropicRequest
buildStreamingRequestWithReasoning configs msgs = buildStreamingRequestGeneric TestModels.anthropicSonnet45OAuth (Model TestModels.ClaudeSonnet45 Provider.AnthropicOAuth) (def, ((), ((), ()))) configs msgs

-- Generic helper to build streaming request with explicit composable provider
buildStreamingRequestGeneric :: forall model s. HasStreaming (Model model Provider.AnthropicOAuth)
                           => ComposableProvider (Model model Provider.AnthropicOAuth) s
                           -> Model model Provider.AnthropicOAuth
                           -> s
                           -> [ModelConfig (Model model Provider.AnthropicOAuth)]
                           -> [Message (Model model Provider.AnthropicOAuth)]
                           -> AnthropicRequest
buildStreamingRequestGeneric composableProvider modelValue s configs msgs =
  let req = snd $ toProviderRequest composableProvider modelValue configs s msgs
  in Proto.enableAnthropicStreaming req

-- | Check that a parsed SSE body contains an event with the given SSE event type.
hasEventType :: BSL.ByteString -> Text -> Bool
hasEventType body evType =
    any (\ev -> sseEventType ev == Just evType) $
        parseSSEComplete (BSL.toStrict body)

-- | Check that any event's JSON data contains the string value anywhere in the object tree.
hasDataValue :: BSL.ByteString -> Text -> Bool
hasDataValue body value =
    any (containsValue . Aeson.decodeStrict . sseEventData) $
        parseSSEComplete (BSL.toStrict body)
  where
    containsValue Nothing                  = False
    containsValue (Just (Aeson.String s))  = s == value
    containsValue (Just (Aeson.Object km)) = any (containsValue . Just) (KM.elems km)
    containsValue (Just (Aeson.Array  vs)) = any (containsValue . Just) vs
    containsValue (Just _)                 = False

spec :: ResponseProvider AnthropicRequest BSL.ByteString -> Spec
spec getResponse = do
  describe "Anthropic Streaming Responses (SSE Format)" $ do

    it "sends streaming request and receives SSE response with message_stop event" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic
          configs = [MaxTokens 100]
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
      sseBody `shouldSatisfy` (`hasEventType` "message_stop")

    it "sends streaming request with tools and receives SSE response with tool_use event" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic
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
          configs = [MaxTokens 2048, Tools [toolDef]]
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
      sseBody `shouldSatisfy` (`hasEventType` "message_stop")

    it "sends streaming request with extended thinking and receives SSE response with thinking_delta events" $ do
      -- Use ClaudeSonnet45 for this test since reasoning requires HasReasoning instance
      let model = Model ClaudeSonnet45 Provider.Anthropic
          configs = [MaxTokens 16000, Reasoning True]
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
      sseBody `shouldSatisfy` (`hasDataValue` "thinking_delta")

      -- Check for message_stop event (indicates complete response)
      sseBody `shouldSatisfy` (`hasEventType` "message_stop")

    it "sends streaming request with thinking and tools, receives SSE response with thinking and tool_use events" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic
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
          configs = [MaxTokens 16000, Tools [toolDef], Reasoning True]
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
      sseBody `shouldSatisfy` (`hasDataValue` "thinking_delta")

      -- Check for tool_use content block
      sseBody `shouldSatisfy` (`hasDataValue` "tool_use")

      -- Check for message_stop event (indicates complete response)
      sseBody `shouldSatisfy` (`hasEventType` "message_stop")

    it "handles multiple content blocks in order (thinking and tool_use)" $ do
      let model = Model ClaudeSonnet45 Provider.Anthropic
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get weather"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object ["location" .= object ["type" .= ("string" :: Text)]]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [MaxTokens 16000, Tools [toolDef], Reasoning True]
          msgs = [UserText "What's the weather in Paris?"]
          req = Provider.withMagicSystemPrompt $
                 buildStreamingRequestWithReasoning configs msgs

      Proto.stream req `shouldBe` Just True
      Proto.thinking req `shouldNotBe` Nothing

      sseBody <- getResponse req
      BSL.null sseBody `shouldBe` False

      -- Response should have thinking_delta and tool_use content blocks
      sseBody `shouldSatisfy` (`hasDataValue` "thinking_delta")
      sseBody `shouldSatisfy` (`hasDataValue` "tool_use")
