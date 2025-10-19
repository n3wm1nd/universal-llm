{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-x-partial -Wno-unused-imports -Wno-name-shadowing #-}

module OpenAIProtocolSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import Data.Aeson (Value, object, (.=))
import UniversalLLM
import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.Providers.OpenAI
import UniversalLLM.Models.GPT4o

spec :: Spec
spec = do
  describe "OpenAI Protocol - Request Translation" $ do

    it "translates basic user message to OpenAI format" $ do
      let msg = UserText "Hello, world!" :: Message GPT4o OpenAI
          oaiMsg = convertMessage msg
      role oaiMsg `shouldBe` "user"
      content oaiMsg `shouldBe` Just "Hello, world!"
      tool_calls oaiMsg `shouldBe` Nothing

    it "translates assistant message to OpenAI format" $ do
      let msg = AssistantText "Hi there!" :: Message GPT4o OpenAI
          oaiMsg = convertMessage msg
      role oaiMsg `shouldBe` "assistant"
      content oaiMsg `shouldBe` Just "Hi there!"

    it "translates system message to OpenAI format" $ do
      let msg = SystemText "You are helpful." :: Message GPT4o OpenAI
          oaiMsg = convertMessage msg
      role oaiMsg `shouldBe` "system"
      content oaiMsg `shouldBe` Just "You are helpful."

    it "builds request with model name and messages" $ do
      let msgs = [UserText "Test message" :: Message GPT4o OpenAI]
          req = toRequest OpenAI GPT4o [] msgs
      model req `shouldBe` "gpt-4o"
      length (messages req) `shouldBe` 1
      role (head $ messages req) `shouldBe` "user"

    it "applies temperature config to request" $ do
      let msgs = [UserText "Test" :: Message GPT4o OpenAI]
          configs = [Temperature 0.7]
          req = toRequest OpenAI GPT4o configs msgs
      temperature req `shouldBe` Just 0.7

    it "applies max_tokens config to request" $ do
      let msgs = [UserText "Test" :: Message GPT4o OpenAI]
          configs = [MaxTokens 100]
          req = toRequest OpenAI GPT4o configs msgs
      max_tokens req `shouldBe` Just 100

    it "applies seed config to request" $ do
      let msgs = [UserText "Test" :: Message GPT4o OpenAI]
          configs = [Seed 42]
          req = toRequest OpenAI GPT4o configs msgs
      seed req `shouldBe` Just 42

    it "applies system prompt config to request" $ do
      let msgs = [UserText "Hello" :: Message GPT4o OpenAI]
          configs = [SystemPrompt "You are helpful."]
          req = toRequest OpenAI GPT4o configs msgs
      length (messages req) `shouldBe` 2
      role (head $ messages req) `shouldBe` "system"
      content (head $ messages req) `shouldBe` Just "You are helpful."

    it "applies multiple configs to request" $ do
      let msgs = [UserText "Test" :: Message GPT4o OpenAI]
          configs = [Temperature 0.5, MaxTokens 200, Seed 123]
          req = toRequest OpenAI GPT4o configs msgs
      temperature req `shouldBe` Just 0.5
      max_tokens req `shouldBe` Just 200
      seed req `shouldBe` Just 123

    it "translates tool definitions to OpenAI format" $ do
      let toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get weather for a location"
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
          oaiTool = toOpenAIToolDef toolDef

      tool_type oaiTool `shouldBe` "function"
      name (function oaiTool) `shouldBe` "get_weather"
      description (function oaiTool) `shouldBe` "Get weather for a location"

    it "includes tools in request when configured" $ do
      let msgs = [UserText "What's the weather?" :: Message GPT4o OpenAI]
          toolDef = ToolDefinition "test_tool" "Test" (object [])
          configs = [Tools [toolDef]]
          req = toRequest OpenAI GPT4o configs msgs

      case tools req of
        Nothing -> expectationFailure "Expected tools in request"
        Just ts -> do
          length ts `shouldBe` 1
          tool_type (head ts) `shouldBe` "function"

  describe "OpenAI Protocol - Response Translation" $ do

    it "parses successful text response" $ do
      let response = OpenAISuccess $ OpenAISuccessResponse
            [ OpenAIChoice $ OpenAIMessage "assistant" (Just "Hello!") Nothing Nothing ]
          result = fromResponse @OpenAI @GPT4o response

      case result of
        Left err -> expectationFailure $ "Expected success but got error: " ++ show err
        Right msgs -> do
          length msgs `shouldBe` 1
          case head msgs of
            AssistantText txt -> txt `shouldBe` "Hello!"
            _ -> expectationFailure "Expected AssistantText"

    it "parses error response" $ do
      let response = OpenAIError $ OpenAIErrorResponse $
            OpenAIErrorDetail 400 "Bad request" "invalid_request_error"
          result = fromResponse @OpenAI @GPT4o response

      case result of
        Right _ -> expectationFailure "Expected error but got success"
        Left (ProviderError code msg) -> do
          code `shouldBe` 400
          T.isInfixOf "Bad request" msg `shouldBe` True
        Left _ -> expectationFailure "Expected ProviderError"

    it "handles empty choices as error" $ do
      let response = OpenAISuccess $ OpenAISuccessResponse []
          result = fromResponse @OpenAI @GPT4o response

      case result of
        Right _ -> expectationFailure "Expected error for empty choices"
        Left (ParseError msg) -> T.isInfixOf "No choices" msg `shouldBe` True
        Left _ -> expectationFailure "Expected ParseError"

    it "handles response with no content and no tool calls as error" $ do
      let response = OpenAISuccess $ OpenAISuccessResponse
            [ OpenAIChoice $ OpenAIMessage "assistant" Nothing Nothing Nothing ]
          result = fromResponse @OpenAI @GPT4o response

      case result of
        Right _ -> expectationFailure "Expected error"
        Left (ParseError msg) -> T.isInfixOf "No content or tool calls" msg `shouldBe` True
        Left _ -> expectationFailure "Expected ParseError"

  describe "OpenAI Protocol - Tool Call Handling" $ do

    it "converts OpenAI tool call to internal ToolCall" $ do
      let oaiToolCall = OpenAIToolCall
            { callId = "call_123"
            , toolCallType = "function"
            , toolFunction = OpenAIToolFunction
                { toolFunctionName = "get_weather"
                , toolFunctionArguments = "{\"location\":\"Paris\"}"
                }
            }
          toolCall = convertToolCall oaiToolCall

      getToolCallId toolCall `shouldBe` "call_123"
      getToolCallName toolCall `shouldBe` "get_weather"
      -- Verify parameters parsed correctly
      case toolCall of
        ToolCall _ _ (Aeson.Object obj) ->
          case KM.lookup "location" obj of
            Just (Aeson.String "Paris") -> return ()
            _ -> expectationFailure "Expected location to be 'Paris'"
        _ -> expectationFailure "Expected parameters to be an object"

    it "handles malformed tool call arguments as InvalidToolCall" $ do
      let oaiToolCall = OpenAIToolCall
            { callId = "call_456"
            , toolCallType = "function"
            , toolFunction = OpenAIToolFunction
                { toolFunctionName = "test"
                , toolFunctionArguments = "not valid json"
                }
            }
          toolCall = convertToolCall oaiToolCall

      -- Should create InvalidToolCall for malformed JSON
      case toolCall of
        InvalidToolCall tcId tcName tcArgs err -> do
          tcId `shouldBe` "call_456"
          tcName `shouldBe` "test"
          tcArgs `shouldBe` "not valid json"  -- Original string preserved
          T.isInfixOf "Malformed JSON" err `shouldBe` True
        ToolCall{} -> expectationFailure "Expected InvalidToolCall for malformed JSON"

    it "converts internal ToolCall to OpenAI format" $ do
      let params = Aeson.Object $ KM.fromList [("city", Aeson.String "London")]
          toolCall = ToolCall "call_789" "get_weather" params
          oaiToolCall = convertFromToolCall toolCall

      callId oaiToolCall `shouldBe` "call_789"
      toolCallType oaiToolCall `shouldBe` "function"
      toolFunctionName (toolFunction oaiToolCall) `shouldBe` "get_weather"
      T.isInfixOf "London" (toolFunctionArguments $ toolFunction oaiToolCall) `shouldBe` True

    it "parses tool call response correctly" $ do
      -- Canned response based on real llama-cpp server data
      let oaiToolCall = OpenAIToolCall
            { callId = "JsvEr09tquvDy6veg6KkfepTLmKspQ1u"
            , toolCallType = "function"
            , toolFunction = OpenAIToolFunction
                { toolFunctionName = "get_weather"
                , toolFunctionArguments = "{\"location\":\"Paris\"}"
                }
            }
          response = OpenAISuccess $ OpenAISuccessResponse
            [ OpenAIChoice $ OpenAIMessage "assistant" Nothing (Just [oaiToolCall]) Nothing ]
          result = fromResponse @OpenAI @GPT4o response

      case result of
        Left err -> expectationFailure $ "Expected success but got: " ++ show err
        Right msgs -> do
          length msgs `shouldBe` 1
          case head msgs of
            AssistantTool calls -> do
              length calls `shouldBe` 1
              let call = head calls
              getToolCallId call `shouldBe` "JsvEr09tquvDy6veg6KkfepTLmKspQ1u"
              getToolCallName call `shouldBe` "get_weather"
            _ -> expectationFailure "Expected AssistantTool message"

    it "translates tool result message to OpenAI format" $ do
      let params = Aeson.Object $ KM.fromList [("location", Aeson.String "Paris")]
          toolCall = ToolCall "call_123" "get_weather" params
          toolResult = ToolResult toolCall (Right $ object ["temp" .= (22 :: Int)])
          oaiMsg = convertMessage @GPT4o (ToolResultMsg toolResult)

      role oaiMsg `shouldBe` "tool"
      tool_call_id oaiMsg `shouldBe` Just "call_123"
      case content oaiMsg of
        Nothing -> expectationFailure "Expected content in tool result message"
        Just c -> T.isInfixOf "temp" c `shouldBe` True

    it "handles multiple tool calls in single response" $ do
      let call1 = OpenAIToolCall "call_1" "function"
            (OpenAIToolFunction "tool1" "{\"arg\":\"val1\"}")
          call2 = OpenAIToolCall "call_2" "function"
            (OpenAIToolFunction "tool2" "{\"arg\":\"val2\"}")
          response = OpenAISuccess $ OpenAISuccessResponse
            [ OpenAIChoice $ OpenAIMessage "assistant" Nothing (Just [call1, call2]) Nothing ]
          result = fromResponse @OpenAI @GPT4o response

      case result of
        Left err -> expectationFailure $ "Expected success: " ++ show err
        Right msgs -> do
          length msgs `shouldBe` 1
          case head msgs of
            AssistantTool calls -> length calls `shouldBe` 2
            _ -> expectationFailure "Expected AssistantTool"

  describe "OpenAI Protocol - InvalidToolCall Execution" $ do

    it "executeToolCall returns error for InvalidToolCall" $ do
      let invalidCall = InvalidToolCall "call_bad" "broken_tool" "{bad json}" "Bad JSON"
          result = executeToolCall ([] :: [LLMTool IO]) invalidCall

      executed <- result
      case toolResultOutput executed of
        Left err -> err `shouldBe` "Bad JSON"
        Right _ -> expectationFailure "Expected error result for InvalidToolCall"
      toolResultCall executed `shouldBe` invalidCall

    it "converts InvalidToolCall back to OpenAI format with original args" $ do
      let invalidCall = InvalidToolCall "call_err" "bad_tool" "malformed{json" "Parse error"
          oaiCall = convertFromToolCall invalidCall

      callId oaiCall `shouldBe` "call_err"
      toolFunctionName (toolFunction oaiCall) `shouldBe` "bad_tool"
      -- Original invalid string is preserved
      toolFunctionArguments (toolFunction oaiCall) `shouldBe` "malformed{json"
