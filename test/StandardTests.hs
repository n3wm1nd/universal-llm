{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Standard Test Suites
--
-- This module defines reusable test suites that can be run against any model.
-- Each test suite is a function that takes a composable provider and runs
-- a series of tests against it.
module StandardTests
  ( StandardTest(..)
  , text
  , tools
  , reasoning
  , reasoningWithTools
  , openAIReasoningDetailsPreservation
  ) where

import Test.Hspec
import qualified Data.Text as T
import Data.Text (Text)
import Data.Aeson (object, (.=), Value)
import Autodocodec (toJSONVia, codec, HasCodec(..))
import qualified Autodocodec
import Control.Monad (when)
import Control.Monad.Catch (MonadCatch, SomeException, catch)
import UniversalLLM.Core.Types
import UniversalLLM.Core.Tools (LLMTool(..), llmToolToDefinition, executeToolCallFromList, ToolFunction(..), ToolParameter(..))
import TestCache (ResponseProvider)
import qualified UniversalLLM.Protocols.OpenAI as OpenAI
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse(..), OpenAISuccessResponse(..), OpenAIChoice(..), OpenAIMessage(role, content, reasoning_content, reasoning_details, tool_calls), OpenAIReasoningConfig(..), OpenAIToolCall(callId))
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider

-- | A standard test is just a function that runs hspec tests
newtype StandardTest provider model state = StandardTest
  ( ComposableProvider provider model state
    -> provider
    -> model
    -> state
    -> ResponseProvider (ProviderRequest provider) (ProviderResponse provider)
    -> Spec
  )

-- ============================================================================
-- Basic Text Tests
-- ============================================================================

text :: ( Monoid (ProviderRequest provider)
        , SupportsMaxTokens provider
        )
     => StandardTest provider model state
text = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Basic Text" $ do
    it "sends message and receives response" $ do
      -- Use larger token limit to accommodate reasoning models
      let configs = [MaxTokens 2048]
          msgs = [UserText "What is 2+2?"]
          (_, req) = toProviderRequest cp provider model configs initialState msgs

      resp <- getResponse req

      let parsedMsgs = either (error . show) snd $ fromProviderResponse cp provider model configs initialState resp

      -- Should get back at least one assistant message
      length parsedMsgs `shouldSatisfy` (> 0)

    it "maintains conversation history" $ do
      -- Use larger token limit to accommodate reasoning models
      let configs = [MaxTokens 2048]
          msgs1 = [UserText "What is 2+2?"]
          (state1, req1) = toProviderRequest cp provider model configs initialState msgs1

      resp1 <- getResponse req1
      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp1

      -- First response should contain messages
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- Continue conversation
      let msgs2 = msgs1 ++ parsedMsgs1 ++ [UserText "What about 3+3?"]
          (_, req2) = toProviderRequest cp provider model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

      -- Second response should also contain messages
      length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Tool Tests
-- ============================================================================

-- | Safe wrapper for executing tools that catches exceptions
-- Similar to how Runix.LLM.ToolExecution uses runFail for Sem (Fail effect)
-- This wraps executeToolCallFromList to catch runtime exceptions:
-- - error calls
-- - fail calls (which throw exceptions in IO and similar monads)
-- - pattern match failures
-- - any other runtime exceptions from tools
-- Works for any monad with MonadCatch (IO, ExceptT, etc.)
safeExecuteTool :: MonadCatch m => [LLMTool m] -> ToolCall -> m ToolResult
safeExecuteTool tools tc =
  catch (executeToolCallFromList tools tc) $ \(e :: SomeException) ->
    return $ ToolResult tc (Left $ T.pack $ show e)

-- Test tool types and implementations

-- Parameter types
newtype Location = Location Text
  deriving stock (Show, Eq)
  deriving (HasCodec) via Text

instance ToolParameter Location where
  paramName _ _ = "location"
  paramDescription _ = "City name"

-- Result types
data WeatherResult = WeatherResult
  { weatherTemp :: Int
  , weatherCondition :: Text
  } deriving stock (Show, Eq)

instance HasCodec WeatherResult where
  codec = Autodocodec.object "WeatherResult" $
    WeatherResult
      <$> Autodocodec.requiredField "temperature" "Temperature in Celsius" Autodocodec..= weatherTemp
      <*> Autodocodec.requiredField "condition" "Weather condition" Autodocodec..= weatherCondition

instance ToolParameter WeatherResult where
  paramName _ _ = "weather_result"
  paramDescription _ = "weather information"

instance ToolFunction WeatherResult where
  toolFunctionName _ = "get_weather"
  toolFunctionDescription _ = "Get current weather for a location"

-- Success tool - returns weather data
getWeatherSuccess :: Location -> IO WeatherResult
getWeatherSuccess (Location _location) = do
  return $ WeatherResult 20 "sunny"

-- Failure result type
newtype FailingToolResult = FailingToolResult Text
  deriving stock (Show, Eq)
  deriving (HasCodec) via Text

instance ToolParameter FailingToolResult where
  paramName _ _ = "result"
  paramDescription _ = "tool result (will always fail)"

instance ToolFunction FailingToolResult where
  toolFunctionName _ = "always_fail"
  toolFunctionDescription _ = "A tool that always fails for testing error handling"

-- Failure tool - always fails using monadic fail
alwaysFail :: Location -> IO FailingToolResult
alwaysFail (Location _location) = do
  fail "This tool intentionally fails"

tools :: ( Monoid (ProviderRequest provider)
         , SupportsMaxTokens provider
         , HasTools model provider
         )
      => StandardTest provider model state
tools = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Tool Calling" $ do
    it "completes tool calling flow with successful tool execution" $ do
      -- Define tools using Approach A (ToolFunction)
      let tools = [LLMTool getWeatherSuccess]
          toolDefs = map llmToolToDefinition tools
          configs = [MaxTokens 2048, Tools toolDefs]
          msgs = [UserText "What's the weather in London?"]
          (state1, req1) = toProviderRequest cp provider model configs initialState msgs

      -- First request should trigger tool use
      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp1

      -- Should get back at least one message
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- Extract tool calls if any
      let toolCalls = [ tc | AssistantTool tc <- parsedMsgs1 ]

      -- If we got tool calls, execute them and complete the flow
      when (not $ null toolCalls) $ do
        -- Execute tools - safeExecuteTool catches exceptions and returns them as Left
        toolResults <- mapM (safeExecuteTool tools) toolCalls
        let toolResultMsgs = map ToolResultMsg toolResults

        -- For this test, verify all tools succeeded
        let successCount = length [() | ToolResult _ (Right _) <- toolResults]
            failures = [err | ToolResult _ (Left err) <- toolResults]
        if successCount /= length toolResults
          then expectationFailure $ "Expected all " ++ show (length toolResults) ++ " tools to succeed, but " ++ show successCount ++ " succeeded. Failures: " ++ show failures
          else successCount `shouldBe` length toolResults

        -- Send tool results back to LLM (exactly the same whether success or failure)
        let msgs2 = msgs ++ parsedMsgs1 ++ toolResultMsgs
            (_, req2) = toProviderRequest cp provider model configs state2 msgs2

        resp2 <- getResponse req2
        let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

        -- LLM should respond after receiving tool results
        length parsedMsgs2 `shouldSatisfy` (> 0)

    it "handles tool execution failures gracefully" $ do
      -- Define a tool that always fails
      let tools = [LLMTool alwaysFail]
          toolDefs = map llmToolToDefinition tools
          configs = [MaxTokens 2048, Tools toolDefs]
          msgs = [UserText "Use the always_fail tool with location 'test'"]
          (state1, req1) = toProviderRequest cp provider model configs initialState msgs

      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp1

      -- Extract tool calls if any
      let toolCalls = [ tc | AssistantTool tc <- parsedMsgs1 ]

      -- If we got tool calls, execute them and complete the flow (IDENTICAL to success test)
      when (not $ null toolCalls) $ do
        -- Execute tools - safeExecuteTool catches exceptions and returns them as Left
        toolResults <- mapM (safeExecuteTool tools) toolCalls
        let toolResultMsgs = map ToolResultMsg toolResults

        -- For this test, verify all tools failed
        let failures = [err | ToolResult _ (Left err) <- toolResults]
        length failures `shouldSatisfy` (> 0)
        -- The error message should be present
        let hasErrorMessage = any (T.isInfixOf "fail") failures
        hasErrorMessage `shouldBe` True

        -- Send tool results back to LLM (exactly the same whether success or failure)
        let msgs2 = msgs ++ parsedMsgs1 ++ toolResultMsgs
            (_, req2) = toProviderRequest cp provider model configs state2 msgs2

        resp2 <- getResponse req2
        let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

        -- LLM should respond after receiving tool results (even if they're errors)
        length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Reasoning Tests
-- ============================================================================

reasoning :: ( Monoid (ProviderRequest provider)
             , SupportsMaxTokens provider
             , HasReasoning model provider
             )
          => StandardTest provider model state
reasoning = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Reasoning" $ do
    it "handles reasoning messages" $ do
      let configs = [MaxTokens 4096, Reasoning True]
          msgs = [UserText "Think step by step: What is 15 * 23?"]
          (state1, req) = toProviderRequest cp provider model configs initialState msgs

      resp <- getResponse req

      let (state2, parsedMsgs) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp

      -- Should get back at least one message
      length parsedMsgs `shouldSatisfy` (> 0)

      -- Should get at least one AssistantReasoning message when reasoning is enabled
      let reasoningMsgs = [txt | AssistantReasoning txt <- parsedMsgs]
      length reasoningMsgs `shouldSatisfy` (> 0)

      -- Should handle reasoning in round-trip (sending AssistantReasoning back to LLM)
      let msgs2 = msgs ++ parsedMsgs ++ [UserText "Now what is 20 * 30?"]
          (_, req2) = toProviderRequest cp provider model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

      length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Combined Tests
-- ============================================================================

reasoningWithTools :: ( Monoid (ProviderRequest provider)
                      , SupportsMaxTokens provider
                      , HasReasoning model provider
                      , HasTools model provider
                      )
                   => StandardTest provider model state
reasoningWithTools = StandardTest $ \cp provider model initialState getResponse -> do
  describe "Reasoning + Tools" $ do
    it "handles reasoning with tool calls" $ do
      -- First request: Ask a question that requires reasoning and tool use
      let toolDef = ToolDefinition "list_files" "List files matching a pattern"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 4096, Reasoning True, Tools [toolDef]]
          msgs = [UserText "Think carefully: What files match the pattern '*.md'? Use the available tools."]
          (state1, req1) = toProviderRequest cp provider model configs initialState msgs

      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp1

      -- Should get back messages (reasoning and/or tool calls)
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- If there are tool calls, add mock tool results
      let toolResults = [ ToolResultMsg (ToolResult tc (Right (object ["result" .= ("success" :: Text)])))
                        | AssistantTool tc <- parsedMsgs1 ]

      -- Continue conversation with tool results
      -- This is where reasoning_details preservation is critical
      let msgs2 = msgs ++ parsedMsgs1 ++ toolResults ++ [UserText "Thank you, now tell me more about those files."]
          (_, req2) = toProviderRequest cp provider model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

      -- Should successfully handle the continuation
      length parsedMsgs2 `shouldSatisfy` (> 0)

-- OpenAI-specific test to verify reasoning_details preservation
openAIReasoningDetailsPreservation :: ( Monoid (ProviderRequest provider)
                                      , SupportsMaxTokens provider
                                      , HasReasoning model provider
                                      , HasTools model provider
                                      , ProviderRequest provider ~ OpenAIRequest
                                      , ProviderResponse provider ~ OpenAIResponse
                                      )
                                   => StandardTest provider model state
openAIReasoningDetailsPreservation = StandardTest $ \cp provider model initialState getResponse -> do
  describe "OpenAI Reasoning Details Preservation" $ do
    it "preserves reasoning_details when they are present in API responses" $ do
      -- First request: Ask a question that requires reasoning and tool use
      let toolDef = ToolDefinition "list_files" "List files matching a pattern"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 4096, Reasoning True, Tools [toolDef]]
          msgs = [UserText "Think carefully: What files match the pattern '*.md'? Use the available tools."]
          (state1, req1) = toProviderRequest cp provider model configs initialState msgs

      -- ASSERTION: Request should have reasoning field populated
      case OpenAI.reasoning req1 of
        Nothing -> error "Request does not have reasoning field set, but Reasoning True was in configs"
        Just reasoningCfg -> do
          reasoning_enabled reasoningCfg `shouldBe` Just True
          -- Should have either effort or max_tokens (but not both for OpenRouter)
          let hasEffort = case reasoningCfg.reasoning_effort of { Just _ -> True; Nothing -> False }
              hasMaxTokens = case reasoningCfg.reasoning_max_tokens of { Just _ -> True; Nothing -> False }
          (hasEffort || hasMaxTokens) `shouldBe` True

      resp1 <- getResponse req1

      -- Extract ALL messages from resp1 as JSON Values
      let resp1Messages = case resp1 of
            OpenAISuccess (OpenAISuccessResponse choices) ->
              [toJSONVia codec msg | OpenAIChoice msg <- choices]
            _ -> []

      -- Extract reasoning_details from first response
      let (hasReasoningDetails, hasReasoningContent, firstMsg) = case resp1 of
            OpenAISuccess (OpenAISuccessResponse (OpenAIChoice msg:_)) ->
              (case reasoning_details msg of { Just _ -> True; Nothing -> False }
              ,case reasoning_content msg of { Just _ -> True; Nothing -> False }
              ,Just msg)
            _ -> (False, False, Nothing)

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp provider model configs state1 resp1

      -- Should get back messages (reasoning and/or tool calls)
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- If there are tool calls, add mock tool results
      -- Use realistic mock data based on the tool name
      let mockToolResult tc = case getToolCallName tc of
            "list_files" -> object ["files" .= (["README.md", "GUIDE.md"] :: [Text])]
            _ -> object ["result" .= ("success" :: Text)]
          toolResults = [ ToolResultMsg (ToolResult tc (Right (mockToolResult tc)))
                        | AssistantTool tc <- parsedMsgs1 ]

      -- Continue conversation with tool results
      let msgs2 = msgs ++ parsedMsgs1 ++ toolResults ++ [UserText "Thank you, now tell me more about those files."]
          (_, req2) = toProviderRequest cp provider model configs state2 msgs2

      -- CRITICAL: ALL messages from resp1 must appear verbatim in req2 (as JSON Values)
      -- The conversation structure is:
      --   msgs (initial user messages) ++ parsedMsgs1 (assistant from resp1) ++ toolResults ++ [new user msg]
      -- So the assistant message from resp1 should be at index: length msgs
      let assistantStartIdx = length msgs
          assistantMessages = take (length resp1Messages) $ drop assistantStartIdx (OpenAI.messages req2)
          assistantMessagesAsJson = map (toJSONVia codec) assistantMessages

      -- The messages from resp1 MUST match the corresponding messages in req2 exactly
      assistantMessagesAsJson `shouldBe` resp1Messages

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp provider model configs state2 resp2

      -- Should successfully handle the continuation
      length parsedMsgs2 `shouldSatisfy` (> 0)
