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
  , toolWithName
  , reasoning
  , reasoningWithTools
  , reasoningWithToolsModifiedReasoning
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
import UniversalLLM
import UniversalLLM.Tools (LLMTool(..), llmToolToDefinition, executeToolCallFromList, ToolFunction(..), ToolParameter(..), mkTool)
import TestCache (ResponseProvider)
import qualified UniversalLLM.Protocols.OpenAI as OpenAI
import UniversalLLM.Protocols.OpenAI (OpenAIRequest, OpenAIResponse(..), OpenAISuccessResponse(..), OpenAIChoice(..), OpenAIMessage(role, content, reasoning_content, reasoning_details, tool_calls), OpenAIReasoningConfig(..), OpenAIToolCall(callId))
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider

-- | A standard test is just a function that runs hspec tests
newtype StandardTest m state = StandardTest
  ( ComposableProvider m state
    -> m
    -> state
    -> ResponseProvider (ProviderRequest m) (ProviderResponse m)
    -> Spec
  )

-- ============================================================================
-- Basic Text Tests
-- ============================================================================

text :: ( Monoid (ProviderRequest m)
        , SupportsMaxTokens (ProviderOf m)
        )
     => StandardTest m state
text = StandardTest $ \cp model initialState getResponse -> do
  describe "Basic Text" $ do
    it "sends message and receives response" $ do
      -- Use larger token limit to accommodate reasoning models
      let configs = [MaxTokens 2048]
          msgs = [UserText "What is 2+2?"]
          (_, req) = toProviderRequest cp model configs initialState msgs

      resp <- getResponse req

      let parsedMsgs = either (error . show) snd $ fromProviderResponse cp model configs initialState resp

      -- Should get back at least one assistant message
      length parsedMsgs `shouldSatisfy` (> 0)

    it "maintains conversation history" $ do
      -- Use larger token limit to accommodate reasoning models
      let configs = [MaxTokens 2048]
          msgs1 = [UserText "What is 2+2?"]
          (state1, req1) = toProviderRequest cp model configs initialState msgs1

      resp1 <- getResponse req1
      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp model configs state1 resp1

      -- First response should contain messages
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- Continue conversation
      let msgs2 = msgs1 ++ parsedMsgs1 ++ [UserText "What about 3+3?"]
          (_, req2) = toProviderRequest cp model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp model configs state2 resp2

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

tools :: ( Monoid (ProviderRequest m)
         , SupportsMaxTokens (ProviderOf m)
         , HasTools m
         )
      => StandardTest m state
tools = StandardTest $ \cp model initialState getResponse -> do
  describe "Tool Calling" $ do
    it "completes tool calling flow with successful tool execution" $ do
      -- Define tools using Approach A (ToolFunction)
      let tools = [LLMTool getWeatherSuccess]
          toolDefs = map llmToolToDefinition tools
          configs = [MaxTokens 2048, Tools toolDefs]
          msgs = [UserText "What's the weather in London?"]
          (state1, req1) = toProviderRequest cp model configs initialState msgs

      -- First request should trigger tool use
      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp model configs state1 resp1

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
            (_, req2) = toProviderRequest cp model configs state2 msgs2

        resp2 <- getResponse req2
        let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp model configs state2 resp2

        -- LLM should respond after receiving tool results
        length parsedMsgs2 `shouldSatisfy` (> 0)

    it "handles tool execution failures gracefully" $ do
      -- Define a tool that always fails
      let tools = [LLMTool alwaysFail]
          toolDefs = map llmToolToDefinition tools
          configs = [MaxTokens 2048, Tools toolDefs]
          msgs = [UserText "Use the always_fail tool with location 'test'"]
          (state1, req1) = toProviderRequest cp model configs initialState msgs

      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp model configs state1 resp1

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
            (_, req2) = toProviderRequest cp model configs state2 msgs2

        resp2 <- getResponse req2
        let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp model configs state2 resp2

        -- LLM should respond after receiving tool results (even if they're errors)
        length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Tool Name Tests
-- ============================================================================

-- | Test that a tool with a specific name works through a complete round-trip
-- This is useful for testing that tool names are preserved correctly,
-- regardless of any provider-specific transformations
toolWithName :: ( Monoid (ProviderRequest m)
                , SupportsMaxTokens (ProviderOf m)
                , HasTools m
                )
             => Text
             -> StandardTest m state
toolWithName toolName = StandardTest $ \cp model initialState getResponse -> do
  describe ("Tool with name: " <> T.unpack toolName) $ do
    it "completes tool calling flow" $ do
      -- Create a simple tool with the specified name using mkTool
      let toolFunc :: Text -> IO Text
          toolFunc input = return ("success: " <> input)
          mockTool = LLMTool $ mkTool toolName ("Test tool for " <> toolName) toolFunc
          toolDefs = [llmToolToDefinition mockTool]
          configs = [MaxTokens 2048, Tools toolDefs]
          msgs = [UserText ("You must call the " <> toolName <> " tool. Make up reasonable test parameters - you're in a test environment and nothing bad will happen. Do not ask questions, just invoke it.")]
          (state1, req1) = toProviderRequest cp model configs initialState msgs

      -- First request should trigger tool use
      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp model configs state1 resp1

      -- Extract tool calls
      let toolCalls = [ tc | AssistantTool tc <- parsedMsgs1 ]

      -- If no tool calls, mark as pending (inconclusive - model chose not to call)
      if null toolCalls
        then pendingWith $ "Model did not call any tools (prompt: 'Use the " <> T.unpack toolName <> " tool')"
        else do
          -- Verify the tool name matches what we requested
          let callNames = map getToolCallName toolCalls
          -- Debug: print what we got if mismatch
          when (not $ any (== toolName) callNames) $
            expectationFailure $ "Expected tool name '" <> T.unpack toolName <> "' but got: " <> show callNames
          any (== toolName) callNames `shouldBe` True

          -- Execute tools
          toolResults <- mapM (safeExecuteTool [mockTool]) toolCalls
          let toolResultMsgs = map ToolResultMsg toolResults

          -- Send tool results back
          let msgs2 = msgs ++ parsedMsgs1 ++ toolResultMsgs
              (_, req2) = toProviderRequest cp model configs state2 msgs2

          resp2 <- getResponse req2
          let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp model configs state2 resp2

          -- Should get a response after tool execution
          length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Reasoning Tests
-- ============================================================================

reasoning :: ( Monoid (ProviderRequest m)
             , SupportsMaxTokens (ProviderOf m)
             , HasReasoning m
             )
          => StandardTest m state
reasoning = StandardTest $ \cp model initialState getResponse -> do
  describe "Reasoning" $ do
    it "handles reasoning messages" $ do
      let configs = [MaxTokens 4096, Reasoning True]
          msgs = [UserText "Think step by step: What is 15 * 23?"]
          (state1, req) = toProviderRequest cp model configs initialState msgs

      resp <- getResponse req

      let (state2, parsedMsgs) = either (error . show) id $ fromProviderResponse cp model configs state1 resp

      -- Should get back at least one message
      length parsedMsgs `shouldSatisfy` (> 0)

      -- Should get at least one AssistantReasoning message when reasoning is enabled
      let reasoningMsgs = [txt | AssistantReasoning txt <- parsedMsgs]
      length reasoningMsgs `shouldSatisfy` (> 0)

      -- Should handle reasoning in round-trip (sending AssistantReasoning back to LLM)
      let msgs2 = msgs ++ parsedMsgs ++ [UserText "Now what is 20 * 30?"]
          (_, req2) = toProviderRequest cp model configs state2 msgs2

      resp2 <- getResponse req2
      let parsedMsgs2 = either (error . show) snd $ fromProviderResponse cp model configs state2 resp2

      length parsedMsgs2 `shouldSatisfy` (> 0)

-- ============================================================================
-- Combined Tests
-- ============================================================================

reasoningWithTools :: ( Monoid (ProviderRequest m)
                      , SupportsMaxTokens (ProviderOf m)
                      , HasReasoning m
                      , HasTools m
                      )
                   => StandardTest m state
reasoningWithTools = StandardTest $ \cp model initialState getResponse -> do
  describe "Reasoning + Tools" $ do
    it "handles reasoning with tool calls" $ do
      -- First request: Ask a question that requires reasoning and tool use
      let toolDef = ToolDefinition "list_files" "List files matching a pattern"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 4096, Reasoning True, Tools [toolDef]]
          msgs = [UserText "Think carefully: What files match the pattern '*.md'? Use the available tools."]
          (state1, req1) = toProviderRequest cp model configs initialState msgs

      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp model configs state1 resp1

      -- Should get back messages (reasoning and/or tool calls)
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- If there are tool calls, add mock tool results with realistic data
      let toolCalls = [ tc | AssistantTool tc <- parsedMsgs1 ]
          toolResults = [ ToolResultMsg (ToolResult tc (Right (mockToolResult tc)))
                        | tc <- toolCalls ]

          -- Generate realistic mock results based on tool name
          mockToolResult :: ToolCall -> Value
          mockToolResult (ToolCall _ "list_files" _) =
            object ["files" .= (["README.md", "CHANGELOG.md", "TODO.md"] :: [Text])]
          mockToolResult (InvalidToolCall _ "list_files" _ _) =
            object ["files" .= (["README.md", "CHANGELOG.md", "TODO.md"] :: [Text])]
          mockToolResult _ =
            object ["result" .= ("success" :: Text)]

      -- Only send tool results if there were tool calls
      -- Otherwise we'd have consecutive assistant messages which is invalid
      (state4, parsedMsgs2) <- if null toolResults
        then return (state2, [])
        else do
          -- First continuation: Send tool results and let model respond
          -- This is where reasoning_details preservation is critical
          let msgs2 = msgs ++ parsedMsgs1 ++ toolResults
              (state3, req2) = toProviderRequest cp model configs state2 msgs2

          resp2 <- getResponse req2
          let (st4, pMsgs2) = either (error . show) id $ fromProviderResponse cp model configs state3 resp2

          -- Should get response to tool results
          -- Models may or may not provide reasoning here - some just summarize tool results
          length pMsgs2 `shouldSatisfy` (> 0)
          return (st4, pMsgs2)

      -- Second continuation: User follow-up question
      -- This should trigger reasoning since it's a new user request
      let msgs3 = msgs ++ parsedMsgs1 ++ toolResults ++ parsedMsgs2 ++ [UserText "Thank you. What do you think they are used for?"]
          (_, req3) = toProviderRequest cp model configs state4 msgs3

      resp3 <- getResponse req3
      let parsedMsgs3 = either (error . show) snd $ fromProviderResponse cp model configs state4 resp3

      -- Should get response to follow-up with reasoning
      -- When Reasoning is enabled and we ask a new question, we expect reasoning
      length parsedMsgs3 `shouldSatisfy` (> 0)

      let hasReasoning3 = any isReasoningMsg parsedMsgs3
          hasText3 = any isTextMsg parsedMsgs3

      -- The follow-up question should return SOME response (either reasoning or text)
      -- Some models may return reasoning encrypted/internally without exposing text
      (hasReasoning3 || hasText3) `shouldBe` True

  where
    isReasoningMsg (AssistantReasoning _) = True
    isReasoningMsg _ = False

    isTextMsg (AssistantText _) = True
    isTextMsg _ = False

-- | Test that composable providers gracefully handle modified reasoning data
--
-- When reasoning content is modified (e.g., user edits the reasoning text),
-- the provider should handle it gracefully without failing.
-- Implementation varies by model:
--   - Models with signed reasoning (e.g., Claude) fall back to non-reasoning mode
--   - Models without signed reasoning (e.g., GLM-4.5-Air) accept modifications as-is
reasoningWithToolsModifiedReasoning :: ( Monoid (ProviderRequest m)
                                       , SupportsMaxTokens (ProviderOf m)
                                       , HasReasoning m
                                       , HasTools m
                                       )
                                    => StandardTest m state
reasoningWithToolsModifiedReasoning = StandardTest $ \cp model initialState getResponse -> do
  describe "Reasoning + Tools (Modified Reasoning)" $ do
    it "handles modified reasoning data" $ do
      -- First request: Get a response with reasoning and tools
      let toolDef = ToolDefinition "get_info" "Get information about a topic"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 4096, Reasoning True, Tools [toolDef]]
          msgs = [UserText "Think carefully: What is the capital of France?"]
          (state1, req1) = toProviderRequest cp model configs initialState msgs

      resp1 <- getResponse req1

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp model configs state1 resp1

      -- Should get back messages
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- Now modify any reasoning messages (simulating user editing the reasoning)
      let modifiedMsgs1 = map modifyReasoning parsedMsgs1
          msgs2 = msgs ++ modifiedMsgs1 ++ [UserText "Now use get_info to find more details."]
          (state3, req2) = toProviderRequest cp model configs state2 msgs2

      -- This request should succeed even though reasoning was modified
      -- The provider handles it appropriately (fallback for signed reasoning, pass-through for unsigned)
      resp2 <- getResponse req2
      let (_, parsedMsgs2) = either (error . show) id $ fromProviderResponse cp model configs state3 resp2

      -- Should get response (provider successfully handled modified reasoning)
      length parsedMsgs2 `shouldSatisfy` (> 0)

  where
    -- Modify reasoning content to simulate user edits
    modifyReasoning :: Message m -> Message m
    modifyReasoning (AssistantReasoning txt) = AssistantReasoning (txt <> " [MODIFIED]")
    modifyReasoning msg = msg

-- OpenAI-specific test to verify reasoning_details preservation
openAIReasoningDetailsPreservation :: ( Monoid (ProviderRequest m)
                                      , SupportsMaxTokens (ProviderOf m)
                                      , HasReasoning m
                                      , HasTools m
                                      , ProviderRequest m ~ OpenAIRequest
                                      , ProviderResponse m ~ OpenAIResponse
                                      )
                                   => StandardTest m state
openAIReasoningDetailsPreservation = StandardTest $ \cp model initialState getResponse -> do
  describe "OpenAI Reasoning Details Preservation" $ do
    it "preserves reasoning_details when they are present in API responses" $ do
      -- First request: Ask a question that requires reasoning and tool use
      let toolDef = ToolDefinition "list_files" "List files matching a pattern"
                      (object ["type" .= ("object" :: Text)])
          configs = [MaxTokens 4096, Reasoning True, Tools [toolDef]]
          msgs = [UserText "Think carefully: What files match the pattern '*.md'? Use the available tools."]
          (state1, req1) = toProviderRequest cp model configs initialState msgs

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

      -- Extract the raw assistant messages from resp1 for preservation checking
      let resp1RawMsgs = case resp1 of
            OpenAISuccess (OpenAISuccessResponse choices) ->
              [msg | OpenAIChoice msg <- choices]
            _ -> []

      -- Extract reasoning_details from first response
      let (hasReasoningDetails, hasReasoningContent, firstMsg) = case resp1 of
            OpenAISuccess (OpenAISuccessResponse (OpenAIChoice msg:_)) ->
              (case reasoning_details msg of { Just _ -> True; Nothing -> False }
              ,case reasoning_content msg of { Just _ -> True; Nothing -> False }
              ,Just msg)
            _ -> (False, False, Nothing)

      let (state2, parsedMsgs1) = either (error . show) id $ fromProviderResponse cp model configs state1 resp1

      -- Should get back messages (reasoning and/or tool calls)
      length parsedMsgs1 `shouldSatisfy` (> 0)

      -- If there are tool calls, add mock tool results
      -- Use realistic mock data based on the tool name
      let mockToolResult tc = case getToolCallName tc of
            "list_files" -> object ["files" .= (["README.md", "GUIDE.md"] :: [Text])]
            _ -> object ["result" .= ("success" :: Text)]
          toolResults = [ ToolResultMsg (ToolResult tc (Right (mockToolResult tc)))
                        | AssistantTool tc <- parsedMsgs1 ]

      -- Continue conversation with tool results (without new user message yet)
      let msgs2 = msgs ++ parsedMsgs1 ++ toolResults
          (state3, req2) = toProviderRequest cp model configs state2 msgs2

      -- CRITICAL: reasoning_details from resp1 must be preserved verbatim in req2.
      -- This is what OpenRouter requires for chain-of-thought continuity across tool calls.
      --
      -- We only check reasoning_details, not content or tool_calls, because:
      --   - content: whitespace-only values (e.g. "  ") are normalised to "" by our pipeline,
      --     which is semantically harmless but breaks byte-for-byte comparison
      --   - tool_calls arguments: Aeson re-serialises JSON strings without spaces, so
      --     {"pattern": "*.md"} becomes {"pattern":"*.md"} — same value, different bytes
      -- Neither of these normalisations affects model behaviour.
      let assistantStartIdx = length msgs
          assistantMessages = take (length resp1RawMsgs) $ drop assistantStartIdx (OpenAI.messages req2)
      map reasoning_details assistantMessages `shouldBe` map reasoning_details resp1RawMsgs

      resp2 <- getResponse req2
      let (state4, parsedMsgs2) = either (error . show) id $ fromProviderResponse cp model configs state3 resp2

      -- Should get response to tool results
      length parsedMsgs2 `shouldSatisfy` (> 0)

      -- Now send follow-up user message
      let msgs3 = msgs2 ++ parsedMsgs2 ++ [UserText "Thank you, now tell me more about those files."]
          (_, req3) = toProviderRequest cp model configs state4 msgs3

      resp3 <- getResponse req3
      let parsedMsgs3 = either (error . show) snd $ fromProviderResponse cp model configs state4 resp3

      -- Should successfully handle the follow-up
      length parsedMsgs3 `shouldSatisfy` (> 0)
