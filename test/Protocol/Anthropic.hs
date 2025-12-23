{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedRecordDot #-}

{- |
Module: Protocol.Anthropic

Low-level protocol helpers for Anthropic wire protocol testing.

= Purpose

This module provides building blocks for protocol-level capability probes.
It is NOT for testing our abstractions (use StandardTests for that).

Provides:
- Helper functions to build requests (hides Aeson/JSON complexity)
- Functions to extract data from responses
- Composable error checking (catch API errors early)
- Assertions for protocol tests

= Design Principles

* Protocol-agnostic naming - functions should work similarly across protocols
* Hide complexity - no Aeson manipulation in tests
* Descriptive error messages - failures should be immediately clear
* Keep functions minimal - add more as needed, don't over-engineer

= Usage Pattern

Building a request:
@
let req = simpleUserRequest "What is 2+2?"
    withTools = req { tools = Just [simpleTool "calculator" "Do math"] }
@

Checking responses:
@
resp <- makeRequest req
let text = getAssistantText . expectSuccess $ resp
assertHasAssistantText resp  -- use in protocol tests
@

-}

module Protocol.Anthropic where

import UniversalLLM.Protocols.Anthropic
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import Test.Hspec (Expectation, HasCallStack, shouldSatisfy)

-- ============================================================================
-- Request Helpers
--
-- Functions to build requests without dealing with protocol details.
-- Named to be protocol-agnostic where possible.
-- ============================================================================

-- | Create a simple request with a user message
--
-- Common pattern: start with this, then add model/other config
simpleUserRequest :: Text -> AnthropicRequest
simpleUserRequest txt = mempty
  { messages = [AnthropicMessage "user" [AnthropicTextBlock txt Nothing]]
  , model = "claude-sonnet-4-5-20250929"  -- Default model
  , max_tokens = 4096  -- Required field (must be > budget_tokens when reasoning enabled)
  }

-- | Create a request with consecutive user messages
--
-- Used to test if the API accepts multiple user messages in a row
consecutiveUserMessages :: Text -> Text -> AnthropicRequest
consecutiveUserMessages msg1 msg2 = mempty
  { messages =
      [ AnthropicMessage "user" [AnthropicTextBlock msg1 Nothing]
      , AnthropicMessage "user" [AnthropicTextBlock msg2 Nothing]
      ]
  , model = "claude-sonnet-4-5-20250929"
  , max_tokens = 4096
  }

-- | Create a request starting with an assistant message
--
-- Used to test if the API accepts history starting with assistant (no initial user message)
startsWithAssistant :: AnthropicRequest
startsWithAssistant = mempty
  { messages =
      [ AnthropicMessage "assistant" [AnthropicTextBlock "I'm a helpful assistant." Nothing]
      , AnthropicMessage "user" [AnthropicTextBlock "What is 2+2?" Nothing]
      ]
  , model = "claude-sonnet-4-5-20250929"
  , max_tokens = 4096
  }

-- | Create a simple tool definition
simpleTool :: Text -> Text -> AnthropicToolDefinition
simpleTool name description = AnthropicToolDefinition
  { anthropicToolName = name
  , anthropicToolDescription = description
  , anthropicToolInputSchema = Aeson.object
      [ "type" Aeson..= ("object" :: Text)
      , "properties" Aeson..= Aeson.object
          [ "location" Aeson..= Aeson.object
              [ "type" Aeson..= ("string" :: Text)
              , "description" Aeson..= ("Location to get weather for" :: Text)
              ]
          ]
      , "required" Aeson..= (["location"] :: [Text])
      ]
  , anthropicToolCacheControl = Nothing
  }

-- | Create a weather tool definition
weatherTool :: AnthropicToolDefinition
weatherTool = simpleTool "get_weather" "Get current weather for a location"

-- | Enable reasoning (extended thinking) on a request
--
-- Sets budget_tokens to a reasonable default (2000 tokens, which is half of the
-- default max_tokens of 4096, bounded between 1024-5000)
enableReasoning :: AnthropicRequest -> AnthropicRequest
enableReasoning req = req
  { thinking = Just AnthropicThinkingConfig
      { thinkingType = "enabled"
      , thinkingBudgetTokens = Just 2000  -- Reasonable default for testing
      }
  }

-- ============================================================================
-- Response Helpers
--
-- Functions to extract data from responses and check for errors.
-- Use 'expectSuccess' when you expect a successful response.
-- ============================================================================

-- | Expect a success response - unwrap and return success response
--
-- Use this when you expect the request to succeed:
-- >>> text <- getAssistantText . expectSuccess $ resp
expectSuccess :: HasCallStack => AnthropicResponse -> AnthropicSuccessResponse
expectSuccess (AnthropicSuccess resp)
  | responseStopReason resp `elem` [Just "end_turn", Just "tool_use", Just "max_tokens"] = resp
  | otherwise = error $ "Unexpected stop reason: " ++ show (responseStopReason resp)
expectSuccess (AnthropicError err) = error $ "Expected success but got error: " ++ T.unpack (errorMessage err)

-- | Extract assistant text from success response
--
-- Throws if no text content blocks found
getAssistantText :: AnthropicSuccessResponse -> Text
getAssistantText resp =
  let textBlocks = [txt | AnthropicTextBlock txt _ <- responseContent resp]
  in if null textBlocks
     then error "Response has no text content blocks"
     else T.intercalate "\n" textBlocks

-- | Extract thinking content from success response
getThinkingContent :: AnthropicSuccessResponse -> Text
getThinkingContent resp =
  let thinkingBlocks = [txt | AnthropicThinkingBlock{thinkingText = txt} <- responseContent resp]
  in if null thinkingBlocks
     then error "Response has no thinking content blocks"
     else T.intercalate "\n" thinkingBlocks

-- | Check if success response has tool use blocks
hasToolUse :: AnthropicSuccessResponse -> Bool
hasToolUse resp = any isToolUse (responseContent resp)
  where
    isToolUse (AnthropicToolUseBlock _ _ _ _) = True
    isToolUse _ = False

-- | Check if success response has thinking blocks
hasThinking :: AnthropicSuccessResponse -> Bool
hasThinking resp = any isThinking (responseContent resp)
  where
    isThinking AnthropicThinkingBlock{} = True
    isThinking _ = False

-- ============================================================================
-- Protocol Assertions
--
-- Assertions for use in protocol capability probes.
-- These check specific protocol behaviors and throw descriptive errors.
-- ============================================================================

-- | Assert that response contains assistant text
assertHasAssistantText :: HasCallStack => AnthropicResponse -> Expectation
assertHasAssistantText resp = do
  let success = expectSuccess resp
      text = getAssistantText success
  T.length text `shouldSatisfy` (> 0)

-- | Assert that response contains tool use blocks
assertHasToolCalls :: HasCallStack => AnthropicResponse -> Expectation
assertHasToolCalls resp = do
  let success = expectSuccess resp
  hasToolUse success `shouldSatisfy` id

-- | Assert that response contains thinking content
assertHasReasoning :: HasCallStack => AnthropicResponse -> Expectation
assertHasReasoning resp = do
  let success = expectSuccess resp
  hasThinking success `shouldSatisfy` id
