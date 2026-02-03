{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Protocol.AnthropicOAuthBlacklist

Systematic identification of OAuth-blocked tool names.

= Purpose

Identify which tool names trigger the OAuth credential rejection.
Based on cost-based blocking theory: expensive built-in tools are reserved.

= Test Strategy

For each suspected tool name:
1. Test with reserved name → expect OAuth error
2. Test with prefixed name (mcp_*) → expect success
3. If both match expectations, tool name is confirmed blacklisted

= Known Claude Code Tools

Based on current Claude Code implementation, these tools exist:
- File operations: read, write, edit, glob, grep, notebookedit
- Execution: bash, bashoutput, killshell, executecode
- Meta: task, skill, slashcommand, exitplanmode
- UI: todowrite, askuserquestion
- Web: webfetch, websearch
- Diagnostics: getdiagnostics
- MCP: listmcpresource, readmcpresource

-}

module Protocol.AnthropicOAuthBlacklist where

import UniversalLLM.Protocols.Anthropic
import UniversalLLM.Providers.Anthropic (oauthBlacklistedToolNames)
import Protocol.Anthropic (simpleUserRequest)
import qualified Data.Aeson as Aeson
import Data.Text (Text)
import qualified Data.Text as T
import Test.Hspec (Spec, describe, it, HasCallStack, shouldBe, expectationFailure)

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- | Create a minimal tool definition
makeTool :: Text -> AnthropicToolDefinition
makeTool name = AnthropicToolDefinition
  { anthropicToolName = name
  , anthropicToolDescription = "Test tool"
  , anthropicToolInputSchema = Aeson.object [("type", "object")]
  , anthropicToolCacheControl = Nothing
  }

-- | Check if error is the OAuth credential error
isOAuthError :: AnthropicResponse -> Bool
isOAuthError (AnthropicError err) =
  "only authorized for use with Claude Code" `T.isInfixOf` errorMessage err
isOAuthError _ = False

-- | Check if response succeeded
isSuccess :: AnthropicResponse -> Bool
isSuccess (AnthropicSuccess _) = True
isSuccess _ = False

-- ============================================================================
-- Blacklist Detection Probes
-- ============================================================================

-- | Test if a specific tool name is blacklisted
--
-- A tool name is blacklisted if:
-- 1. Using the reserved name triggers OAuth error
-- 2. Using a prefixed version (mcp_*) succeeds
testToolNameBlacklisted :: HasCallStack
                        => (AnthropicRequest -> IO AnthropicResponse)
                        -> Text
                        -> Spec
testToolNameBlacklisted makeRequest toolName = do
  it (T.unpack toolName ++ " is blacklisted") $ do
    -- Test 1: Reserved name should fail with OAuth error
    let reservedTool = makeTool toolName
    let req1 = (simpleUserRequest "test") { tools = Just [reservedTool] }
    resp1 <- makeRequest req1

    -- Test 2: Prefixed name should succeed
    let prefixedName = "mcp_" <> toolName
    let prefixedTool = makeTool prefixedName
    let req2 = (simpleUserRequest "test") { tools = Just [prefixedTool] }
    resp2 <- makeRequest req2

    -- Verify both conditions
    case (isOAuthError resp1, isSuccess resp2) of
      (True, True) ->
        -- Confirmed blacklisted
        return ()
      (False, True) ->
        expectationFailure $ T.unpack toolName ++ " is NOT blacklisted (reserved name works)"
      (True, False) ->
        expectationFailure $ T.unpack toolName ++ " might be blacklisted, but prefixed version also fails"
      (False, False) ->
        expectationFailure $ T.unpack toolName ++ " test inconclusive (both failed for other reasons)"

-- | Test all known blacklisted tools
--
-- Uses the canonical list from UniversalLLM.Providers.Anthropic
blacklistProbes :: HasCallStack
                => (AnthropicRequest -> IO AnthropicResponse)
                -> Spec
blacklistProbes makeRequest = do
  describe "OAuth Blacklisted Tools" $ do
    -- Test each tool name from the canonical list
    mapM_ (testToolNameBlacklisted makeRequest) oauthBlacklistedToolNames
