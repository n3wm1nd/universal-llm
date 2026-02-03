{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RecordWildCards #-}

module UniversalLLM.Providers.Anthropic where

import UniversalLLM
import UniversalLLM.Serialization
import UniversalLLM.Protocols.Anthropic
import Data.Text (Text)
import qualified Data.Text as Text
import qualified Data.Text.Encoding as TE
import Data.Aeson (Value)
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Default (Default(..))

-- Anthropic provider (phantom type)
data Anthropic = Anthropic deriving (Show, Eq)

-- | Anthropic OAuth provider (with tool name workarounds)
-- Automatically prefixes blacklisted tool names with "runix_" to avoid OAuth restrictions
data AnthropicOAuth = AnthropicOAuth deriving (Show, Eq)

-- | State for managing thinking block signatures
-- Maps thinking text to its signature metadata so we can echo it back verbatim
newtype AnthropicReasoningState = AnthropicReasoningState
  { signatureMap :: Map Text Value
  } deriving (Show, Eq)

instance Default AnthropicReasoningState where
  def = AnthropicReasoningState Map.empty

-- | State for tracking OAuth tool name mappings
-- Maps prefixed names back to their original names
data OAuthToolsState = OAuthToolsState
  { prefixedToOriginal :: Map Text Text  -- "runix_read" -> "read"
  } deriving (Show, Eq)

instance Default OAuthToolsState where
  def = OAuthToolsState Map.empty

-- ============================================================================
-- Helper Functions for AnthropicRequest Manipulation
-- ============================================================================

-- | Modify system blocks in a request
modifySystemBlocks :: (Maybe [AnthropicSystemBlock] -> Maybe [AnthropicSystemBlock])
                   -> AnthropicRequest -> AnthropicRequest
modifySystemBlocks f req = req { system = f (system req) }

-- | Modify messages in a request
modifyMessages :: ([AnthropicMessage] -> [AnthropicMessage])
               -> AnthropicRequest -> AnthropicRequest
modifyMessages f req = req { messages = f (messages req) }

-- | Modify tool definitions in a request
modifyToolDefinitions :: (Maybe [AnthropicToolDefinition] -> Maybe [AnthropicToolDefinition])
                      -> AnthropicRequest -> AnthropicRequest
modifyToolDefinitions f req = req { tools = f (tools req) }

-- | Set system blocks (replaces existing)
setSystemBlocks :: [AnthropicSystemBlock] -> AnthropicRequest -> AnthropicRequest
setSystemBlocks blocks = modifySystemBlocks (const (Just blocks))

-- | Set tool definitions (replaces existing)
setToolDefinitions :: [AnthropicToolDefinition] -> AnthropicRequest -> AnthropicRequest
setToolDefinitions defs = modifyToolDefinitions (const (Just defs))

-- | Add cache control to conversation history
-- Applies cache control to the last content block of an earlier message
-- This caches the conversation prefix while leaving room for new turns
addConversationCacheControl :: AnthropicRequest -> AnthropicRequest
addConversationCacheControl req =
  let msgs = messages req
      -- Apply cache control to last block of second-to-last message (if exists)
      -- This ensures we cache conversation history while allowing the latest turn to vary
      updatedMsgs = case reverse msgs of
        (lastMsg : secondToLastMsg : rest) ->
          let cachedMsg = addCacheToLastBlock secondToLastMsg
          in reverse (lastMsg : cachedMsg : rest)
        _ -> msgs  -- Less than 2 messages, no caching needed
  in req { messages = updatedMsgs }
  where
    addCacheToLastBlock :: AnthropicMessage -> AnthropicMessage
    addCacheToLastBlock msg =
      case reverse (content msg) of
        [] -> msg
        (lastBlock : restBlocks) ->
          let cachedBlock = addCacheControlToBlock lastBlock
          in msg { content = reverse (cachedBlock : restBlocks) }

    addCacheControlToBlock :: AnthropicContentBlock -> AnthropicContentBlock
    addCacheControlToBlock (AnthropicTextBlock txt _) =
      AnthropicTextBlock txt (Just (CacheControl "ephemeral" "5m"))
    addCacheControlToBlock (AnthropicToolUseBlock tid tname tinput _) =
      AnthropicToolUseBlock tid tname tinput (Just (CacheControl "ephemeral" "5m"))
    addCacheControlToBlock (AnthropicToolResultBlock rid rcontent _) =
      AnthropicToolResultBlock rid rcontent (Just (CacheControl "ephemeral" "5m"))
    addCacheControlToBlock block@AnthropicThinkingBlock{} =
      block { thinkingCacheControl = Just (CacheControl "ephemeral" "5m") }

-- | Append a content block to messages, grouping with last message if same role
appendContentBlock :: Text -> AnthropicContentBlock -> AnthropicRequest -> AnthropicRequest
appendContentBlock msgRole block = modifyMessages (appendBlockToMessages msgRole block)
  where
    appendBlockToMessages :: Text -> AnthropicContentBlock -> [AnthropicMessage] -> [AnthropicMessage]
    appendBlockToMessages r b [] = [AnthropicMessage r [b]]
    appendBlockToMessages r b msgs =
      let lastMsg = last msgs
          initMsgs = init msgs
      in if role lastMsg == r
         then initMsgs <> [lastMsg { content = content lastMsg <> [b] }]
         else msgs <> [AnthropicMessage r [b]]

-- ============================================================================
-- Data Conversion Functions
-- ============================================================================

-- | Convert a ToolCall to an AnthropicContentBlock
fromToolCall :: ToolCall -> AnthropicContentBlock
fromToolCall (ToolCall tcId tcName tcParams) = AnthropicToolUseBlock tcId tcName tcParams Nothing
fromToolCall (InvalidToolCall tcId tcName rawArgs err) =
  -- InvalidToolCall cannot be directly converted to Anthropic's tool_use format
  -- Instead, convert it to a text message describing the error
  let errorText = "Tool call error for " <> tcName <> " (id: " <> tcId <> "): " <> err
                  <> "\nRaw arguments: " <> rawArgs
  in AnthropicTextBlock errorText Nothing

-- | Convert a ToolResult to an AnthropicContentBlock
fromToolResult :: ToolResult -> AnthropicContentBlock
fromToolResult (ToolResult toolCall output) =
  let callId = getToolCallId toolCall
      resultContent = case output of
        Left errMsg -> errMsg
        Right jsonVal -> TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ jsonVal
  in AnthropicToolResultBlock callId resultContent Nothing

-- | Convert text to a text content block
fromText :: Text -> AnthropicContentBlock
fromText txt = AnthropicTextBlock txt Nothing

-- ============================================================================
-- Regular Anthropic Provider
-- ============================================================================

-- Declare Anthropic parameter support
instance SupportsTemperature Anthropic
instance SupportsMaxTokens Anthropic
instance SupportsSystemPrompt Anthropic
instance SupportsStreaming Anthropic
-- Note: Anthropic does NOT support Seed or JSON mode

-- Anthropic capabilities are now declared per-model (see model files)

-- Provider typeclass implementation (just type associations)
instance Provider (Model aiModel Anthropic) where
  type ProviderRequest (Model aiModel Anthropic) = AnthropicRequest
  type ProviderResponse (Model aiModel Anthropic) = AnthropicResponse

-- Text message encoder - groups messages incrementally
handleTextMessage :: ProviderRequest m ~ AnthropicRequest => MessageEncoder m
handleTextMessage msg req = case msg of
  UserText txt -> if Text.null txt then req else appendContentBlock "user" (fromText txt) req
  AssistantText txt -> if Text.null txt then req else appendContentBlock "assistant" (fromText txt) req
  SystemText txt -> if Text.null txt then req else appendContentBlock "user" (fromText txt) req
  _ -> req

-- Tool message encoder - groups tool blocks incrementally
handleToolMessage :: (ProviderRequest m ~ AnthropicRequest, HasTools m) => MessageEncoder m
handleToolMessage msg req = case msg of
  AssistantTool toolCall -> appendContentBlock "assistant" (fromToolCall toolCall) req
  ToolResultMsg result -> appendContentBlock "user" (fromToolResult result) req
  _ -> req

-- Composable providers for Anthropic

-- Base provider: handles text messages and basic configuration
baseComposableProvider :: forall m. (ModelName m, ProviderRequest m ~ AnthropicRequest, ProviderResponse m ~ AnthropicResponse) => ComposableProvider m ()
baseComposableProvider model configs _s = noopHandler
  { cpPureMessageRequest = ensureUserFirstPure
  , cpToRequest = \msg req ->
      let req' = req { model = modelName model
                     , max_tokens = case [mt | MaxTokens mt <- configs] of { (mt:_) -> mt; [] -> max_tokens req }
                     , temperature = case [t | Temperature t <- configs] of { (t:_) -> Just t; [] -> temperature req }
                     }
      in handleTextMessage msg req'
  , cpConfigHandler = \req ->
      let systemPrompts = [sp | SystemPrompt sp <- configs]
          -- Create system blocks with cache control on the last block
          sysBlocks = case systemPrompts of
            [] -> []
            prompts ->
              let allButLast = [AnthropicSystemBlock sp "text" Nothing | sp <- init prompts]
                  lastBlock = AnthropicSystemBlock (last prompts) "text" (Just (CacheControl "ephemeral" "5m"))
              in if length prompts == 1
                 then [lastBlock]
                 else allButLast ++ [lastBlock]
          req1 = if null sysBlocks then req else setSystemBlocks sysBlocks req
          -- Add cache control to conversation history
          req2 = addConversationCacheControl req1
          streamEnabled = case [s | Streaming s <- configs] of
            (s:_) -> Just s
            [] -> stream req2
      in req2 { stream = streamEnabled }
  , cpFromResponse = parseTextResponse
  , cpSerializeMessage = serializeBaseMessage
  , cpDeserializeMessage = deserializeBaseMessage
  }
  where
    ensureUserFirstPure msgs =
      case msgs of
        [] -> []
        (m1:_) -> if messageDirection m1 == User then msgs else UserText " " : msgs

    parseTextResponse (AnthropicError err) =
      Left $ ModelError $ errorMessage err
    parseTextResponse (AnthropicSuccess resp) =
      case responseContent resp of
        (AnthropicTextBlock txt _ : rest) ->
          -- First block is text, extract it
          Right (Just (AssistantText txt, AnthropicSuccess resp { responseContent = rest }))
        _ -> Right Nothing  -- First block is not text, let another provider handle it

-- Standalone tools provider
anthropicTools :: forall m s . (HasTools m, ProviderRequest m ~ AnthropicRequest, ProviderResponse m ~ AnthropicResponse) => ComposableProvider m s
anthropicTools _m configs _s = noopHandler
  { cpToRequest = handleToolMessage
  , cpConfigHandler = \req ->
      let toolDefs = [defs | Tools defs <- configs]
          -- Add cache control to the last tool definition
          anthropicToolDefs = withToolCacheControl (map toAnthropicToolDef (concat toolDefs))
      in if null toolDefs then req else setToolDefinitions anthropicToolDefs req
  , cpFromResponse = parseToolResponse
  , cpSerializeMessage = serializeToolMessages
  , cpDeserializeMessage = deserializeToolMessages
  }
  where
    parseToolResponse (AnthropicError err) =
      Left $ ModelError $ errorMessage err
    parseToolResponse (AnthropicSuccess resp) =
      case responseContent resp of
        (AnthropicToolUseBlock tid tname tinput _ : rest) ->
          -- First block is a tool call, extract it
          Right (Just (AssistantTool (ToolCall tid tname tinput), AnthropicSuccess resp { responseContent = rest }))
        _ -> Right Nothing  -- First block is not a tool call, let another provider handle it

-- Standalone reasoning provider
anthropicReasoning :: forall m. (HasReasoning m, ProviderRequest m ~ AnthropicRequest, ProviderResponse m ~ AnthropicResponse) => ComposableProvider m AnthropicReasoningState
anthropicReasoning _m configs state = noopHandler
  { cpToRequest = handleReasoningMessageWithState state
  , cpConfigHandler = \req ->
      let reasoningEnabled = not $ any isReasoningFalse configs
          maxTokensFromConfig = case [t | MaxTokens t <- configs] of
            (t:_) -> t
            [] -> 2048
          thinkingBudget = max 1024 (min 5000 (maxTokensFromConfig `div` 2))
          -- Check if we're in a tool call sequence that requires signature chain
          needsSignatureChain = isInToolCallSequence req
          hasCompleteChain = hasCompleteSignatureChain req state
          -- Only enable reasoning if either:
          -- 1. We're not in a tool call sequence, OR
          -- 2. We're in a tool call sequence AND have complete signature chain
          canEnableReasoning = reasoningEnabled && (not needsSignatureChain || hasCompleteChain)
      in if canEnableReasoning
         then req { thinking = Just $ AnthropicThinkingConfig "enabled" (Just thinkingBudget) }
         else req
  , cpPostResponse = storeSignatureFromResponse
  , cpFromResponse = parseReasoningResponse
  , cpSerializeMessage = UniversalLLM.Serialization.serializeReasoningMessages
  , cpDeserializeMessage = UniversalLLM.Serialization.deserializeReasoningMessages
  }
  where
    isReasoningFalse (Reasoning False) = True
    isReasoningFalse _ = False

    -- Check if we're in an active tool call sequence
    -- This is true when the most recent assistant message contains tool calls
    -- and we're about to send tool results
    isInToolCallSequence :: AnthropicRequest -> Bool
    isInToolCallSequence req =
      case reverse (messages req) of
        -- If last message is user with tool results, check if previous assistant had tool calls
        (userMsg : assistantMsg : _) | role userMsg == "user" && role assistantMsg == "assistant" ->
          -- Check if user message has tool results
          let hasToolResults = any isToolResult (content userMsg)
              -- Check if assistant message has tool calls
              hasToolCalls = any isToolUse (content assistantMsg)
          in hasToolResults && hasToolCalls
        _ -> False
      where
        isToolResult (AnthropicToolResultBlock _ _ _) = True
        isToolResult _ = False
        isToolUse (AnthropicToolUseBlock _ _ _ _) = True
        isToolUse _ = False

    -- Check if the current tool call sequence has complete signature chain
    -- This means all thinking blocks from the assistant message have signatures in state
    hasCompleteSignatureChain :: AnthropicRequest -> AnthropicReasoningState -> Bool
    hasCompleteSignatureChain req st =
      case reverse (messages req) of
        (_ : assistantMsg : _) | role assistantMsg == "assistant" ->
          -- Get all thinking blocks from assistant message
          let thinkingBlocks = [txt | AnthropicThinkingBlock txt _ _ <- content assistantMsg]
              -- Check if all have signatures in state
              allHaveSignatures = all (\txt -> Map.member txt (signatureMap st)) thinkingBlocks
          in null thinkingBlocks || allHaveSignatures  -- Empty is OK, or all must have sigs
        _ -> True  -- No tool call sequence, so no constraint

    -- Store signature in state after receiving response
    storeSignatureFromResponse :: ProviderResponse m -> AnthropicReasoningState -> AnthropicReasoningState
    storeSignatureFromResponse (AnthropicError _) st = st
    storeSignatureFromResponse (AnthropicSuccess resp) st =
      case responseContent resp of
        (AnthropicThinkingBlock{..} : _) ->
          -- Store the signature in state for this thinking text
          AnthropicReasoningState $ Map.insert thinkingText thinkingSignature (signatureMap st)
        _ -> st

    -- Parse response and extract thinking block (no state modification here)
    parseReasoningResponse (AnthropicError err) =
      Left $ ModelError $ errorMessage err
    parseReasoningResponse (AnthropicSuccess resp) =
      case responseContent resp of
        (AnthropicThinkingBlock{..} : rest) ->
          -- Extract thinking text, signature is stored via cpPostResponse
          Right (Just (AssistantReasoning thinkingText, AnthropicSuccess resp { responseContent = rest }))
        _ -> Right Nothing  -- First block is not thinking, let another provider handle it

    -- Lookup signature when creating request
    handleReasoningMessageWithState :: AnthropicReasoningState -> MessageEncoder m
    handleReasoningMessageWithState st msg req = case msg of
      AssistantReasoning thinking ->
        -- Lookup signature from state map
        case Map.lookup thinking (signatureMap st) of
          Just signature ->
            -- Found signature in state - this is an echoed thinking block from a previous API response
            -- Create a proper thinking block with the signature
            appendContentBlock "assistant"
              (AnthropicThinkingBlock
                { thinkingText = thinking
                , thinkingSignature = signature
                , thinkingCacheControl = Nothing
                }) req
          Nothing ->
            -- No signature found - this is a reasoning message that wasn't from the API
            -- (e.g., deserialized from storage or created programmatically)
            -- Since we can't create thinking blocks without signatures, convert to regular text
            appendContentBlock "assistant" (AnthropicTextBlock thinking Nothing) req
      _ -> req

-- These are removed - use the typeclass methods withTools and withReasoning instead
-- They're defined in the HasTools/HasReasoning instances

-- | Add magic system prompt for OAuth authentication
-- Prepends the Claude Code authentication prompt to user's system prompts
withMagicSystemPrompt :: AnthropicRequest -> AnthropicRequest
withMagicSystemPrompt request =
  let magicBlock = AnthropicSystemBlock "You are Claude Code, Anthropic's official CLI for Claude." "text" Nothing
      combinedSystem = case system request of
        Nothing -> [magicBlock]
        Just userBlocks -> magicBlock : userBlocks
  in request { system = Just combinedSystem }

-- | Headers for OAuth authentication (Claude Code subscription)
-- Returns headers as [(Text, Text)] for transport-agnostic usage
-- Note: Content-Type is NOT included here - it's added by the RestAPI layer
-- When using these headers directly (e.g., in tests), you must add Content-Type yourself
oauthHeaders :: Text -> [(Text, Text)]
oauthHeaders token =
  [ ("Authorization", "Bearer " <> token)
  , ("anthropic-version", "2023-06-01")
  , ("anthropic-beta", "oauth-2025-04-20,interleaved-thinking-2025-05-14,claude-code-20250219")
  , ("User-Agent", "hs-universal-llm (prerelease-dev)")
  --, ("User-Agent", "claude-cli/2.1.2 (external,cli)")
  ]

-- | Headers for API key authentication (console.anthropic.com)
-- Returns headers as [(Text, Text)] for transport-agnostic usage
-- Note: Content-Type is NOT included here - it's added by the RestAPI layer
-- When using these headers directly (e.g., in tests), you must add Content-Type yourself
apiKeyHeaders :: Text -> [(Text, Text)]
apiKeyHeaders apiKey =
  [ ("x-api-key", apiKey)
  , ("anthropic-version", "2023-06-01")
  , ("User-Agent", "hs-universal-llm (prerelease-dev)")
  ]

-- | Tool names reserved by Claude Code that trigger OAuth credential rejection
--
-- These tool names are blocked when using OAuth credentials because they represent
-- expensive server-side operations (file access, shell execution) that have costs
-- beyond token usage. Third-party clients must use prefixed names (e.g., "mcp_read")
-- to avoid triggering the OAuth restriction.
--
-- Note: This list is based on empirical testing and may change as Anthropic updates
-- their OAuth policies.
-- | Tool names reserved by Claude Code that trigger OAuth credential rejection
--
-- These tool names are blocked when using OAuth credentials because they represent
-- expensive server-side operations (file access, shell execution) that have costs
-- beyond token usage. Third-party clients must use prefixed names (e.g., "runix_read")
-- to avoid triggering the OAuth restriction.
--
-- This list is empirically verified by tests in Protocol.AnthropicOAuthBlacklist.
-- Only names that actually trigger OAuth rejection are included here.
oauthBlacklistedToolNames :: [Text]
oauthBlacklistedToolNames =
  [ -- Original Claude Code names
    "read"
  , "write"
  , "edit"
  , "glob"
  , "grep"
  , "bash"
  , "task"
  , "todowrite"
  -- Suffixed variants (testing)
  , "read_file"
  ]

-- ============================================================================
-- OAuth Provider (with tool name workarounds)
-- ============================================================================

-- OAuth provider instances (same request/response types, different behavior)
instance SupportsTemperature AnthropicOAuth
instance SupportsMaxTokens AnthropicOAuth
instance SupportsSystemPrompt AnthropicOAuth
instance SupportsStreaming AnthropicOAuth

instance Provider (Model aiModel AnthropicOAuth) where
  type ProviderRequest (Model aiModel AnthropicOAuth) = AnthropicRequest
  type ProviderResponse (Model aiModel AnthropicOAuth) = AnthropicResponse

-- Prefix for blacklisted tool names
toolNamePrefix :: Text
toolNamePrefix = "runix_"

-- | Check if a tool name is blacklisted
isBlacklisted :: Text -> Bool
isBlacklisted name = name `elem` oauthBlacklistedToolNames

-- | Prefix a tool name if it's blacklisted
prefixIfBlacklisted :: Text -> Text
prefixIfBlacklisted name
  | isBlacklisted name = toolNamePrefix <> name
  | otherwise = name

-- | Check if a tool name has our prefix
hasPrefix :: Text -> Bool
hasPrefix name = toolNamePrefix `Text.isPrefixOf` name

-- | Remove prefix from a tool name if it has one
unprefix :: Text -> Text
unprefix name
  | hasPrefix name = Text.drop (Text.length toolNamePrefix) name
  | otherwise = name

-- | Prefix tool names in tool definitions
prefixToolDefinitions :: [AnthropicToolDefinition] -> ([AnthropicToolDefinition], Map Text Text)
prefixToolDefinitions defs =
  let prefixDef def =
        let origName = anthropicToolName def
            prefixedName = prefixIfBlacklisted origName
            mapping = if origName /= prefixedName
                      then Map.singleton prefixedName origName
                      else Map.empty
        in (def { anthropicToolName = prefixedName }, mapping)
      (defs', mappings) = unzip $ map prefixDef defs
  in (defs', Map.unions mappings)

-- | Prefix tool names in content blocks
prefixContentBlock :: AnthropicContentBlock -> AnthropicContentBlock
prefixContentBlock block@(AnthropicToolUseBlock tid tname tinput cache) =
  AnthropicToolUseBlock tid (prefixIfBlacklisted tname) tinput cache
prefixContentBlock block = block

-- | Prefix tool names in all messages of a request
prefixRequestMessages :: AnthropicRequest -> AnthropicRequest
prefixRequestMessages req =
  let prefixMessage msg = msg { content = map prefixContentBlock (content msg) }
  in req { messages = map prefixMessage (messages req) }

-- OAuth tools provider with prefix/unprefix logic
-- Note: Per-model instances for BaseComposableProvider and HasTools should be defined
-- in test files or application code, similar to regular Anthropic models
anthropicOAuthTools :: forall m. (HasTools m, ProviderRequest m ~ AnthropicRequest, ProviderResponse m ~ AnthropicResponse) => ComposableProvider m OAuthToolsState
anthropicOAuthTools _m configs state = noopHandler
  { cpConfigHandler = \req ->
      let -- Extract tool definitions from configs
          toolDefs = [defs | Tools defs <- configs]
          -- Convert to Anthropic format
          anthropicToolDefs = map toAnthropicToolDef (concat toolDefs)
          -- Prefix blacklisted tool names and collect mappings
          (prefixedDefs, _newMappings) = prefixToolDefinitions anthropicToolDefs
          -- Add cache control
          finalDefs = withToolCacheControl prefixedDefs
          -- Set tool definitions
          req1 = if null toolDefs then req else req { tools = Just finalDefs }
          -- Prefix tool names in existing messages
          req2 = prefixRequestMessages req1
      in req2

  , cpPreRequest = \req state ->
      -- Extract mappings from configs (since cpConfigHandler hasn't run yet!)
      let toolDefs = [defs | Tools defs <- configs]
          anthropicToolDefs = map toAnthropicToolDef (concat toolDefs)
          (prefixedDefs, mappings) = prefixToolDefinitions anthropicToolDefs
      in state { prefixedToOriginal = Map.union mappings (prefixedToOriginal state) }

  , cpToRequest = \msg req -> case msg of
      AssistantTool toolCall ->
        let prefixedCall = case toolCall of
              ToolCall tcId tcName tcParams ->
                ToolCall tcId (prefixIfBlacklisted tcName) tcParams
              InvalidToolCall tcId tcName rawArgs err ->
                InvalidToolCall tcId (prefixIfBlacklisted tcName) rawArgs err
            block = fromToolCall prefixedCall
        in appendContentBlock "assistant" block req
      ToolResultMsg result ->
        let block = fromToolResult result
        in appendContentBlock "user" block req
      _ -> req

  , cpFromResponse = \resp -> case resp of
      AnthropicError err ->
        Left $ ModelError $ errorMessage err
      AnthropicSuccess anthropicResp ->
        case responseContent anthropicResp of
          (AnthropicToolUseBlock tid tname tinput _ : rest) ->
            -- Unprefix the tool name before returning
            let origName = Map.findWithDefault tname tname (prefixedToOriginal state)
                toolCall = ToolCall tid origName tinput
                remaining = anthropicResp { responseContent = rest }
            in Right (Just (AssistantTool toolCall, AnthropicSuccess remaining))
          _ -> Right Nothing  -- Not a tool block, let other handlers try

  , cpSerializeMessage = serializeToolMessages
  , cpDeserializeMessage = deserializeToolMessages
  }

-- Note: For OAuth models, use the same anthropicReasoning provider
-- Define per-model HasReasoning instances in test files or application code
