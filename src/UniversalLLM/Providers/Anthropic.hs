{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module UniversalLLM.Providers.Anthropic where

import UniversalLLM.Core.Types
import UniversalLLM.Core.Serialization
import UniversalLLM.Protocols.Anthropic
import Data.Text (Text)
import qualified Data.Text as Text

-- Anthropic provider (phantom type)
data Anthropic = Anthropic deriving (Show, Eq)

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
fromToolCall (ToolCall tcId tcName tcParams) = AnthropicToolUseBlock tcId tcName tcParams
fromToolCall (InvalidToolCall tcId tcName rawArgs err) =
  -- InvalidToolCall cannot be directly converted to Anthropic's tool_use format
  -- Instead, convert it to a text message describing the error
  let errorText = "Tool call error for " <> tcName <> " (id: " <> tcId <> "): " <> err
                  <> "\nRaw arguments: " <> rawArgs
  in AnthropicTextBlock errorText

-- | Convert a ToolResult to an AnthropicContentBlock
fromToolResult :: ToolResult -> AnthropicContentBlock
fromToolResult (ToolResult toolCall output) =
  let callId = getToolCallId toolCall
      resultContent = case output of
        Left errMsg -> errMsg
        Right jsonVal -> Text.pack $ show jsonVal
  in AnthropicToolResultBlock callId resultContent

-- | Convert text to a text content block
fromText :: Text -> AnthropicContentBlock
fromText = AnthropicTextBlock

-- ============================================================================
-- Message Handlers
-- ============================================================================

-- Declare Anthropic parameter support
instance SupportsTemperature Anthropic
instance SupportsMaxTokens Anthropic
instance SupportsSystemPrompt Anthropic
-- Note: Anthropic does NOT support Seed or JSON mode

-- Anthropic capabilities are now declared per-model (see model files)

-- Provider typeclass implementation (just type associations)
instance Provider Anthropic model where
  type ProviderRequest Anthropic = AnthropicRequest
  type ProviderResponse Anthropic = AnthropicResponse

-- Message handlers for Anthropic

-- Base handler: model name and basic config
-- Updates the request with model name and config
handleBase :: forall provider model . (ProviderRequest provider ~ AnthropicRequest, ModelName provider model) => MessageHandler provider model
handleBase _provider modelType configs _msg req =
  req { model = modelName @provider modelType
      , max_tokens = case [mt | MaxTokens mt <- configs] of { (mt:_) -> mt; [] -> max_tokens req }
      , temperature = case [t | Temperature t <- configs] of { (t:_) -> Just t; [] -> temperature req }
      }

-- System prompt config handler (from config)
-- This is a ConfigHandler, not a MessageHandler - it runs after all messages are processed
handleSystemPrompt :: ProviderRequest provider ~ AnthropicRequest => ConfigHandler provider model
handleSystemPrompt = \_provider _model configs req ->
  let systemPrompts = [sp | SystemPrompt sp <- configs]
      sysBlocks = [AnthropicSystemBlock sp "text" | sp <- systemPrompts]
  in if null sysBlocks
     then req
     else setSystemBlocks sysBlocks req

-- Text message handler - groups messages incrementally
handleTextMessages :: ProviderRequest provider ~ AnthropicRequest => MessageHandler provider model
handleTextMessages = \_provider _model _configs msg req -> case msg of
  UserText txt -> appendContentBlock "user" (fromText txt) req
  AssistantText txt -> appendContentBlock "assistant" (fromText txt) req
  SystemText txt -> appendContentBlock "user" (fromText txt) req
  _ -> req

-- Tools handler - groups tool blocks incrementally
handleTools :: ProviderRequest provider ~ AnthropicRequest => MessageHandler provider model
handleTools = \_provider _model configs msg req -> case msg of
  AssistantTool toolCall -> appendContentBlock "assistant" (fromToolCall toolCall) req
  ToolResultMsg result -> appendContentBlock "user" (fromToolResult result) req
  _ ->
    -- Apply tool definitions from config (for any message)
    let toolDefs = [defs | Tools defs <- configs]
    in if null toolDefs
       then req
       else setToolDefinitions (map toAnthropicToolDef (concat toolDefs)) req


-- Composable providers for Anthropic

-- Base provider: handles text messages and basic configuration
baseComposableProvider :: forall model provider . (ProviderRequest provider ~ AnthropicRequest, ProviderResponse provider ~ AnthropicResponse, ModelName provider model) => ComposableProvider provider model
baseComposableProvider = ComposableProvider
  { cpToRequest = handleBase >>> handleTextMessages
  , cpConfigHandler = handleSystemPrompt
  , cpFromResponse = parseTextResponse
  , cpSerializeMessage = serializeBaseMessage
  , cpDeserializeMessage = deserializeBaseMessage
  }
  where
    parseTextResponse _provider _model _configs _history acc (AnthropicError _err) = acc
    parseTextResponse _provider _model _configs _history acc (AnthropicSuccess resp) =
      case [txt | AnthropicTextBlock txt <- responseContent resp] of
        (txt:_) -> acc <> [AssistantText txt]
        [] -> acc

-- Tools capability combinator
anthropicWithTools :: forall model provider . (ProviderRequest provider ~ AnthropicRequest, ProviderResponse provider ~ AnthropicResponse, HasTools model provider) => ComposableProvider provider model -> ComposableProvider provider model
anthropicWithTools base = base `chainProviders` toolsProvider
  where
    toolsProvider = ComposableProvider
      { cpToRequest = handleTools
      , cpConfigHandler = \_provider _model _configs req -> req  -- No config handling needed
      , cpFromResponse = parseToolResponse
      , cpSerializeMessage = serializeToolMessages
      , cpDeserializeMessage = deserializeToolMessages
      }
    parseToolResponse _provider _model _configs _history acc (AnthropicError _) = acc
    parseToolResponse _provider _model _configs _history acc (AnthropicSuccess resp) =
      acc <> [AssistantTool (ToolCall tid tname tinput) | AnthropicToolUseBlock tid tname tinput <- responseContent resp]

-- Ensures Anthropic's API constraint: conversations must start with a user message
-- If the first message is from assistant, prepends an empty user message
-- This should be composed at the END of the provider chain
ensureUserFirst :: ProviderRequest provider ~ AnthropicRequest => ComposableProvider provider model -> ComposableProvider provider model
ensureUserFirst base = base `chainProviders` ensureUserFirstProvider
  where
    ensureUserFirstProvider = ComposableProvider
      { cpToRequest = \_provider _model _configs _msg req -> req  -- No message handling needed
      , cpConfigHandler = \_provider _model _configs req ->
          -- Anthropic API requires first message to be from user
          case messages req of
            [] -> req
            (firstMsg:_) -> if role firstMsg == "user"
              then req
              else req { messages = AnthropicMessage "user" [AnthropicTextBlock ""] : messages req }
      , cpFromResponse = \_provider _model _configs _history acc _resp -> acc  -- No-op for response parsing
      , cpSerializeMessage = \_ -> Nothing  -- Let base handle serialization
      , cpDeserializeMessage = \_ -> Nothing  -- Let base handle deserialization
      }

-- Default ProviderImplementation for basic text-only models
-- Models with capabilities (tools, vision, etc.) should provide their own instances
instance {-# OVERLAPPABLE #-} ModelName Anthropic model => ProviderImplementation Anthropic model where
  getComposableProvider = ensureUserFirst $ baseComposableProvider


-- | Add magic system prompt for OAuth authentication
-- Prepends the Claude Code authentication prompt to user's system prompts
withMagicSystemPrompt :: AnthropicRequest -> AnthropicRequest
withMagicSystemPrompt request =
  let magicBlock = AnthropicSystemBlock "You are Claude Code, Anthropic's official CLI for Claude." "text"
      combinedSystem = case system request of
        Nothing -> [magicBlock]
        Just userBlocks -> magicBlock : userBlocks
  in request { system = Just combinedSystem }

-- | Headers for OAuth authentication (Claude Code subscription)
-- Returns headers as [(Text, Text)] for transport-agnostic usage
oauthHeaders :: Text -> [(Text, Text)]
oauthHeaders token =
  [ ("Content-Type", "application/json")
  , ("Authorization", "Bearer " <> token)
  , ("anthropic-version", "2023-06-01")
  , ("anthropic-beta", "oauth-2025-04-20")
  , ("User-Agent", "hs-universal-llm (prerelease-dev)")
  ]

-- | Headers for API key authentication (console.anthropic.com)
-- Returns headers as [(Text, Text)] for transport-agnostic usage
apiKeyHeaders :: Text -> [(Text, Text)]
apiKeyHeaders apiKey =
  [ ("Content-Type", "application/json")
  , ("x-api-key", apiKey)
  , ("anthropic-version", "2023-06-01")
  , ("User-Agent", "hs-universal-llm (prerelease-dev)")
  ]