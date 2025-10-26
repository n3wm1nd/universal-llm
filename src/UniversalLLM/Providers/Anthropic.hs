{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}

module UniversalLLM.Providers.Anthropic where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.Anthropic
import Data.Text (Text)
import qualified Data.Text as Text

-- Anthropic provider (phantom type)
data Anthropic = Anthropic deriving (Show, Eq)

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
handleBase :: ModelName Anthropic model => MessageHandler Anthropic model
handleBase _provider modelType configs _msg req =
  req { model = modelName @Anthropic modelType
      , max_tokens = case [mt | MaxTokens mt <- configs] of { (mt:_) -> mt; [] -> max_tokens req }
      , temperature = case [t | Temperature t <- configs] of { (t:_) -> Just t; [] -> temperature req }
      }

-- System prompt config handler (from config)
-- This is a ConfigHandler, not a MessageHandler - it runs after all messages are processed
handleSystemPrompt :: ConfigHandler Anthropic model
handleSystemPrompt = \_provider _model configs req ->
  let systemPrompts = [sp | SystemPrompt sp <- configs]
      sysBlocks = [AnthropicSystemBlock sp "text" | sp <- systemPrompts]
  in if null sysBlocks
     then req  -- Don't modify if no system prompts
     else req { system = Just sysBlocks }

-- Text message handler - groups messages incrementally
handleTextMessages :: MessageHandler Anthropic model
handleTextMessages = \_provider _model _configs msg req -> case msg of
  UserText txt -> req { messages = appendBlock (messages req) "user" (AnthropicTextBlock txt) }
  AssistantText txt -> req { messages = appendBlock (messages req) "assistant" (AnthropicTextBlock txt) }
  SystemText txt -> req { messages = appendBlock (messages req) "user" (AnthropicTextBlock txt) }
  _ -> req
  where
    -- Append a block to messages, grouping with last message if same role
    appendBlock :: [AnthropicMessage] -> Text -> AnthropicContentBlock -> [AnthropicMessage]
    appendBlock [] r b = [AnthropicMessage r [b]]
    appendBlock msgs r b =
      let lastMsg = last msgs
          initMsgs = init msgs
      in if role lastMsg == r
         then initMsgs <> [lastMsg { content = content lastMsg <> [b] }]
         else msgs <> [AnthropicMessage r [b]]

-- Tools handler - groups tool blocks incrementally
handleTools :: MessageHandler Anthropic model
handleTools = \_provider _model configs msg req -> case msg of
  AssistantTool (ToolCall tcId tcName tcParams) ->
    req { messages = appendToolBlock (messages req) "assistant" (AnthropicToolUseBlock tcId tcName tcParams) }
  AssistantTool (InvalidToolCall tcId tcName rawArgs err) ->
    -- InvalidToolCall cannot be directly converted to Anthropic's tool_use format
    -- Instead, convert it to a text message describing the error
    -- This allows the conversation to continue rather than crashing
    let errorText = "Tool call error for " <> tcName <> " (id: " <> tcId <> "): " <> err
                    <> "\nRaw arguments: " <> rawArgs
    in req { messages = appendToolBlock (messages req) "assistant" (AnthropicTextBlock errorText) }
  ToolResultMsg (ToolResult toolCall output) ->
    let callId = getToolCallId toolCall
        resultContent = case output of
          Left errMsg -> errMsg
          Right jsonVal -> Text.pack $ show jsonVal
    in req { messages = appendToolBlock (messages req) "user" (AnthropicToolResultBlock callId resultContent) }
  _ ->
    -- Apply tool definitions from config (for any message)
    let toolDefs = [defs | Tools defs <- configs]
    in if null toolDefs
       then req
       else req { tools = Just (map toAnthropicToolDef (concat toolDefs)) }
  where
    -- Append a block to messages, grouping with last message if same role
    appendToolBlock :: [AnthropicMessage] -> Text -> AnthropicContentBlock -> [AnthropicMessage]
    appendToolBlock [] r b = [AnthropicMessage r [b]]
    appendToolBlock msgs r b =
      let lastMsg = last msgs
          initMsgs = init msgs
      in if role lastMsg == r
         then initMsgs <> [lastMsg { content = content lastMsg <> [b] }]
         else msgs <> [AnthropicMessage r [b]]


-- Composable providers for Anthropic

-- Base provider: handles text messages and basic configuration
baseComposableProvider :: forall model. ModelName Anthropic model => ComposableProvider Anthropic model
baseComposableProvider = ComposableProvider
  { cpToRequest = handleBase >>> handleTextMessages
  , cpConfigHandler = handleSystemPrompt
  , cpFromResponse = parseTextResponse
  }
  where
    parseTextResponse _provider _model _configs _history acc (AnthropicError _err) = acc
    parseTextResponse _provider _model _configs _history acc (AnthropicSuccess resp) =
      case [txt | AnthropicTextBlock txt <- responseContent resp] of
        (txt:_) -> acc <> [AssistantText txt]
        [] -> acc

-- Tools capability combinator
anthropicWithTools :: forall model. HasTools model Anthropic => ComposableProvider Anthropic model -> ComposableProvider Anthropic model
anthropicWithTools base = base `chainProviders` toolsProvider
  where
    toolsProvider = ComposableProvider
      { cpToRequest = handleTools
      , cpConfigHandler = \_provider _model _configs req -> req  -- No config handling needed
      , cpFromResponse = parseToolResponse
      }
    parseToolResponse _provider _model _configs _history acc (AnthropicError _) = acc
    parseToolResponse _provider _model _configs _history acc (AnthropicSuccess resp) =
      acc <> [AssistantTool (ToolCall tid tname tinput) | AnthropicToolUseBlock tid tname tinput <- responseContent resp]

-- Ensures Anthropic's API constraint: conversations must start with a user message
-- If the first message is from assistant, prepends an empty user message
-- This should be composed at the END of the provider chain
ensureUserFirst :: ComposableProvider Anthropic model -> ComposableProvider Anthropic model
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