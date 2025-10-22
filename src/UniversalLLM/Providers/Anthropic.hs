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
instance ModelName Anthropic model => Provider Anthropic model where
  type ProviderRequest Anthropic = AnthropicRequest
  type ProviderResponse Anthropic = AnthropicResponse

-- Message handlers for Anthropic

-- Base handler: model name and basic config
-- Sets model name always. Config values (max_tokens, temperature) are only
-- set if explicitly provided in this message's config.
-- IMPORTANT: Never reference req's own fields - only set fields based on external inputs
handleBase :: ModelName Anthropic model => MessageHandler Anthropic model
handleBase = MessageHandler $ \_provider model configs _msg req ->
  req { model = modelName @Anthropic model
      , max_tokens = case [mt | MaxTokens mt <- configs] of { (mt:_) -> mt; [] -> max_tokens req }
      , temperature = case [t | Temperature t <- configs] of { (t:_) -> Just t; [] -> temperature req }
      }

-- System prompt handler (from config)
handleSystemPrompt :: MessageHandler Anthropic model
handleSystemPrompt = MessageHandler $ \_provider _model configs _msg req ->
  let systemPrompts = [sp | SystemPrompt sp <- configs]
      sysBlocks = [AnthropicSystemBlock sp "text" | sp <- systemPrompts]
  in if null sysBlocks
     then req  -- Don't modify if no system prompts
     else req { system = Just sysBlocks }

-- Text message handler - groups messages incrementally
handleTextMessages :: MessageHandler Anthropic model
handleTextMessages = MessageHandler $ \_provider _model _configs msg req -> case msg of
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
handleTools = MessageHandler $ \_provider _model configs msg req -> case msg of
  AssistantTool (ToolCall tcId tcName tcParams) ->
    req { messages = appendToolBlock (messages req) "assistant" (AnthropicToolUseBlock tcId tcName tcParams) }
  AssistantTool (InvalidToolCall tcId tcName _rawArgs _err) ->
    error $ "Cannot convert InvalidToolCall to Anthropic request - malformed tool call: "
         <> Text.unpack tcName <> " (id: " <> Text.unpack tcId <> ")"
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

baseComposableProvider :: forall model. ModelName Anthropic model => ComposableProvider Anthropic model
baseComposableProvider = ComposableProvider
  { cpToRequest = handleBase <> handleSystemPrompt <> handleTextMessages
  , cpFromResponse = parseTextResponse
  }
  where
    parseTextResponse _provider _model _configs _history acc (AnthropicError _err) = acc
    parseTextResponse _provider _model _configs _history acc (AnthropicSuccess resp) =
      case [txt | AnthropicTextBlock txt <- responseContent resp] of
        (txt:_) -> acc <> [AssistantText txt]
        [] -> acc

toolsComposableProvider :: forall model. HasTools model Anthropic => ComposableProvider Anthropic model
toolsComposableProvider = ComposableProvider
  { cpToRequest = handleTools
  , cpFromResponse = parseToolResponse
  }
  where
    parseToolResponse _provider _model _configs _history acc (AnthropicError _) = acc
    parseToolResponse _provider _model _configs _history acc (AnthropicSuccess resp) =
      acc <> [AssistantTool (ToolCall tid tname tinput) | AnthropicToolUseBlock tid tname tinput <- responseContent resp]

-- Helper function to build Anthropic request from messages (replacement for old toRequest)


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