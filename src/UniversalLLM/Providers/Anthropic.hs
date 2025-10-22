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

-- Composable providers for Anthropic
-- Note: Anthropic requires grouping all messages at once (alternating user/assistant)
-- so we use a simpler approach: build the full request from all messages at the end

baseComposableProvider :: forall model. ModelName Anthropic model => ComposableProvider Anthropic model
baseComposableProvider = ComposableProvider
  { cpToRequest = mempty  -- No incremental building - Anthropic needs all messages
  , cpFromResponse = parseTextResponse
  }
  where
    parseTextResponse _provider _model _configs _history acc (AnthropicError err) = acc
    parseTextResponse _provider _model _configs _history acc (AnthropicSuccess resp) =
      case [txt | AnthropicTextBlock txt <- responseContent resp] of
        (txt:_) -> acc <> [AssistantText txt]
        [] -> acc

toolsComposableProvider :: forall model. HasTools model Anthropic => ComposableProvider Anthropic model
toolsComposableProvider = ComposableProvider
  { cpToRequest = mempty
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