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

-- Declare Anthropic capabilities
instance HasTools Anthropic
instance SupportsTemperature Anthropic
instance SupportsMaxTokens Anthropic
instance SupportsSystemPrompt Anthropic
-- Note: Anthropic does NOT support Seed or JSON mode

-- Provider typeclass implementation (just type associations)
instance ModelName Anthropic model => Provider Anthropic model where
  type ProviderRequest Anthropic = AnthropicRequest
  type ProviderResponse Anthropic = AnthropicResponse

-- Composable providers for Anthropic
-- Note: Anthropic requires grouping all messages at once (alternating user/assistant)
-- so we use a simpler approach: build the full request from all messages at the end

baseComposableProvider :: forall model. (ModelName Anthropic model, HasTools model) => ComposableProvider Anthropic model
baseComposableProvider = ComposableProvider
  { cpToRequest = mempty  -- No incremental building - Anthropic needs all messages
  , cpFromResponse = parseTextResponse
  }
  where
    parseTextResponse acc (AnthropicError err) = acc
    parseTextResponse acc (AnthropicSuccess resp) =
      case [txt | AnthropicTextBlock txt <- responseContent resp] of
        (txt:_) -> acc <> [AssistantText txt]
        [] -> acc

toolsComposableProvider :: forall model. (ModelName Anthropic model, HasTools model, HasTools Anthropic) => ComposableProvider Anthropic model
toolsComposableProvider = ComposableProvider
  { cpToRequest = mempty
  , cpFromResponse = parseToolResponse
  }
  where
    parseToolResponse acc (AnthropicError _) = acc
    parseToolResponse acc (AnthropicSuccess resp) =
      acc <> [AssistantTool (ToolCall tid tname tinput) | AnthropicToolUseBlock tid tname tinput <- responseContent resp]

-- Helper function to build Anthropic request from messages (replacement for old toRequest)
-- This is what users will call instead of using the composable provider directly
buildAnthropicRequest :: ModelName Anthropic model
                      => Anthropic
                      -> model
                      -> [ModelConfig Anthropic model]
                      -> [Message model Anthropic]
                      -> AnthropicRequest
buildAnthropicRequest _provider mdl configs msgs =
  let (systemMsg, otherMsgs) = extractSystem msgs
      systemFromConfig = case [sp | SystemPrompt sp <- configs] of
        (sp:_) -> Just [AnthropicSystemBlock sp "text"]
        [] -> Nothing
      toolsFromConfig = case [defs | Tools defs <- configs] of
        (defs:_) -> Just (map toAnthropicToolDef defs)
        [] -> Nothing
      baseRequest = AnthropicRequest
        { model = modelName @Anthropic mdl
        , messages = groupMessages otherMsgs
        , max_tokens = case [mt | MaxTokens mt <- configs] of { (mt:_) -> mt; [] -> 1000 }
        , temperature = case [t | Temperature t <- configs] of { (t:_) -> Just t; [] -> Nothing }
        , system = systemMsg <> systemFromConfig
        , tools = toolsFromConfig
        }
  in baseRequest
-- Helper function to parse Anthropic response (replacement for old fromResponse)
parseAnthropicResponse :: (HasTools model, HasTools Anthropic)
                       => AnthropicResponse
                       -> Either LLMError [Message model Anthropic]
parseAnthropicResponse (AnthropicError err) =
  Left $ ProviderError 0 $ errorMessage err <> " (" <> errorType err <> ")"
parseAnthropicResponse (AnthropicSuccess resp) =
  case responseContent resp of
    [] -> Left $ ParseError "Empty response content"
    allBlocks ->
      let textMsgs = [AssistantText txt | AnthropicTextBlock txt <- allBlocks]
          toolMsgs = [AssistantTool (ToolCall tid tname tinput) | AnthropicToolUseBlock tid tname tinput <- allBlocks]
          allMsgs = textMsgs <> toolMsgs
      in if null allMsgs
         then Left $ ParseError "No text or tool content in response"
         else Right allMsgs

-- Note: SystemText messages are reminders/context, not the system prompt
-- The system prompt comes from the SystemPrompt config, not from messages
extractSystem :: [Message model Anthropic] -> (Maybe [AnthropicSystemBlock], [Message model Anthropic])
extractSystem (SystemText txt : rest) =
  -- System text messages should be converted to user messages in Anthropic
  -- as Anthropic doesn't support system messages in the message array
  (Nothing, UserText txt : rest)
extractSystem msgs = (Nothing, msgs)

-- Convert MessageDirection to Anthropic role text
toRoleText :: MessageDirection -> Text
toRoleText User = "user"
toRoleText Assistant = "assistant"

-- Group consecutive messages by direction (User vs Assistant)
-- Anthropic expects alternating roles with multiple content blocks per message
groupMessages :: [Message model Anthropic] -> [AnthropicMessage]
groupMessages [] = []
groupMessages (msg:msgs) =
  let msgDir = messageDirection msg
      (sameDir, rest) = span (\m -> messageDirection m == msgDir) msgs
      blocks = concatMap messageToBlocks (msg:sameDir)
  in AnthropicMessage (toRoleText msgDir) blocks : groupMessages rest
  where

    messageToBlocks :: Message model Anthropic -> [AnthropicContentBlock]
    messageToBlocks (UserText txt) = [AnthropicTextBlock txt]
    messageToBlocks (SystemText txt) = [AnthropicTextBlock txt]  -- System messages are client reminders
    messageToBlocks (AssistantText txt) = [AnthropicTextBlock txt]
    messageToBlocks (AssistantTool (ToolCall tcId tcName tcParams)) =
      [AnthropicToolUseBlock tcId tcName tcParams]
    messageToBlocks (AssistantTool (InvalidToolCall tcId tcName _rawArgs _err)) =
      error $ "Cannot convert InvalidToolCall to Anthropic request - malformed tool call: "
           <> Text.unpack tcName <> " (id: " <> Text.unpack tcId <> ")"
    messageToBlocks (ToolResultMsg (ToolResult toolCall output)) =
      let callId = getToolCallId toolCall
          resultContent = case output of
            Left errMsg -> errMsg
            Right jsonVal -> Text.pack $ show jsonVal  -- TODO: better JSON to text conversion
      in [AnthropicToolResultBlock callId resultContent]
    -- Catch-all for unsupported message types
    messageToBlocks msg = error $ "Unsupported message type for Anthropic provider: " ++ show msg


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