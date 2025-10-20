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

-- Apply configuration to Anthropic request
instance ApplyConfig AnthropicRequest Anthropic model where
  applyConfig configs req = foldr applyOne req configs
    where
      applyOne (Temperature t) r = r { temperature = Just t }
      applyOne (MaxTokens mt) r = r { max_tokens = mt }
      applyOne (SystemPrompt sp) r = r { system = Just sp }
      applyOne (Tools toolDefs) r = r { tools = Just (map toAnthropicToolDef toolDefs) }
      applyOne _ r = r  -- Catch-all for future config options (e.g., Seed not supported by Anthropic)

-- Provider typeclass implementation
instance (ModelName Anthropic model, ProtocolHandleTools AnthropicContentBlock model Anthropic)
         => Provider Anthropic model where
  type ProviderRequest Anthropic = AnthropicRequest
  type ProviderResponse Anthropic = AnthropicResponse

  toRequest = toRequest'
  fromResponse = fromResponse'
-- Pure functions: no IO, just transformations
toRequest' :: forall model. ModelName Anthropic model
          => Anthropic -> model -> [ModelConfig Anthropic model] -> [Message model Anthropic] -> AnthropicRequest
toRequest' _provider mdl configs msgs =
  let (systemMsg, otherMsgs) = extractSystem msgs
      baseRequest = AnthropicRequest
        { model = modelName @Anthropic mdl
        , messages = groupMessages otherMsgs
        , max_tokens = 1000  -- default, will be overridden by config if present
        , temperature = Nothing
        , system = systemMsg
        , tools = Nothing
        }
  in applyConfig configs baseRequest

fromResponse' :: forall model. ProtocolHandleTools AnthropicContentBlock model Anthropic => AnthropicResponse -> Either LLMError [Message model Anthropic]
fromResponse' (AnthropicError err) =
  Left $ ProviderError 0 $ errorMessage err <> " (" <> errorType err <> ")"
fromResponse' (AnthropicSuccess resp) =
  case responseContent resp of
    [] -> Left $ ProviderError 0 "Empty response content"
    [AnthropicTextBlock txt] -> Right [AssistantText txt]
    allBlocks ->
      -- Check for text blocks first
      case [txt | AnthropicTextBlock txt <- allBlocks] of
        (txt:_) -> Right [AssistantText txt]
        [] ->
          -- No text blocks, try tool calls (will be empty list for non-tool models)
          let toolMessages = handleToolCalls @AnthropicContentBlock @model @Anthropic allBlocks
          in if null toolMessages
             then Left $ ProviderError 0 "No text or tool content in response"
             else Right toolMessages

extractSystem :: [Message model Anthropic] -> (Maybe Text, [Message model Anthropic])
extractSystem (SystemText txt : rest) = (Just txt, rest)
extractSystem msgs = (Nothing, msgs)

-- Message direction for grouping
data MessageDirection = User | Assistant
  deriving (Eq, Show)

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
    messageDirection :: Message model Anthropic -> MessageDirection
    messageDirection (UserText _) = User
    messageDirection (UserImage _ _) = User
    messageDirection (ToolResultMsg _) = User
    messageDirection (SystemText _) = User  -- System messages are autonomous client reminders, treated as user direction
    messageDirection (AssistantText _) = Assistant
    messageDirection (AssistantTool _) = Assistant
    messageDirection (AssistantJSON _) = Assistant  -- Will never match (Anthropic lacks HasJSON)

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
-- Prepends the Claude Code authentication prompt to user's system prompt
withMagicSystemPrompt :: AnthropicRequest -> AnthropicRequest
withMagicSystemPrompt request =
  let magicPrompt = "You are Claude Code, Anthropic's official CLI for Claude."
      combinedSystem = case system request of
        Nothing -> magicPrompt
        Just userPrompt -> magicPrompt <> "\n\n" <> userPrompt
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