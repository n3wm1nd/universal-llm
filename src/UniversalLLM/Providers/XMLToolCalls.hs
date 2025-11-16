{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE GADTs #-}

module UniversalLLM.Providers.XMLToolCalls
  ( -- * Strategy A: Native Tool Support + XML Response Parsing
    -- For providers that understand tool definitions natively but models respond with XML
    withXMLResponseParsing
    -- * Strategy B: Full XML Tool Support
    -- For providers with no native tool support - everything via prompts
  , withFullXMLToolSupport
    -- * Helper Functions
  , toolCallToXMLArgs
  ) where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.ToolCall.XML
import Data.Maybe (mapMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Autodocodec.Aeson.Compat as Compat

-- ============================================================================
-- Common Helper Functions
-- ============================================================================

-- | Convert ToolCall parameters (JSON) to XML argument pairs
toolCallToXMLArgs :: Aeson.Value -> [XMLArgPair]
toolCallToXMLArgs (Aeson.Object obj) =
  [ XMLArgPair (Key.toText k) (valueToText v)
  | (k, v) <- Compat.toList obj
  ]
toolCallToXMLArgs _ = []

valueToText :: Aeson.Value -> T.Text
valueToText (Aeson.String s) = s
valueToText v = T.pack (show v)  -- Fallback for non-string values

-- | Convert ToolCall to XML text
toolCallToXMLText :: ToolCall -> Text
toolCallToXMLText (ToolCall _ tcName params) =
  let args = toolCallToXMLArgs params
      xmlToolCall = XMLToolCall tcName args
  in encodeXMLToolCall xmlToolCall
toolCallToXMLText (InvalidToolCall _ tcName rawArgs _) =
  -- Best effort: wrap in tags
  wrapInTag "tool_call" (tcName <> "\n" <> rawArgs)

-- | Parse XML tool calls from assistant text response
-- Returns (list of tool call messages, cleaned text without tool calls)
parseXMLFromResponse :: forall model provider. HasTools model provider
                     => Text -> ([Message model provider], Text)
parseXMLFromResponse txt =
  let (xmlBlocks, cleanedText) = extractAndRemoveXMLToolCalls txt
      parsedCalls = mapMaybe parseXMLToolCall xmlBlocks
      toolCallMsgs = map (AssistantTool . xmlToolCallToToolCall) parsedCalls
  in (toolCallMsgs, cleanedText)

-- ============================================================================
-- Strategy A: Native Tool Support + XML Response Parsing Only
-- ============================================================================
-- Provider understands OpenAI tool format, but model responds with XML
-- Example: LlamaCpp with certain models

-- | Parse XML tool calls from responses (provider has native tool support)
-- This ONLY handles response parsing - tool definitions use native format
withXMLResponseParsing :: forall provider model.
                          (HasTools model provider,
                           ProviderResponse provider ~ OpenAIResponse)
                       => ComposableProvider provider model
                       -> ComposableProvider provider model
withXMLResponseParsing base = base `chainProviders` xmlResponseParser
  where
    xmlResponseParser _p _m _configs = noopHandler
      { cpFromResponse = parseResponse
      , cpPureMessageResponse = extractXMLFromMessages
      }

    parseResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        [] -> Nothing
        (OpenAIChoice msg:_) ->
          case content msg of
            Nothing -> Nothing
            Just txt -> Just (AssistantText txt, OpenAISuccess (OpenAISuccessResponse []))
    parseResponse _ = Nothing

    extractXMLFromMessages :: [Message model provider] -> [Message model provider]
    extractXMLFromMessages acc =
      let (toolCalls, cleanedMsgs) = foldl processMessage ([], []) acc
      in cleanedMsgs <> toolCalls
      where
        processMessage (calls, msgs) (AssistantText txt) =
          let (toolCallMsgs, cleanedText) = parseXMLFromResponse txt
          in if null toolCallMsgs
             then (calls, msgs <> [AssistantText txt])
             else
               let cleanMsg = if T.null (T.strip cleanedText) then [] else [AssistantText cleanedText]
               in (calls <> toolCallMsgs, msgs <> cleanMsg)
        processMessage (calls, msgs) msg = (calls, msgs <> [msg])

-- ============================================================================
-- Strategy B: Full XML Tool Support (no native tool support)
-- ============================================================================
-- Provider has no tool concept - everything via prompts and text
-- Example: Basic OpenAI-compatible servers

-- | Full XML tool support
-- Handles: system prompt injection, tool results as text, XML response parsing
withFullXMLToolSupport :: forall provider model.
                          (HasTools model provider,
                           ProviderRequest provider ~ OpenAIRequest,
                           ProviderResponse provider ~ OpenAIResponse)
                       => ComposableProvider provider model
                       -> ComposableProvider provider model
withFullXMLToolSupport base = base `chainProviders` xmlFullSupport
  where
    xmlFullSupport _p _m configs = noopHandler
      { cpPureMessageRequest = convertToolsToXML
      , cpConfigHandler = injectToolDefinitions configs
      , cpFromResponse = parseResponse
      , cpPureMessageResponse = extractXMLMessagesFromResponse
      }

    -- Convert tool calls and results to XML before encoding
    convertToolsToXML :: [Message model provider] -> [Message model provider]
    convertToolsToXML = map convertMessage
      where
        convertMessage msg = case msg of
          -- Tool results become XML text messages (will be encoded as user message by base)
          ToolResultMsg result ->
            let xmlResult = toolResultToXML result
                xmlText = encodeXMLToolResult xmlResult
            in UserText xmlText  -- Will get encoded as user's tool result response

          -- Tool calls become XML text (will be encoded as assistant message by base)
          AssistantTool toolCall ->
            let xmlCall = toolCallToXMLText toolCall
            in AssistantText xmlCall

          _ -> msg

    -- Inject tool definitions into system prompt
    injectToolDefinitions :: [ModelConfig provider model] -> ConfigEncoder provider model
    injectToolDefinitions cfgs req =
      let toolDefs = concat [defs | Tools defs <- cfgs]
      in if null toolDefs
           then req
           else
             let toolPrompt = formatToolDefinitionsAsXML toolDefs
                 (sysMsgs, otherMsgs) = span isSystemMessage (messages req)
                 updatedSysMsg = case sysMsgs of
                   [] -> [OpenAIMessage "system" (Just toolPrompt) Nothing Nothing Nothing]
                   (OpenAIMessage msgRole (Just existingContent) rc tc tcid : rest) ->
                     OpenAIMessage msgRole (Just (existingContent <> "\n\n" <> toolPrompt)) rc tc tcid : rest
                   (first:rest) -> first : rest
             in req { messages = updatedSysMsg <> otherMsgs }
      where
        isSystemMessage (OpenAIMessage "system" _ _ _ _) = True
        isSystemMessage _ = False

    -- Parse XML from responses
    parseResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        [] -> Nothing
        (OpenAIChoice msg:_) ->
          case content msg of
            Nothing -> Nothing
            Just txt -> Just (AssistantText txt, OpenAISuccess (OpenAISuccessResponse []))
    parseResponse _ = Nothing

    -- Extract XML tool calls from parsed text messages
    extractXMLMessagesFromResponse :: [Message model provider] -> [Message model provider]
    extractXMLMessagesFromResponse acc =
      let (toolCalls, cleanedMsgs) = foldl processMessage ([], []) acc
      in cleanedMsgs <> toolCalls
      where
        processMessage (calls, msgs) (AssistantText txt) =
          let (toolCallMsgs, cleanedText) = parseXMLFromResponse txt
          in if null toolCallMsgs
             then (calls, msgs <> [AssistantText txt])
             else
               let cleanMsg = if T.null (T.strip cleanedText) then [] else [AssistantText cleanedText]
               in (calls <> toolCallMsgs, msgs <> cleanMsg)
        processMessage (calls, msgs) msg = (calls, msgs <> [msg])