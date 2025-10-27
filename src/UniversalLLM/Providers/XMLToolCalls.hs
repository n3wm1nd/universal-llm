{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE GADTs #-}

module UniversalLLM.Providers.XMLToolCalls
  ( -- * Composable Providers for XML Tool Calls
    withXMLToolCalls
  , withXMLToolCallsAndSystemPrompt
  ) where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import UniversalLLM.ToolCall.XML
import Data.Maybe (mapMaybe)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Autodocodec.Aeson.Compat as Compat

-- | Add XML tool call support to a provider (response parsing only)
-- This parses XML <tool_call> blocks from AssistantText responses and converts them to AssistantTool messages
-- Also converts ToolResultMsg to UserText when building requests
withXMLToolCalls :: forall provider model.
                    (HasTools model provider,
                     ProviderRequest provider ~ OpenAIRequest,
                     ProviderResponse provider ~ OpenAIResponse)
                 => ComposableProvider provider model
                 -> ComposableProvider provider model
withXMLToolCalls base = base `chainProviders` xmlProvider
  where
    xmlProvider = ComposableProvider
      { cpToRequest = handleXMLMessages
      , cpConfigHandler = \_provider _model _configs req -> req
      , cpFromResponse = parseXMLToolCalls
      }

    -- Convert AssistantTool and ToolResultMsg to XML format when building requests
    handleXMLMessages :: MessageHandler provider model
    handleXMLMessages _provider _model _configs msg req = case msg of
      -- Convert tool results to XML user messages
      ToolResultMsg result ->
        let xmlResult = toolResultToXML result
            xmlText = encodeXMLToolResult xmlResult
        in req { messages = messages req <> [OpenAIMessage "user" (Just xmlText) Nothing Nothing Nothing] }

      -- Convert tool calls back to XML in assistant messages
      AssistantTool toolCall ->
        let xmlCall = case toolCall of
              ToolCall _ tcName params ->
                -- Convert ToolCall to XMLToolCall, then to XML text
                let args = toolCallParamsToXMLArgs params
                    xmlToolCall = XMLToolCall tcName args
                in encodeXMLToolCall xmlToolCall
              InvalidToolCall _ tcName rawArgs _ ->
                -- Best effort: encode invalid call as text
                "<tool_call>" <> tcName <> "\n" <> rawArgs <> "\n</tool_call>"
        in req { messages = messages req <> [OpenAIMessage "assistant" (Just xmlCall) Nothing Nothing Nothing] }

      _ -> req

    -- Helper: Convert JSON params to XML arg pairs (simplified - just string conversion)
    toolCallParamsToXMLArgs :: Aeson.Value -> [XMLArgPair]
    toolCallParamsToXMLArgs (Aeson.Object obj) =
      [ XMLArgPair (Key.toText k) (valueToText v)
      | (k, v) <- Compat.toList obj
      ]
    toolCallParamsToXMLArgs _ = []

    valueToText :: Aeson.Value -> T.Text
    valueToText (Aeson.String s) = s
    valueToText v = T.pack (show v)  -- Fallback for non-string values

    -- Parse XML tool calls from AssistantText responses
    parseXMLToolCalls :: ResponseParser provider model
    parseXMLToolCalls _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case content msg of
            Nothing -> acc
            Just txt ->
              let -- Extract XML tool calls AND get cleaned text (single pass)
                  (xmlBlocks, cleanedText) = extractAndRemoveXMLToolCalls txt
                  -- Parse each block
                  parsedCalls = mapMaybe parseXMLToolCall xmlBlocks
                  -- Convert to ToolCall messages
                  toolCallMsgs = map (AssistantTool . xmlToolCallToToolCall) parsedCalls
              in if null toolCallMsgs
                   then acc  -- No tool calls, let base parser's text through
                   else
                     -- Remove any AssistantText from acc (added by base parser)
                     -- and replace with our cleaned version + tool calls
                     let accWithoutText = filter (not . isAssistantText) acc
                         textMsg = if T.null (T.strip cleanedText)
                                   then []
                                   else [AssistantText cleanedText]
                     in accWithoutText <> textMsg <> toolCallMsgs
    parseXMLToolCalls _provider _model _configs _history acc _ = acc

    isAssistantText :: Message model provider -> Bool
    isAssistantText (AssistantText _) = True
    isAssistantText _ = False

-- | Add XML tool call support with system prompt injection
-- This version also injects tool definitions into the system prompt
withXMLToolCallsAndSystemPrompt :: forall provider model.
                                   (HasTools model provider,
                                    ProviderRequest provider ~ OpenAIRequest,
                                    ProviderResponse provider ~ OpenAIResponse)
                                => ComposableProvider provider model
                                -> ComposableProvider provider model
withXMLToolCallsAndSystemPrompt base = base `chainProviders` xmlProviderWithPrompt
  where
    xmlProviderWithPrompt = ComposableProvider
      { cpToRequest = handleXMLMessages
      , cpConfigHandler = injectToolDefinitions
      , cpFromResponse = parseXMLToolCallsWithPrompt
      }

    -- Convert AssistantTool and ToolResultMsg to XML format when building requests
    handleXMLMessages :: MessageHandler provider model
    handleXMLMessages _provider _model _configs msg req = case msg of
      -- Convert tool results to XML user messages
      ToolResultMsg result ->
        let xmlResult = toolResultToXML result
            xmlText = encodeXMLToolResult xmlResult
        in req { messages = messages req <> [OpenAIMessage "user" (Just xmlText) Nothing Nothing Nothing] }

      -- Convert tool calls back to XML in assistant messages
      AssistantTool toolCall ->
        let xmlCall = case toolCall of
              ToolCall _ tcName params ->
                -- Convert ToolCall to XMLToolCall, then to XML text
                let args = toolCallParamsToXMLArgs params
                    xmlToolCall = XMLToolCall tcName args
                in encodeXMLToolCall xmlToolCall
              InvalidToolCall _ tcName rawArgs _ ->
                -- Best effort: encode invalid call as text
                "<tool_call>" <> tcName <> "\n" <> rawArgs <> "\n</tool_call>"
        in req { messages = messages req <> [OpenAIMessage "assistant" (Just xmlCall) Nothing Nothing Nothing] }

      _ -> req

    -- Helper: Convert JSON params to XML arg pairs
    toolCallParamsToXMLArgs :: Aeson.Value -> [XMLArgPair]
    toolCallParamsToXMLArgs (Aeson.Object obj) =
      [ XMLArgPair (Key.toText k) (valueToText v)
      | (k, v) <- Compat.toList obj
      ]
    toolCallParamsToXMLArgs _ = []

    valueToText :: Aeson.Value -> T.Text
    valueToText (Aeson.String s) = s
    valueToText v = T.pack (show v)  -- Fallback for non-string values

    -- Inject tool definitions into system prompt
    injectToolDefinitions :: ConfigHandler provider model
    injectToolDefinitions _provider _model configs req =
      let toolDefs = concat [defs | Tools defs <- configs]
      in if null toolDefs
           then req
           else
             let toolPrompt = formatToolDefinitionsAsXML toolDefs
                 -- Find existing system message or create new one
                 (sysMsgs, otherMsgs) = span isSystemMessage (messages req)
                 updatedSysMsg = case sysMsgs of
                   [] -> [OpenAIMessage "system" (Just toolPrompt) Nothing Nothing Nothing]
                   (OpenAIMessage msgRole (Just existingContent) rc tc tcid : rest) ->
                     OpenAIMessage msgRole (Just (existingContent <> "\n\n" <> toolPrompt)) rc tc tcid : rest
                   (first:rest) -> first : rest  -- Keep first as-is if no content
             in req { messages = updatedSysMsg <> otherMsgs }
      where
        isSystemMessage (OpenAIMessage "system" _ _ _ _) = True
        isSystemMessage _ = False

    -- Parse XML tool calls from AssistantText responses
    parseXMLToolCallsWithPrompt :: ResponseParser provider model
    parseXMLToolCallsWithPrompt _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case content msg of
            Nothing -> acc
            Just txt ->
              let -- Extract XML tool calls AND get cleaned text (single pass)
                  (xmlBlocks, cleanedText) = extractAndRemoveXMLToolCalls txt
                  -- Parse each block
                  parsedCalls = mapMaybe parseXMLToolCall xmlBlocks
                  -- Convert to ToolCall messages
                  toolCallMsgs = map (AssistantTool . xmlToolCallToToolCall) parsedCalls
              in if null toolCallMsgs
                   then acc  -- No tool calls, let base parser's text through
                   else
                     -- Remove any AssistantText from acc (added by base parser)
                     -- and replace with our cleaned version + tool calls
                     let accWithoutText = filter (not . isAssistantTextMsg) acc
                         textMsg = if T.null (T.strip cleanedText)
                                   then []
                                   else [AssistantText cleanedText]
                     in accWithoutText <> textMsg <> toolCallMsgs
    parseXMLToolCallsWithPrompt _provider _model _configs _history acc _ = acc

    isAssistantTextMsg :: Message model provider -> Bool
    isAssistantTextMsg (AssistantText _) = True
    isAssistantTextMsg _ = False
