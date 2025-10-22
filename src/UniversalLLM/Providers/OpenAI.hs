{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}

module UniversalLLM.Providers.OpenAI where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL

-- OpenAI provider (phantom type)
data OpenAI = OpenAI deriving (Show, Eq)

-- Declare OpenAI parameter support
instance SupportsTemperature OpenAI
instance SupportsMaxTokens OpenAI
instance SupportsSeed OpenAI
instance SupportsSystemPrompt OpenAI

-- OpenAI capabilities are now declared per-model (see model files)

instance Provider OpenAI model where
  type ProviderRequest OpenAI = OpenAIRequest
  type ProviderResponse OpenAI = OpenAIResponse 

-- Helper: Get last message from request
lastMessage :: OpenAIRequest -> Maybe OpenAIMessage
lastMessage req = case messages req of
  [] -> Nothing
  msgs -> Just (last msgs)

-- Helper: Modify last message in request
modifyLastMessage :: OpenAIRequest -> (OpenAIMessage -> OpenAIMessage) -> OpenAIRequest
modifyLastMessage req f = case messages req of
  [] -> req
  msgs -> req { messages = init msgs <> [f (last msgs)] }

-- Base handler: model name and basic config
-- Updates the request with model name and config
handleBase :: ModelName OpenAI model => MessageHandler OpenAI model
handleBase _provider model configs _msg req =
  req { model = modelName @OpenAI model
      , temperature = getFirst [t | Temperature t <- configs]
      , max_tokens = getFirst [mt | MaxTokens mt <- configs]
      , seed = getFirst [s | Seed s <- configs]
      }
  where
    getFirst [] = Nothing
    getFirst (x:_) = Just x

-- System prompt handler (from config)
handleSystemPrompt :: MessageHandler OpenAI model
handleSystemPrompt =  \_provider _model configs _msg req ->
  let systemPrompts = [sp | SystemPrompt sp <- configs]
      sysMessages = [OpenAIMessage "system" (Just sp) Nothing Nothing Nothing | sp <- systemPrompts]
  in req { messages = sysMessages <> messages req }

-- Basic text message handler
handleTextMessages :: MessageHandler OpenAI model
handleTextMessages =  \_provider _model _configs msg req -> case msg of
  UserText txt ->
    case lastMessage req of
      Just (OpenAIMessage "user" (Just existingContent) Nothing Nothing Nothing) ->
        -- Append to existing user message (merge consecutive user messages)
        modifyLastMessage req $ \msg -> msg { content = Just (existingContent <> "\n" <> txt) }
      _ ->
        -- Create new user message
        req { messages = messages req <> [OpenAIMessage "user" (Just txt) Nothing Nothing Nothing] }
  AssistantText txt ->
    case lastMessage req of
      Just (OpenAIMessage "assistant" (Just existingContent) Nothing Nothing Nothing) ->
        -- Append to existing assistant message (only if no tool calls)
        modifyLastMessage req $ \msg -> msg { content = Just (existingContent <> "\n" <> txt) }
      _ ->
        -- Create new assistant message
        req { messages = messages req <> [OpenAIMessage "assistant" (Just txt) Nothing Nothing Nothing] }
  SystemText txt ->
    req { messages = messages req <> [OpenAIMessage "system" (Just txt) Nothing Nothing Nothing] }
  _ -> req  -- Not a text message

-- Tools handler
handleTools :: MessageHandler OpenAI model
handleTools =  \_provider _model configs msg req -> case msg of
  AssistantTool call ->
    case lastMessage req of
      Just (OpenAIMessage "assistant" _ _ (Just existingCalls) _) ->
        -- Append to existing tool calls
        modifyLastMessage req $ \msg ->
          msg { tool_calls = Just (existingCalls <> [convertFromToolCall call]) }
      _ ->
        -- Create new assistant message with tool call
        req { messages = messages req <> [OpenAIMessage "assistant" Nothing Nothing (Just [convertFromToolCall call]) Nothing] }
  ToolResultMsg result ->
    let resultCallId = getToolCallId (toolResultCall result)
        resultContent = either id (TE.decodeUtf8 . BSL.toStrict . Aeson.encode) $ toolResultOutput result
    in req { messages = messages req <> [OpenAIMessage "tool" (Just resultContent) Nothing Nothing (Just resultCallId)] }
  _ ->
    -- Apply tool definitions from config (for any message)
    let toolDefs = [defs | Tools defs <- configs]
    in if null toolDefs
       then req
       else req { tools = Just (map toOpenAIToolDef (concat toolDefs)) }

-- JSON handler
handleJSON :: MessageHandler OpenAI model
handleJSON =  \_provider _model _configs msg req -> case msg of
  UserRequestJSON txt schema ->
    req { messages = messages req <> [OpenAIMessage "user" (Just txt) Nothing Nothing Nothing]
        , response_format = Just $ OpenAIResponseFormat "json_schema" (Just schema)
        }
  AssistantJSON jsonVal ->
    let jsonText = TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ jsonVal
    in req { messages = messages req <> [OpenAIMessage "assistant" (Just jsonText) Nothing Nothing Nothing] }
  _ -> req

-- Reasoning handler
handleReasoning :: MessageHandler OpenAI model
handleReasoning =  \_provider _model _configs msg req -> case msg of
  AssistantReasoning txt ->
    req { messages = messages req <> [OpenAIMessage "assistant" Nothing (Just txt) Nothing Nothing] }
  _ -> req

-- Composable providers (bidirectional handlers)

-- Base composable provider: model name, basic config, text messages
baseComposableProvider :: forall model. ModelName OpenAI model => ComposableProvider OpenAI model
baseComposableProvider = ComposableProvider
  { cpToRequest = handleBase >>> handleSystemPrompt >>> handleTextMessages
  , cpFromResponse = parseTextResponse
  }
  where
    parseTextResponse _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse choices)) =
      case choices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case content msg of
            Just txt -> acc <> [AssistantText txt]
            Nothing -> acc
    parseTextResponse _provider _model _configs _history acc _ = acc

-- Reasoning composable provider
reasoningComposableProvider :: forall model. HasReasoning model OpenAI => ComposableProvider OpenAI model
reasoningComposableProvider = ComposableProvider
  { cpToRequest = UniversalLLM.Providers.OpenAI.handleReasoning
  , cpFromResponse = parseReasoningResponse
  }
  where
    parseReasoningResponse _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse choices)) =
      case choices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case reasoning_content msg of
            Just reasoningTxt -> acc <> [AssistantReasoning reasoningTxt]
            Nothing -> acc
    parseReasoningResponse _provider _model _configs _history acc _ = acc

-- Tools composable provider
toolsComposableProvider :: forall model. HasTools model OpenAI => ComposableProvider OpenAI model
toolsComposableProvider = ComposableProvider
  { cpToRequest = handleTools
  , cpFromResponse = parseToolResponse
  }
  where
    parseToolResponse _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse choices)) =
      case choices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case tool_calls msg of
            Just calls -> acc <> map (AssistantTool . convertToolCall) calls
            Nothing -> acc
    parseToolResponse _provider _model _configs _history acc _ = acc

-- JSON composable provider
-- Transforms AssistantText messages that contain valid JSON into AssistantJSON
-- Only transforms if JSON was explicitly requested (checks response format in request)
jsonComposableProvider :: forall model. HasJSON model OpenAI => ComposableProvider OpenAI model
jsonComposableProvider = ComposableProvider
  { cpToRequest = handleJSON
  , cpFromResponse = parseJSONResponse
  }
  where
    parseJSONResponse _provider _model _configs history acc resp =
      -- Only transform to JSON if:
      -- 1. The last user message explicitly requested JSON mode (UserRequestJSON), AND
      -- 2. Response indicates success (not an error), AND
      -- 3. The text actually parses as valid JSON
      if lastMessageRequestedJSON history && isSuccessResponse resp
        then map transformTextToJSON acc
        else acc

    lastMessageRequestedJSON :: [Message model OpenAI] -> Bool
    lastMessageRequestedJSON msgs =
      case lastUserMessage msgs of
        Just (UserRequestJSON _ _) -> True
        _ -> False
      where
        -- Get the last user message (ignoring assistant messages)
        lastUserMessage [] = Nothing
        lastUserMessage [msg] = if isUserMessage msg then Just msg else Nothing
        lastUserMessage (msg:rest) =
          case lastUserMessage rest of
            Nothing -> if isUserMessage msg then Just msg else Nothing
            found -> found

        isUserMessage (UserText _) = True
        isUserMessage (UserImage _ _) = True
        isUserMessage (UserRequestJSON _ _) = True
        isUserMessage _ = False

    isSuccessResponse :: OpenAIResponse -> Bool
    isSuccessResponse (OpenAISuccess _) = True
    isSuccessResponse (OpenAIError _) = False

    transformTextToJSON :: Message model OpenAI -> Message model OpenAI
    transformTextToJSON (AssistantText txt) =
      -- Only transform if it's valid JSON and not empty
      case Aeson.eitherDecodeStrict (TE.encodeUtf8 txt) of
        Right jsonVal ->
          -- Verify it's actually structured data, not just a JSON string literal
          case jsonVal of
            Aeson.Object _ -> AssistantJSON jsonVal
            Aeson.Array _ -> AssistantJSON jsonVal
            Aeson.Null -> AssistantJSON jsonVal
            Aeson.Number _ -> AssistantJSON jsonVal
            Aeson.Bool _ -> AssistantJSON jsonVal
            Aeson.String _ -> AssistantJSON jsonVal  -- Even plain strings are valid JSON responses
        Left _ -> AssistantText txt  -- Keep as text if parsing fails
    transformTextToJSON other = other

-- Default ProviderImplementation for basic text-only models
-- Models with capabilities (tools, json, reasoning, etc.) should provide their own instances
instance {-# OVERLAPPABLE #-} ModelName OpenAI model => ProviderImplementation OpenAI model where
  getComposableProvider = baseComposableProvider