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

-- Declare OpenAI capabilities
instance HasTools OpenAI
instance HasJSON OpenAI
instance HasReasoning OpenAI
instance SupportsTemperature OpenAI
instance SupportsMaxTokens OpenAI
instance SupportsSeed OpenAI
instance SupportsSystemPrompt OpenAI

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
handleBase :: ModelName OpenAI model => MessageHandler OpenAI model
handleBase = MessageHandler $ \_provider model configs _msg req ->
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
handleSystemPrompt = MessageHandler $ \_provider _model configs _msg req ->
  let systemPrompts = [sp | SystemPrompt sp <- configs]
      sysMessages = [OpenAIMessage "system" (Just sp) Nothing Nothing Nothing | sp <- systemPrompts]
  in req { messages = sysMessages <> messages req }

-- Basic text message handler
handleTextMessages :: MessageHandler OpenAI model
handleTextMessages = MessageHandler $ \_provider _model _configs msg req -> case msg of
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
handleTools :: HasTools model => MessageHandler OpenAI model
handleTools = MessageHandler $ \_provider _model configs msg req -> case msg of
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
handleJSON :: (HasJSON model, HasJSON OpenAI) => MessageHandler OpenAI model
handleJSON = MessageHandler $ \_provider _model _configs msg req -> case msg of
  UserRequestJSON txt schema ->
    req { messages = messages req <> [OpenAIMessage "user" (Just txt) Nothing Nothing Nothing]
        , response_format = Just $ OpenAIResponseFormat "json_schema" (Just schema)
        }
  AssistantJSON jsonVal ->
    let jsonText = TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ jsonVal
    in req { messages = messages req <> [OpenAIMessage "assistant" (Just jsonText) Nothing Nothing Nothing] }
  _ -> req

-- Reasoning handler
handleReasoning :: (HasReasoning model, HasReasoning OpenAI) => MessageHandler OpenAI model
handleReasoning = MessageHandler $ \_provider _model _configs msg req -> case msg of
  AssistantReasoning txt ->
    req { messages = messages req <> [OpenAIMessage "assistant" Nothing (Just txt) Nothing Nothing] }
  _ -> req

-- Composable providers (bidirectional handlers)

-- Base composable provider: model name, basic config, text messages
baseComposableProvider :: forall model. ModelName OpenAI model => ComposableProvider OpenAI model
baseComposableProvider = ComposableProvider
  { cpToRequest = handleBase <> handleSystemPrompt <> handleTextMessages
  , cpFromResponse = parseTextResponse
  }
  where
    parseTextResponse acc (OpenAISuccess (OpenAISuccessResponse choices)) =
      case choices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case content msg of
            Just txt -> acc <> [AssistantText txt]
            Nothing -> acc
    parseTextResponse acc _ = acc

-- Reasoning composable provider
reasoningComposableProvider :: forall model. (HasReasoning model, HasReasoning OpenAI) => ComposableProvider OpenAI model
reasoningComposableProvider = ComposableProvider
  { cpToRequest = UniversalLLM.Providers.OpenAI.handleReasoning
  , cpFromResponse = \acc _resp -> acc -- FIXME: this needs to implement putting the thinking/reasoning response into AssistantReasoning message
  }

-- Tools composable provider
toolsComposableProvider :: forall model. (HasTools model, HasTools OpenAI) => ComposableProvider OpenAI model
toolsComposableProvider = ComposableProvider
  { cpToRequest = handleTools
  , cpFromResponse = parseToolResponse
  }
  where
    parseToolResponse acc (OpenAISuccess (OpenAISuccessResponse choices)) =
      case choices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case tool_calls msg of
            Just calls -> acc <> map (AssistantTool . convertToolCall) calls
            Nothing -> acc
    parseToolResponse acc _ = acc

-- JSON composable provider
-- Transforms AssistantText messages that contain valid JSON into AssistantJSON
jsonComposableProvider :: forall model. (HasJSON model, HasJSON OpenAI) => ComposableProvider OpenAI model
jsonComposableProvider = ComposableProvider
  { cpToRequest = handleJSON
  , cpFromResponse = parseJSONResponse
  }
  where
    parseJSONResponse acc _resp = map transformTextToJSON acc

    transformTextToJSON :: Message OpenAI model -> Message OpenAI model
    transformTextToJSON (AssistantText txt) =
      case Aeson.eitherDecodeStrict (TE.encodeUtf8 txt) of
        Right jsonVal -> AssistantJSON jsonVal
        Left _ -> AssistantText txt
    transformTextToJSON other = other

-- Full composable provider for models with tools and JSON (like GPT4o)
fullComposableProvider :: forall model. (ModelName OpenAI model, HasTools model, HasJSON model) => ComposableProvider OpenAI model
fullComposableProvider = baseComposableProvider <> toolsComposableProvider <> jsonComposableProvider