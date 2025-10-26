{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}

module UniversalLLM.Providers.OpenAI where

import UniversalLLM.Core.Types
import UniversalLLM.Protocols.OpenAI
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL

-- OpenAI provider (official OpenAI API)
data OpenAI = OpenAI deriving (Show, Eq)

-- OpenAI-compatible providers
-- These all use OpenAI protocol but may have provider-specific quirks
data OpenAICompatible = OpenAICompatible deriving (Show, Eq)  -- Generic OpenAI-compatible
data OpenRouter = OpenRouter deriving (Show, Eq)              -- OpenRouter aggregator
data LlamaCpp = LlamaCpp deriving (Show, Eq)                  -- llama.cpp server
data Ollama = Ollama deriving (Show, Eq)                      -- Ollama
data VLLM = VLLM deriving (Show, Eq)                          -- vLLM
data LiteLLM = LiteLLM deriving (Show, Eq)                    -- LiteLLM proxy

-- Declare parameter support for all OpenAI-compatible providers
-- All support the same parameter set (temperature, max_tokens, seed, system_prompt)
instance SupportsTemperature OpenAI
instance SupportsMaxTokens OpenAI
instance SupportsSeed OpenAI
instance SupportsSystemPrompt OpenAI

instance SupportsTemperature OpenAICompatible
instance SupportsMaxTokens OpenAICompatible
instance SupportsSeed OpenAICompatible
instance SupportsSystemPrompt OpenAICompatible

instance SupportsTemperature OpenRouter
instance SupportsMaxTokens OpenRouter
instance SupportsSeed OpenRouter
instance SupportsSystemPrompt OpenRouter

instance SupportsTemperature LlamaCpp
instance SupportsMaxTokens LlamaCpp
instance SupportsSeed LlamaCpp
instance SupportsSystemPrompt LlamaCpp

instance SupportsTemperature Ollama
instance SupportsMaxTokens Ollama
instance SupportsSeed Ollama
instance SupportsSystemPrompt Ollama

instance SupportsTemperature VLLM
instance SupportsMaxTokens VLLM
instance SupportsSeed VLLM
instance SupportsSystemPrompt VLLM

instance SupportsTemperature LiteLLM
instance SupportsMaxTokens LiteLLM
instance SupportsSeed LiteLLM
instance SupportsSystemPrompt LiteLLM

-- OpenAI capabilities are now declared per-model (see model files)

-- All OpenAI-compatible providers use the same request/response types
instance Provider OpenAI model where
  type ProviderRequest OpenAI = OpenAIRequest
  type ProviderResponse OpenAI = OpenAIResponse

instance Provider OpenAICompatible model where
  type ProviderRequest OpenAICompatible = OpenAIRequest
  type ProviderResponse OpenAICompatible = OpenAIResponse

instance Provider OpenRouter model where
  type ProviderRequest OpenRouter = OpenAIRequest
  type ProviderResponse OpenRouter = OpenAIResponse

instance Provider LlamaCpp model where
  type ProviderRequest LlamaCpp = OpenAIRequest
  type ProviderResponse LlamaCpp = OpenAIResponse

instance Provider Ollama model where
  type ProviderRequest Ollama = OpenAIRequest
  type ProviderResponse Ollama = OpenAIResponse

instance Provider VLLM model where
  type ProviderRequest VLLM = OpenAIRequest
  type ProviderResponse VLLM = OpenAIResponse

instance Provider LiteLLM model where
  type ProviderRequest LiteLLM = OpenAIRequest
  type ProviderResponse LiteLLM = OpenAIResponse 

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
-- Polymorphic over any provider that uses OpenAI protocol
handleBase :: forall provider model. (ModelName provider model, ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleBase _provider modelType configs _msg req =
  req { model = modelName @provider modelType
      , temperature = getFirst [t | Temperature t <- configs]
      , max_tokens = getFirst [mt | MaxTokens mt <- configs]
      , seed = getFirst [s | Seed s <- configs]
      }
  where
    getFirst [] = Nothing
    getFirst (x:_) = Just x

-- System prompt config handler (from config)
-- Polymorphic over any provider that uses OpenAI protocol
-- This is a ConfigHandler, not a MessageHandler - it runs after all messages are processed
configureSystemPrompt :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => ConfigHandler provider model
configureSystemPrompt = \_provider _model configs req ->
  let systemPrompts = [sp | SystemPrompt sp <- configs]
      sysMessages = [OpenAIMessage "system" (Just sp) Nothing Nothing Nothing | sp <- systemPrompts]
  in req { messages = sysMessages <> messages req }

-- Basic text message handler
-- Polymorphic over any provider that uses OpenAI protocol
handleTextMessages :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleTextMessages =  \_provider _model _configs msg req -> case msg of
  UserText txt ->
    case lastMessage req of
      Just (OpenAIMessage "user" (Just existingContent) Nothing Nothing Nothing) ->
        -- Append to existing user message (merge consecutive user messages)
        modifyLastMessage req $ \m -> m { content = Just (existingContent <> "\n" <> txt) }
      _ ->
        -- Create new user message
        req { messages = messages req <> [OpenAIMessage "user" (Just txt) Nothing Nothing Nothing] }
  AssistantText txt ->
    case lastMessage req of
      Just (OpenAIMessage "assistant" (Just existingContent) Nothing Nothing Nothing) ->
        -- Append to existing assistant message (only if no tool calls)
        modifyLastMessage req $ \m -> m { content = Just (existingContent <> "\n" <> txt) }
      _ ->
        -- Create new assistant message
        req { messages = messages req <> [OpenAIMessage "assistant" (Just txt) Nothing Nothing Nothing] }
  SystemText txt ->
    req { messages = messages req <> [OpenAIMessage "system" (Just txt) Nothing Nothing Nothing] }
  _ -> req  -- Not a text message

-- Tools handler
-- Polymorphic over any provider that uses OpenAI protocol
handleTools :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleTools =  \_provider _model configs msg req -> case msg of
  AssistantTool toolCall ->
    case lastMessage req of
      Just (OpenAIMessage "assistant" _ _ (Just existingCalls) _) ->
        -- Append to existing tool calls
        modifyLastMessage req $ \m ->
          m { tool_calls = Just (existingCalls <> [convertFromToolCall toolCall]) }
      _ ->
        -- Create new assistant message with tool call
        req { messages = messages req <> [OpenAIMessage "assistant" Nothing Nothing (Just [convertFromToolCall toolCall]) Nothing] }
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
-- Polymorphic over any provider that uses OpenAI protocol
handleJSON :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
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
-- Polymorphic over any provider that uses OpenAI protocol
handleReasoning :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleReasoning =  \_provider _model _configs msg req -> case msg of
  AssistantReasoning txt ->
    req { messages = messages req <> [OpenAIMessage "assistant" Nothing (Just txt) Nothing Nothing] }
  _ -> req

-- Composable providers (bidirectional handlers)

-- Base composable provider: model name, basic config, text messages
-- Polymorphic over any provider that uses OpenAI protocol
baseComposableProvider :: forall provider model. (ModelName provider model, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model
baseComposableProvider = ComposableProvider
  { cpToRequest = handleBase >>> handleTextMessages
  , cpConfigHandler = configureSystemPrompt
  , cpFromResponse = parseTextResponse
  }
  where
    parseTextResponse _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case content msg of
            Just txt -> acc <> [AssistantText txt]
            Nothing -> acc
    parseTextResponse _provider _model _configs _history acc _ = acc

-- Reasoning capability combinator
-- Polymorphic over any provider that uses OpenAI protocol
openAIWithReasoning :: forall provider model. (HasReasoning model provider, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model -> ComposableProvider provider model
openAIWithReasoning base = base `chainProviders` reasoningProvider
  where
    reasoningProvider = ComposableProvider
      { cpToRequest = UniversalLLM.Providers.OpenAI.handleReasoning
      , cpConfigHandler = \_provider _model _configs req -> req  -- No config handling needed
      , cpFromResponse = parseReasoningResponse
      }
    parseReasoningResponse _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case reasoning_content msg of
            Just reasoningTxt -> acc <> [AssistantReasoning reasoningTxt]
            Nothing -> acc
    parseReasoningResponse _provider _model _configs _history acc _ = acc

-- Tools capability combinator
-- Polymorphic over any provider that uses OpenAI protocol
openAIWithTools :: forall provider model. (HasTools model provider, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model -> ComposableProvider provider model
openAIWithTools base = base `chainProviders` toolsProvider
  where
    toolsProvider = ComposableProvider
      { cpToRequest = handleTools
      , cpConfigHandler = \_provider _model _configs req -> req  -- No config handling needed
      , cpFromResponse = parseToolResponse
      }
    parseToolResponse _provider _model _configs _history acc (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        [] -> acc
        (OpenAIChoice msg:_) ->
          case tool_calls msg of
            Just calls -> acc <> map (AssistantTool . convertToolCall) calls
            Nothing -> acc
    parseToolResponse _provider _model _configs _history acc _ = acc

-- JSON capability combinator
-- Transforms AssistantText messages that contain valid JSON into AssistantJSON
-- Only transforms if JSON was explicitly requested (checks response format in request)
-- Polymorphic over any provider that uses OpenAI protocol
openAIWithJSON :: forall provider model. (HasJSON model provider, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model -> ComposableProvider provider model
openAIWithJSON base = base `chainProviders` jsonProvider
  where
    jsonProvider = ComposableProvider
      { cpToRequest = handleJSON
      , cpConfigHandler = \_provider _model _configs req -> req  -- No config handling needed
      , cpFromResponse = parseJSONResponse
      }

    parseJSONResponse _provider _model _configs history acc resp =
      -- Only transform to JSON if:
      -- 1. The last user message explicitly requested JSON mode (UserRequestJSON), AND
      -- 2. Response indicates success (not an error), AND
      -- 3. The text actually parses as valid JSON
      if lastMessageRequestedJSON history && isSuccessResponse resp
        then map transformTextToJSON acc
        else acc

    lastMessageRequestedJSON :: [Message model provider] -> Bool
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

    transformTextToJSON :: Message model provider -> Message model provider
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