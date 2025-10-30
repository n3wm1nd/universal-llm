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
import Data.Text (Text)
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BSL

-- OpenAI provider (official OpenAI API)
data OpenAI = OpenAI deriving (Show, Eq)

-- ============================================================================
-- Helper Functions for OpenAIRequest Manipulation
-- ============================================================================

-- | Modify messages in a request
modifyMessages :: ([OpenAIMessage] -> [OpenAIMessage]) -> OpenAIRequest -> OpenAIRequest
modifyMessages f req = req { messages = f (messages req) }

-- | Modify tool definitions in a request
modifyToolDefinitions :: (Maybe [OpenAIToolDefinition] -> Maybe [OpenAIToolDefinition])
                      -> OpenAIRequest -> OpenAIRequest
modifyToolDefinitions f req = req { tools = f (tools req) }

-- | Append a message to the request
appendMessage :: OpenAIMessage -> OpenAIRequest -> OpenAIRequest
appendMessage msg = modifyMessages (<> [msg])

-- | Modify the last message in a request (if it exists)
modifyLastMessage :: (OpenAIMessage -> OpenAIMessage) -> OpenAIRequest -> OpenAIRequest
modifyLastMessage f req = case messages req of
  [] -> req
  msgs -> req { messages = init msgs <> [f (last msgs)] }

-- | Set tool definitions (replaces existing)
setToolDefinitions :: [OpenAIToolDefinition] -> OpenAIRequest -> OpenAIRequest
setToolDefinitions defs = modifyToolDefinitions (const (Just defs))

-- | Set response format
setResponseFormat :: OpenAIResponseFormat -> OpenAIRequest -> OpenAIRequest
setResponseFormat fmt req = req { response_format = Just fmt }

-- ============================================================================
-- Data Conversion Functions
-- ============================================================================

-- | Create a user message with text content
userMessage :: Text -> OpenAIMessage
userMessage txt = OpenAIMessage "user" (Just txt) Nothing Nothing Nothing

-- | Create an assistant message with text content
assistantMessage :: Text -> OpenAIMessage
assistantMessage txt = OpenAIMessage "assistant" (Just txt) Nothing Nothing Nothing

-- | Create a system message with text content
systemMessage :: Text -> OpenAIMessage
systemMessage txt = OpenAIMessage "system" (Just txt) Nothing Nothing Nothing

-- | Create an assistant message with reasoning content
reasoningMessage :: Text -> OpenAIMessage
reasoningMessage txt = OpenAIMessage "assistant" Nothing (Just txt) Nothing Nothing

-- | Create an assistant message with tool calls
toolCallMessage :: [OpenAIToolCall] -> OpenAIMessage
toolCallMessage calls = OpenAIMessage "assistant" Nothing Nothing (Just calls) Nothing

-- | Create a tool result message
toolResultMessage :: Text -> Text -> OpenAIMessage
toolResultMessage tcId contentTxt = OpenAIMessage "tool" (Just contentTxt) Nothing Nothing (Just tcId)

-- | Append text to a message's content (only if roles match and message has text content)
appendToMessageIfSameRole :: Text -> Text -> OpenAIMessage -> Maybe OpenAIMessage
appendToMessageIfSameRole targetRole txt (OpenAIMessage msgRole (Just existingContent) Nothing Nothing Nothing)
  | msgRole == targetRole = Just $ OpenAIMessage msgRole (Just (existingContent <> "\n" <> txt)) Nothing Nothing Nothing
appendToMessageIfSameRole _ _ _ = Nothing

-- | Append a tool call to a message's tool calls
appendToolCallToMessage :: OpenAIToolCall -> OpenAIMessage -> Maybe OpenAIMessage
appendToolCallToMessage tc (OpenAIMessage "assistant" msgContent reasoning (Just existingCalls) tcid) =
  Just $ OpenAIMessage "assistant" msgContent reasoning (Just (existingCalls <> [tc])) tcid
appendToolCallToMessage _ _ = Nothing

-- ============================================================================
-- Message Handlers
-- ============================================================================

-- OpenAI-compatible providers
-- These all use OpenAI protocol but may have provider-specific quirks
data OpenAICompatible = OpenAICompatible deriving (Show, Eq)  -- Generic OpenAI-compatible
data OpenRouter = OpenRouter deriving (Show, Eq)              -- OpenRouter aggregator
data LlamaCpp = LlamaCpp deriving (Show, Eq)                  -- llama.cpp server
data Ollama = Ollama deriving (Show, Eq)                      -- Ollama
data VLLM = VLLM deriving (Show, Eq)                          -- vLLM
data LiteLLM = LiteLLM deriving (Show, Eq)                    -- LiteLLM proxy

-- Declare parameter support for all OpenAI-compatible providers
-- All support the same parameter set (temperature, max_tokens, seed, system_prompt, stop)
instance SupportsTemperature OpenAI
instance SupportsMaxTokens OpenAI
instance SupportsSeed OpenAI
instance SupportsSystemPrompt OpenAI
instance SupportsStop OpenAI

instance SupportsTemperature OpenAICompatible
instance SupportsMaxTokens OpenAICompatible
instance SupportsSeed OpenAICompatible
instance SupportsSystemPrompt OpenAICompatible
instance SupportsStop OpenAICompatible

instance SupportsTemperature OpenRouter
instance SupportsMaxTokens OpenRouter
instance SupportsSeed OpenRouter
instance SupportsSystemPrompt OpenRouter
instance SupportsStop OpenRouter

instance SupportsTemperature LlamaCpp
instance SupportsMaxTokens LlamaCpp
instance SupportsSeed LlamaCpp
instance SupportsSystemPrompt LlamaCpp
instance SupportsStop LlamaCpp

instance SupportsTemperature Ollama
instance SupportsMaxTokens Ollama
instance SupportsSeed Ollama
instance SupportsSystemPrompt Ollama
instance SupportsStop Ollama

instance SupportsTemperature VLLM
instance SupportsMaxTokens VLLM
instance SupportsSeed VLLM
instance SupportsSystemPrompt VLLM
instance SupportsStop VLLM

instance SupportsTemperature LiteLLM
instance SupportsMaxTokens LiteLLM
instance SupportsSeed LiteLLM
instance SupportsSystemPrompt LiteLLM
instance SupportsStop LiteLLM

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
      sysMessages = map systemMessage systemPrompts
  in modifyMessages (sysMessages <>) req

-- Basic text message handler
-- Polymorphic over any provider that uses OpenAI protocol
handleTextMessages :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleTextMessages =  \_provider _model _configs msg req -> case msg of
  UserText txt ->
    case lastMessage req >>= appendToMessageIfSameRole "user" txt of
      Just updatedMsg -> modifyLastMessage (const updatedMsg) req
      Nothing -> appendMessage (userMessage txt) req
  AssistantText txt ->
    case lastMessage req >>= appendToMessageIfSameRole "assistant" txt of
      Just updatedMsg -> modifyLastMessage (const updatedMsg) req
      Nothing -> appendMessage (assistantMessage txt) req
  SystemText txt ->
    appendMessage (systemMessage txt) req
  _ -> req  -- Not a text message

-- Tools handler
-- Polymorphic over any provider that uses OpenAI protocol
handleTools :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleTools =  \_provider _model configs msg req -> case msg of
  AssistantTool toolCall ->
    let tc = convertFromToolCall toolCall
    in case lastMessage req >>= appendToolCallToMessage tc of
      Just updatedMsg -> modifyLastMessage (const updatedMsg) req
      Nothing -> appendMessage (toolCallMessage [tc]) req
  ToolResultMsg result ->
    let resultCallId = getToolCallId (toolResultCall result)
        resultContent = either id (TE.decodeUtf8 . BSL.toStrict . Aeson.encode) $ toolResultOutput result
    in appendMessage (toolResultMessage resultCallId resultContent) req
  _ ->
    -- Apply tool definitions from config (for any message)
    let toolDefs = [defs | Tools defs <- configs]
    in if null toolDefs
       then req
       else setToolDefinitions (map toOpenAIToolDef (concat toolDefs)) req

-- JSON handler
-- Polymorphic over any provider that uses OpenAI protocol
handleJSON :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleJSON =  \_provider _model _configs msg req -> case msg of
  UserRequestJSON txt schema ->
    appendMessage (userMessage txt) req
      & setResponseFormat (OpenAIResponseFormat "json_schema" (Just schema))
  AssistantJSON jsonVal ->
    let jsonText = TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ jsonVal
    in appendMessage (assistantMessage jsonText) req
  _ -> req
  where
    (&) = flip ($)

-- Reasoning handler
-- Polymorphic over any provider that uses OpenAI protocol
handleReasoning :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageHandler provider model
handleReasoning =  \_provider _model _configs msg req -> case msg of
  AssistantReasoning txt -> appendMessage (reasoningMessage txt) req
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

-- ============================================================================
-- Completion Interface (Legacy /v1/completions endpoint)
-- ============================================================================

-- All OpenAI-compatible providers use the same completion request/response types
instance Provider OpenAI model => CompletionProvider OpenAI model where
  type CompletionRequest OpenAI = OpenAICompletionRequest
  type CompletionResponse OpenAI = OpenAICompletionResponse

instance Provider OpenAICompatible model => CompletionProvider OpenAICompatible model where
  type CompletionRequest OpenAICompatible = OpenAICompletionRequest
  type CompletionResponse OpenAICompatible = OpenAICompletionResponse

instance Provider OpenRouter model => CompletionProvider OpenRouter model where
  type CompletionRequest OpenRouter = OpenAICompletionRequest
  type CompletionResponse OpenRouter = OpenAICompletionResponse

instance Provider LlamaCpp model => CompletionProvider LlamaCpp model where
  type CompletionRequest LlamaCpp = OpenAICompletionRequest
  type CompletionResponse LlamaCpp = OpenAICompletionResponse

instance Provider Ollama model => CompletionProvider Ollama model where
  type CompletionRequest Ollama = OpenAICompletionRequest
  type CompletionResponse Ollama = OpenAICompletionResponse

instance Provider VLLM model => CompletionProvider VLLM model where
  type CompletionRequest VLLM = OpenAICompletionRequest
  type CompletionResponse VLLM = OpenAICompletionResponse

instance Provider LiteLLM model => CompletionProvider LiteLLM model where
  type CompletionRequest LiteLLM = OpenAICompletionRequest
  type CompletionResponse LiteLLM = OpenAICompletionResponse

-- Helper: Modify completion request fields
modifyCompletionRequest :: (OpenAICompletionRequest -> OpenAICompletionRequest)
                        -> OpenAICompletionRequest
                        -> OpenAICompletionRequest
modifyCompletionRequest = id

-- Prompt handler: Set the prompt and model name
handlePrompt :: forall provider model.
                (
                 CompletionRequest provider ~ OpenAICompletionRequest,
                 ModelName provider model)
             =>PromptHandler provider model
handlePrompt _provider mdl _configs prmpt req =
  req { completionModel = modelName @provider mdl
      , prompt = prmpt
      }

-- Config handler: Apply temperature, max_tokens, and stop sequences
configureCompletion :: forall provider model.
                       (CompletionRequest provider ~ OpenAICompletionRequest)
                    =>CompletionConfigHandler provider model
configureCompletion _provider _model configs req = foldl applyConfig req configs
  where
    applyConfig :: OpenAICompletionRequest -> ModelConfig provider model -> OpenAICompletionRequest
    applyConfig r (Temperature temp) = r { completionTemperature = Just temp }
    applyConfig r (MaxTokens maxTok) = r { completionMaxTokens = Just maxTok }
    applyConfig r (Stop stopSeqs) = r { stop = Just stopSeqs }
    applyConfig r _ = r  -- Other configs don't apply to completions

-- Response parser: Extract text from first completion choice
parseCompletionResponse :: forall provider model.
                           (CompletionResponse provider ~ OpenAICompletionResponse)
                        =>CompletionParser provider model
parseCompletionResponse _provider _model _configs _prompt resp =
  case resp of
    OpenAICompletionSuccess successResp ->
      case completionChoices successResp of
        [] -> ""  -- No choices returned
        (choice:_) -> completionText choice  -- Return first choice
    OpenAICompletionError _err -> ""  -- Return empty on error (could be improved)

-- Base completion provider
baseCompletionProvider :: forall provider model.
                          (CompletionRequest provider ~ OpenAICompletionRequest,
                           CompletionResponse provider ~ OpenAICompletionResponse,
                           ModelName provider model)
                       =>ComposableCompletionProvider provider model
baseCompletionProvider = ComposableCompletionProvider
  { ccpToRequest = handlePrompt @provider @model
  , ccpConfigHandler = configureCompletion @provider @model
  , ccpFromResponse = parseCompletionResponse @provider @model
  }

-- Default CompletionProviderImplementation for all models
instance {-# OVERLAPPABLE #-} ModelName OpenAI model => CompletionProviderImplementation OpenAI model where
  getComposableCompletionProvider = baseCompletionProvider

instance {-# OVERLAPPABLE #-} ModelName OpenAICompatible model => CompletionProviderImplementation OpenAICompatible model where
  getComposableCompletionProvider = baseCompletionProvider

instance {-# OVERLAPPABLE #-} ModelName OpenRouter model => CompletionProviderImplementation OpenRouter model where
  getComposableCompletionProvider = baseCompletionProvider

instance {-# OVERLAPPABLE #-} ModelName LlamaCpp model => CompletionProviderImplementation LlamaCpp model where
  getComposableCompletionProvider = baseCompletionProvider

instance {-# OVERLAPPABLE #-} ModelName Ollama model => CompletionProviderImplementation Ollama model where
  getComposableCompletionProvider = baseCompletionProvider

instance {-# OVERLAPPABLE #-} ModelName VLLM model => CompletionProviderImplementation VLLM model where
  getComposableCompletionProvider = baseCompletionProvider

instance {-# OVERLAPPABLE #-} ModelName LiteLLM model => CompletionProviderImplementation LiteLLM model where
  getComposableCompletionProvider = baseCompletionProvider