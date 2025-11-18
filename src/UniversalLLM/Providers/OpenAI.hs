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
{-# LANGUAGE InstanceSigs #-}

module UniversalLLM.Providers.OpenAI where

import UniversalLLM.Core.Types
import UniversalLLM.Core.Serialization
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
instance SupportsStreaming OpenAI

instance SupportsTemperature OpenAICompatible
instance SupportsMaxTokens OpenAICompatible
instance SupportsSeed OpenAICompatible
instance SupportsSystemPrompt OpenAICompatible
instance SupportsStop OpenAICompatible
instance SupportsStreaming OpenAICompatible

instance SupportsTemperature OpenRouter
instance SupportsMaxTokens OpenRouter
instance SupportsSeed OpenRouter
instance SupportsSystemPrompt OpenRouter
instance SupportsStop OpenRouter
instance SupportsStreaming OpenRouter

instance SupportsTemperature LlamaCpp
instance SupportsMaxTokens LlamaCpp
instance SupportsSeed LlamaCpp
instance SupportsSystemPrompt LlamaCpp
instance SupportsStop LlamaCpp
instance SupportsStreaming LlamaCpp

instance SupportsTemperature Ollama
instance SupportsMaxTokens Ollama
instance SupportsSeed Ollama
instance SupportsSystemPrompt Ollama
instance SupportsStop Ollama
instance SupportsStreaming Ollama

instance SupportsTemperature VLLM
instance SupportsMaxTokens VLLM
instance SupportsSeed VLLM
instance SupportsSystemPrompt VLLM
instance SupportsStop VLLM
instance SupportsStreaming VLLM

instance SupportsTemperature LiteLLM
instance SupportsMaxTokens LiteLLM
instance SupportsSeed LiteLLM
instance SupportsSystemPrompt LiteLLM
instance SupportsStop LiteLLM
instance SupportsStreaming LiteLLM

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

-- Text message encoder
handleTextMessage :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageEncoder provider model
handleTextMessage msg req = case msg of
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

-- Tool message encoder
handleToolMessage :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageEncoder provider model
handleToolMessage msg req = case msg of
  AssistantTool toolCall ->
    let tc = convertFromToolCall toolCall
    in case lastMessage req >>= appendToolCallToMessage tc of
      Just updatedMsg -> modifyLastMessage (const updatedMsg) req
      Nothing -> appendMessage (toolCallMessage [tc]) req
  ToolResultMsg result ->
    let resultCallId = getToolCallId (toolResultCall result)
        resultContent = either id (TE.decodeUtf8 . BSL.toStrict . Aeson.encode) $ toolResultOutput result
    in appendMessage (toolResultMessage resultCallId resultContent) req
  _ -> req

-- JSON message encoder
handleJSONMessage :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageEncoder provider model
handleJSONMessage msg req = case msg of
  UserRequestJSON txt schema ->
    appendMessage (userMessage txt) req
      & setResponseFormat (OpenAIResponseFormat "json_schema" (Just schema))
  AssistantJSON jsonVal ->
    let jsonText = TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ jsonVal
    in appendMessage (assistantMessage jsonText) req
  _ -> req
  where
    (&) = flip ($)

-- Reasoning message encoder
handleReasoningMessage :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => MessageEncoder provider model
handleReasoningMessage msg req = case msg of
  AssistantReasoning txt -> appendMessage (reasoningMessage txt) req
  _ -> req

-- Composable providers

-- Base composable provider: model name, basic config, text messages
baseComposableProvider :: forall provider model. (ModelName provider model, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model ()
baseComposableProvider _p m configs _s = noopHandler
  { cpToRequest = \msg req ->
      let req' = req { model = modelName @provider m
                     , temperature = getFirst [t | Temperature t <- configs]
                     , max_tokens = getFirst [mt | MaxTokens mt <- configs]
                     , seed = getFirst [s | Seed s <- configs]
                     }
      in handleTextMessage msg req'
  , cpConfigHandler = \req ->
      let systemPrompts = [sp | SystemPrompt sp <- configs]
          sysMessages = map systemMessage systemPrompts
          req1 = modifyMessages (sysMessages <>) req
          streamEnabled = case [s | Streaming s <- configs] of
            (s:_) -> Just s
            [] -> stream req1
      in req1 { stream = streamEnabled }
  , cpFromResponse = parseTextResponse
  , cpSerializeMessage = serializeBaseMessage
  , cpDeserializeMessage = deserializeBaseMessage
  }
  where
    getFirst [] = Nothing
    getFirst (x:_) = Just x

    parseTextResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        (OpenAIChoice msg:rest) ->
          case content msg of
            Just txt ->
              -- Extract text but preserve reasoning_content and other fields in the choice
              let updatedMsg = msg { content = Nothing }
                  newChoices = OpenAIChoice updatedMsg : rest
              in Just (AssistantText txt, OpenAISuccess (OpenAISuccessResponse newChoices))
            Nothing -> Nothing
        [] -> Nothing
    parseTextResponse _ = Nothing

-- Standalone reasoning provider
openAIReasoning :: forall provider model state. (HasReasoning model provider, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model state
openAIReasoning _p _m _configs _s = noopHandler
  { cpToRequest = handleReasoningMessage
  , cpFromResponse = parseReasoningResponse
  , cpPureMessageResponse = orderReasoningBeforeText
  , cpSerializeMessage = serializeReasoningMessages
  , cpDeserializeMessage = deserializeReasoningMessages
  }
  where
    parseReasoningResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        (OpenAIChoice msg:rest) ->
          case reasoning_content msg of
            Just txt ->
              -- Extract reasoning but preserve content and other fields in the choice
              let updatedMsg = msg { reasoning_content = Nothing }
                  newChoices = OpenAIChoice updatedMsg : rest
              in Just (AssistantReasoning txt, OpenAISuccess (OpenAISuccessResponse newChoices))
            Nothing -> Nothing
        [] -> Nothing
    parseReasoningResponse _ = Nothing

    -- Move reasoning messages before text in the same sequence
    -- When we encounter reasoning after text, put reasoning first, then text
    orderReasoningBeforeText :: [Message model provider] -> [Message model provider]
    orderReasoningBeforeText = go [] []
      where
        go accum reasoning [] = reasoning ++ accum
        go accum reasoning (m@(AssistantReasoning _) : rest) =
          let hasText = any isAssistantText accum
          in if hasText
             -- Text comes before reasoning, so put reasoning first in output
             then [m] ++ reasoning ++ accum ++ go [] [] rest
             else go accum (reasoning ++ [m]) rest
        go accum reasoning (m@(AssistantText _) : rest) =
          go (accum ++ [m]) reasoning rest
        go accum reasoning (m : rest) =
          go (accum ++ [m]) reasoning rest

    isAssistantText (AssistantText _) = True
    isAssistantText _ = False

-- Standalone tools provider
openAITools :: forall provider model state. (HasTools model provider, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model state
openAITools _p _m configs _s = noopHandler
  { cpToRequest = \msg req ->
      let req' = handleToolMessage msg req
          toolDefs = [defs | Tools defs <- configs]
      in if null toolDefs then req' else setToolDefinitions (map toOpenAIToolDef (concat toolDefs)) req'
  , cpFromResponse = parseToolResponse
  , cpSerializeMessage = serializeToolMessages
  , cpDeserializeMessage = deserializeToolMessages
  }
  where
    parseToolResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        (OpenAIChoice msg:rest) ->
          case tool_calls msg of
            Just (tc:remainingTCs) ->
              -- Extract first tool call but preserve remaining tool calls and other fields
              let updatedMsg = msg { tool_calls = if null remainingTCs then Nothing else Just remainingTCs }
                  newChoices = OpenAIChoice updatedMsg : rest
              in Just (AssistantTool (convertToolCall tc), OpenAISuccess (OpenAISuccessResponse newChoices))
            _ -> Nothing
        [] -> Nothing
    parseToolResponse _ = Nothing

-- Standalone JSON provider
openAIJSON :: forall provider model state. (HasJSON model provider, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => ComposableProvider provider model state
openAIJSON _p _m _configs _s = noopHandler
  { cpToRequest = handleJSONMessage
  , cpFromResponse = \_ -> Nothing  -- Let base handler parse it
  , cpPureMessageResponse = convertTextToJSON
  , cpSerializeMessage = serializeJSONMessages
  , cpDeserializeMessage = deserializeJSONMessages
  }
  where
    -- FIXME: In the new composable provider architecture, we don't currently have access to the request
    -- or message history in the pure message response handler. We should track whether JSON mode was
    -- actually requested so we only parse as JSON when appropriate. For now, we do optimistic parsing:
    -- if the text is valid JSON, convert it to AssistantJSON. This means we might incorrectly parse
    -- regular text that happens to be JSON as a JSON response. This should be fixed when we have
    -- a way to track request state through the response parsing pipeline.

    -- Convert AssistantText messages to AssistantJSON if the text is valid JSON
    convertTextToJSON :: [Message model provider] -> [Message model provider]
    convertTextToJSON = map convertMessage
      where
        convertMessage (AssistantText txt) =
          case Aeson.decode (BSL.fromStrict (TE.encodeUtf8 txt)) of
            Just jsonVal -> AssistantJSON jsonVal
            Nothing -> AssistantText txt  -- Keep as text if not valid JSON
        convertMessage msg = msg

-- These are removed - use the typeclass methods withTools, withReasoning, etc. instead
-- They're defined in the HasTools/HasReasoning/HasJSON instances

-- ============================================================================
-- Test Helper Functions (Convenience Wrappers)
-- ============================================================================

-- | Convenience wrapper: Apply base composable provider's model name and config handling (without message processing)
handleBase :: forall provider model. (ModelName provider model, ProviderRequest provider ~ OpenAIRequest) => provider -> model -> [ModelConfig provider model] -> MessageEncoder provider model
handleBase _p m configs _msg req =
  let req' = req { model = modelName @provider m
                 , temperature = getFirst [t | Temperature t <- configs]
                 , max_tokens = getFirst [mt | MaxTokens mt <- configs]
                 , seed = getFirst [s | Seed s <- configs]
                 }
  in req'
  where
    getFirst [] = Nothing
    getFirst (x:_) = Just x

-- | Convenience wrapper: Apply text message handler
handleTextMessages :: forall provider model. (ProviderRequest provider ~ OpenAIRequest) => provider -> model -> [ModelConfig provider model] -> MessageEncoder provider model
handleTextMessages _p _m _configs = handleTextMessage

-- | Convenience wrapper: Apply reasoning message handler
handleReasoning :: forall provider model s. (HasReasoning model provider, ProviderRequest provider ~ OpenAIRequest, ProviderResponse provider ~ OpenAIResponse) => provider -> model -> [ModelConfig provider model] -> s -> MessageEncoder provider model
handleReasoning p m configs s msg req =
  let cp = openAIReasoning @provider @model
      handlers = cp p m configs s
  in cpToRequest handlers msg req


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