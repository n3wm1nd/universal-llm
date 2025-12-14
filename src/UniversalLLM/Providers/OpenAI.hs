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
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import Data.Aeson (object, (.=), Value)
import qualified Data.Aeson as Value
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Vector as V
import qualified Data.ByteString.Lazy as BSL
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Default (Default(..))
import Data.Maybe (isJust)
import Control.Applicative ((<|>))

-- OpenAI provider (official OpenAI API)
data OpenAI = OpenAI deriving (Show, Eq)

-- | State for managing OpenRouter reasoning_details
-- Stores reasoning_details from API responses so we can echo them back verbatim
-- This is required by OpenRouter for models like Nova and Gemini when using tool calls with reasoning
data OpenRouterReasoningState = OpenRouterReasoningState
  { reasoningTextToDetails :: Map Text Value.Value  -- Map from reasoning text to its details
  , toolCallToDetails :: Map Text Value.Value       -- Map from tool call ID to reasoning_details
  } deriving (Show, Eq)

instance Default OpenRouterReasoningState where
  def = OpenRouterReasoningState Map.empty Map.empty

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
userMessage txt = defaultOpenAIMessage
  { role = "user"
  , content = Just txt
  }

-- | Create an assistant message with text content
assistantMessage :: Text -> OpenAIMessage
assistantMessage txt = defaultOpenAIMessage
  { role = "assistant"
  , content = Just txt
  }

-- | Create a system message with text content
systemMessage :: Text -> OpenAIMessage
systemMessage txt = defaultOpenAIMessage
  { role = "system"
  , content = Just txt
  }

-- | Create an assistant message with reasoning content
-- OpenAI spec requires 'content' or 'tool_calls' to be present, so we provide empty content
-- when there's only reasoning. The reasoning goes in the separate 'reasoning_content' field.
reasoningMessage :: Text -> OpenAIMessage
reasoningMessage txt = defaultOpenAIMessage
  { role = "assistant"
  , content = Just ""
  , reasoning_content = Just txt
  }

-- | Create an assistant message with tool calls
-- Per OpenAI spec, assistant messages with tool_calls should have null content (not omitted)
-- This is different from reasoning messages which need empty string.
toolCallMessage :: [OpenAIToolCall] -> OpenAIMessage
toolCallMessage calls = defaultOpenAIMessage
  { role = "assistant"
  , content = Just ""  -- Must include empty content for verbatim preservation
  , tool_calls = Just calls
  }

-- | Create a tool result message
toolResultMessage :: Text -> Text -> OpenAIMessage
toolResultMessage tcId contentTxt = defaultOpenAIMessage
  { role = "tool"
  , content = Just contentTxt
  , tool_call_id = Just tcId
  }

-- | Append text to a message's content (only if roles match)
-- Allows merging text with messages that have tool_calls (for combined responses)
appendToMessageIfSameRole :: Text -> Text -> OpenAIMessage -> Maybe OpenAIMessage
appendToMessageIfSameRole targetRole txt msg@OpenAIMessage{ role = msgRole, content = existingContent, reasoning_content = Nothing, tool_call_id = Nothing }
  | msgRole == targetRole =
      let newContent = case existingContent of
            Just existing -> Just (existing <> "\n" <> txt)
            Nothing -> Just txt
      in Just $ msg { content = newContent }
appendToMessageIfSameRole _ _ _ = Nothing

-- | Append a tool call to a message's tool calls
-- Allows adding tool calls to messages that have text content (for combined responses)
appendToolCallToMessage :: OpenAIToolCall -> OpenAIMessage -> Maybe OpenAIMessage
appendToolCallToMessage tc msg@OpenAIMessage{ role = "assistant", tool_calls = existingCalls } =
  let newCalls = case existingCalls of
        Just existing -> Just (existing <> [tc])
        Nothing -> Just [tc]
  in Just $ msg { tool_calls = newCalls }
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
instance Provider (Model aiModel OpenAI) where
  type ProviderRequest (Model aiModel OpenAI) = OpenAIRequest
  type ProviderResponse (Model aiModel OpenAI) = OpenAIResponse

instance Provider (Model aiModel OpenAICompatible) where
  type ProviderRequest (Model aiModel OpenAICompatible) = OpenAIRequest
  type ProviderResponse (Model aiModel OpenAICompatible) = OpenAIResponse

instance Provider (Model aiModel OpenRouter) where
  type ProviderRequest (Model aiModel OpenRouter) = OpenAIRequest
  type ProviderResponse (Model aiModel OpenRouter) = OpenAIResponse

instance Provider (Model aiModel LlamaCpp) where
  type ProviderRequest (Model aiModel LlamaCpp) = OpenAIRequest
  type ProviderResponse (Model aiModel LlamaCpp) = OpenAIResponse

instance Provider (Model aiModel Ollama) where
  type ProviderRequest (Model aiModel Ollama) = OpenAIRequest
  type ProviderResponse (Model aiModel Ollama) = OpenAIResponse

instance Provider (Model aiModel VLLM) where
  type ProviderRequest (Model aiModel VLLM) = OpenAIRequest
  type ProviderResponse (Model aiModel VLLM) = OpenAIResponse

instance Provider (Model aiModel LiteLLM) where
  type ProviderRequest (Model aiModel LiteLLM) = OpenAIRequest
  type ProviderResponse (Model aiModel LiteLLM) = OpenAIResponse 

-- Helper: Get last message from request
lastMessage :: OpenAIRequest -> Maybe OpenAIMessage
lastMessage req = case messages req of
  [] -> Nothing
  msgs -> Just (last msgs)

-- Text message encoder
handleTextMessage :: forall m. (ProviderRequest m ~ OpenAIRequest) => MessageEncoder m
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
handleToolMessage :: forall m. (ProviderRequest m ~ OpenAIRequest) => MessageEncoder m
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
handleJSONMessage :: forall m. (ProviderRequest m ~ OpenAIRequest) => MessageEncoder m
handleJSONMessage msg req = case msg of
  UserRequestJSON txt schema ->
    let wrappedSchema = object
          [ "name" .= ("response" :: Text)
          , "strict" .= True
          , "schema" .= schema
          ]
    in appendMessage (userMessage txt) req
      & setResponseFormat (OpenAIResponseFormat "json_schema" (Just wrappedSchema))
  AssistantJSON jsonVal ->
    let jsonText = TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ jsonVal
    in appendMessage (assistantMessage jsonText) req
  _ -> req
  where
    (&) = flip ($)

-- Reasoning message encoder
handleReasoningMessage :: forall m. (ProviderRequest m ~ OpenAIRequest) => MessageEncoder m
handleReasoningMessage msg req = case msg of
  AssistantReasoning txt -> appendMessage (reasoningMessage txt) req
  _ -> req

-- Composable providers

-- Base composable provider: model name, basic config, text messages
baseComposableProvider :: forall m. (ModelName m, ProviderRequest m ~ OpenAIRequest, ProviderResponse m ~ OpenAIResponse) => ComposableProvider m ()
baseComposableProvider modelProxy configs _s = noopHandler
  { cpToRequest = \msg req ->
      let reqWithModel = req { model = modelName modelProxy
                     , temperature = getFirst [t | Temperature t <- configs]
                     , max_tokens = getFirst [mt | MaxTokens mt <- configs]
                     , seed = getFirst [s | Seed s <- configs]
                     }
      in handleTextMessage msg reqWithModel
  , cpConfigHandler = \req ->
      let systemPrompts = [sp | SystemPrompt sp <- configs]
          sysMessages = map systemMessage systemPrompts
          reqWithSysMessages = modifyMessages (sysMessages <>) req
          streamEnabled = case [s | Streaming s <- configs] of
            (s:_) -> Just s
            [] -> stream reqWithSysMessages
      in reqWithSysMessages { stream = streamEnabled }
  , cpFromResponse = parseTextResponse
  , cpSerializeMessage = serializeBaseMessage
  , cpDeserializeMessage = deserializeBaseMessage
  }
  where
    getFirst [] = Nothing
    getFirst (x:_) = Just x

    parseTextResponse (OpenAIError err) =
      Left $ ModelError $ errorMessage (errorDetail err)
    parseTextResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        (OpenAIChoice msg:rest) ->
          case content msg of
            Just txt | not (T.null txt) ->
              -- Extract text but preserve reasoning_content and other fields in the choice
              -- Skip empty content entirely (prevents duplicate assistant messages)
              let updatedMsg = msg { content = Nothing }
                  newChoices = OpenAIChoice updatedMsg : rest
              in Right (Just (AssistantText txt, OpenAISuccess (OpenAISuccessResponse newChoices)))
            _ -> Right Nothing
        [] -> Right Nothing

-- Standalone reasoning provider
openAIReasoning :: forall m state. (HasReasoning m, ProviderRequest m ~ OpenAIRequest, ProviderResponse m ~ OpenAIResponse) => ComposableProvider m state
openAIReasoning _m configs _s = noopHandler
  { cpToRequest = handleReasoningMessage
  , cpConfigHandler = \req ->
      let -- Only enable reasoning if the last message is from the user
          lastMessageIsUser = case reverse (messages req) of
            (msg:_) -> role msg == "user"
            [] -> False

          -- Only set reasoning field if last message is from user
          reasoningConfig = case [r | Reasoning r <- configs] of
            (True:_) | lastMessageIsUser -> Just OpenAIReasoningConfig
              { reasoning_enabled = Just True
              , reasoning_max_tokens = Nothing
              , reasoning_effort = Just "low"
              , reasoning_exclude = Just False
              }
            _ -> Nothing
      in req { reasoning = reasoningConfig }
  , cpFromResponse = parseReasoningResponse
  , cpPureMessageResponse = orderReasoningBeforeText
  , cpSerializeMessage = serializeReasoningMessages
  , cpDeserializeMessage = deserializeReasoningMessages
  }
  where
    parseReasoningResponse (OpenAIError err) =
      Left $ ModelError $ errorMessage (errorDetail err)
    parseReasoningResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        (OpenAIChoice msg:rest) ->
          case reasoning_content msg of
            Just txt | not (T.null txt) ->
              -- Extract reasoning but preserve content and other fields in the choice
              -- Skip empty reasoning (prevents empty AssistantReasoning messages)
              let updatedMsg = msg { reasoning_content = Nothing }
                  newChoices = OpenAIChoice updatedMsg : rest
              in Right (Just (AssistantReasoning txt, OpenAISuccess (OpenAISuccessResponse newChoices)))
            _ -> Right Nothing
        [] -> Right Nothing

    -- Move reasoning messages before text in the same sequence
    -- When we encounter reasoning after text, put reasoning first, then text
    -- Also filter out empty reasoning and text messages
    orderReasoningBeforeText :: [Message m] -> [Message m]
    orderReasoningBeforeText = go [] [] . filter (not . isEmptyMessage)
      where
        go accum reasonMsgs [] = reasonMsgs ++ accum
        go accum reasonMsgs (msg@(AssistantReasoning _) : rest) =
          let hasText = any isAssistantText accum
          in if hasText
             -- Text comes before reasoning, so put reasoning first in output
             then [msg] ++ reasonMsgs ++ accum ++ go [] [] rest
             else go accum (reasonMsgs ++ [msg]) rest
        go accum reasonMsgs (msg@(AssistantText _) : rest) =
          go (accum ++ [msg]) reasonMsgs rest
        go accum reasonMsgs (msg : rest) =
          go (accum ++ [msg]) reasonMsgs rest

    isAssistantText (AssistantText _) = True
    isAssistantText _ = False

    isEmptyMessage (AssistantText txt) = T.null (T.strip txt)
    isEmptyMessage (AssistantReasoning txt) = T.null (T.strip txt)
    isEmptyMessage _ = False

-- OpenRouter reasoning provider - handles reasoning_details preservation
-- This is specifically for OpenRouter which requires reasoning_details to be preserved
-- across multi-turn conversations, especially when using tool calls with reasoning models
openRouterReasoning :: forall m. (HasReasoning m, ProviderRequest m ~ OpenAIRequest, ProviderResponse m ~ OpenAIResponse, ReasoningState m ~ OpenRouterReasoningState) => ComposableProvider m OpenRouterReasoningState
openRouterReasoning _m configs state = noopHandler
  { cpToRequest = handleReasoningMessageWithState state
  , cpConfigHandler = \req ->
      let -- Verify and add reasoning_details based on chain-of-thought verification
          req1 = verifyAndAddReasoningDetails state req

          -- Check if any message has reasoning_details (after verification)
          hasReasoningDetails = any (\msg -> case reasoning_details msg of
                                       Just _ -> True
                                       Nothing -> False) (messages req1)

          -- Check if last message is a user message
          lastMessageRole = case reverse (messages req1) of
            (msg:_) -> role msg
            [] -> ""
          isUserMessage = lastMessageRole == "user"

          -- Set reasoning field only if:
          -- 1. Reasoning True in configs
          -- 2. Last message is a user message (we're requesting new reasoning)
          -- 3. No messages have reasoning_details (not continuing a reasoning conversation)
          reasoningConfig = case [r | Reasoning r <- configs] of
            (True:_) | isUserMessage && not hasReasoningDetails -> Just OpenAIReasoningConfig
              { reasoning_enabled = Just True
              , reasoning_max_tokens = Nothing
              , reasoning_effort = Just "low"
              , reasoning_exclude = Just False
              }
            _ -> Nothing

          req2 = req1 { reasoning = reasoningConfig }
      in req2
  , cpFromResponse = parseReasoningResponse
  , cpPostResponse = storeReasoningDetailsFromResponse
  , cpPureMessageResponse = orderReasoningBeforeText
  , cpSerializeMessage = serializeReasoningMessages
  , cpDeserializeMessage = deserializeReasoningMessages
  }
  where
    parseReasoningResponse (OpenAIError err) =
      Left $ ModelError $ errorMessage (errorDetail err)
    parseReasoningResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        (OpenAIChoice msg:rest) ->
          -- Try reasoning_content first (OpenAI format)
          case reasoning_content msg of
            Just txt | not (T.null txt) ->
              -- Extract reasoning but preserve content and other fields in the choice
              -- Skip empty reasoning (prevents empty AssistantReasoning messages)
              let updatedMsg = msg { reasoning_content = Nothing }
                  newChoices = OpenAIChoice updatedMsg : rest
              in Right (Just (AssistantReasoning txt, OpenAISuccess (OpenAISuccessResponse newChoices)))
            _ ->
              -- Try reasoning_details (OpenRouter format)
              case extractReasoningFromDetails (reasoning_details msg) of
                Just txt | not (T.null txt) ->
                  -- Clear reasoning_details after extraction to prevent infinite unfold loop
                  -- (State preservation happens in cpPostResponse via storeReasoningDetailsFromResponse)
                  -- Skip empty reasoning (prevents empty AssistantReasoning messages)
                  let updatedMsg = msg { reasoning_details = Nothing }
                      newChoices = OpenAIChoice updatedMsg : rest
                  in Right (Just (AssistantReasoning txt, OpenAISuccess (OpenAISuccessResponse newChoices)))
                _ -> Right Nothing
        [] -> Right Nothing

    -- Extract reasoning text from OpenRouter reasoning_details array
    extractReasoningFromDetails :: Maybe Value -> Maybe Text
    extractReasoningFromDetails Nothing = Nothing
    extractReasoningFromDetails (Just (Aeson.Array arr)) =
      -- reasoning_details is an array of objects with "text" fields
      -- Concatenate all text fields
      let texts = [txt | Aeson.Object obj <- V.toList arr
                       , Just (Aeson.String txt) <- [KM.lookup "text" obj]]
      in if null texts
         then Nothing
         else Just (T.intercalate "\n\n" texts)
    extractReasoningFromDetails _ = Nothing

    -- Store reasoning_details from response for later echo-back
    storeReasoningDetailsFromResponse :: ProviderResponse m -> OpenRouterReasoningState -> OpenRouterReasoningState
    storeReasoningDetailsFromResponse (OpenAIError _) st = st
    storeReasoningDetailsFromResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) st =
      case respChoices of
        (OpenAIChoice msg:_) ->
          case reasoning_details msg of
            Just details ->
              let -- Extract reasoning text from either reasoning_content (OpenAI) or reasoning_details (OpenRouter)
                  reasoningText = case reasoning_content msg of
                    Just txt -> Just txt
                    Nothing -> extractReasoningFromDetails (Just details)
                  -- Store by reasoning text if we extracted it
                  st1 = case reasoningText of
                    Just txt -> st { reasoningTextToDetails = Map.insert txt details (reasoningTextToDetails st) }
                    Nothing -> st
                  -- Store by tool call ID(s) if present
                  st2 = case tool_calls msg of
                    Just tcs -> foldl (\s tc -> s { toolCallToDetails = Map.insert (callId tc) details (toolCallToDetails s) }) st1 tcs
                    Nothing -> st1
              in st2
            Nothing -> st
        [] -> st

    -- Lookup reasoning_details when creating request
    handleReasoningMessageWithState :: OpenRouterReasoningState -> MessageEncoder m
    handleReasoningMessageWithState st msg req = case msg of
      AssistantReasoning reasoningTxt ->
        -- Lookup reasoning_details from state map by reasoning text
        case Map.lookup reasoningTxt (reasoningTextToDetails st) of
          Just details ->
            -- Found details in state - this is an echoed reasoning from a previous API response
            -- Create a proper message with the reasoning_details preserved
            -- Do NOT set reasoning_content - OpenRouter uses reasoning_details, not reasoning_content
            appendMessage
              (defaultOpenAIMessage
                { role = "assistant"
                , content = Just ""  -- Empty content when there's only reasoning
                , reasoning_content = Nothing  -- Don't add reasoning_content when using reasoning_details
                , reasoning_details = Just details  -- Preserve the details!
                }) req
          Nothing ->
            -- No details found - this is a reasoning message that wasn't from the API
            -- (e.g., deserialized from storage or created programmatically)
            -- Just use reasoning_content without details
            appendMessage (reasoningMessage reasoningTxt) req
      _ -> req

    -- Verify chain-of-thought and add reasoning_details
    -- Split messages into chunks by user messages, verify each chunk has complete reasoning_details
    -- Only add reasoning_details to chunks where ALL reasoning can be looked up
    verifyAndAddReasoningDetails :: OpenRouterReasoningState -> OpenAIRequest -> OpenAIRequest
    verifyAndAddReasoningDetails st req =
      let chunks = chunkMessagesByUser (messages req)
          verifiedChunks = map (verifyAndSignChunk st) chunks
          processedMsgs = concatMap (processChunk st) verifiedChunks
      in req { messages = processedMsgs }

    -- Chunk messages by user messages
    -- Returns [[UserMsg, AssistantMsg1, ToolMsg1, ...], [UserMsg, AssistantMsg2, ...], ...]
    chunkMessagesByUser :: [OpenAIMessage] -> [[OpenAIMessage]]
    chunkMessagesByUser = foldr addToChunks []
      where
        addToChunks msg [] = [[msg]]
        addToChunks msg chunks@(currentChunk:rest) =
          if role msg == "user"
            then [msg] : chunks  -- Start new chunk
            else (msg : currentChunk) : rest  -- Add to current chunk

    -- Verify a chunk and mark it as signed or unsigned
    verifyAndSignChunk :: OpenRouterReasoningState -> [OpenAIMessage] -> (Bool, [OpenAIMessage])
    verifyAndSignChunk st chunk =
      let canLookupAll = all (canLookupReasoningDetails st) chunk
      in (canLookupAll, chunk)

    -- Check if we can lookup reasoning_details for a message
    canLookupReasoningDetails :: OpenRouterReasoningState -> OpenAIMessage -> Bool
    canLookupReasoningDetails st msg
      | role msg /= "assistant" = True  -- Non-assistant messages don't need reasoning_details
      | otherwise =
          let hasReasoningContent = case reasoning_content msg of
                Just txt -> Map.member txt (reasoningTextToDetails st)
                Nothing -> True  -- No reasoning_content means we don't need to look it up
              hasToolCallDetails = case tool_calls msg of
                Just (tc:_) -> Map.member (callId tc) (toolCallToDetails st)
                Just [] -> True  -- Empty tool calls list means we don't need to look up details
                Nothing -> True  -- No tool calls means we don't need to look up details
          in hasReasoningContent && hasToolCallDetails

    -- Process a verified chunk: add reasoning_details if signed, filter AssistantReasoning if unsigned
    processChunk :: OpenRouterReasoningState -> (Bool, [OpenAIMessage]) -> [OpenAIMessage]
    processChunk st (True, chunk) = map (addReasoningDetailsToMessage st) chunk
    processChunk _  (False, chunk) = filter (not . isAssistantReasoningMessage) chunk

    -- Check if a message is an assistant message with only reasoning_content
    isAssistantReasoningMessage :: OpenAIMessage -> Bool
    isAssistantReasoningMessage msg =
      role msg == "assistant" && isJust (reasoning_content msg)

    -- Add reasoning_details to a single message if it has tool calls or reasoning_content that we have details for
    addReasoningDetailsToMessage :: OpenRouterReasoningState -> OpenAIMessage -> OpenAIMessage
    addReasoningDetailsToMessage st msg
      | role msg /= "assistant" = msg
      | otherwise =
          let -- Try to lookup by tool call ID first
              detailsFromToolCall = case tool_calls msg of
                Just (tc:_) -> Map.lookup (callId tc) (toolCallToDetails st)
                Just [] -> Nothing  -- Empty tool calls list means no details to lookup
                Nothing -> Nothing
              -- Try to lookup by reasoning_content
              detailsFromReasoning = case reasoning_content msg of
                Just txt -> Map.lookup txt (reasoningTextToDetails st)
                Nothing -> Nothing
              -- Use whichever we found (tool call takes precedence)
              details = detailsFromToolCall <|> detailsFromReasoning
          in case details of
               Just d -> msg { reasoning_details = Just d }
               Nothing -> msg

    -- Move reasoning messages before text in the same sequence
    -- When we encounter reasoning after text, put reasoning first, then text
    -- Also filter out empty reasoning and text messages
    orderReasoningBeforeText :: [Message m] -> [Message m]
    orderReasoningBeforeText = go [] [] . filter (not . isEmptyMessage)
      where
        go accum reasonMsgs [] = reasonMsgs ++ accum
        go accum reasonMsgs (msg@(AssistantReasoning _) : rest) =
          let hasText = any isAssistantText accum
          in if hasText
             -- Text comes before reasoning, so put reasoning first in output
             then [msg] ++ reasonMsgs ++ accum ++ go [] [] rest
             else go accum (reasonMsgs ++ [msg]) rest
        go accum reasonMsgs (msg@(AssistantText _) : rest) =
          go (accum ++ [msg]) reasonMsgs rest
        go accum reasonMsgs (msg : rest) =
          go (accum ++ [msg]) reasonMsgs rest

    isAssistantText (AssistantText _) = True
    isAssistantText _ = False

    isEmptyMessage (AssistantText txt) = T.null (T.strip txt)
    isEmptyMessage (AssistantReasoning txt) = T.null (T.strip txt)
    isEmptyMessage _ = False

-- Standalone tools provider
openAITools :: forall m . (HasTools m, ProviderRequest m ~ OpenAIRequest, ProviderResponse m ~ OpenAIResponse) => ComposableProvider m ()
openAITools _m configs _s = noopHandler
  { cpToRequest = \msg req ->
      let req' = handleToolMessage msg req
          toolDefs = [defs | Tools defs <- configs]
      in if null toolDefs then req' else setToolDefinitions (map toOpenAIToolDef (concat toolDefs)) req'
  , cpFromResponse = parseToolResponse
  , cpSerializeMessage = serializeToolMessages
  , cpDeserializeMessage = deserializeToolMessages
  }
  where
    parseToolResponse (OpenAIError err) =
      Left $ ModelError $ errorMessage (errorDetail err)
    parseToolResponse (OpenAISuccess (OpenAISuccessResponse respChoices)) =
      case respChoices of
        (OpenAIChoice msg:rest) ->
          case tool_calls msg of
            Just (tc:remainingTCs) ->
              -- Extract first tool call but preserve remaining tool calls and other fields
              let updatedMsg = msg { tool_calls = if null remainingTCs then Nothing else Just remainingTCs }
                  newChoices = OpenAIChoice updatedMsg : rest
              in Right (Just (AssistantTool (convertToolCall tc), OpenAISuccess (OpenAISuccessResponse newChoices)))
            _ -> Right Nothing
        [] -> Right Nothing

-- Standalone JSON provider
openAIJSON :: forall m . (HasJSON m, ProviderRequest m ~ OpenAIRequest, ProviderResponse m ~ OpenAIResponse) => ComposableProvider m ()
openAIJSON _m _configs _s = noopHandler
  { cpToRequest = handleJSONMessage
  , cpFromResponse = \_ -> Right Nothing  -- Let base handler parse it
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
    convertTextToJSON :: [Message m] -> [Message m]
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
handleBase :: forall m. (ModelName m, ProviderRequest m ~ OpenAIRequest) => m -> [ModelConfig m] -> MessageEncoder m
handleBase modelProxy configs _msg req =
  let reqWithModel = req { model = modelName modelProxy
                         , temperature = getFirst [t | Temperature t <- configs]
                         , max_tokens = getFirst [mt | MaxTokens mt <- configs]
                         , seed = getFirst [s | Seed s <- configs]
                         }
  in reqWithModel
  where
    getFirst [] = Nothing
    getFirst (x:_) = Just x


-- ============================================================================
-- Completion Interface (Legacy /v1/completions endpoint)
-- ============================================================================

-- All OpenAI-compatible providers use the same completion request/response types
instance Provider m => CompletionProvider m where
  type CompletionRequest m = OpenAICompletionRequest
  type CompletionResponse m = OpenAICompletionResponse

-- Helper: Modify completion request fields
modifyCompletionRequest :: (OpenAICompletionRequest -> OpenAICompletionRequest)
                        -> OpenAICompletionRequest
                        -> OpenAICompletionRequest
modifyCompletionRequest = id

-- Prompt handler: Set the prompt and model name
handlePrompt :: forall m.
                (ModelName m)
             =>PromptHandler m
handlePrompt modelProxy _configs promptTxt req =
  req { completionModel = modelName modelProxy
      , prompt = promptTxt
      }

-- Config handler: Apply temperature, max_tokens, and stop sequences
configureCompletion :: forall m.
                       ()
                    =>CompletionConfigHandler m
configureCompletion _m configs req = foldl applyConfig req configs
  where
    applyConfig :: OpenAICompletionRequest -> ModelConfig m -> OpenAICompletionRequest
    applyConfig r (Temperature temp) = r { completionTemperature = Just temp }
    applyConfig r (MaxTokens maxTok) = r { completionMaxTokens = Just maxTok }
    applyConfig r (Stop stopSeqs) = r { stop = Just stopSeqs }
    applyConfig r _ = r  -- Other configs don't apply to completions

-- Response parser: Extract text from first completion choice
parseCompletionResponse :: forall m.
                           ()
                        =>CompletionParser m
parseCompletionResponse _m _configs _prompt resp =
  case resp of
    OpenAICompletionSuccess successResp ->
      case completionChoices successResp of
        [] -> ""  -- No choices returned
        (choice:_) -> completionText choice  -- Return first choice
    OpenAICompletionError _err -> ""  -- Return empty on error (could be improved)

-- Base completion provider
baseCompletionProvider :: forall m.
                          (ModelName m)
                       =>ComposableCompletionProvider m
baseCompletionProvider = ComposableCompletionProvider
  { ccpToRequest = handlePrompt @m
  , ccpConfigHandler = configureCompletion @m
  , ccpFromResponse = parseCompletionResponse @m
  }

-- Default CompletionProviderImplementation for all models
instance {-# OVERLAPPABLE #-} (Provider m, ModelName m) => CompletionProviderImplementation m where
  getComposableCompletionProvider = baseCompletionProvider