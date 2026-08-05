{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedRecordDot #-}

{- |
Module: Protocol.OpenAI

Low-level protocol helpers for OpenAI wire protocol testing.

= Purpose

This module provides building blocks for protocol-level capability probes.
It is NOT for testing our abstractions (use StandardTests for that).

Provides:
- Helper functions to build requests (hides Aeson/JSON complexity)
- Functions to extract data from responses
- Composable error checking (catch API errors early)
- Assertions for protocol tests

= Design Principles

* Protocol-agnostic naming - functions should work similarly across protocols
* Hide complexity - no Aeson manipulation in tests
* Use mempty with record updates - resilient to field additions
* Descriptive error messages - failures should be immediately clear
* Keep functions minimal - add more as needed, don't over-engineer

= Usage Pattern

Building a request:
@
let req = (simpleUserRequest "What is 2+2?") { model = "gpt-4" }
    withTools = req { tools = Just [simpleTool "calculator" "Do math"] }
@

Checking responses:
@
resp <- makeRequest req
let text = getAssistantText . checkError $ resp  -- checkError catches API errors first
assertHasAssistantText resp  -- use in protocol tests
@

-}

module Protocol.OpenAI where

import UniversalLLM.Protocols.OpenAI
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Vector as V
import Test.Hspec (Spec, it, shouldSatisfy, Expectation, HasCallStack)

-- ============================================================================
-- Empty/Default Objects
--
-- Internal defaults for building objects with record updates.
-- These ensure we handle all fields and can swap to mempty if instances added.
-- ============================================================================

-- | Empty message - replace with mempty if OpenAIMessage gets Monoid instance
emptyMessage :: OpenAIMessage
emptyMessage = OpenAIMessage
  { role = ""
  , content = Nothing
  , reasoning_content = Nothing
  , reasoning_details = Nothing
  , tool_calls = Nothing
  , tool_call_id = Nothing
  }

-- | Empty reasoning config - internal, use 'enableReasoning' instead
emptyReasoningConfig :: OpenAIReasoningConfig
emptyReasoningConfig = OpenAIReasoningConfig
  { reasoning_enabled = Nothing
  , reasoning_effort = Nothing
  , reasoning_max_tokens = Nothing
  , reasoning_exclude = Nothing
  }

-- ============================================================================
-- Request Helpers
--
-- Functions to build requests without dealing with protocol details.
-- Named to be protocol-agnostic where possible.
-- ============================================================================

-- | Create a user message
userMessage :: Text -> OpenAIMessage
userMessage txt = emptyMessage { role = "user", content = Just (TextContent txt) }

-- | Create a simple request with a user message
--
-- Common pattern: start with this, then set model field
-- >>> simpleUserRequest "What is 2+2?" & \r -> r { model = "gpt-4" }
simpleUserRequest :: Text -> OpenAIRequest
simpleUserRequest txt = mempty { messages = [userMessage txt] }

-- | Create a request with consecutive user messages
--
-- Used to test if the API accepts multiple user messages in a row
consecutiveUserMessages :: Text -> Text -> OpenAIRequest
consecutiveUserMessages msg1 msg2 = mempty { messages = [userMessage msg1, userMessage msg2] }

-- | Create an assistant message
assistantMessage :: Text -> OpenAIMessage
assistantMessage txt = emptyMessage { role = "assistant", content = Just (TextContent txt) }

-- | Create a request starting with an assistant message
--
-- Used to test if the API accepts history starting with assistant (no initial user message)
startsWithAssistant :: OpenAIRequest
startsWithAssistant = mempty
  { messages =
      [ assistantMessage "I'm a helpful assistant."
      , userMessage "What is 2+2?"
      ]
  }

-- | Create a simple tool definition with parameters
--
-- Example: simpleTool "get_weather" "Get current weather" [("location", "string", "City name")]
--
-- This creates a complete, valid tool definition with properly specified parameters.
-- Each parameter is (name, type, description), and all parameters are marked as required.
simpleTool :: Text -> Text -> [(Text, Text, Text)] -> OpenAIToolDefinition
simpleTool name description params = OpenAIToolDefinition
  { tool_type = "function"
  , function = OpenAIFunction
      { name = name
      , description = description
      , parameters = Aeson.object
          [ "type" Aeson..= ("object" :: Text)
          , "properties" Aeson..= Aeson.object
              [ Key.fromText paramName Aeson..= Aeson.object
                  [ "type" Aeson..= paramType
                  , "description" Aeson..= paramDesc
                  ]
              | (paramName, paramType, paramDesc) <- params
              ]
          , "required" Aeson..= ([paramName | (paramName, _, _) <- params] :: [Text])
          ]
      }
  }

-- | Create a weather tool definition with proper parameters
--
-- This creates a complete, valid tool definition that models can actually use.
-- The location parameter is properly specified with type and description.
weatherTool :: OpenAIToolDefinition
weatherTool = simpleTool "get_weather" "Get current weather for a location"
  [("location", "string", "City name")]

-- | Create a request asking for structured JSON output matching a schema
--
-- Wraps the schema the same way 'UniversalLLM.Providers.OpenAI.handleJSONMessage' does
-- (name/strict/schema wrapper), so the wire request matches what the real JSON message
-- encoder produces.
jsonRequest :: Text -> Aeson.Value -> OpenAIRequest
jsonRequest txt schema = mempty
  { messages = [userMessage txt]
  , response_format = Just (OpenAIResponseFormat "json_schema" (Just wrappedSchema))
  }
  where
    wrappedSchema = Aeson.object
      [ "name" Aeson..= ("response" :: Text)
      , "strict" Aeson..= True
      , "schema" Aeson..= schema
      ]

-- | A minimal object schema requesting a "colors" array - reused by JSON mode probes
colorsSchema :: Aeson.Value
colorsSchema = Aeson.object
  [ "type" Aeson..= ("object" :: Text)
  , "properties" Aeson..= Aeson.object
      [ "colors" Aeson..= Aeson.object
          [ "type" Aeson..= ("array" :: Text)
          , "items" Aeson..= Aeson.object ["type" Aeson..= ("string" :: Text)]
          ]
      ]
  , "required" Aeson..= (["colors"] :: [Text])
  ]

-- | A deliberately unusual, nested schema for a "list some colors" style prompt
--
-- The obvious/guessable shape for "list 3 primary colors" is @{"colors": [...]}@ -
-- 'colorsSchema' can't tell real schema-following apart from a model just producing
-- its "natural" answer that happens to satisfy a plausible-looking schema. This schema
-- uses unrelated key names and a nesting level nothing in the prompt suggests, so
-- matching it is only possible if the model (or backend) actually reads and follows
-- @response_format@ rather than pattern-matching the prompt.
oddShapeColorsSchema :: Aeson.Value
oddShapeColorsSchema = Aeson.object
  [ "type" Aeson..= ("object" :: Text)
  , "properties" Aeson..= Aeson.object
      [ "result" Aeson..= Aeson.object
          [ "type" Aeson..= ("object" :: Text)
          , "properties" Aeson..= Aeson.object
              [ "hue_list" Aeson..= Aeson.object
                  [ "type" Aeson..= ("array" :: Text)
                  , "items" Aeson..= Aeson.object ["type" Aeson..= ("string" :: Text)]
                  ]
              ]
          , "required" Aeson..= (["hue_list"] :: [Text])
          ]
      ]
  , "required" Aeson..= (["result"] :: [Text])
  ]

-- | Assert that response content is bare JSON matching 'oddShapeColorsSchema' exactly:
-- @{"result": {"hue_list": [...]}}@ with at least 3 entries
--
-- Used by 'Protocol.OpenAITests.jsonSchemaShapeCompliance' to distinguish real
-- schema-following from a model just producing its guessable "natural" answer for a
-- prompt like "list 3 colors" - see 'oddShapeColorsSchema'.
assertMatchesOddShapeSchema :: HasCallStack => OpenAIResponse -> Expectation
assertMatchesOddShapeSchema resp = do
  let text = getAssistantText . expectSuccess $ resp
  case Aeson.decode (BSL.fromStrict (TE.encodeUtf8 (T.strip text))) of
    Just (Aeson.Object obj) ->
      case KM.lookup "result" obj of
        Just (Aeson.Object resultObj) ->
          case KM.lookup "hue_list" resultObj of
            Just (Aeson.Array arr) -> length arr `shouldSatisfy` (>= 3)
            _ -> error $ "\"result\" object missing \"hue_list\" array, raw content: " <> T.unpack text
        _ -> error $ "Response missing nested \"result\" object (schema not followed), raw content: " <> T.unpack text
    _ -> error $ "Content is not bare JSON matching the schema, raw content: " <> T.unpack text

-- | Enable reasoning on a request
enableReasoning :: OpenAIRequest -> OpenAIRequest
enableReasoning req = req
  { reasoning = Just $ emptyReasoningConfig { reasoning_enabled = Just True }
  }

-- | Disable reasoning on a request
disableReasoning :: OpenAIRequest -> OpenAIRequest
disableReasoning req = req
  { reasoning = Just $ emptyReasoningConfig { reasoning_enabled = Just False }
  }

-- | Disable thinking via llama.cpp's chat_template_kwargs
disableThinkingLlamaCpp :: OpenAIRequest -> OpenAIRequest
disableThinkingLlamaCpp req = req
  { chat_template_kwargs = Just $ Aeson.object ["enable_thinking" Aeson..= False] }

-- | Enable thinking via llama.cpp's chat_template_kwargs
enableThinkingLlamaCpp :: OpenAIRequest -> OpenAIRequest
enableThinkingLlamaCpp req = req
  { chat_template_kwargs = Just $ Aeson.object ["enable_thinking" Aeson..= True] }

-- | Enable reasoning with effort field only (no enabled flag)
enableReasoningEffortOnly :: OpenAIRequest -> OpenAIRequest
enableReasoningEffortOnly req = req
  { reasoning = Just $ emptyReasoningConfig { reasoning_effort = Just "low" }
  }

-- | Enable reasoning with enabled=true and effort (full config as openAIReasoning sends)
enableReasoningFull :: OpenAIRequest -> OpenAIRequest
enableReasoningFull req = req
  { reasoning = Just $ emptyReasoningConfig
      { reasoning_enabled = Just True
      , reasoning_effort = Just "low"
      , reasoning_exclude = Just False
      }
  }

-- | Create an assistant message with a tool call
--
-- Example: assistantToolCallMessage "call_123" "get_weather" "{\"location\": \"London\"}"
assistantToolCallMessage :: Text -> Text -> Text -> OpenAIMessage
assistantToolCallMessage callId functionName arguments = emptyMessage
  { role = "assistant"
  , tool_calls = Just [OpenAIToolCall
      { callId = callId
      , toolCallType = "function"
      , toolFunction = OpenAIToolFunction
          { toolFunctionName = functionName
          , toolFunctionArguments = arguments
          }
      }]
  }

-- | Create a tool response message
--
-- Example: toolResponseMessage "call_123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
toolResponseMessage :: Text -> Text -> OpenAIMessage
toolResponseMessage callId result = emptyMessage
  { role = "tool"
  , tool_call_id = Just callId
  , content = Just (TextContent result)
  }

-- | Create a system message
systemMsg :: Text -> OpenAIMessage
systemMsg txt = emptyMessage { role = "system", content = Just (TextContent txt) }

-- | Create a request with system message mid-conversation
--
-- History: user message, then system message, then user message.
-- Tests if the API accepts system messages that are not at the beginning.
requestWithSystemMidConversation :: OpenAIRequest
requestWithSystemMidConversation = mempty
  { messages =
      [ userMessage "What is 2+2?"
      , assistantMessage "4"
      , systemMsg "You are a helpful assistant."
      , userMessage "And what is 3+3?"
      ]
  }

-- | Create a request with system message at the beginning
--
-- History: system message, then user message.
-- This is the expected/standard positioning.
requestWithSystemAtStart :: OpenAIRequest
requestWithSystemAtStart = mempty
  { messages =
      [ systemMsg "You are a helpful assistant."
      , userMessage "What is 2+2?"
      ]
  }

-- | Create a request with multiple system messages
--
-- History: three system messages, then user message.
-- Tests if the API accepts more than one system message.
requestWithMultipleSystemMessages :: OpenAIRequest
requestWithMultipleSystemMessages = mempty
  { messages =
      [ systemMsg "You are a helpful assistant."
      , systemMsg "Always respond concisely."
      , systemMsg "Use plain language."
      , userMessage "What is 2+2?"
      ]
  }

-- | Create a request with fabricated tool call history
--
-- This creates a conversation with:
-- 1. User asks question that should trigger tool
-- 2. Assistant makes tool call
-- 3. Tool returns result (automated, from us)
-- (Next message should be assistant responding to the tool result)
--
-- Used to test if the API accepts tool results in the expected format.
-- Includes the tool definition since the tool call references it.
requestWithToolCallHistory :: OpenAIRequest
requestWithToolCallHistory = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      ]
  , tools = Just [weatherTool]
  }

-- | Create a request with tool call history but tool no longer available
--
-- This tests if the API accepts tool calls in history where the tool definition
-- is no longer in the current tool set. This happens when tools are added/removed
-- between requests.
--
-- History contains a get_weather tool call, but only calculator tool is available now.
requestWithStaleToolInHistory :: OpenAIRequest
requestWithStaleToolInHistory = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      , assistantMessage "It's 72 degrees and sunny in London."
      , userMessage "Now calculate 5 * 7"
      ]
  , tools = Just [simpleTool "calculator" "Perform calculations" [("expression", "string", "Math expression")]]
  }

-- | Create a request with tool call result but no tools defined
--
-- This tests if the API accepts tool results when tools field is empty/null.
-- Some APIs (like Nova) require tool definitions even when processing results.
requestWithToolResultNoTools :: OpenAIRequest
requestWithToolResultNoTools = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      ]
  , tools = Nothing  -- No tools defined
  }

-- | Create a request with tool call result but the specific tool no longer available
--
-- This tests if the API accepts tool results when the tool that was called
-- is no longer in the tool set. Different from requestWithStaleToolInHistory
-- which has the tool call + result deep in history - this is immediately after
-- the tool call returns.
requestWithToolResultToolGone :: OpenAIRequest
requestWithToolResultToolGone = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      ]
  , tools = Just [simpleTool "calculator" "Perform calculations" [("expression", "string", "Math expression")]]
  }

-- | Create a request with old tool call in history, new user message, tool still available
--
-- This tests if tools in history must remain in the tool set even after
-- the conversation has moved on. The get_weather call is in history but conversation
-- has continued - does the tool need to stay available?
requestWithOldToolCallStillAvailable :: OpenAIRequest
requestWithOldToolCallStillAvailable = mempty
  { messages =
      [ userMessage "What's the weather in London?"
      , assistantToolCallMessage "call_abc123" "get_weather" "{\"location\": \"London\"}"
      , toolResponseMessage "call_abc123" "{\"temperature\": 72, \"condition\": \"sunny\"}"
      , assistantMessage "It's 72 degrees and sunny in London."
      , userMessage "Now calculate 5 * 7"
      ]
  , tools = Just [weatherTool]  -- weather still available, though conversation moved to math
  }

-- | Create a request with tool result from a previous response
--
-- This extracts the assistant message from the response (preserving all fields
-- including reasoning_details), adds a mock tool result addressed to the real
-- tool_call_id the model returned, and creates a new request.
--
-- Used to test if reasoning/metadata is preserved through tool call chains.
requestWithToolResult :: HasCallStack => OpenAIResponse -> OpenAIRequest
requestWithToolResult resp =
  let assistantMsg = getFirstMessage . expectSuccess $ resp
      callId = case assistantMsg.tool_calls of
        Just (tc:_) -> tc.callId
        _ -> error $ "requestWithToolResult: assistant message has no tool_calls, full message: " ++ show assistantMsg
  in mempty
    { messages =
        [ userMessage "Use the get_weather function to check the weather in London."
        , assistantMsg  -- Preserve all fields (including reasoning_details)
        , toolResponseMessage callId "{\"temperature\": 72, \"condition\": \"sunny\"}"
        ]
    }

-- ============================================================================
-- Response Helpers
--
-- Functions to extract data from responses and check for errors.
-- Use 'expectSuccess' when you expect a successful response.
-- Use 'expectError' when testing error handling.
-- ============================================================================

-- | Expect a success response - unwrap and return OpenAISuccessResponse
--
-- Use this when you expect the request to succeed:
-- >>> text <- getAssistantText . expectSuccess $ resp
--
-- Note: This throws on OpenAIError, which is a VALID protocol response.
-- If you want to test error handling, use 'expectError' instead.
expectSuccess :: HasCallStack => OpenAIResponse -> OpenAISuccessResponse
expectSuccess (OpenAIError (OpenAIErrorResponse details)) =
  let typeInfo = case errorType details of
        Just t -> " (" <> t <> ")"
        Nothing -> ""
  in error $ T.unpack $ "Expected success but got provider error: " <> errorMessage details <> typeInfo
expectSuccess (OpenAISuccess success) = success

-- | Expect an error response - unwrap and return OpenAIErrorResponse
--
-- Use this when testing error handling:
-- >>> errResp <- expectError resp
-- >>> errorMessage (errorDetail errResp) `shouldContain` "invalid"
expectError :: OpenAIResponse -> OpenAIErrorResponse
expectError (OpenAISuccess _) =
  error "Expected provider error but got success response"
expectError (OpenAIError err) = err

-- | Check if response is an error - throw if so, otherwise pass through
--
-- DEPRECATED: Use 'expectSuccess' instead for clarity.
-- This name is confusing because it doesn't "check" - it throws on error.
{-# DEPRECATED checkError "Use expectSuccess instead" #-}
checkError :: OpenAIResponse -> OpenAIResponse
checkError resp = OpenAISuccess (expectSuccess resp)

-- | Extract the single message from success response
--
-- Throws if no choices or multiple choices. Use this to avoid repeated pattern matching:
-- >>> msg <- getFirstMessage . expectSuccess $ resp
getFirstMessage :: OpenAISuccessResponse -> OpenAIMessage
getFirstMessage (OpenAISuccessResponse []) =
  error "No choices in response"
getFirstMessage (OpenAISuccessResponse [OpenAIChoice msg _]) = msg
getFirstMessage (OpenAISuccessResponse choices) =
  error $ "Expected exactly one choice, got " ++ show (length choices)

-- | Extract the finish_reason of the single choice ("stop", "length", etc.)
--
-- Use this to distinguish a natural stop from truncation when a probe gets less
-- content than expected - "length" means the token budget ran out mid-generation.
getFinishReason :: OpenAISuccessResponse -> Maybe Text
getFinishReason (OpenAISuccessResponse [OpenAIChoice _ fr]) = fr
getFinishReason _ = Nothing

-- | Extract assistant text from success response
--
-- Throws if no choices or no content. Use after expectSuccess:
-- >>> getAssistantText . expectSuccess $ resp
getAssistantText :: OpenAISuccessResponse -> Text
getAssistantText success =
  let msg = getFirstMessage success
      fr = getFinishReason success
  in case contentText msg.content of
    Just txt -> txt
    Nothing -> error $ "Assistant message has no content (finish_reason: " ++ show fr ++ "), full message: " ++ show msg

-- | Extract error details from error response
--
-- Throws if response is not an error. Use to test error handling:
-- >>> details <- getErrorDetail . expectError $ resp
-- >>> errorMessage details `shouldContain` "invalid"
getErrorDetail :: OpenAIErrorResponse -> OpenAIErrorDetail
getErrorDetail (OpenAIErrorResponse details) = details

-- | Extract the underlying upstream provider's raw error message from
-- OpenRouter's @error.metadata.raw@ field.
--
-- OpenRouter wraps every upstream error behind a generic @errorMessage@
-- (usually the unhelpful \"Provider returned error\"). The real reason lives
-- in @metadata.raw@, itself a JSON string containing the upstream provider's
-- own error body. Returns 'Nothing' if there's no metadata, or if @raw@
-- isn't present/parseable -- callers should fall back to 'errorMessage' then.
errorMetadataRawMessage :: OpenAIErrorDetail -> Maybe Text
errorMetadataRawMessage detail = do
  Aeson.Object metaObj <- errorMetadata detail
  Aeson.String rawStr <- KM.lookup "raw" metaObj
  rawValue <- Aeson.decode (BSL.fromStrict (TE.encodeUtf8 rawStr))
  case rawValue of
    Aeson.Object rawObj -> do
      Aeson.Object errObj <- KM.lookup "error" rawObj
      Aeson.String msg <- KM.lookup "message" errObj
      return msg
    _ -> Nothing

-- ============================================================================
-- Protocol Assertions
--
-- Assertions for use in protocol capability probes.
-- These check specific protocol behaviors and throw descriptive errors.
-- ============================================================================

-- | Assert that response was successful (not an error)
--
-- Use this when you just need to verify no error occurred.
-- Throws if provider returned error, succeeds if response is valid.
-- Useful for tests that just need to check acceptance without inspecting content.
wasSuccessful :: HasCallStack => OpenAIResponse -> Expectation
wasSuccessful resp = expectSuccess resp `seq` return ()

-- | Assert that response is a provider error (not a success)
--
-- Use this to test error handling - verifies we got an OpenAIError response
assertIsProviderError :: HasCallStack => OpenAIResponse -> Expectation
assertIsProviderError resp = do
  let details = getErrorDetail . expectError $ resp
  -- Verify it has error details
  T.length (errorMessage details) `shouldSatisfy` (> 0)

-- | Assert that response contains assistant text
--
-- Use this in protocol tests to verify basic text responses
assertHasAssistantText :: HasCallStack => OpenAIResponse -> Expectation
assertHasAssistantText resp = do
  let text = getAssistantText . expectSuccess $ resp
  T.length text `shouldSatisfy` (> 0)

-- | Assert that response contains tool calls
assertHasToolCalls :: HasCallStack => OpenAIResponse -> Expectation
assertHasToolCalls resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.tool_calls of
    Just (_:_) -> return ()
    Just [] -> error "Message has empty tool_calls list"
    Nothing -> error "Message has no tool_calls field"

-- | Assert that response contains reasoning content
assertHasReasoningContent :: HasCallStack => OpenAIResponse -> Expectation
assertHasReasoningContent resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.reasoning_content of
    Just txt | T.length txt > 0 -> return ()
    Just _ -> error "Message has empty reasoning_content"
    Nothing -> error "Message has no reasoning_content field"

-- | Assert that response contains reasoning_details
--
-- Some providers (like OpenRouter) put reasoning in reasoning_details instead
assertHasReasoningDetails :: HasCallStack => OpenAIResponse -> Expectation
assertHasReasoningDetails resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.reasoning_details of
    Just _ -> return ()  -- It's a Value, just check it exists
    Nothing -> error $
      "Message has no reasoning_details field. Diagnostic -- reasoning_content: "
      ++ show msg.reasoning_content
      ++ ", tool_calls: " ++ show (length <$> msg.tool_calls)
      ++ ", content: " ++ show (contentText msg.content)

-- | Assert that response has NO reasoning data (neither reasoning_content nor reasoning_details)
--
-- Used for testing "hidden reasoning" models that accept reasoning config but
-- don't expose reasoning in responses.
assertNoReasoningData :: HasCallStack => OpenAIResponse -> Expectation
assertNoReasoningData resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case (msg.reasoning_content, msg.reasoning_details) of
    (Nothing, Nothing) -> return ()
    (Just txt, _) | T.length txt == 0 ->
      case msg.reasoning_details of
        Nothing -> return ()
        Just _ -> error "Message has reasoning_details (expected hidden reasoning)"
    (Just _, _) -> error "Message has reasoning_content (expected hidden reasoning)"
    (Nothing, Just _) -> error "Message has reasoning_details (expected hidden reasoning)"

-- | Assert that response has encrypted reasoning in reasoning_details
--
-- Used for testing models that return reasoning_details with encrypted data
-- (type: "reasoning.encrypted"). This is common for models via OpenRouter
-- where reasoning is signed/encrypted by the provider.
assertHasEncryptedReasoning :: HasCallStack => OpenAIResponse -> Expectation
assertHasEncryptedReasoning resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case msg.reasoning_details of
    Nothing -> error "Message has no reasoning_details field"
    Just val -> do
      -- Check if it's an array containing encrypted reasoning entries
      case val of
        Aeson.Array arr -> do
          let hasEncrypted = any isEncryptedReasoning (V.toList arr)
          if hasEncrypted
            then return ()
            else error "reasoning_details exists but contains no encrypted reasoning entries"
        _ -> error "reasoning_details is not an array"
  where
    isEncryptedReasoning (Aeson.Object obj) =
      case KM.lookup (Key.fromString "type") obj of
        Just (Aeson.String t) -> "reasoning.encrypted" `T.isInfixOf` t
        _ -> False
    isEncryptedReasoning _ = False

-- | Assert that response contains an XML-style tool call in the content
--
-- Some models (like GLM-4.5 via llama.cpp) return tool calls as XML in the content field
-- instead of using the proper tool_calls field. This checks for the presence of
-- <tool_call>function_name in the response body.
assertHasXMLToolCall :: HasCallStack => Text -> OpenAIResponse -> Expectation
assertHasXMLToolCall functionName resp = do
  let msg = getFirstMessage . expectSuccess $ resp
  case contentText msg.content of
    Just txt | T.isInfixOf ("<tool_call>" <> functionName) txt -> return ()
    Just txt -> error $ "Message content does not contain <tool_call>" <> T.unpack functionName <> ", got: " <> T.unpack txt
    Nothing -> error "Message has no content field"

-- | Assert that response content is valid JSON matching the requested shape
--
-- Checks the RAW wire content (not our AssistantJSON abstraction). Providers that wrap
-- structured output in markdown fences, add a preamble, or otherwise deviate from bare
-- JSON will fail this even though they returned "JSON mode" content - that distinction
-- is exactly what this probe exists to surface.
assertHasValidJSONContent :: HasCallStack => OpenAIResponse -> Expectation
assertHasValidJSONContent resp = do
  let text = getAssistantText . expectSuccess $ resp
  case Aeson.decode (BSL.fromStrict (TE.encodeUtf8 text)) of
    Just (Aeson.Object _) -> return ()
    Just _ -> error $ "Content parsed as JSON but wasn't an object, raw content: " <> T.unpack text
    Nothing -> error $ "Content is not valid JSON (wire-level), raw content: " <> T.unpack text

-- | Assert that response content is fenced/wrapped JSON, NOT bare JSON
--
-- This is the inverse of 'assertHasValidJSONContent' - used as a negative probe
-- (see 'Protocol.OpenAITests.wrapsJSONInFence') to document and continuously verify a
-- known provider quirk: JSON-mode output arrives wrapped in a markdown code fence
-- (and often trailing prose) despite a strict schema being requested. If this ever
-- starts failing, the provider has fixed its JSON-mode output and the
-- 'UniversalLLM.Providers.OpenAI.openAIJSONWithFencedFallback' workaround may no
-- longer be needed for that model.
assertHasFencedJSONContent :: HasCallStack => OpenAIResponse -> Expectation
assertHasFencedJSONContent resp = do
  let success = expectSuccess resp
      text = getAssistantText success
  case Aeson.decode (BSL.fromStrict (TE.encodeUtf8 (T.strip text))) of
    Just (Aeson.Object _) ->
      error $ "Expected fenced/wrapped JSON (known quirk) but got bare JSON - provider may have fixed JSON mode, raw content: " <> T.unpack text
    _ ->
      if "```" `T.isInfixOf` text
        then return ()
        else error $ "Expected a markdown code fence (known quirk) but found neither bare nor fenced JSON, finish_reason: "
               <> show (getFinishReason success) <> ", raw content: " <> T.unpack text

-- | Assert that JSON-mode output landed entirely in reasoning_details, with content
-- staying empty even at a clean finish_reason: "stop"
--
-- Negative probe (see 'Protocol.OpenAITests.jsonStaysInReasoningDetails') documenting a
-- GLM-4.7/GLM-4.7-Flash-via-OpenRouter-specific quirk, distinct from the fenced-content
-- quirk on GLM-4.5-Air (see 'assertHasFencedJSONContent'): the model never emits final
-- content for JSON-schema requests at all. Confirmed not to be token-budget truncation
-- by checking finish_reason is "stop", not "length". If this ever starts failing, the
-- provider may have started populating content and 'UniversalLLM.Providers.OpenAI.HasJSON'
-- could potentially be re-enabled for these models (pending a fix that can read the
-- answer out of reasoning_details, or the provider populating content directly).
assertJSONStaysInReasoningDetails :: HasCallStack => OpenAIResponse -> Expectation
assertJSONStaysInReasoningDetails resp = do
  let success = expectSuccess resp
      msg = getFirstMessage success
      fr = getFinishReason success
  case (msg.content, msg.reasoning_details) of
    (Nothing, Just _) ->
      if fr == Just "stop"
        then return ()
        else error $ "Expected finish_reason: stop (to rule out truncation) but got: " <> show fr
    (Just _, _) ->
      error $ "Expected content: null (known quirk) but got content - provider may have fixed this, full message: " <> show msg
    (Nothing, Nothing) ->
      error $ "Expected the answer in reasoning_details (known quirk) but found neither content nor reasoning_details, finish_reason: " <> show fr

-- | Create a vision request with a user message containing a base64-encoded image
visionRequest :: Text -> Text -> Text -> OpenAIRequest
visionRequest modelName mediaType b64Data = mempty
  { model = modelName
  , messages =
      [ emptyMessage
          { role = "user"
          , content = Just (PartsContent
              [ OpenAITextPart "Describe this image briefly."
              , OpenAIImagePart mediaType b64Data
              ])
          }
      ]
  }

-- | Create a vision request asking what specific object is in the image
visionIdentifyRequest :: Text -> Text -> Text -> OpenAIRequest
visionIdentifyRequest modelName mediaType b64Data = mempty
  { model = modelName
  , messages =
      [ emptyMessage
          { role = "user"
          , content = Just (PartsContent
              [ OpenAITextPart "What is the main object in this image? Answer in one sentence."
              , OpenAIImagePart mediaType b64Data
              ])
          }
      ]
  }

-- | Create a multi-image request asking the model to compare two images
visionCompareRequest :: Text -> Text -> Text -> Text -> Text -> OpenAIRequest
visionCompareRequest modelName mediaType1 b64Data1 mediaType2 b64Data2 = mempty
  { model = modelName
  , messages =
      [ emptyMessage
          { role = "user"
          , content = Just (PartsContent
              [ OpenAITextPart "I am showing you two images. Is the second image a horizontally mirrored version of the first? Answer yes or no, then briefly explain."
              , OpenAIImagePart mediaType1 b64Data1
              , OpenAIImagePart mediaType2 b64Data2
              ])
          }
      ]
  }

-- | Assert that response describes image content (non-empty assistant text)
assertDescribesImage :: HasCallStack => OpenAIResponse -> Expectation
assertDescribesImage = assertHasAssistantText

-- | Assert that response mentions a specific word (case-insensitive)
assertMentions :: HasCallStack => Text -> OpenAIResponse -> Expectation
assertMentions word resp = do
  let txt = T.toLower $ getAssistantText . expectSuccess $ resp
  if T.isInfixOf (T.toLower word) txt
    then return ()
    else error $ "Expected response to mention '" <> T.unpack word <> "' but got: " <> T.unpack txt

-- | Assert that response does NOT mention a specific word (case-insensitive)
assertDoesNotMention :: HasCallStack => Text -> OpenAIResponse -> Expectation
assertDoesNotMention word resp = do
  let txt = T.toLower $ getAssistantText . expectSuccess $ resp
  if T.isInfixOf (T.toLower word) txt
    then error $ "Expected response NOT to mention '" <> T.unpack word <> "' but it did: " <> T.unpack txt
    else return ()

-- | Assert that response confirms affirmatively (contains "yes")
assertConfirmsYes :: HasCallStack => OpenAIResponse -> Expectation
assertConfirmsYes resp = do
  let txt = T.toLower $ getAssistantText . expectSuccess $ resp
  if T.isInfixOf "yes" txt
    then return ()
    else error $ "Expected affirmative response (containing 'yes') but got: " <> T.unpack txt
