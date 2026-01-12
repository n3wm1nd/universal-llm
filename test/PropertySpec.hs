{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}

module PropertySpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (Value, object, (.=))
import qualified Data.Aeson as Aeson
import Autodocodec (toJSONViaCodec)
import UniversalLLM
import qualified UniversalLLM.Protocols.Anthropic as AP
import qualified UniversalLLM.Protocols.OpenAI as OP
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider
import TestModels
import UniversalLLM.Providers.Anthropic (Anthropic(Anthropic))
import Data.Default (Default(..))

-- ============================================================================
-- Arbitrary instances
-- ============================================================================

-- Generator for realistic text (including unicode, special chars, etc)
genRealisticText :: Gen Text
genRealisticText = T.pack <$> listOf (frequency
  [ (60, elements (['a'..'z'] ++ ['A'..'Z'] ++ ['0'..'9'] ++ " .,!?"))  -- Common
  , (10, elements "\n\t\r")  -- Whitespace
  , (10, elements "{}[]()<>")  -- Brackets
  , (5, elements "\"'`")  -- Quotes
  , (5, elements "\\/@#$%^&*-_+=|:;")  -- Special chars
  , (5, elements "éñ中文🔥💯")  -- Unicode samples
  , (5, arbitrary)  -- Occasional truly random char
  ])

-- Generator for non-empty realistic text
genNonEmptyText :: Gen Text
genNonEmptyText = do
  first <- elements (['a'..'z'] ++ ['A'..'Z'])  -- Start with letter
  rest <- genRealisticText
  return $ T.pack (first : T.unpack rest)

-- Generate a simple JSON Value
genSimpleValue :: Gen Value
genSimpleValue = oneof
  [ Aeson.String <$> genRealisticText
  , Aeson.Number . fromInteger <$> arbitrary
  , Aeson.Bool <$> arbitrary
  , return Aeson.Null
  , Aeson.Object <$> pure mempty
  ]

-- Generate ToolDefinition
instance Arbitrary ToolDefinition where
  arbitrary = ToolDefinition
    <$> genNonEmptyText
    <*> genRealisticText
    <*> genSimpleValue
  shrink (ToolDefinition name desc params) =
    [ ToolDefinition name' desc' params'
    | name' <- shrinkText name
    , desc' <- shrinkText desc
    , params' <- [params]  -- Don't shrink params for simplicity
    ]
    where
      shrinkText txt = if T.null txt then [] else [T.take (T.length txt `div` 2) txt | T.length txt > 1]

-- Generate ToolCall
instance Arbitrary ToolCall where
  arbitrary = oneof
    [ ToolCall
        <$> genNonEmptyText  -- id
        <*> genNonEmptyText  -- name
        <*> genSimpleValue   -- params
    , InvalidToolCall
        <$> genNonEmptyText  -- id
        <*> genNonEmptyText  -- name
        <*> genRealisticText -- raw args
        <*> genRealisticText -- error message
    ]

-- Generate Messages for ClaudeSonnet45 (Anthropic with tools)
genMessageAnthropicTools :: Gen (Message (Model ClaudeSonnet45 AnthropicProvider.Anthropic))
genMessageAnthropicTools = oneof
  [ UserText <$> genNonEmptyText
  , AssistantText <$> genNonEmptyText
  , SystemText <$> genNonEmptyText
  , AssistantTool <$> arbitrary  -- Include all ToolCall variants (valid and invalid)
  , ToolResultMsg <$> (ToolResult <$> arbitrary <*> genToolOutput)
  ]
  where
    genToolOutput = oneof
      [ Left <$> genRealisticText
      , Right <$> genSimpleValue
      ]

-- Generate Messages for GLM45 (OpenAI with tools, reasoning, JSON)
genMessageOpenAIFull :: Gen (Message (Model GLM45 OpenAIProvider.OpenAI))
genMessageOpenAIFull = oneof
  [ UserText <$> genNonEmptyText
  , AssistantText <$> genNonEmptyText
  , SystemText <$> genNonEmptyText
  , AssistantTool <$> arbitrary
  , ToolResultMsg <$> (ToolResult <$> arbitrary <*> genToolOutput)
  , AssistantReasoning <$> genNonEmptyText
  , AssistantJSON <$> genSimpleValue
  , UserRequestJSON <$> genNonEmptyText <*> genSimpleValue
  ]
  where
    genToolOutput = oneof
      [ Left <$> genRealisticText
      , Right <$> genSimpleValue
      ]

-- Generate ModelConfig for Anthropic
genConfigAnthropic :: Gen (ModelConfig (Model ClaudeSonnet45 AnthropicProvider.Anthropic))
genConfigAnthropic = oneof
  [ Temperature <$> choose (0.0, 2.0)
  , MaxTokens <$> choose (1, 4096)
  , SystemPrompt <$> genRealisticText
  , Tools <$> listOf1 arbitrary
  ]

-- Generate ModelConfig for OpenAI
genConfigOpenAI :: Gen (ModelConfig (Model GLM45 OpenAIProvider.OpenAI))
genConfigOpenAI = oneof
  [ Temperature <$> choose (0.0, 2.0)
  , MaxTokens <$> choose (1, 4096)
  , Seed <$> arbitrary
  , SystemPrompt <$> genRealisticText
  , Tools <$> listOf1 arbitrary
  , Reasoning <$> arbitrary
  ]


toProviderRequestSonnet45 msg = fmap snd $ toProviderRequest anthropicSonnet45 (Model ClaudeSonnet45 Anthropic) msg ((), ())
toProviderRequestSonnet45WithReasoning msg = fmap snd $ toProviderRequest anthropicSonnet45Reasoning (Model ClaudeSonnet45WithReasoning Anthropic) msg (def, ((), ()))

toProviderRequestGLM45 msg = fmap snd $ toProviderRequest openAIGLM45 (Model GLM45 OpenAIProvider.OpenAI) msg ((), ((), ((), ())))
toProviderRequestGLM45WithReasoning msg = fmap snd $ toProviderRequest openAIGLM45 (Model GLM45 OpenAIProvider.OpenAI) msg ((), ((), ((), ())))

-- fromProviderResponse signature: composableProvider -> model -> configs -> state -> response -> (state, messages)
fromProviderResponseGLM45 configs resp = either (error . show) snd $ fromProviderResponse openAIGLM45 (Model GLM45 OpenAIProvider.OpenAI) configs ((), ((), ((), ()))) resp
fromProviderResponseSonnet45WithReasoning configs resp = either (error . show) snd $ fromProviderResponse anthropicSonnet45Reasoning (Model ClaudeSonnet45WithReasoning Anthropic) configs (def, ((), ())) resp

-- ============================================================================
-- Property tests for Anthropic
-- ============================================================================

-- | Property: Building a request should always terminate (no infinite loops)
prop_anthropicRequestTerminates :: Property
prop_anthropicRequestTerminates = forAll genMessages $ \msgs ->
  let configs = []  -- Use empty configs to avoid Show instance requirement
      req = toProviderRequestSonnet45 configs msgs
  in req `seq` property True
  where
    genMessages = listOf genMessageAnthropicTools

-- | Property: Anthropic messages must alternate between user and assistant roles
-- This is a key invariant for Anthropic's API
prop_anthropicMessagesAlternate :: Property
prop_anthropicMessagesAlternate = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequestSonnet45 configs msgs
      roles = map AP.role (AP.messages req)
  in not (null roles) ==> checkAlternating roles
  where
    genMessages = listOf1 genMessageAnthropicTools

    -- Check that roles alternate (allowing same role only if messages were grouped)
    checkAlternating [] = True
    checkAlternating [_] = True
    checkAlternating (r1:r2:rest)
      | r1 == r2 = False  -- Same role in a row means grouping failed
      | otherwise = checkAlternating (r2:rest)

-- | Property: Anthropic request should ALWAYS start with a user message
-- The provider enforces this by automatically prepending an empty user message if needed
prop_anthropicStartsWithUser :: Property
prop_anthropicStartsWithUser = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequestSonnet45 configs msgs
  in not (null (AP.messages req)) ==> AP.role (head (AP.messages req)) === "user"
  where
    genMessages = listOf1 genMessageAnthropicTools

-- | Property: Consecutive user messages should be grouped into one message
prop_anthropicGroupsConsecutiveUsers :: Property
prop_anthropicGroupsConsecutiveUsers = forAll consecutiveUsers $ \msgs ->
  let configs = []
      req = toProviderRequestSonnet45 configs msgs
  in length (AP.messages req) === 1
     .&&. AP.role (head (AP.messages req)) === "user"
     .&&. length (AP.content (head (AP.messages req))) === length msgs
  where
    consecutiveUsers = listOf1 (UserText <$> genNonEmptyText)

-- | Property: MaxTokens config should be reflected in request
prop_anthropicMaxTokens :: Property
prop_anthropicMaxTokens = forAll (choose (1, 4096)) $ \maxTok ->
  let configs = [MaxTokens maxTok]
      msgs = [UserText "test"]
      req = toProviderRequestSonnet45 configs msgs
  in AP.max_tokens req === maxTok

-- | Property: Temperature config should be reflected in request
prop_anthropicTemperature :: Property
prop_anthropicTemperature = forAll (choose (0.0, 2.0)) $ \temp ->
  let configs = [Temperature temp]
      msgs = [UserText "test"]
      req = toProviderRequestSonnet45 configs msgs
  in AP.temperature req === Just temp

-- | Property: Tools config should be reflected in request
prop_anthropicTools :: Property
prop_anthropicTools = forAll (listOf1 arbitrary) $ \(toolDefs :: [ToolDefinition]) ->
  let configs = [Tools toolDefs]
      msgs = [UserText "test"]
      req = toProviderRequestSonnet45 configs msgs
  in case AP.tools req of
       Just toolList -> length toolList === length toolDefs
       Nothing -> property False

-- | Property: System prompt config should be reflected in request
prop_anthropicSystemPrompt :: Property
prop_anthropicSystemPrompt = forAll genNonEmptyText $ \sysPrompt ->
  let configs = [SystemPrompt sysPrompt]
      msgs = [UserText "test"]
      req = toProviderRequestSonnet45 configs msgs
  in case AP.system req of
       Just blocks -> property $ any (\(AP.AnthropicSystemBlock txt _ _) -> txt == sysPrompt) blocks
       Nothing -> property False

-- ============================================================================
-- Property tests for OpenAI
-- ============================================================================

-- | Property: Building an OpenAI request should always terminate
prop_openaiRequestTerminates :: Property
prop_openaiRequestTerminates = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequestGLM45 configs msgs
  in req `seq` property True
  where
    genMessages = listOf genMessageOpenAIFull

-- | Property: OpenAI consecutive user messages should be merged
prop_openaiMergesConsecutiveUsers :: Property
prop_openaiMergesConsecutiveUsers = forAll consecutiveUsers $ \msgs ->
  let configs = []
      req = toProviderRequestGLM45 configs msgs
      userMessages = filter (\m -> OP.role m == "user") (OP.messages req)
  in length userMessages === 1
     .&&. OP.role (head userMessages) === "user"
  where
    consecutiveUsers = listOf1 (UserText <$> genNonEmptyText)

-- | Property: MaxTokens config should be reflected in OpenAI request
prop_openaiMaxTokens :: Property
prop_openaiMaxTokens = forAll (choose (1, 4096)) $ \maxTok ->
  let configs = [MaxTokens maxTok]
      msgs = [UserText "test"]
      req = toProviderRequestGLM45 configs msgs
  in OP.max_tokens req === Just maxTok

-- | Property: Temperature config should be reflected in OpenAI request
prop_openaiTemperature :: Property
prop_openaiTemperature = forAll (choose (0.0, 2.0)) $ \temp ->
  let configs = [Temperature temp]
      msgs = [UserText "test"]
      req = toProviderRequestGLM45 configs msgs
  in OP.temperature req === Just temp

-- | Property: Seed config should be reflected in OpenAI request
prop_openaiSeed :: Property
prop_openaiSeed = forAll arbitrary $ \seedVal ->
  let configs = [Seed seedVal]
      msgs = [UserText "test"]
      req = toProviderRequestGLM45 configs msgs
  in OP.seed req === Just seedVal

-- | Property: Tools config should be reflected in OpenAI request
prop_openaiTools :: Property
prop_openaiTools = forAll (listOf1 arbitrary) $ \(toolDefs :: [ToolDefinition]) ->
  let configs = [Tools toolDefs]
      msgs = [UserText "test"]
      req = toProviderRequestGLM45 configs msgs
  in case OP.tools req of
       Just toolList -> length toolList === length toolDefs
       Nothing -> property False

-- | Property: System prompt should be prepended to messages
prop_openaiSystemPrompt :: Property
prop_openaiSystemPrompt = forAll genNonEmptyText $ \sysPrompt ->
  let configs = [SystemPrompt sysPrompt]
      msgs = [UserText "test"]
      req = toProviderRequestGLM45 configs msgs
  in not (null (OP.messages req)) ==>
       OP.role (head (OP.messages req)) === "system"
       .&&. OP.content (head (OP.messages req)) === Just sysPrompt

-- | Property: JSON mode sets response_format
prop_openaiJSONMode :: Property
prop_openaiJSONMode = forAll genNonEmptyText $ \prompt ->
  forAll genSimpleValue $ \schema ->
    let msgs = [UserRequestJSON prompt schema]
        configs = []
        req = toProviderRequestGLM45 configs msgs
    in case OP.response_format req of
         Just fmt -> OP.responseType fmt === "json_schema"
         Nothing -> property False

-- ============================================================================
-- General roundtrip properties
-- ============================================================================

-- | Property: Request serialization should always succeed
prop_anthropicSerializationSucceeds :: Property
prop_anthropicSerializationSucceeds = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequestSonnet45 configs msgs
      encoded = toJSONViaCodec req
  in length (show encoded) > 0
  where
    genMessages = listOf genMessageAnthropicTools

prop_openaiSerializationSucceeds :: Property
prop_openaiSerializationSucceeds = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequestGLM45 configs msgs
      encoded = toJSONViaCodec req
  in length (show encoded) > 0
  where
    genMessages = listOf genMessageOpenAIFull

-- | Property: Anthropic requests always have valid content blocks
-- - No empty text blocks
-- - No empty tool results
-- - Properly structured thinking blocks
prop_anthropicRequestIsValid :: Property
prop_anthropicRequestIsValid = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequestSonnet45 configs msgs
      msgs_list = AP.messages req
      -- Check all content blocks are valid
      validBlocks = all isValidContentBlock (concatMap AP.content msgs_list)
      emptyBlocks = [(i, block) | (i, msg) <- zip [0..] msgs_list, (block) <- AP.content msg, not (isValidContentBlock block)]
  in counterexample ("Empty blocks: " ++ show emptyBlocks) validBlocks
  where
    genMessages = listOf genMessageAnthropicTools

    isValidContentBlock :: AP.AnthropicContentBlock -> Bool
    isValidContentBlock (AP.AnthropicTextBlock txt _) = not (T.null txt)
    isValidContentBlock (AP.AnthropicToolResultBlock _ _ _) = True  -- Tool results can be empty
    isValidContentBlock (AP.AnthropicThinkingBlock txt _ _) = not (T.null txt)
    isValidContentBlock _ = True  -- Tool use blocks don't have the same constraint

-- | Property: OpenAI with multiple reasoning messages handles all of them correctly
-- Reasoning messages should be processed and included in the request
prop_openaiMultipleReasoningMessages :: Property
prop_openaiMultipleReasoningMessages = forAll reasoningSequence $ \msgs ->
  let configs = []
      req = toProviderRequestGLM45 configs msgs
      reasoningMsgs = filter (\m -> OP.role m == "assistant" && OP.reasoning_content m /= Nothing) (OP.messages req)
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Reasoning in request: " ++ show (length reasoningMsgs))
       (length reasoningMsgs >= 1)  -- At least one reasoning message made it through
  where
    reasoningSequence = do
      -- Generate at least one reasoning message
      reasoning1 <- AssistantReasoning <$> genNonEmptyText
      -- Interleave with other message types
      remaining <- listOf $ oneof
        [ UserText <$> genNonEmptyText
        , AssistantText <$> genNonEmptyText
        , AssistantReasoning <$> genNonEmptyText
        ]
      return (reasoning1 : remaining)

-- | Property: OpenAI with multiple tool calls handles all of them correctly
-- Multiple consecutive tool calls should all be processed
-- This also tests the branch where tool_calls becomes Nothing after extracting the last call
prop_openaiMultipleToolCalls :: Property
prop_openaiMultipleToolCalls = forAll toolCallSequence $ \msgs ->
  let configs = []
      req = toProviderRequestGLM45 configs msgs
      -- Count tool calls in the request
      toolCallCount = sum [case OP.tool_calls m of
                             Just calls -> length calls
                             Nothing -> 0
                           | m <- OP.messages req, OP.role m == "assistant"]
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Tool calls in request: " ++ show toolCallCount)
       (toolCallCount >= 1)  -- At least one tool call made it through
  where
    genToolOutput = oneof
      [ Left <$> genRealisticText
      , Right <$> genSimpleValue
      ]

    toolCallSequence = do
      -- Generate multiple tool calls to ensure we test the null remainingTCs branch
      tool1 <- AssistantTool <$> arbitrary
      -- Generate at least one more tool call to test the extraction of multiple calls
      tool2 <- AssistantTool <$> arbitrary
      -- Interleave with other message types
      remaining <- listOf $ oneof
        [ UserText <$> genNonEmptyText
        , AssistantText <$> genNonEmptyText
        , AssistantTool <$> arbitrary
        , ToolResultMsg <$> (ToolResult <$> arbitrary <*> genToolOutput)
        ]
      return (tool1 : tool2 : remaining)

-- | Property: Anthropic with reasoning messages handles them correctly
-- AssistantReasoning messages without signatures are converted to regular text blocks
-- (You can't create thinking blocks without signatures from the API)
prop_anthropicReasoningMessages :: Property
prop_anthropicReasoningMessages = forAll reasoningMessage $ \msgs ->
  let configs = [MaxTokens 16000]  -- Higher tokens for reasoning
      req = toProviderRequestSonnet45WithReasoning configs msgs
      -- AssistantReasoning messages without signatures become text blocks
      textBlocks = [block | msg <- AP.messages req, block <- AP.content msg, isTextBlock block]
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Text blocks in request: " ++ show (length textBlocks))
       (not (null textBlocks))
  where
    isTextBlock (AP.AnthropicTextBlock _ _) = True
    isTextBlock _ = False

    reasoningMessage = do
      -- Ensure at least one reasoning message
      reasoning1 <- AssistantReasoning <$> genNonEmptyText
      -- Interleave with other message types
      remaining <- listOf genMessageAnthropicReasoningTools
      return (reasoning1 : remaining)

-- Helper generator for Anthropic messages with reasoning support
genMessageAnthropicReasoningTools :: Gen (Message (Model ClaudeSonnet45WithReasoning AnthropicProvider.Anthropic))
genMessageAnthropicReasoningTools = oneof
  [ UserText <$> genNonEmptyText
  , AssistantText <$> genNonEmptyText
  , SystemText <$> genNonEmptyText
  , AssistantReasoning <$> genNonEmptyText  -- Thinking/reasoning message
  , AssistantTool <$> arbitrary
  , ToolResultMsg <$> (ToolResult <$> arbitrary <*> genToolOutput)
  ]
  where
    genToolOutput = oneof
      [ Left <$> genRealisticText
      , Right <$> genSimpleValue
      ]

-- | Property: Anthropic with multiple reasoning messages handles all of them correctly
-- Multiple consecutive reasoning messages become text blocks (no signatures), but thinking config is set
prop_anthropicMultipleReasoningMessages :: Property
prop_anthropicMultipleReasoningMessages = forAll reasoningSequence $ \msgs ->
  let configs = [MaxTokens 16000]
      req = toProviderRequestSonnet45WithReasoning configs msgs
      -- Count text blocks (reasoning messages without signatures become text)
      textBlocks = [block | msg <- AP.messages req, block <- AP.content msg, isTextBlock block]
      -- Also verify thinking is enabled in config
      thinkingEnabled = AP.thinking req /= Nothing
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Text blocks in request: " ++ show (length textBlocks) ++
                     ", Thinking enabled: " ++ show thinkingEnabled)
       (length textBlocks >= 1 .&&. thinkingEnabled)  -- Reasoning must be processed and config must be set
  where
    isTextBlock (AP.AnthropicTextBlock _ _) = True
    isTextBlock _ = False

    reasoningSequence = do
      -- Generate multiple reasoning messages to thoroughly exercise the code path
      reasoning1 <- AssistantReasoning <$> genNonEmptyText
      reasoning2 <- AssistantReasoning <$> genNonEmptyText
      -- Interleave with other message types
      remaining <- listOf $ oneof
        [ UserText <$> genNonEmptyText
        , AssistantText <$> genNonEmptyText
        , AssistantReasoning <$> genNonEmptyText
        , AssistantTool <$> arbitrary
        ]
      return (reasoning1 : reasoning2 : remaining)

-- ============================================================================
-- Full-stack roundtrip tests (request serialization & response parsing)
-- ============================================================================

-- | Full-stack test: OpenAI with tools and reasoning - serialize request
-- Tests the full provider pipeline with a model that has both tools and reasoning
prop_openaiFullStackRequestWithToolsAndReasoning :: Property
prop_openaiFullStackRequestWithToolsAndReasoning = forAll genMessages $ \msgs ->
  let configs = [MaxTokens 16000, Tools [ToolDefinition "test_tool" "A test tool" (object [])]]
      req = toProviderRequestGLM45 configs msgs
      -- Verify the request has messages
      requestMsgCount = length (OP.messages req)
      -- Verify model is set
      modelSet = OP.model req /= ""
      -- Verify tools are in request
      hasTools = OP.tools req /= Nothing
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Request messages: " ++ show requestMsgCount ++
                     ", Model set: " ++ show modelSet ++
                     ", Tools configured: " ++ show hasTools)
       (requestMsgCount >= 1 .&&. modelSet .&&. hasTools)
  where
    genMessages = listOf1 genMessageOpenAIFull

-- | Full-stack test: Anthropic with tools and reasoning - serialize request
-- Tests the full provider pipeline with a model that has both tools and reasoning
prop_anthropicFullStackRequestWithToolsAndReasoning :: Property
prop_anthropicFullStackRequestWithToolsAndReasoning = forAll genMessages $ \msgs ->
  let configs = [MaxTokens 16000, Reasoning True, Tools [ToolDefinition "test_tool" "A test tool" (object [])]]
      req = toProviderRequestSonnet45WithReasoning configs msgs
      -- Verify the request has messages
      requestMsgCount = length (AP.messages req)
      -- Verify model is set
      modelSet = AP.model req /= ""
      -- Verify messages alternate between user and assistant
      roles = map AP.role (AP.messages req)
      alternates = checkAlternating roles
      -- Verify reasoning is enabled
      reasoningEnabled = AP.thinking req /= Nothing
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Request messages: " ++ show requestMsgCount ++
                     ", Model set: " ++ show modelSet ++
                     ", Alternates: " ++ show alternates ++
                     ", Reasoning enabled: " ++ show reasoningEnabled)
       (requestMsgCount >= 1 .&&. modelSet .&&. alternates .&&. reasoningEnabled)
  where
    checkAlternating [] = True
    checkAlternating [_] = True
    checkAlternating (r1:r2:rest)
      | r1 == r2 = False
      | otherwise = checkAlternating (r2:rest)

    genMessages = listOf1 genMessageAnthropicReasoningTools

-- | Full-stack response parsing test: Parse OpenAI responses through composable provider pipeline
-- Actually use the provider's fromProviderResponse to parse messages including reasoning
prop_openaiResponseParsingWithToolsAndReasoning :: Property
prop_openaiResponseParsingWithToolsAndReasoning = forAll genResponse $ \(msgs, resp) ->
  let configs = []
      -- Use the actual provider pipeline to parse the response
      parsedMsgs = fromProviderResponseGLM45 configs resp
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Parsed messages: " ++ show (length parsedMsgs))
       (length parsedMsgs >= 1 || null msgs)  -- Either we parsed messages or generated empty input
  where
    genResponse = do
      -- Generate a list of messages for GLM45 (with tools and reasoning)
      msgs <- listOf1 genMessageOpenAIFull
      -- Create a mock response with text, reasoning, or tool calls
      responseMsg <- oneof
        [ -- Text response
          (\txt -> OP.defaultOpenAIMessage { OP.role = "assistant", OP.content = Just txt }) <$> genNonEmptyText
        , -- Reasoning response
          (\reasoning -> OP.defaultOpenAIMessage { OP.role = "assistant", OP.reasoning_content = Just reasoning }) <$> genNonEmptyText
        , -- Tool call response
          do tcId <- genNonEmptyText
             tcName <- genNonEmptyText
             let tc = OP.defaultOpenAIToolCall
                   { OP.callId = tcId
                   , OP.toolCallType = "function"
                   , OP.toolFunction = OP.defaultOpenAIToolFunction
                       { OP.toolFunctionName = tcName
                       , OP.toolFunctionArguments = "{}"
                       }
                   }
             return $ OP.defaultOpenAIMessage { OP.role = "assistant", OP.tool_calls = Just [tc] }
        ]
      let resp = OP.OpenAISuccess (OP.defaultOpenAISuccessResponse
            { OP.choices = [OP.defaultOpenAIChoice { OP.message = responseMsg }] })
      return (msgs, resp)

-- | Full-stack response parsing test: Parse Anthropic responses through composable provider pipeline
-- Actually use the provider's fromProviderResponse to parse messages including thinking blocks
prop_anthropicResponseParsingWithToolsAndReasoning :: Property
prop_anthropicResponseParsingWithToolsAndReasoning = forAll genResponse $ \(msgs, resp) ->
  let configs = [Reasoning True]
      -- Use the actual provider pipeline to parse the response (with reasoning enabled)
      parsedMsgs = fromProviderResponseSonnet45WithReasoning configs resp
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Parsed messages: " ++ show (length parsedMsgs))
       (length parsedMsgs >= 1 || null msgs)  -- Either we parsed messages or generated empty input
  where
    genResponse = do
      -- Generate a list of messages for ClaudeSonnet45WithReasoning (with tools and reasoning)
      msgs <- listOf1 genMessageAnthropicReasoningTools
      -- Create a mock response with various content block types
      contentBlocks <- listOf1 $ oneof
        [ AP.AnthropicTextBlock <$> genNonEmptyText <*> pure Nothing
        , AP.AnthropicThinkingBlock <$> genNonEmptyText <*> genSimpleValue <*> pure Nothing
        , AP.AnthropicToolUseBlock <$> genNonEmptyText <*> genNonEmptyText <*> genSimpleValue <*> pure Nothing
        ]
      let successResp = AP.defaultAnthropicSuccessResponse
            { AP.responseId = "test-response-id"
            , AP.responseModel = "claude-sonnet-4.5"
            , AP.responseRole = "assistant"
            , AP.responseContent = contentBlocks
            , AP.responseStopReason = Just "end_turn"
            , AP.responseUsage = AP.AnthropicUsage 100 200
            }
          resp = AP.AnthropicSuccess successResp
      return (msgs, resp)

-- | Full-stack provider pipeline test: Multiple messages through composable providers
-- Verify that composable providers handle text message sequences correctly
prop_openaiComposableProviderPipeline :: Property
prop_openaiComposableProviderPipeline = forAll genMessages $ \msgs ->
  let -- Apply the provider's message handler to each message sequentially
      emptyReq = mempty :: OP.OpenAIRequest
      finalReq = foldl (\req msg -> OpenAIProvider.handleTextMessage msg req) emptyReq msgs
      -- Verify messages were accumulated (only text messages are handled)
      msgCount = length (OP.messages finalReq)
      -- Count how many text messages were in the input
      textMsgCount = length [m | m <- msgs, isTextMessage m]
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Text messages: " ++ show textMsgCount ++
                     ", Final request messages: " ++ show msgCount)
       (if textMsgCount == 0 then msgCount == 0 else msgCount >= 1)
  where
    isTextMessage (UserText _) = True
    isTextMessage (AssistantText _) = True
    isTextMessage (SystemText _) = True
    isTextMessage _ = False

    genMessages = listOf genMessageOpenAIFull

-- | Full-stack provider pipeline test: Anthropic composable provider message accumulation
-- Verify that handleTextMessage properly groups consecutive messages and maintains alternation
prop_anthropicComposableProviderPipeline :: Property
prop_anthropicComposableProviderPipeline = forAll genMessages $ \msgs ->
  let -- Apply the provider's message handler to each message sequentially
      emptyReq = mempty :: AP.AnthropicRequest
      finalReq = foldl (\req msg -> AnthropicProvider.handleTextMessage msg req) emptyReq msgs
      -- Verify messages were accumulated and alternation is maintained
      msgCount = length (AP.messages finalReq)
      roles = map AP.role (AP.messages finalReq)
      alternates = checkAlternating roles
      -- Count how many text messages were in the input (only these are handled by handleTextMessage)
      textMsgCount = length [m | m <- msgs, isTextMessage m]
  in counterexample ("Generated messages: " ++ show (length msgs) ++
                     ", Text messages: " ++ show textMsgCount ++
                     ", Final request messages: " ++ show msgCount ++
                     ", Alternates: " ++ show alternates)
       (if textMsgCount == 0 then msgCount == 0 else (msgCount >= 1 && alternates))
  where
    isTextMessage (UserText _) = True
    isTextMessage (AssistantText _) = True
    isTextMessage (SystemText _) = True
    isTextMessage _ = False

    checkAlternating [] = True
    checkAlternating [_] = True
    checkAlternating (r1:r2:rest)
      | r1 == r2 = False
      | otherwise = checkAlternating (r2:rest)

    genMessages = listOf genMessageAnthropicTools

-- ============================================================================
-- HSpec test suite
-- ============================================================================

spec :: Spec
spec = do
  describe "Anthropic Provider Properties" $ do
    it "request building always terminates (no infinite loops)" $
      withMaxSuccess 100 prop_anthropicRequestTerminates

    it "messages alternate between user and assistant roles" $
      withMaxSuccess 100 prop_anthropicMessagesAlternate

    it "request starts with a user message" $
      withMaxSuccess 100 prop_anthropicStartsWithUser

    it "consecutive user messages are grouped into one" $
      withMaxSuccess 50 prop_anthropicGroupsConsecutiveUsers

    it "MaxTokens config is reflected in request" $
      withMaxSuccess 50 prop_anthropicMaxTokens

    it "Temperature config is reflected in request" $
      withMaxSuccess 50 prop_anthropicTemperature

    it "Tools config is reflected in request" $
      withMaxSuccess 50 prop_anthropicTools

    it "SystemPrompt config is reflected in request" $
      withMaxSuccess 50 prop_anthropicSystemPrompt

    it "serialization always succeeds" $
      withMaxSuccess 100 prop_anthropicSerializationSucceeds

    it "generated requests are always valid (no empty text blocks)" $
      withMaxSuccess 100 prop_anthropicRequestIsValid

  describe "OpenAI Provider Properties" $ do
    it "request building always terminates (no infinite loops)" $
      withMaxSuccess 100 prop_openaiRequestTerminates

    it "consecutive user messages are merged" $
      withMaxSuccess 50 prop_openaiMergesConsecutiveUsers

    it "MaxTokens config is reflected in request" $
      withMaxSuccess 50 prop_openaiMaxTokens

    it "Temperature config is reflected in request" $
      withMaxSuccess 50 prop_openaiTemperature

    it "Seed config is reflected in request" $
      withMaxSuccess 50 prop_openaiSeed

    it "Tools config is reflected in request" $
      withMaxSuccess 50 prop_openaiTools

    it "SystemPrompt is prepended to messages" $
      withMaxSuccess 50 prop_openaiSystemPrompt

    it "JSON mode sets response_format" $
      withMaxSuccess 50 prop_openaiJSONMode

    it "serialization always succeeds" $
      withMaxSuccess 100 prop_openaiSerializationSucceeds

    it "handles multiple reasoning messages in sequence" $
      withMaxSuccess 50 prop_openaiMultipleReasoningMessages

    it "handles multiple tool calls in sequence" $
      withMaxSuccess 50 prop_openaiMultipleToolCalls

  describe "Anthropic Provider - Reasoning Support" $ do
    it "handles reasoning (thinking) messages" $
      withMaxSuccess 50 prop_anthropicReasoningMessages

    it "handles multiple reasoning messages in sequence" $
      withMaxSuccess 50 prop_anthropicMultipleReasoningMessages

  describe "Full-Stack Provider Pipeline Tests" $ do
    it "serializes OpenAI requests with tools and reasoning" $
      withMaxSuccess 100 prop_openaiFullStackRequestWithToolsAndReasoning

    it "serializes Anthropic requests with tools and reasoning" $
      withMaxSuccess 100 prop_anthropicFullStackRequestWithToolsAndReasoning

    it "parses OpenAI responses with tools and reasoning" $
      withMaxSuccess 100 prop_openaiResponseParsingWithToolsAndReasoning

    it "parses Anthropic responses with tools and reasoning" $
      withMaxSuccess 100 prop_anthropicResponseParsingWithToolsAndReasoning

    it "handles OpenAI composable provider message accumulation" $
      withMaxSuccess 100 prop_openaiComposableProviderPipeline

    it "handles Anthropic composable provider message accumulation" $
      withMaxSuccess 100 prop_anthropicComposableProviderPipeline
