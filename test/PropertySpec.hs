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
import UniversalLLM.Core.Types
import qualified UniversalLLM.Protocols.Anthropic as AP
import qualified UniversalLLM.Protocols.OpenAI as OP
import qualified UniversalLLM.Providers.Anthropic as AnthropicProvider
import qualified UniversalLLM.Providers.OpenAI as OpenAIProvider
import TestModels

-- ============================================================================
-- Arbitrary instances
-- ============================================================================

-- Generator for safe text (avoiding control characters that might cause issues)
genSafeText :: Gen Text
genSafeText = T.pack <$> listOf (elements (['a'..'z'] ++ ['A'..'Z'] ++ ['0'..'9'] ++ " .,!?"))

-- Generator for non-empty safe text
genNonEmptyText :: Gen Text
genNonEmptyText = T.pack <$> listOf1 (elements (['a'..'z'] ++ ['A'..'Z'] ++ ['0'..'9'] ++ " .,!?"))

-- Generate a simple JSON Value
genSimpleValue :: Gen Value
genSimpleValue = oneof
  [ Aeson.String <$> genSafeText
  , Aeson.Number . fromInteger <$> arbitrary
  , Aeson.Bool <$> arbitrary
  , return Aeson.Null
  , Aeson.Object <$> pure mempty
  ]

-- Generate ToolDefinition
instance Arbitrary ToolDefinition where
  arbitrary = ToolDefinition
    <$> genNonEmptyText
    <*> genSafeText
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
        <*> genSafeText      -- raw args
        <*> genSafeText      -- error message
    ]

-- Generate Messages for ClaudeSonnet45 (Anthropic with tools)
genMessageAnthropicTools :: Gen (Message ClaudeSonnet45 AnthropicProvider.Anthropic)
genMessageAnthropicTools = oneof
  [ UserText <$> genNonEmptyText
  , AssistantText <$> genNonEmptyText
  , SystemText <$> genNonEmptyText
  , AssistantTool <$> arbitrary  -- Include all ToolCall variants (valid and invalid)
  , ToolResultMsg <$> (ToolResult <$> arbitrary <*> genToolOutput)
  ]
  where
    genToolOutput = oneof
      [ Left <$> genSafeText
      , Right <$> genSimpleValue
      ]

-- Generate Messages for GLM45 (OpenAI with tools, reasoning, JSON)
genMessageOpenAIFull :: Gen (Message GLM45 OpenAIProvider.OpenAI)
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
      [ Left <$> genSafeText
      , Right <$> genSimpleValue
      ]

-- Generate ModelConfig for Anthropic
genConfigAnthropic :: Gen (ModelConfig AnthropicProvider.Anthropic ClaudeSonnet45)
genConfigAnthropic = oneof
  [ Temperature <$> choose (0.0, 2.0)
  , MaxTokens <$> choose (1, 4096)
  , SystemPrompt <$> genSafeText
  , Tools <$> listOf1 arbitrary
  ]

-- Generate ModelConfig for OpenAI
genConfigOpenAI :: Gen (ModelConfig OpenAIProvider.OpenAI GLM45)
genConfigOpenAI = oneof
  [ Temperature <$> choose (0.0, 2.0)
  , MaxTokens <$> choose (1, 4096)
  , Seed <$> arbitrary
  , SystemPrompt <$> genSafeText
  , Tools <$> listOf1 arbitrary
  , Reasoning <$> arbitrary
  ]

-- ============================================================================
-- Property tests for Anthropic
-- ============================================================================

-- | Property: Building a request should always terminate (no infinite loops)
prop_anthropicRequestTerminates :: Property
prop_anthropicRequestTerminates = forAll genMessages $ \msgs ->
  let configs = []  -- Use empty configs to avoid Show instance requirement
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
  in req `seq` property True
  where
    genMessages = listOf genMessageAnthropicTools

-- | Property: Anthropic messages must alternate between user and assistant roles
-- This is a key invariant for Anthropic's API
prop_anthropicMessagesAlternate :: Property
prop_anthropicMessagesAlternate = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
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
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
  in not (null (AP.messages req)) ==> AP.role (head (AP.messages req)) === "user"
  where
    genMessages = listOf1 genMessageAnthropicTools

-- | Property: Consecutive user messages should be grouped into one message
prop_anthropicGroupsConsecutiveUsers :: Property
prop_anthropicGroupsConsecutiveUsers = forAll consecutiveUsers $ \msgs ->
  let configs = []
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
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
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
  in AP.max_tokens req === maxTok

-- | Property: Temperature config should be reflected in request
prop_anthropicTemperature :: Property
prop_anthropicTemperature = forAll (choose (0.0, 2.0)) $ \temp ->
  let configs = [Temperature temp]
      msgs = [UserText "test"]
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
  in AP.temperature req === Just temp

-- | Property: Tools config should be reflected in request
prop_anthropicTools :: Property
prop_anthropicTools = forAll (listOf1 arbitrary) $ \(toolDefs :: [ToolDefinition]) ->
  let configs = [Tools toolDefs]
      msgs = [UserText "test"]
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
  in case AP.tools req of
       Just toolList -> length toolList === length toolDefs
       Nothing -> property False

-- | Property: System prompt config should be reflected in request
prop_anthropicSystemPrompt :: Property
prop_anthropicSystemPrompt = forAll genNonEmptyText $ \sysPrompt ->
  let configs = [SystemPrompt sysPrompt]
      msgs = [UserText "test"]
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
  in case AP.system req of
       Just blocks -> property $ any (\(AP.AnthropicSystemBlock txt _) -> txt == sysPrompt) blocks
       Nothing -> property False

-- ============================================================================
-- Property tests for OpenAI
-- ============================================================================

-- | Property: Building an OpenAI request should always terminate
prop_openaiRequestTerminates :: Property
prop_openaiRequestTerminates = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
  in req `seq` property True
  where
    genMessages = listOf genMessageOpenAIFull

-- | Property: OpenAI consecutive user messages should be merged
prop_openaiMergesConsecutiveUsers :: Property
prop_openaiMergesConsecutiveUsers = forAll consecutiveUsers $ \msgs ->
  let configs = []
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
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
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
  in OP.max_tokens req === Just maxTok

-- | Property: Temperature config should be reflected in OpenAI request
prop_openaiTemperature :: Property
prop_openaiTemperature = forAll (choose (0.0, 2.0)) $ \temp ->
  let configs = [Temperature temp]
      msgs = [UserText "test"]
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
  in OP.temperature req === Just temp

-- | Property: Seed config should be reflected in OpenAI request
prop_openaiSeed :: Property
prop_openaiSeed = forAll arbitrary $ \seedVal ->
  let configs = [Seed seedVal]
      msgs = [UserText "test"]
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
  in OP.seed req === Just seedVal

-- | Property: Tools config should be reflected in OpenAI request
prop_openaiTools :: Property
prop_openaiTools = forAll (listOf1 arbitrary) $ \(toolDefs :: [ToolDefinition]) ->
  let configs = [Tools toolDefs]
      msgs = [UserText "test"]
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
  in case OP.tools req of
       Just toolList -> length toolList === length toolDefs
       Nothing -> property False

-- | Property: System prompt should be prepended to messages
prop_openaiSystemPrompt :: Property
prop_openaiSystemPrompt = forAll genNonEmptyText $ \sysPrompt ->
  let configs = [SystemPrompt sysPrompt]
      msgs = [UserText "test"]
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
  in not (null (OP.messages req)) ==>
       OP.role (head (OP.messages req)) === "system"
       .&&. OP.content (head (OP.messages req)) === Just sysPrompt

-- | Property: JSON mode sets response_format
prop_openaiJSONMode :: Property
prop_openaiJSONMode = forAll genNonEmptyText $ \prompt ->
  forAll genSimpleValue $ \schema ->
    let msgs = [UserRequestJSON prompt schema]
        configs = []
        req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
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
      req = toProviderRequest AnthropicProvider.Anthropic ClaudeSonnet45 configs msgs
      encoded = toJSONViaCodec req
  in length (show encoded) > 0
  where
    genMessages = listOf genMessageAnthropicTools

prop_openaiSerializationSucceeds :: Property
prop_openaiSerializationSucceeds = forAll genMessages $ \msgs ->
  let configs = []
      req = toProviderRequest OpenAIProvider.OpenAI GLM45 configs msgs
      encoded = toJSONViaCodec req
  in length (show encoded) > 0
  where
    genMessages = listOf genMessageOpenAIFull

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
