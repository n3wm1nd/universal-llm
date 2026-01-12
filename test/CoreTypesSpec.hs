{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module CoreTypesSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (Value, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KeyMap
import Data.Functor.Identity (Identity(..))
import Autodocodec (HasCodec(..), named, requiredField')
import qualified Autodocodec as AC
import UniversalLLM
import UniversalLLM.Tools

-- ============================================================================
-- Test Data Types and Tools
-- ============================================================================

-- Simple input type for testing
data SimpleInput = SimpleInput
  { inputText :: Text
  , inputNumber :: Int
  } deriving (Show, Eq)

instance HasCodec SimpleInput where
  codec = named "SimpleInput" $ AC.object "SimpleInput" $
    SimpleInput
      <$> requiredField' "text" AC..= inputText
      <*> requiredField' "number" AC..= inputNumber

-- Simple output type
data SimpleOutput = SimpleOutput
  { outputResult :: Text
  } deriving (Show, Eq)

instance HasCodec SimpleOutput where
  codec = named "SimpleOutput" $ AC.object "SimpleOutput" $
    SimpleOutput
      <$> requiredField' "result" AC..= outputResult

instance ToolParameter SimpleOutput where
  paramName _ n = "output_" <> T.pack (show n)
  paramDescription _ = "an output value"

-- Tool function for echo
echoTool :: Text -> Int -> Identity SimpleOutput
echoTool text number = return $ SimpleOutput $ text <> " (number: " <> T.pack (show number) <> ")"

-- Wrapped version with metadata
echoToolWrapped :: ToolWrapped (Text -> Int -> Identity SimpleOutput) (Text, (Int, ()))
echoToolWrapped = mkToolWithMeta "echo" "Echoes the input text" echoTool
                    "text" "the text to echo"
                    "number" "a number to include"

-- ============================================================================
-- Generators
-- ============================================================================

genSimpleInput :: Gen SimpleInput
genSimpleInput = SimpleInput
  <$> (T.pack <$> listOf1 (elements (['a'..'z'] ++ ['A'..'Z'] ++ " ")))
  <*> arbitrary

genValidToolCall :: Text -> Gen ToolCall
genValidToolCall name = do
  input <- genSimpleInput
  let params = Aeson.object
        [ "text" .= inputText input
        , "number" .= inputNumber input
        ]
  callId <- T.pack <$> listOf1 (elements (['a'..'z'] ++ ['0'..'9']))
  return $ ToolCall callId name params

genInvalidToolCall :: Gen ToolCall
genInvalidToolCall = do
  callId <- T.pack <$> listOf1 (elements (['a'..'z'] ++ ['0'..'9']))
  name <- T.pack <$> listOf1 (elements (['a'..'z']))
  args <- T.pack <$> arbitrary
  errMsg <- T.pack <$> listOf1 (elements (['a'..'z'] ++ " "))
  return $ InvalidToolCall callId name args errMsg

-- ============================================================================
-- Property Tests
-- ============================================================================

-- | Property: InvalidToolCall should always return an error
prop_invalidToolCallReturnsError :: Property
prop_invalidToolCallReturnsError = forAll genInvalidToolCall $ \invalidCall ->
  let Identity result = executeToolCallFromList @Identity [] invalidCall
  in case toolResultOutput result of
       Left err -> property $ not (T.null err)  -- Should have an error message
       Right _ -> property False  -- Should never succeed

-- | Property: Calling a non-existent tool should return "Tool not found" error
prop_nonExistentToolReturnsError :: Property
prop_nonExistentToolReturnsError = forAll (genValidToolCall "nonexistent") $ \toolCall ->
  let tools = [LLMTool echoToolWrapped]  -- Available tools don't include "nonexistent"
      Identity result = executeToolCallFromList tools toolCall
  in case toolResultOutput result of
       Left err -> property $ T.isInfixOf "Tool not found" err
       Right _ -> property False

-- | Property: Invalid parameters should return parsing error
prop_invalidParametersReturnError :: Property
prop_invalidParametersReturnError = property $
  let toolCall = ToolCall "call_123" "echo" (Aeson.object ["wrong_field" .= ("test" :: Text)])
      tools = [LLMTool echoToolWrapped]
      Identity result = executeToolCallFromList tools toolCall
  in case toolResultOutput result of
       Left err -> property $ T.isInfixOf "parameter" err || T.isInfixOf "Parameter" err
       Right _ -> property False

-- | Property: Valid tool call with valid parameters should succeed
prop_validToolCallSucceeds :: Property
prop_validToolCallSucceeds = forAll genSimpleInput $ \input ->
  let params = Aeson.object
        [ "text" .= inputText input
        , "number" .= inputNumber input
        ]
      toolCall = ToolCall "call_456" "echo" params
      tools = [LLMTool echoToolWrapped]
      Identity result = executeToolCallFromList tools toolCall
  in case toolResultOutput result of
       Left _ -> property False
       Right output ->
         case output of
           Aeson.Object obj ->
             case KeyMap.lookup (Key.fromText "result") obj of
               Just (Aeson.String resultText) ->
                 T.isInfixOf (inputText input) resultText
                 .&&. T.isInfixOf (T.pack $ show $ inputNumber input) resultText
               _ -> property False
           _ -> property False

-- | Property: Tool result should preserve the original tool call
prop_toolResultPreservesCall :: Property
prop_toolResultPreservesCall = forAll (genValidToolCall "echo") $ \toolCall ->
  let tools = [LLMTool echoToolWrapped]
      Identity result = executeToolCallFromList tools toolCall
  in toolResultCall result === toolCall

-- | Property: getToolCallName should extract the correct name
prop_getToolCallNameCorrect :: Property
prop_getToolCallNameCorrect =
  forAll (elements ["echo", "test", "compute"]) $ \name ->
  forAll (genValidToolCall name) $ \toolCall ->
    getToolCallName toolCall === name

-- | Property: getToolCallName works for InvalidToolCall too
prop_getToolCallNameWorksForInvalid :: Property
prop_getToolCallNameWorksForInvalid = forAll genInvalidToolCall $ \invalidCall ->
  let name = getToolCallName invalidCall
  in not (T.null name)

-- | Property: toToolDefinition extracts correct metadata
prop_toToolDefinitionCorrect :: Property
prop_toToolDefinitionCorrect = property $
  let toolDef = toToolDefinition echoToolWrapped
  in toolDefName toolDef === "echo"
     .&&. toolDefDescription toolDef === "Echoes the input text"
     .&&. (toolDefParameters toolDef /= Aeson.Null)  -- Should have schema

-- | Property: Tool definition parameters should be valid JSON schema
prop_toolDefinitionHasValidSchema :: Property
prop_toolDefinitionHasValidSchema = property $
  let toolDef = toToolDefinition echoToolWrapped
  in case toolDefParameters toolDef of
       Aeson.Object _ -> property True  -- Valid schema object
       _ -> property False

-- ============================================================================
-- HSpec Test Suite
-- ============================================================================

spec :: Spec
spec = do
  describe "Core Types - executeToolCall" $ do
    describe "Error Handling" $ do
      it "returns error for InvalidToolCall" $
        withMaxSuccess 100 prop_invalidToolCallReturnsError

      it "returns 'Tool not found' for non-existent tools" $
        withMaxSuccess 100 prop_nonExistentToolReturnsError

      it "returns 'Invalid parameters' for wrong parameter schema" $
        prop_invalidParametersReturnError

    describe "Successful Execution" $ do
      it "executes valid tool calls successfully" $
        withMaxSuccess 100 prop_validToolCallSucceeds

      it "preserves the original tool call in the result" $
        withMaxSuccess 100 prop_toolResultPreservesCall

    describe "Helper Functions" $ do
      it "getToolCallName extracts correct name from ToolCall" $
        withMaxSuccess 50 prop_getToolCallNameCorrect

      it "getToolCallName works for InvalidToolCall" $
        withMaxSuccess 50 prop_getToolCallNameWorksForInvalid

      it "toToolDefinition extracts correct metadata" $
        prop_toToolDefinitionCorrect

      it "toToolDefinition produces valid JSON schema" $
        prop_toolDefinitionHasValidSchema

  describe "Core Types - Unit Tests" $ do
    it "executes echo tool with specific input" $ do
      let input = SimpleInput "hello" 42
          params = Aeson.object ["text" .= ("hello" :: Text), "number" .= (42 :: Int)]
          toolCall = ToolCall "test_call" "echo" params
          tools = [LLMTool echoToolWrapped]
          Identity result = executeToolCallFromList tools toolCall

      case toolResultOutput result of
        Left err -> expectationFailure $ "Expected success, got error: " ++ T.unpack err
        Right output ->
          case output of
            Aeson.Object obj ->
              case KeyMap.lookup (Key.fromText "result") obj of
                Just (Aeson.String resultText) -> do
                  T.isInfixOf "hello" resultText `shouldBe` True
                  T.isInfixOf "42" resultText `shouldBe` True
                _ -> expectationFailure "Expected string result"
            _ -> expectationFailure "Expected object output"

    it "handles missing required field" $ do
      let params = Aeson.object ["text" .= ("hello" :: Text)]  -- Missing "number"
          toolCall = ToolCall "test_call" "echo" params
          tools = [LLMTool echoToolWrapped]
          Identity result = executeToolCallFromList tools toolCall

      case toolResultOutput result of
        Left err -> T.isInfixOf "Missing parameter" err `shouldBe` True
        Right _ -> expectationFailure "Expected parameter parsing error"

    it "handles extra unexpected fields gracefully" $ do
      let params = Aeson.object
            [ "text" .= ("hello" :: Text)
            , "number" .= (42 :: Int)
            , "extra_field" .= ("ignored" :: Text)
            ]
          toolCall = ToolCall "test_call" "echo" params
          tools = [LLMTool echoToolWrapped]
          Identity result = executeToolCallFromList tools toolCall

      -- Extra fields should be ignored, tool should still work
      case toolResultOutput result of
        Left err -> expectationFailure $ "Should ignore extra fields, got error: " ++ T.unpack err
        Right _ -> return ()
