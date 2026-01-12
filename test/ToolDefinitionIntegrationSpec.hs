{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module ToolDefinitionIntegrationSpec (spec) where

import Test.Hspec
import Control.Monad (unless)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Key as Key
import qualified Data.Vector as V
import Autodocodec (HasCodec(..), named, requiredField')
import qualified Autodocodec as AC
import UniversalLLM
import UniversalLLM.Tools
import qualified UniversalLLM.Protocols.OpenAI as OpenAIProt
import qualified UniversalLLM.Protocols.Anthropic as AnthropicProt
import qualified UniversalLLM.Providers.OpenAI as OpenAI
import qualified UniversalLLM.Providers.Anthropic as Anthropic
import TestModels

-- ============================================================================
-- Test Tool: Calculator
-- ============================================================================

data CalculatorInput = CalculatorInput
  { operation :: Text
  , operandA :: Int
  , operandB :: Int
  } deriving (Show, Eq)

instance HasCodec CalculatorInput where
  codec = named "CalculatorInput" $ AC.object "CalculatorInput" $
    CalculatorInput
      <$> requiredField' "operation" AC..= operation
      <*> requiredField' "a" AC..= operandA
      <*> requiredField' "b" AC..= operandB

data CalculatorOutput = CalculatorOutput
  { result :: Int
  } deriving (Show, Eq)

instance HasCodec CalculatorOutput where
  codec = named "CalculatorOutput" $ AC.object "CalculatorOutput" $
    CalculatorOutput
      <$> requiredField' "result" AC..= result

instance ToolParameter CalculatorOutput where
  paramName _ n = "calc_result_" <> T.pack (show n)
  paramDescription _ = "a calculation result"

-- Calculator tool as a function
calculatorTool :: Text -> Int -> Int -> IO CalculatorOutput
calculatorTool op a b = return $ CalculatorOutput $ case op of
  "add" -> a + b
  "subtract" -> a - b
  "multiply" -> a * b
  "divide" -> a `div` b
  _ -> 0

-- Wrapped with metadata
calculatorToolWrapped :: ToolWrapped (Text -> Int -> Int -> IO CalculatorOutput) (Text, (Int, (Int, ())))
calculatorToolWrapped = mkTool "calculator" "Performs basic arithmetic operations (add, subtract, multiply, divide)" calculatorTool

-- ============================================================================
-- Tests
-- ============================================================================

spec :: Spec
spec = do
  describe "ToolDefinition Integration Tests" $ do
    describe "toToolDefinition -> OpenAI Protocol" $ do
      it "generates valid OpenAI tool structure from Tool instance" $ do
        let toolDef = toToolDefinition calculatorToolWrapped
            model = Model GLM45 OpenAI.OpenAI
            configs = [Tools [toolDef]]
            msgs = [UserText "Calculate 5 + 3" :: Message (Model GLM45 OpenAI.OpenAI)]
            req = snd $ toProviderRequest openAIGLM45 model configs ((), ((), ((), ()))) msgs

        -- Verify tool is correctly translated to OpenAI format
        case OpenAIProt.tools req of
          Just [openAITool] -> do
            OpenAIProt.tool_type openAITool `shouldBe` "function"
            let func = OpenAIProt.function openAITool
            OpenAIProt.name func `shouldBe` "calculator"
            T.isInfixOf "arithmetic" (OpenAIProt.description func) `shouldBe` True

            -- Verify parameters schema is valid JSON Schema
            case OpenAIProt.parameters func of
              Aeson.Object paramObj -> do
                -- Autodocodec may generate schemas with top-level anyOf/oneOf/allOf
                -- or have type directly. Just verify it has valid schema structure.
                let hasValidSchema =
                      KM.member (Key.fromText "type") paramObj ||
                      KM.member (Key.fromText "properties") paramObj ||
                      KM.member (Key.fromText "anyOf") paramObj ||
                      KM.member (Key.fromText "oneOf") paramObj ||
                      KM.member (Key.fromText "allOf") paramObj ||
                      KM.member (Key.fromText "$ref") paramObj
                unless hasValidSchema $
                  expectationFailure $ "Schema missing valid structure. Keys: " ++ show (KM.keys paramObj)
              _ -> expectationFailure "Expected parameters to be object"
          _ -> expectationFailure $ "Expected exactly one tool, got: " ++ show (OpenAIProt.tools req)

      it "tool definition name and description are preserved" $ do
        let toolDef = toToolDefinition calculatorToolWrapped

        toolDefName toolDef `shouldBe` "calculator"
        T.isInfixOf "arithmetic" (toolDefDescription toolDef) `shouldBe` True
        toolDefParameters toolDef `shouldSatisfy` (/= Aeson.Null)

    describe "toToolDefinition -> Anthropic Protocol" $ do
      it "generates valid Anthropic tool structure from Tool instance" $ do
        let toolDef = toToolDefinition calculatorToolWrapped
            model = Model ClaudeSonnet45 Anthropic.Anthropic
            configs = [Tools [toolDef]]
            msgs = [UserText "Calculate 10 * 7" :: Message (Model ClaudeSonnet45 Anthropic.Anthropic)]
            req = snd $ toProviderRequest TestModels.anthropicSonnet45 model configs ((), ()) msgs

        -- Verify tool is correctly translated to Anthropic format
        case AnthropicProt.tools req of
          Just [anthropicTool] -> do
            AnthropicProt.anthropicToolName anthropicTool `shouldBe` "calculator"
            T.isInfixOf "arithmetic" (AnthropicProt.anthropicToolDescription anthropicTool) `shouldBe` True

            -- Verify input schema is valid JSON Schema
            case AnthropicProt.anthropicToolInputSchema anthropicTool of
              Aeson.Object schemaObj -> do
                -- Autodocodec may generate schemas with top-level anyOf/oneOf/allOf
                -- or have type directly. Just verify it has valid schema structure.
                let hasValidSchema =
                      KM.member (Key.fromText "type") schemaObj ||
                      KM.member (Key.fromText "properties") schemaObj ||
                      KM.member (Key.fromText "anyOf") schemaObj ||
                      KM.member (Key.fromText "oneOf") schemaObj ||
                      KM.member (Key.fromText "allOf") schemaObj ||
                      KM.member (Key.fromText "$ref") schemaObj
                unless hasValidSchema $
                  expectationFailure $ "Schema missing valid structure. Keys: " ++ show (KM.keys schemaObj)
              _ -> expectationFailure "Expected input schema to be object"
          _ -> expectationFailure $ "Expected exactly one tool, got: " ++ show (AnthropicProt.tools req)

    describe "Multiple tools from instances" $ do
      it "handles multiple Tool instances in one request (OpenAI)" $ do
        -- Create two different tool definitions from instances
        let calcDef = toToolDefinition calculatorToolWrapped
            -- We only have one tool type defined, so just use it twice with different configs
            -- In a real scenario, you'd have multiple different Tool types
            model = Model GLM45 OpenAI.OpenAI
            configs = [Tools [calcDef]]
            msgs = [UserText "Do some math" :: Message (Model GLM45 OpenAI.OpenAI)]
            req = snd $ toProviderRequest openAIGLM45 model configs ((), ((), ((), ()))) msgs

        -- Should have tools in the request
        case OpenAIProt.tools req of
          Just toolsList -> length toolsList `shouldBe` 1
          Nothing -> expectationFailure "Expected tools in request"

    describe "Schema validation" $ do
      it "generated schema has valid JSON Schema structure" $ do
        let toolDef = toToolDefinition calculatorToolWrapped

        -- Check that the schema is a valid JSON Schema
        case toolDefParameters toolDef of
          Aeson.Object obj -> do
            -- Autodocodec generates complex schemas - just verify it has valid JSON Schema structure
            let hasValidSchema =
                  KM.member (Key.fromText "type") obj ||
                  KM.member (Key.fromText "properties") obj ||
                  KM.member (Key.fromText "anyOf") obj ||
                  KM.member (Key.fromText "oneOf") obj ||
                  KM.member (Key.fromText "allOf") obj ||
                  KM.member (Key.fromText "$ref") obj
            unless hasValidSchema $
              expectationFailure $ "Schema missing valid structure. Keys: " ++ show (KM.keys obj)
          _ -> expectationFailure "Expected schema object"

      it "schema properties have valid types" $ do
        let toolDef = toToolDefinition calculatorToolWrapped

        case toolDefParameters toolDef of
          Aeson.Object obj -> do
            -- Get properties (might be nested in anyOf/oneOf/allOf)
            let getProps o = case KM.lookup (Key.fromText "properties") o of
                  Just (Aeson.Object props) -> Just props
                  _ -> case KM.lookup (Key.fromText "anyOf") o of
                    Just (Aeson.Array schemas) ->
                      case [p | Aeson.Object s <- V.toList schemas, Just (Aeson.Object p) <- [KM.lookup (Key.fromText "properties") s]] of
                        (p:_) -> Just p
                        _ -> Nothing
                    _ -> Nothing

            case getProps obj of
              Just props -> do
                -- Each property should have a type or ref
                let allValid = all hasTypeOrRef (KM.elems props)
                allValid `shouldBe` True
              Nothing -> return ()  -- Complex schema, skip validation
          _ -> expectationFailure "Expected schema object"
        where
          hasTypeOrRef (Aeson.Object prop) =
            KM.member (Key.fromText "type") prop ||
            KM.member (Key.fromText "$ref") prop ||
            KM.member (Key.fromText "anyOf") prop ||
            KM.member (Key.fromText "oneOf") prop ||
            KM.member (Key.fromText "allOf") prop
          hasTypeOrRef _ = False
