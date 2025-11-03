{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}

module ToolsSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Proxy (Proxy(..))
import Autodocodec (HasCodec(..))


import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (Value, Object)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM

import UniversalLLM.Core.Tools

-- Test the TupleSchema instances
spec :: Spec
spec = describe "Tools" $ do

  describe "TupleSchema" $ do
    it "generates empty schema for ()" $ do
      let schema = tupleToDefaultSchema (Proxy @())
      case Aeson.fromJSON schema of
        Aeson.Success (obj :: Object) -> do
          KM.lookup "type" obj `shouldBe` Just (Aeson.String "object")
          case KM.lookup "properties" obj of
            Just (Aeson.Object props) -> KM.null props `shouldBe` True
            _ -> expectationFailure "properties should be an empty object"
          case KM.lookup "required" obj of
            Just (Aeson.Array arr) -> length arr `shouldBe` 0
            _ -> expectationFailure "required should be an empty array"
        _ -> expectationFailure "Schema should decode to Object"

    it "generates schema for single parameter (Text, ())" $ do
      let schema = tupleToDefaultSchema (Proxy @(Text, ()))
      case Aeson.fromJSON schema of
        Aeson.Success (obj :: Object) -> do
          KM.lookup "type" obj `shouldBe` Just (Aeson.String "object")
          case KM.lookup "properties" obj of
            Just (Aeson.Object props) -> do
              KM.size props `shouldBe` 1
              KM.member "text_0" props `shouldBe` True
            _ -> expectationFailure "properties should be an object with one field"
          case KM.lookup "required" obj of
            Just (Aeson.Array arr) -> length arr `shouldBe` 1
            _ -> expectationFailure "required should have one element"
        _ -> expectationFailure "Schema should decode to Object"

    it "generates schema for two parameters (Text, (Int, ()))" $ do
      let schema = tupleToDefaultSchema (Proxy @(Text, (Int, ())))
      case Aeson.fromJSON schema of
        Aeson.Success (obj :: Object) -> do
          case KM.lookup "properties" obj of
            Just (Aeson.Object props) -> do
              KM.size props `shouldBe` 2
              KM.member "text_0" props `shouldBe` True
              KM.member "number_1" props `shouldBe` True
            _ -> expectationFailure "properties should have two fields"
          case KM.lookup "required" obj of
            Just (Aeson.Array arr) -> length arr `shouldBe` 2
            _ -> expectationFailure "required should have two elements"
        _ -> expectationFailure "Schema should decode to Object"

    it "generates schema for three parameters (Text, (Int, (Bool, ())))" $ do
      let schema = tupleToDefaultSchema (Proxy @(Text, (Int, (Bool, ()))))
      case Aeson.fromJSON schema of
        Aeson.Success (obj :: Object) -> do
          case KM.lookup "properties" obj of
            Just (Aeson.Object props) -> do
              KM.size props `shouldBe` 3
              KM.member "text_0" props `shouldBe` True
              KM.member "number_1" props `shouldBe` True
              KM.member "bool_2" props `shouldBe` True
            _ -> expectationFailure "properties should have three fields"
          case KM.lookup "required" obj of
            Just (Aeson.Array arr) -> length arr `shouldBe` 3
            _ -> expectationFailure "required should have three elements"
        _ -> expectationFailure "Schema should decode to Object"

  describe "ToolWrapped" $ do
    it "preserves tool name and description" $ do
      let tool = mkTool "my_tool" "does something" (return 42 :: IO Int)
      toolName tool `shouldBe` "my_tool"
      toolDescription tool `shouldBe` "does something"

    it "can call wrapped IO action" $ do
      let tool = mkTool "get_number" "returns 42" (return 42 :: IO Int)
      result <- call tool ()
      result `shouldBe` 42

    it "can call wrapped function" $ do
      let addOne :: Int -> IO Int
          addOne x = return (x + 1)
          tool = mkTool "add_one" "adds one" addOne
      result <- call tool (5, ())
      result `shouldBe` 6

  describe "mkToolWithMeta (vary-adic parameter naming)" $ do
    it "works with 0-arity function" $ do
      let tool = mkToolWithMeta "get_number" "returns 42" (return 42 :: IO Int)
      toolName tool `shouldBe` "get_number"
      toolWrapParamMetas tool `shouldBe` []

    it "works with 1-arity function" $ do
      let greet :: Text -> IO Text
          greet name = return $ "Hello, " <> name
          tool = mkToolWithMeta "greet" "greets someone"
                   (\x -> return $ "Hello, " <> x :: IO Text)
                   "name" "person's name"
      toolWrapParamMetas tool `shouldBe` [("name", "person's name")]

    it "works with 2-arity function" $ do
      let add :: Int -> Int -> IO Int
          add x y = return (x + y)
          tool = mkToolWithMeta "add" "adds two numbers" add
                   "x" "first number"
                   "y" "second number"
      toolWrapParamMetas tool `shouldBe` [("x", "first number"), ("y", "second number")]

    it "works with 3-arity function" $ do
      let formatInfo :: Text -> Int -> Bool -> IO Text
          formatInfo name age isStudent =
            return $ name <> " is " <> T.pack (show age) <> " years old"
          tool = mkToolWithMeta "format_info" "formats person info" formatInfo
                   "name" "person's name"
                   "age" "person's age"
                   "isStudent" "whether they are a student"
      toolWrapParamMetas tool `shouldBe`
        [("name", "person's name"), ("age", "person's age"), ("isStudent", "whether they are a student")]

    it "can call vary-adically defined tool" $ do
      let add :: Int -> Int -> IO Int
          add x y = return (x + y)
          tool = mkToolWithMeta "add" "adds two numbers" add
                   "first" "first number"
                   "second" "second number"
      result <- call tool (10, (32, ()))
      result `shouldBe` 42

  describe "TupleParser" $ do
    it "parses empty JSON object to ()" $ do
      let obj = KM.empty
          result = parseJsonToDefaultTuple (Proxy @()) obj
      result `shouldBe` Right ()

    it "parses single parameter from JSON object" $ do
      let obj = KM.fromList [("text_0", Aeson.String "hello")]
          result = parseJsonToDefaultTuple (Proxy @(Text, ())) obj
      result `shouldBe` Right ("hello", ())

    it "parses two parameters from JSON object" $ do
      let obj = KM.fromList [("text_0", Aeson.String "hello"), ("number_1", Aeson.Number 42)]
          result = parseJsonToDefaultTuple (Proxy @(Text, (Int, ()))) obj
      result `shouldBe` Right ("hello", (42, ()))

    it "parses three parameters from JSON object" $ do
      let obj = KM.fromList
            [ ("text_0", Aeson.String "hello")
            , ("number_1", Aeson.Number 42)
            , ("bool_2", Aeson.Bool True)
            ]
          result = parseJsonToDefaultTuple (Proxy @(Text, (Int, (Bool, ())))) obj
      result `shouldBe` Right ("hello", (42, (True, ())))

    it "fails when parameter is missing" $ do
      let obj = KM.fromList [("text_0", Aeson.String "hello")]
          result = parseJsonToDefaultTuple (Proxy @(Text, (Int, ()))) obj
      case result of
        Left err -> T.unpack err `shouldContain` "Missing parameter"
        Right _ -> expectationFailure "Should have failed with missing parameter"

    it "fails when parameter has wrong type" $ do
      let obj = KM.fromList [("text_0", Aeson.Number 42)]  -- Number instead of String
          result = parseJsonToDefaultTuple (Proxy @(Text, ())) obj
      case result of
        Left err -> T.unpack err `shouldContain` "Failed to parse parameter"
        Right _ -> expectationFailure "Should have failed with type error"

  describe "Round-trip tests (toToolDefinition + executeToolCall)" $ do
    it "round-trips a 0-arity IO action" $ do
      let tool = mkTool "get_number" "returns 42" (return 42 :: IO Int)
          toolDef = toToolDefinition tool

      -- Verify the tool definition
      toolDefName toolDef `shouldBe` "get_number"
      toolDefDescription toolDef `shouldBe` "returns 42"

      -- Create a tool call with empty parameters
      let toolCall = ToolCall "call-1" "get_number" (Aeson.object [])

      -- Execute the tool call
      result <- executeToolCall tool toolCall

      -- Verify the result
      case result of
        ToolResult _ (Right jsonResult) -> do
          -- The result should be 42 encoded as JSON
          jsonResult `shouldBe` Aeson.Number 42
        ToolResult _ (Left err) -> expectationFailure $ "Tool call failed: " <> T.unpack err

    it "round-trips a 1-parameter function (Text -> IO Text)" $ do
      let greet :: Text -> IO Text
          greet name = return $ "Hello, " <> name <> "!"
          tool = mkTool "greet" "greets a person by name" greet
          toolDef = toToolDefinition tool

      -- Verify the tool definition
      toolDefName toolDef `shouldBe` "greet"

      -- Verify the schema has the right structure
      case Aeson.fromJSON (toolDefParameters toolDef) of
        Aeson.Success (obj :: Object) -> do
          case KM.lookup "properties" obj of
            Just (Aeson.Object props) -> do
              KM.member "text_0" props `shouldBe` True
            _ -> expectationFailure "properties should be an object"
        _ -> expectationFailure "Schema should decode to Object"

      -- Create a tool call with parameters
      let toolCall = ToolCall "call-2" "greet"
                       (Aeson.object [("text_0", Aeson.String "Alice")])

      -- Execute the tool call
      result <- executeToolCall tool toolCall

      -- Verify the result
      case result of
        ToolResult _ (Right jsonResult) -> do
          jsonResult `shouldBe` Aeson.String "Hello, Alice!"
        ToolResult _ (Left err) -> expectationFailure $ "Tool call failed: " <> T.unpack err

    it "round-trips a 2-parameter function (Int -> Int -> IO Int)" $ do
      let add :: Int -> Int -> IO Int
          add x y = return (x + y)
          tool = mkTool "add" "adds two numbers" add
          toolDef = toToolDefinition tool

      -- Verify the schema has both parameters
      case Aeson.fromJSON (toolDefParameters toolDef) of
        Aeson.Success (obj :: Object) -> do
          case KM.lookup "properties" obj of
            Just (Aeson.Object props) -> do
              KM.size props `shouldBe` 2
              KM.member "number_0" props `shouldBe` True
              KM.member "number_1" props `shouldBe` True
            _ -> expectationFailure "properties should be an object with two fields"
        _ -> expectationFailure "Schema should decode to Object"

      -- Create a tool call with parameters
      let toolCall = ToolCall "call-3" "add"
                       (Aeson.object
                         [ ("number_0", Aeson.Number 10)
                         , ("number_1", Aeson.Number 32)
                         ])

      -- Execute the tool call
      result <- executeToolCall tool toolCall

      -- Verify the result
      case result of
        ToolResult _ (Right jsonResult) -> do
          jsonResult `shouldBe` Aeson.Number 42
        ToolResult _ (Left err) -> expectationFailure $ "Tool call failed: " <> T.unpack err

    it "round-trips a 3-parameter function (Text -> Int -> Bool -> IO Text)" $ do
      let formatInfo :: Text -> Int -> Bool -> IO Text
          formatInfo name age isStudent =
            return $ name <> " is " <> T.pack (show age) <>
                    " years old" <>
                    (if isStudent then " and is a student" else "")
          tool = mkTool "format_info" "formats person information" formatInfo
          toolDef = toToolDefinition tool

      -- Verify the schema has all three parameters
      case Aeson.fromJSON (toolDefParameters toolDef) of
        Aeson.Success (obj :: Object) -> do
          case KM.lookup "properties" obj of
            Just (Aeson.Object props) -> do
              KM.size props `shouldBe` 3
              KM.member "text_0" props `shouldBe` True
              KM.member "number_1" props `shouldBe` True
              KM.member "bool_2" props `shouldBe` True
            _ -> expectationFailure "properties should have three fields"
          case KM.lookup "required" obj of
            Just (Aeson.Array arr) -> length arr `shouldBe` 3
            _ -> expectationFailure "required should have three elements"
        _ -> expectationFailure "Schema should decode to Object"

      -- Create a tool call with all parameters
      let toolCall = ToolCall "call-4" "format_info"
                       (Aeson.object
                         [ ("text_0", Aeson.String "Alice")
                         , ("number_1", Aeson.Number 25)
                         , ("bool_2", Aeson.Bool True)
                         ])

      -- Execute the tool call
      result <- executeToolCall tool toolCall

      -- Verify the result
      case result of
        ToolResult _ (Right jsonResult) -> do
          jsonResult `shouldBe` Aeson.String "Alice is 25 years old and is a student"
        ToolResult _ (Left err) -> expectationFailure $ "Tool call failed: " <> T.unpack err

    it "handles invalid tool calls gracefully" $ do
      let tool = mkTool "add" "adds two numbers" (\x y -> return (x + y :: Int) :: IO Int)
          invalidCall = InvalidToolCall "call-5" "add" "{not valid json" "Invalid JSON"

      result <- executeToolCall tool invalidCall

      case result of
        ToolResult _ (Left err) -> err `shouldBe` "Invalid JSON"
        ToolResult _ (Right _) -> expectationFailure "Should have failed with invalid tool call"

    it "handles missing parameters gracefully" $ do
      let add :: Int -> Int -> IO Int
          add x y = return (x + y)
          tool = mkTool "add" "adds two numbers" add
          -- Create a tool call with only one parameter (missing number_1)
          toolCall = ToolCall "call-6" "add"
                       (Aeson.object [("number_0", Aeson.Number 10)])

      result <- executeToolCall tool toolCall

      case result of
        ToolResult _ (Left err) -> T.unpack err `shouldContain` "Missing parameter"
        ToolResult _ (Right _) -> expectationFailure "Should have failed with missing parameter"

    it "handles wrong parameter types gracefully" $ do
      let add :: Int -> Int -> IO Int
          add x y = return (x + y)
          tool = mkTool "add" "adds two numbers" add
          -- Create a tool call with wrong types (strings instead of numbers)
          toolCall = ToolCall "call-7" "add"
                       (Aeson.object
                         [ ("number_0", Aeson.String "not a number")
                         , ("number_1", Aeson.Number 32)
                         ])

      result <- executeToolCall tool toolCall

      case result of
        ToolResult _ (Left err) -> T.unpack err `shouldContain` "Failed to parse parameter"
        ToolResult _ (Right _) -> expectationFailure "Should have failed with type error"
