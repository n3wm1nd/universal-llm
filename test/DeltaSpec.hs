{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module DeltaSpec (spec) where

import Data.Aeson (Value, object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Text as T
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Vector as V
import Test.Hspec
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (counterexample, NonEmptyList(..))

import UniversalLLM.Protocols.OpenAI.Delta

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

apply :: Value -> [Value] -> Value
apply acc vs = foldl applyDelta acc (map Delta vs)

obj :: [(Aeson.Key, Value)] -> Value
obj = object

arr :: [Value] -> Value
arr = Aeson.Array . V.fromList

num :: Int -> Value
num = Aeson.Number . fromIntegral

str :: String -> Value
str = Aeson.String . T.pack

-- | Parse a chunk the same way streaming code would
chunk :: Value -> Delta
chunk = Delta

-- ---------------------------------------------------------------------------
-- Spec
-- ---------------------------------------------------------------------------

spec :: Spec
spec = do

  describe "parseDelta" $ do
    it "parses valid JSON" $
      parseDelta "{\"foo\":1}" `shouldBe` Just (Delta (obj ["foo" .= num 1]))

    it "returns Nothing for invalid JSON" $
      parseDelta "not json" `shouldBe` Nothing

    it "returns Nothing for empty input" $
      parseDelta "" `shouldBe` Nothing

  describe "applyDelta - null" $ do
    it "null field in delta leaves accumulator field unchanged" $
      let acc   = obj ["a" .= str "hello"]
          delta = obj ["a" .= Aeson.Null]
      in applyDelta acc (chunk delta) `shouldBe` acc

    it "null delta for missing key is a noop" $
      let acc   = obj []
          delta = obj ["a" .= Aeson.Null]
      in applyDelta acc (chunk delta) `shouldBe` acc

  describe "applyDelta - strings" $ do
    it "appends string delta to existing string" $
      let acc   = obj ["s" .= str "hel"]
          delta = obj ["s" .= str "lo"]
      in applyDelta acc (chunk delta) `shouldBe` obj ["s" .= str "hello"]

    it "initializes missing string field" $
      let acc   = obj []
          delta = obj ["s" .= str "hello"]
      in applyDelta acc (chunk delta) `shouldBe` obj ["s" .= str "hello"]

    it "accumulates across multiple deltas" $
      let acc = obj ["s" .= str ""]
          vs  = map (\c -> obj ["s" .= str [c]]) "hello"
      in apply acc vs `shouldBe` obj ["s" .= str "hello"]

  describe "applyDelta - scalars" $ do
    it "numbers replace" $
      let acc   = obj ["n" .= num 1]
          delta = obj ["n" .= num 2]
      in applyDelta acc (chunk delta) `shouldBe` obj ["n" .= num 2]

    it "booleans replace" $
      let acc   = obj ["b" .= Aeson.Bool True]
          delta = obj ["b" .= Aeson.Bool False]
      in applyDelta acc (chunk delta) `shouldBe` obj ["b" .= Aeson.Bool False]

  describe "applyDelta - objects" $ do
    it "recurses into nested objects" $
      let acc   = obj ["inner" .= obj ["a" .= str "foo"]]
          delta = obj ["inner" .= obj ["a" .= str "bar"]]
      in applyDelta acc (chunk delta) `shouldBe` obj ["inner" .= obj ["a" .= str "foobar"]]

    it "adds new keys from delta into accumulator object" $
      let acc   = obj ["a" .= str "x"]
          delta = obj ["b" .= str "y"]
      in applyDelta acc (chunk delta) `shouldBe` obj ["a" .= str "x", "b" .= str "y"]

    it "initializes missing nested object" $
      let acc   = obj []
          delta = obj ["inner" .= obj ["a" .= str "hi"]]
      in applyDelta acc (chunk delta) `shouldBe` obj ["inner" .= obj ["a" .= str "hi"]]

  describe "applyDelta - indexed arrays" $ do
    it "merges element at index 0 into empty accumulator" $
      let acc   = arr []
          delta = arr [obj ["index" .= num 0, "s" .= str "hi"]]
      in applyDelta acc (chunk delta) `shouldBe` arr [obj ["index" .= num 0, "s" .= str "hi"]]

    it "appends string field within indexed element" $
      let acc   = arr [obj ["index" .= num 0, "s" .= str "hel"]]
          delta = arr [obj ["index" .= num 0, "s" .= str "lo"]]
      in applyDelta acc (chunk delta) `shouldBe`
           arr [obj ["index" .= num 0, "s" .= str "hello"]]

    it "pads accumulator when delta index is beyond its length" $
      let acc   = arr []
          delta = arr [obj ["index" .= num 2, "s" .= str "hi"]]
          result = applyDelta acc (chunk delta)
      in case result of
           Aeson.Array v -> do
             V.length v `shouldBe` 3
             v V.! 2 `shouldBe` obj ["index" .= num 2, "s" .= str "hi"]
           _ -> expectationFailure "expected array"

    it "merges multiple indexed elements in one delta" $
      let acc   = arr []
          delta = arr [ obj ["index" .= num 0, "s" .= str "a"]
                      , obj ["index" .= num 1, "s" .= str "b"]
                      ]
      in applyDelta acc (chunk delta) `shouldBe`
           arr [ obj ["index" .= num 0, "s" .= str "a"]
               , obj ["index" .= num 1, "s" .= str "b"]
               ]

    it "accumulates arguments string across indexed array elements" $
      -- Simulates streaming tool call arguments arriving in pieces
      let acc = arr [obj ["index" .= num 0, "function" .= obj ["arguments" .= str ""]]]
          deltas =
            [ arr [obj ["index" .= num 0, "function" .= obj ["arguments" .= str "{\"loc"]]]
            , arr [obj ["index" .= num 0, "function" .= obj ["arguments" .= str "ation\""]]]
            , arr [obj ["index" .= num 0, "function" .= obj ["arguments" .= str ":\"Paris\"}"]]]
            ]
          result = foldl applyDelta acc (map (Delta . id) deltas)
      in result `shouldBe`
           arr [obj ["index" .= num 0, "function" .= obj ["arguments" .= str "{\"location\":\"Paris\"}"]]]

    it "non-indexed arrays fall back to replace" $
      -- Arrays without index keys are not streamed — just replaced
      let acc   = arr [str "old"]
          delta = arr [str "new"]
      in applyDelta acc (chunk delta) `shouldBe` arr [str "new"]

  describe "real OpenAI streaming shapes" $ do
    it "accumulates a choices[].delta text stream" $
      let mkChunk content =
            obj [ "choices" .= arr
                    [ obj [ "index" .= num 0
                           , "delta" .= obj ["content" .= str content]
                           ]
                    ]
                ]
          vs = map mkChunk ["Hello", ",", " world", "!"]
          result = foldl applyDelta Aeson.Null (map chunk vs)
      in case result of
           Aeson.Object root ->
             case KM.lookup "choices" root of
               Just (Aeson.Array choices) ->
                 case choices V.!? 0 of
                   Just (Aeson.Object choice) ->
                     case KM.lookup "delta" choice of
                       Just (Aeson.Object delta) ->
                         KM.lookup "content" delta `shouldBe`
                           Just (Aeson.String "Hello, world!")
                       _ -> expectationFailure "no delta"
                   _ -> expectationFailure "no choice[0]"
               _ -> expectationFailure "no choices"
           _ -> expectationFailure "not an object"

    it "accumulates tool_call arguments across chunks" $
      -- Mirrors the GLM-5 streaming format from the spec
      let initial =
            obj [ "choices" .= arr
                    [ obj [ "index" .= num 0
                           , "delta" .= obj
                               [ "tool_calls" .= arr
                                   [ obj [ "index" .= num 0
                                         , "id" .= str "call_abc"
                                         , "function" .= obj
                                             [ "name" .= str "read_file"
                                             , "arguments" .= str ""
                                             ]
                                         ]
                                   ]
                               ]
                           ]
                    ]
                ]
          argChunk arg =
            obj [ "choices" .= arr
                    [ obj [ "index" .= num 0
                           , "delta" .= obj
                               [ "tool_calls" .= arr
                                   [ obj [ "index" .= num 0
                                         , "function" .= obj ["arguments" .= str arg]
                                         ]
                                   ]
                               ]
                           ]
                    ]
                ]
          vs = initial : map argChunk ["{\"file_path\"", ": \"README", ".md\"}"]
          result = foldl applyDelta Aeson.Null (map chunk vs)
      in case result of
           Aeson.Object root ->
             case KM.lookup "choices" root of
               Just (Aeson.Array choices) ->
                 case choices V.!? 0 of
                   Just (Aeson.Object choice) ->
                     case KM.lookup "delta" choice of
                       Just (Aeson.Object delta) ->
                         case KM.lookup "tool_calls" delta of
                           Just (Aeson.Array tcs) ->
                             case tcs V.!? 0 of
                               Just (Aeson.Object tc) ->
                                 case KM.lookup "function" tc of
                                   Just (Aeson.Object fn) ->
                                     KM.lookup "arguments" fn `shouldBe`
                                       Just (Aeson.String "{\"file_path\": \"README.md\"}")
                                   _ -> expectationFailure "no function"
                               _ -> expectationFailure "no tool_call[0]"
                           _ -> expectationFailure "no tool_calls"
                       _ -> expectationFailure "no delta"
                   _ -> expectationFailure "no choice[0]"
               _ -> expectationFailure "no choices"
           _ -> expectationFailure "not an object"

  describe "properties" $ do
    prop "applying empty object delta is identity" $
      \() ->
        let acc   = obj ["x" .= str "hello", "n" .= num 42]
            delta = obj []
        in applyDelta acc (Delta delta) `shouldBe` acc

    prop "null accumulator absorbs string-only delta" $
      \(NonEmpty s) ->
        let t = Aeson.String (T.pack s)
        in applyDelta Aeson.Null (Delta t) `shouldBe` t
