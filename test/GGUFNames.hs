{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module GGUFNames
  ( canonicalizeGGUFNames
  , canonicalizationTests
  , queryLlamaCppModel
  ) where

import Test.Hspec
import Data.List (intercalate, nub)
import Data.Char (toLower, isDigit, isUpper)
import System.FilePath (takeBaseName)
import Control.Exception (catch, SomeException)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KeyMap
import qualified Data.Vector as V
import qualified TestHTTP

-- | Multi-part file indicators: "0001", "of", "0002"
isMultiPart :: String -> Bool
isMultiPart "of" = True
isMultiPart p = all isDigit p && length p >= 4

-- | Quantization methods: "Q8_K_XL", "Q4_K_M"
isQuantization :: String -> Bool
isQuantization ('Q':d:rest) | isDigit d && '_' `elem` rest = True
isQuantization _ = False

-- | Short version/variant tags: "UD", "A3B"
-- Must NOT be a size parameter (those end in B/M and should be kept)
isVersionTag :: String -> Bool
isVersionTag p = length p <= 3
              && all isUpperOrDigit p
              && any isUpper p
              && not (isSizeParam p)
  where isUpperOrDigit c = isUpper c || isDigit c

-- | Parameter count: "30B", "7B", "1.5B"
isSizeParam :: String -> Bool
isSizeParam p = case reverse p of
  ('B':rest) | not (null rest) && all isDigitOrDot rest -> True
  ('M':rest) | not (null rest) && all isDigitOrDot rest -> True
  _ -> False
  where isDigitOrDot c = isDigit c || c == '.'

-- | Known tuning/training indicators
isTuning :: String -> Bool
isTuning "Instruct" = True
isTuning "Chat" = True
isTuning "Code" = True
isTuning "Coder" = True
isTuning _ = False

-- | Split string on delimiter
splitOn :: Char -> String -> [String]
splitOn _ "" = []
splitOn delim str =
  let (before, remainder) = break (== delim) str
  in before : case remainder of
                [] -> []
                (_:after) -> splitOn delim after

-- | Canonicalize a GGUF filename to possible model names
-- Example: "Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf"
-- Returns: Multiple variants including:
--   - Progressive truncation: "Qwen3-Coder-30B-Instruct", "Qwen3-Coder-30B", "Qwen3-Coder"
--   - Without size params: "Qwen3-Coder-Instruct"
--   - Dash variants: "Qwen-3-Coder"
--   - Lowercase versions of all
canonicalizeGGUFNames :: String -> [String]
canonicalizeGGUFNames filename =
  let base = takeBaseName filename
      parts = splitOn '-' base
      -- Drop junk from the END (quantization, version tags, multi-part)
      cleaned = reverse $ dropWhile isJunk (reverse parts)
      -- Generate variants
      progressive = [intercalate "-" (take n cleaned) | n <- reverse [1..length cleaned], n > 0]
      -- Generate progressive variants on the cleaned list (with version tags removed)
      noVersions = filter (not . isVersionTag) cleaned
      progressiveNoVer = [intercalate "-" (take n noVersions) | n <- reverse [1..length noVersions], n > 0]
      -- Also generate variants with optional parts filtered out
      withoutOptional = [intercalate "-" (filter isCore cleaned)]  -- Drop size + version
      allVariants = nub (progressive ++ progressiveNoVer ++ withoutOptional)
      -- Add dash-in-name variants ("Qwen3" -> "Qwen-3")
      withDashes = concatMap addDashInName allVariants
      -- Add lowercase
      withCase = nub (withDashes ++ map (map toLower) withDashes)
      -- Filter out empty strings
      valid = filter (not . null) withCase
  in valid
  where
    -- Parts to drop from the end only
    isJunk p = isMultiPart p || isQuantization p || isVersionTag p

    -- Core parts to keep when filtering (not size params or version tags)
    isCore p = not (isSizeParam p || isVersionTag p)

    -- Add dash before first digit in parts: "Qwen3-Coder" -> "Qwen-3-Coder"
    addDashInName :: String -> [String]
    addDashInName s =
      let parts' = splitOn '-' s
          transformed = map dashBeforeDigit parts'
          result = intercalate "-" transformed
      in if result == s then [s] else [s, result]

    -- "Qwen3" -> "Qwen-3", but "4.5" stays "4.5"
    dashBeforeDigit :: String -> String
    dashBeforeDigit part = case break isDigit part of
      (prefix, suffix) | not (null prefix) && not (null suffix) && head suffix /= '.'
        -> prefix ++ "-" ++ suffix
      _ -> part

-- | Query llama.cpp server for loaded model information
-- Returns the list of possible canonicalized model names
queryLlamaCppModel :: String -> IO (Maybe [String])
queryLlamaCppModel baseUrl = do
  let url = baseUrl ++ "/v1/models"
  catch (do
    response <- TestHTTP.httpGet url
    case Aeson.decode response of
      Just (Aeson.Object obj) -> do
        -- llama.cpp returns: {"data": [{"id": "model-name.gguf", ...}]}
        case KeyMap.lookup "data" obj of
          Just (Aeson.Array models) | not (V.null models) -> do
            case V.head models of
              Aeson.Object modelObj -> do
                case KeyMap.lookup "id" modelObj of
                  Just (Aeson.String modelId) ->
                    return $ Just (canonicalizeGGUFNames $ T.unpack modelId)
                  _ -> return Nothing
              _ -> return Nothing
          _ -> return Nothing
      _ -> return Nothing
    ) (\(_ :: SomeException) -> return Nothing)

-- | Test suite for canonicalization
canonicalizationTests :: Spec
canonicalizationTests = describe "GGUF Name Canonicalization" $ do
  it "handles GLM-4.5-Air with quantization" $ do
    let result = canonicalizeGGUFNames "GLM-4.5-Air-Q4_K_M.gguf"
    result `shouldContain` ["GLM-4.5-Air"]
    result `shouldContain` ["glm-4.5-air"]
    result `shouldContain` ["GLM-4.5"]
    result `shouldContain` ["GLM"]

  it "handles Qwen with size param and variants" $ do
    let result = canonicalizeGGUFNames "Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf"
    result `shouldContain` ["Qwen3-Coder-30B-Instruct"]
    result `shouldContain` ["Qwen3-Coder-Instruct"]  -- Without size
    result `shouldContain` ["Qwen3-Coder"]
    result `shouldContain` ["Qwen-3-Coder"]  -- Dash variant
    result `shouldContain` ["qwen-3-coder"]  -- Lowercase
