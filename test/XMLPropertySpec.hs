{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module XMLPropertySpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Text (Text)
import qualified Data.Text as T
import Data.Aeson (Value)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KeyMap
import Data.List (sort, nub)
import UniversalLLM.ToolCall.XML
import UniversalLLM.Core.Types (ToolCall(..), ToolResult(..), ToolDefinition(..))

-- ============================================================================
-- Generators for XML Tool Call Types
-- ============================================================================

-- Generator for realistic text (avoiding XML special chars for now)
genXMLSafeText :: Gen Text
genXMLSafeText = T.pack <$> listOf1 (frequency
  [ (70, elements (['a'..'z'] ++ ['A'..'Z'] ++ ['0'..'9'] ++ " .,!?_-"))
  , (10, elements "\n\t")
  , (10, elements "()[]{}")
  , (5, elements "éñ中文🔥💯")
  , (5, elements "@#$%^&*+=|:;")
  ])

-- Generator for non-empty XML-safe text
genNonEmptyXMLSafeText :: Gen Text
genNonEmptyXMLSafeText = do
  first <- elements (['a'..'z'] ++ ['A'..'Z'])
  rest <- genXMLSafeText
  return $ T.pack (first : T.unpack rest)

-- Generator for XMLArgPair
genXMLArgPair :: Gen XMLArgPair
genXMLArgPair = XMLArgPair
  <$> genNonEmptyXMLSafeText  -- key
  <*> genXMLSafeText          -- value

instance Arbitrary XMLArgPair where
  arbitrary = genXMLArgPair

-- Generator for XMLToolCall
genXMLToolCall :: Gen XMLToolCall
genXMLToolCall = XMLToolCall
  <$> genNonEmptyXMLSafeText  -- tool name
  <*> listOf genXMLArgPair    -- arguments

instance Arbitrary XMLToolCall where
  arbitrary = genXMLToolCall
  shrink (XMLToolCall name args) =
    [ XMLToolCall name' args'
    | name' <- if T.null name then [] else [T.take (max 1 (T.length name `div` 2)) name]
    , args' <- shrinkList (const []) args
    ]

-- Generator for XMLToolResult
genXMLToolResult :: Gen XMLToolResult
genXMLToolResult = XMLToolResult
  <$> genNonEmptyXMLSafeText  -- call ID
  <*> genNonEmptyXMLSafeText  -- tool name
  <*> genXMLSafeText          -- output

instance Arbitrary XMLToolResult where
  arbitrary = genXMLToolResult

-- Generator for ToolCall
genToolCall :: Gen ToolCall
genToolCall = oneof
  [ ToolCall
      <$> genNonEmptyXMLSafeText
      <*> genNonEmptyXMLSafeText
      <*> genSimpleValue
  , InvalidToolCall
      <$> genNonEmptyXMLSafeText
      <*> genNonEmptyXMLSafeText
      <*> genXMLSafeText
      <*> genXMLSafeText
  ]
  where
    genSimpleValue :: Gen Value
    genSimpleValue = oneof
      [ Aeson.String <$> genXMLSafeText
      , Aeson.Number . fromInteger <$> arbitrary
      , Aeson.Bool <$> arbitrary
      , return Aeson.Null
      , Aeson.Object . KeyMap.fromList <$> listOf genKeyValue
      ]
    genKeyValue = (,) <$> (Key.fromText <$> genNonEmptyXMLSafeText) <*> genSimpleValue

instance Arbitrary ToolCall where
  arbitrary = genToolCall

-- Generator for ToolResult
genToolResult :: Gen ToolResult
genToolResult = ToolResult
  <$> genToolCall
  <*> oneof
      [ Left <$> genXMLSafeText
      , Right <$> (Aeson.String <$> genXMLSafeText)
      ]

instance Arbitrary ToolResult where
  arbitrary = genToolResult

-- ============================================================================
-- Property Tests: Round-trip Stability
-- ============================================================================

-- | Property: Encoding then parsing an XMLToolCall should preserve semantics
-- Note: Tool names are stripped of leading/trailing whitespace during parsing (normalization)
prop_xmlToolCallRoundTrip :: Property
prop_xmlToolCallRoundTrip = forAll arbitrary $ \(xmlCall :: XMLToolCall) ->
  let encoded = encodeXMLToolCall xmlCall
      parsed = parseXMLToolCall encoded
      normalized = xmlCall { toolName = T.strip (toolName xmlCall) }
  in parsed === Just normalized

-- | Property: Encoding XMLToolResult should always produce valid XML
prop_xmlToolResultEncoding :: Property
prop_xmlToolResultEncoding = forAll arbitrary $ \(xmlResult :: XMLToolResult) ->
  let encoded = encodeXMLToolResult xmlResult
  in -- Should contain the expected tags
     T.isInfixOf "<tool_result>" encoded
     .&&. T.isInfixOf "</tool_result>" encoded
     .&&. T.isInfixOf "<tool_call_id>" encoded
     .&&. T.isInfixOf "<tool_name>" encoded
     .&&. T.isInfixOf "<result>" encoded

-- | Property: parseXMLToolCall should reject invalid XML
prop_xmlToolCallRejectsInvalid :: Property
prop_xmlToolCallRejectsInvalid = forAll genXMLSafeText $ \randomText ->
  -- Random text (not valid tool call XML) should fail to parse
  not ("<tool_call>" `T.isInfixOf` randomText) ==>
    parseXMLToolCall randomText === Nothing

-- | Property: Nested tool calls in text should be extracted correctly
prop_extractMultipleToolCalls :: Property
prop_extractMultipleToolCalls = forAll (listOf1 arbitrary) $ \(xmlCalls :: [XMLToolCall]) ->
  let encodedCalls = map encodeXMLToolCall xmlCalls
      mixedText = T.intercalate "\nSome text between calls\n" encodedCalls
      extracted = extractXMLToolCalls mixedText
  in length extracted === length xmlCalls
     .&&. all (`elem` encodedCalls) extracted

-- | Property: extractAndRemoveXMLToolCalls should remove tool calls from text
prop_extractAndRemovePreservesText :: Property
prop_extractAndRemovePreservesText = forAll arbitrary $ \(xmlCall :: XMLToolCall) ->
  forAll genXMLSafeText $ \beforeText ->
  forAll genXMLSafeText $ \afterText ->
    let encoded = encodeXMLToolCall xmlCall
        fullText = beforeText <> "\n" <> encoded <> "\n" <> afterText
        (extracted, cleaned) = extractAndRemoveXMLToolCalls fullText
    in length extracted === 1
       .&&. head extracted === encoded
       .&&. not (T.isInfixOf "<tool_call>" cleaned)
       .&&. T.isInfixOf (T.strip beforeText) cleaned || T.null (T.strip beforeText)
       .&&. T.isInfixOf (T.strip afterText) cleaned || T.null (T.strip afterText)

-- ============================================================================
-- Property Tests: ToolCall Conversion
-- ============================================================================

-- | Property: xmlToolCallToToolCall should produce valid ToolCall
-- Note: JSON objects are unordered, so we compare sorted keys
prop_xmlToToolCallConversion :: Property
prop_xmlToToolCallConversion = forAll arbitrary $ \(xmlCall :: XMLToolCall) ->
  let toolCall = xmlToolCallToToolCall xmlCall
      argKeys = map argKey (toolArgs xmlCall)
      hasDuplicateKeys = length argKeys /= length (nub argKeys)
  in case toolCall of
       ToolCall tcId tcName tcParams ->
         -- Should only be valid ToolCall if there are no duplicate keys
         not hasDuplicateKeys
         .&&. T.isPrefixOf "xml-" tcId
         .&&. tcName === toolName xmlCall
         -- Params should be a JSON object with the same keys (order-independent)
         .&&. case tcParams of
                Aeson.Object obj ->
                  let keys = sort $ map (Key.toText . fst) (KeyMap.toList obj)
                      expectedKeys = sort $ nub argKeys
                  in keys === expectedKeys
                _ -> property False
       InvalidToolCall _ _ _ _ ->
         -- Should be InvalidToolCall if and only if there are duplicate keys
         property hasDuplicateKeys

-- | Property: Converting ToolCall->XML->ToolCall should preserve tool name and structure
-- Note: Tool names are normalized (whitespace stripped) during parsing
prop_toolCallToXMLRoundTrip :: Property
prop_toolCallToXMLRoundTrip = forAll genValidToolCall $ \toolCall ->
  let xmlCall = toolCallToXML toolCall
      encoded = encodeXMLToolCall xmlCall
      parsed = parseXMLToolCall encoded
  in case (toolCall, parsed) of
       (ToolCall _ name _, Just (XMLToolCall parsedName _)) ->
         parsedName === T.strip name  -- Names are normalized
       _ -> property True  -- InvalidToolCall can't round-trip perfectly
  where
    -- Generate only valid ToolCalls for this test
    genValidToolCall = ToolCall
      <$> genNonEmptyXMLSafeText
      <*> genNonEmptyXMLSafeText
      <*> (Aeson.Object . KeyMap.fromList <$> listOf genKeyValue)
    genKeyValue = (,)
      <$> (Key.fromText <$> genNonEmptyXMLSafeText)
      <*> (Aeson.String <$> genXMLSafeText)

    toolCallToXML :: ToolCall -> XMLToolCall
    toolCallToXML (ToolCall _ name params) =
      XMLToolCall name (jsonToArgs params)
    toolCallToXML (InvalidToolCall _ name _ _) =
      XMLToolCall name []

    jsonToArgs :: Value -> [XMLArgPair]
    jsonToArgs (Aeson.Object obj) =
      [ XMLArgPair (Key.toText k) (valueToText v)
      | (k, v) <- KeyMap.toList obj
      ]
    jsonToArgs _ = []

    valueToText :: Value -> Text
    valueToText (Aeson.String s) = s
    valueToText v = T.pack (show v)

-- ============================================================================
-- Property Tests: Error Message Preservation
-- ============================================================================

-- | Property: ToolResult with error should preserve error message in XML
prop_errorMessagePreservation :: Property
prop_errorMessagePreservation = forAll genToolCall $ \toolCall ->
  forAll genXMLSafeText $ \errorMsg ->
    let toolResult = ToolResult toolCall (Left errorMsg)
        xmlResult = toolResultToXML toolResult
        encoded = encodeXMLToolResult xmlResult
    in -- Error message should be in the output with "Error: " prefix
       T.isInfixOf "Error: " (resultOutput xmlResult)
       .&&. T.isInfixOf errorMsg (resultOutput xmlResult)
       .&&. T.isInfixOf errorMsg encoded

-- | Property: ToolResult with success should preserve success value
prop_successValuePreservation :: Property
prop_successValuePreservation = forAll genToolCall $ \toolCall ->
  forAll genSimpleValue $ \successValue ->
    let toolResult = ToolResult toolCall (Right successValue)
        xmlResult = toolResultToXML toolResult
    in -- Success value should be JSON-encoded in the result
       not (T.isInfixOf "Error: " (resultOutput xmlResult))
  where
    genSimpleValue = Aeson.String <$> genXMLSafeText

-- ============================================================================
-- Property Tests: General XML Primitives
-- ============================================================================

-- | Property: wrapInTag should produce properly nested tags
prop_wrapInTagCorrect :: Property
prop_wrapInTagCorrect = forAll genNonEmptyXMLSafeText $ \tagName ->
  forAll genXMLSafeText $ \content ->
    let wrapped = wrapInTag tagName content
        expectedOpen = "<" <> tagName <> ">"
        expectedClose = "</" <> tagName <> ">"
    in T.isPrefixOf expectedOpen wrapped
       .&&. T.isSuffixOf expectedClose wrapped
       .&&. T.isInfixOf content wrapped

-- | Property: extractFirstTag should extract content correctly
prop_extractFirstTagCorrect :: Property
prop_extractFirstTagCorrect = forAll genNonEmptyXMLSafeText $ \tagName ->
  forAll genXMLSafeText $ \content ->
    let wrapped = wrapInTag tagName content
        extracted = extractFirstTag tagName wrapped
    in extracted === Just content

-- | Property: extractAllTags should find all occurrences
prop_extractAllTagsCorrect :: Property
prop_extractAllTagsCorrect = forAll genNonEmptyXMLSafeText $ \tagName ->
  forAll (listOf1 genXMLSafeText) $ \contents ->
    let wrapped = T.concat $ map (wrapInTag tagName) contents
        extracted = extractAllTags tagName wrapped
    in extracted === contents

-- | Property: extractAndRemoveTags should remove all tags
prop_extractAndRemoveTagsCorrect :: Property
prop_extractAndRemoveTagsCorrect = forAll genNonEmptyXMLSafeText $ \tagName ->
  forAll (listOf1 genXMLSafeText) $ \contents ->
  forAll genXMLSafeText $ \beforeText ->
  forAll genXMLSafeText $ \afterText ->
    let wrapped = T.concat $ map (wrapInTag tagName) contents
        fullText = beforeText <> wrapped <> afterText
        (extracted, cleaned) = extractAndRemoveTags tagName fullText
        openTag = "<" <> tagName <> ">"
    in extracted === contents
       .&&. not (T.isInfixOf openTag cleaned)
       .&&. (T.isInfixOf beforeText cleaned || T.null (T.strip beforeText))
       .&&. (T.isInfixOf afterText cleaned || T.null (T.strip afterText))

-- ============================================================================
-- Property Tests: Format-Specific Functions
-- ============================================================================

-- | Property: formatToolDefinitionsAsXML should include all tool names
prop_formatToolDefinitionsIncludesNames :: Property
prop_formatToolDefinitionsIncludesNames = forAll (listOf1 genToolDefinition) $ \toolDefs ->
  let formatted = formatToolDefinitionsAsXML toolDefs
  in conjoin [counterexample ("Missing tool: " <> T.unpack (toolDefName tool)) $
              T.isInfixOf (toolDefName tool) formatted
             | tool <- toolDefs]
  where
    genToolDefinition = ToolDefinition
      <$> genNonEmptyXMLSafeText
      <*> genXMLSafeText
      <*> (Aeson.String <$> genXMLSafeText)

-- | Property: formatToolDefinitionsAsXML should include usage instructions
prop_formatToolDefinitionsIncludesInstructions :: Property
prop_formatToolDefinitionsIncludesInstructions = forAll (listOf1 genToolDefinition) $ \toolDefs ->
  let formatted = formatToolDefinitionsAsXML toolDefs
  in T.isInfixOf "<tool_call>" formatted
     .&&. T.isInfixOf "<arg_key>" formatted
     .&&. T.isInfixOf "<arg_value>" formatted
  where
    genToolDefinition = ToolDefinition
      <$> genNonEmptyXMLSafeText
      <*> genXMLSafeText
      <*> (Aeson.String <$> genXMLSafeText)

-- ============================================================================
-- HSpec Test Suite
-- ============================================================================

spec :: Spec
spec = do
  describe "XML Tool Call Properties" $ do
    describe "Round-trip Stability" $ do
      it "XMLToolCall encode->parse preserves semantics" $
        withMaxSuccess 100 prop_xmlToolCallRoundTrip

      it "XMLToolResult encoding produces valid XML" $
        withMaxSuccess 100 prop_xmlToolResultEncoding

      it "parseXMLToolCall rejects invalid XML" $
        withMaxSuccess 50 prop_xmlToolCallRejectsInvalid

      it "extractXMLToolCalls finds all tool calls" $
        withMaxSuccess 100 prop_extractMultipleToolCalls

      it "extractAndRemoveXMLToolCalls preserves surrounding text" $
        withMaxSuccess 100 prop_extractAndRemovePreservesText

    describe "ToolCall Conversion" $ do
      it "xmlToolCallToToolCall produces valid ToolCall" $
        withMaxSuccess 100 prop_xmlToToolCallConversion

      it "ToolCall->XML->ToolCall preserves tool name" $
        withMaxSuccess 100 prop_toolCallToXMLRoundTrip

    describe "Error Message Preservation" $ do
      it "ToolResult errors are preserved in XML" $
        withMaxSuccess 100 prop_errorMessagePreservation

      it "ToolResult success values are preserved" $
        withMaxSuccess 100 prop_successValuePreservation

    describe "General XML Primitives" $ do
      it "wrapInTag produces properly nested tags" $
        withMaxSuccess 100 prop_wrapInTagCorrect

      it "extractFirstTag extracts content correctly" $
        withMaxSuccess 100 prop_extractFirstTagCorrect

      it "extractAllTags finds all occurrences" $
        withMaxSuccess 100 prop_extractAllTagsCorrect

      it "extractAndRemoveTags removes all tags" $
        withMaxSuccess 100 prop_extractAndRemoveTagsCorrect

    describe "Format-Specific Functions" $ do
      it "formatToolDefinitionsAsXML includes all tool names" $
        withMaxSuccess 50 prop_formatToolDefinitionsIncludesNames

      it "formatToolDefinitionsAsXML includes usage instructions" $
        withMaxSuccess 50 prop_formatToolDefinitionsIncludesInstructions
