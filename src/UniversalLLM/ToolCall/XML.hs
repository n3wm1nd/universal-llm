{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module UniversalLLM.ToolCall.XML
  ( -- * Types
    XMLToolCall(..)
  , XMLToolResult(..)
  , XMLArgPair(..)
    -- * General XML Primitives
  , wrapInTag
  , extractFirstTag
  , extractAllTags
  , extractAndRemoveTags
    -- * Our Tool Call Format (tag names baked in)
  , extractXMLToolCalls
  , extractAndRemoveXMLToolCalls
  , parseXMLToolCall
  , encodeXMLToolCall
  , encodeXMLToolResult
    -- * Conversion
  , xmlToolCallToToolCall
  , toolResultToXML
  , formatToolDefinitionsAsXML
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import Data.Aeson (Value)
import GHC.Generics (Generic)
import Autodocodec
import UniversalLLM (ToolCall(..), ToolResult(..), ToolDefinition(..))
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Text.Encoding as TE
import Data.Hashable (hash)
import Data.List (nub, (\\))

-- ============================================================================
-- Types
-- ============================================================================

-- | XML tool call structure
data XMLToolCall = XMLToolCall
  { toolName :: Text
  , toolArgs :: [XMLArgPair]
  } deriving (Show, Eq, Generic)

-- | Key-value pair for tool arguments
data XMLArgPair = XMLArgPair
  { argKey :: Text
  , argValue :: Text
  } deriving (Show, Eq, Generic)

-- | XML tool result structure
data XMLToolResult = XMLToolResult
  { toolCallId :: Text
  , resultToolName :: Text
  , resultOutput :: Text
  } deriving (Show, Eq, Generic)

instance HasCodec XMLArgPair where
  codec = object "XMLArgPair" $
    XMLArgPair
      <$> requiredField "arg_key" "Argument key" .= argKey
      <*> requiredField "arg_value" "Argument value" .= argValue

instance HasCodec XMLToolCall where
  codec = object "XMLToolCall" $
    XMLToolCall
      <$> requiredField "tool_name" "Tool name" .= toolName
      <*> requiredField "args" "Tool arguments" .= toolArgs

instance HasCodec XMLToolResult where
  codec = object "XMLToolResult" $
    XMLToolResult
      <$> requiredField "tool_call_id" "Tool call ID" .= toolCallId
      <*> requiredField "tool_name" "Tool name" .= resultToolName
      <*> requiredField "result" "Result output" .= resultOutput

-- ============================================================================
-- Core XML Primitives
-- ============================================================================

-- | Wrap content in a tag: content -> <tag>content</tag>
wrapInTag :: Text -> Text -> Text
wrapInTag tagName content = "<" <> tagName <> ">" <> content <> "</" <> tagName <> ">"

-- | Extract first occurrence of a tag's content (without the tags themselves)
-- Returns Nothing if tag not found or not properly closed
extractFirstTag :: Text -> Text -> Maybe Text
extractFirstTag tagName input =
  let openTag = "<" <> tagName <> ">"
      closeTag = "</" <> tagName <> ">"
  in case T.breakOn openTag input of
       (_, "") -> Nothing  -- No opening tag
       (_, rest) ->
         let afterOpen = T.drop (T.length openTag) rest
         in case T.breakOn closeTag afterOpen of
              (_, "") -> Nothing  -- No closing tag
              (content, _) -> Just content

-- | Extract all occurrences of a tag's content (without the tags themselves)
-- Returns list of tag contents
extractAllTags :: Text -> Text -> [Text]
extractAllTags tagName input = go input []
  where
    openTag = "<" <> tagName <> ">"
    closeTag = "</" <> tagName <> ">"

    go txt acc =
      case T.breakOn openTag txt of
        (_, "") -> reverse acc  -- No more opening tags
        (_, rest) ->
          let afterOpen = T.drop (T.length openTag) rest
          in case T.breakOn closeTag afterOpen of
               (_, "") -> reverse acc  -- No closing tag, stop
               (content, afterClose) ->
                 let remaining = T.drop (T.length closeTag) afterClose
                 in go remaining (content : acc)

-- | Extract all tag contents AND remove them from text (single pass)
-- Returns: (tag contents, cleaned text)
extractAndRemoveTags :: Text -> Text -> ([Text], Text)
extractAndRemoveTags tagName input = go input [] []
  where
    openTag = "<" <> tagName <> ">"
    closeTag = "</" <> tagName <> ">"

    go txt beforeAcc contentsAcc =
      case T.breakOn openTag txt of
        (_, "") ->
          -- No more opening tags - return accumulated results
          (reverse contentsAcc, T.concat (reverse (txt : beforeAcc)))
        (before, rest) ->
          let afterOpen = T.drop (T.length openTag) rest
          in case T.breakOn closeTag afterOpen of
               (_, "") ->
                 -- No closing tag, keep remaining text and stop
                 (reverse contentsAcc, T.concat (reverse (txt : beforeAcc)))
               (content, afterClose) ->
                 let remaining = T.drop (T.length closeTag) afterClose
                 in go remaining (before : beforeAcc) (content : contentsAcc)


-- ============================================================================
-- Our Tool Call Format (tag names: tool_call, arg_key, arg_value)
-- ============================================================================

-- | Extract all XML tool call blocks from text (returns full blocks with tags)
-- Uses our standard format with <tool_call> tags
extractXMLToolCalls :: Text -> [Text]
extractXMLToolCalls input =
  let contents = extractAllTags "tool_call" input
  in map (wrapInTag "tool_call") contents

-- | Extract XML tool calls AND return cleaned text (single pass)
-- Returns: (list of full tool call blocks with tags, text with tool calls removed)
extractAndRemoveXMLToolCalls :: Text -> ([Text], Text)
extractAndRemoveXMLToolCalls input =
  let (contents, cleaned) = extractAndRemoveTags "tool_call" input
      fullBlocks = map (wrapInTag "tool_call") contents
  in (fullBlocks, cleaned)

-- ============================================================================
-- Tool Call Parsing (XML → Haskell)
-- ============================================================================

-- | Parse a single XML tool call block
-- Format: <tool_call>tool_name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
-- Uses primitives to build the parser compositionally
parseXMLToolCall :: Text -> Maybe XMLToolCall
parseXMLToolCall block = do
  -- Extract content from the <tool_call> tag
  content <- extractFirstTag "tool_call" block

  -- Tool name is everything before the first <
  let (namePart, rest) = T.break (== '<') content
      name = T.strip namePart

  guard (not $ T.null name)

  -- Extract argument key-value pairs using <arg_key> and <arg_value> tags
  let keys = extractAllTags "arg_key" rest
      values = extractAllTags "arg_value" rest
      args = zipWith XMLArgPair keys values

  return $ XMLToolCall name args
  where
    guard True = Just ()
    guard False = Nothing

-- ============================================================================
-- Tool Call Encoding (Haskell → XML)
-- ============================================================================

-- | Encode a single argument pair to XML using <arg_key> and <arg_value> tags
encodeArgPair :: XMLArgPair -> Text
encodeArgPair (XMLArgPair k v) =
  wrapInTag "arg_key" k <> "\n" <> wrapInTag "arg_value" v

-- | Encode XMLToolCall to XML text
-- Format: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
-- Uses primitives for composable serialization
encodeXMLToolCall :: XMLToolCall -> Text
encodeXMLToolCall (XMLToolCall name args) =
  let argsText = T.intercalate "\n" (map encodeArgPair args)
      content = if null args
                then name
                else name <> "\n" <> argsText
  in wrapInTag "tool_call" content

-- | Encode XMLToolResult to XML text
-- Format: <tool_result><tool_call_id>...</tool_call_id>...</tool_result>
encodeXMLToolResult :: XMLToolResult -> Text
encodeXMLToolResult (XMLToolResult callId name output) =
  wrapInTag "tool_result" $
    wrapInTag "tool_call_id" callId <> "\n" <>
    wrapInTag "tool_name" name <> "\n" <>
    wrapInTag "result" output

-- ============================================================================
-- Conversion Helpers
-- ============================================================================

-- | Convert XMLToolCall to ToolCall (generates deterministic ID from content)
-- Returns InvalidToolCall if there are duplicate parameter names (even with same values)
xmlToolCallToToolCall :: XMLToolCall -> ToolCall
xmlToolCallToToolCall xmlCall@(XMLToolCall name args) =
  let callId = "xml-" <> T.pack (show (abs (hash (encodeXMLToolCall xmlCall))))
      keys = map argKey args
      uniqueKeys = nub keys
      duplicates = keys \\ uniqueKeys
      hasDuplicates = not (null duplicates)
  in if hasDuplicates
     then InvalidToolCall callId name (encodeXMLToolCall xmlCall)
          ("Duplicate parameter names in XML tool call: " <> T.intercalate ", " duplicates)
     else ToolCall callId name (argsToJSON args)
  where
    argsToJSON :: [XMLArgPair] -> Value
    argsToJSON pairs = Aeson.object [Key.fromText (argKey p) Aeson..= argValue p | p <- pairs]

-- | Convert ToolResult to XMLToolResult
toolResultToXML :: ToolResult -> XMLToolResult
toolResultToXML result =
  let callId = getToolCallId (toolResultCall result)
      name = getToolCallName (toolResultCall result)
      output = case toolResultOutput result of
                 Left err -> "Error: " <> err
                 Right val -> TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ val
  in XMLToolResult callId name output
  where
    getToolCallId (ToolCall tcId _ _) = tcId
    getToolCallId (InvalidToolCall tcId _ _ _) = tcId

    getToolCallName (ToolCall _ tcName _) = tcName
    getToolCallName (InvalidToolCall _ tcName _ _) = tcName

-- | Format tool definitions as XML for system prompt injection
formatToolDefinitionsAsXML :: [ToolDefinition] -> Text
formatToolDefinitionsAsXML tools =
  T.unlines
    [ "You have access to the following tools:"
    , ""
    , T.intercalate "\n\n" (map formatToolDef tools)
    , ""
    , "To use a tool, respond with:"
    , "<tool_call>tool_name"
    , "<arg_key>parameter_name</arg_key>"
    , "<arg_value>parameter_value</arg_value>"
    , "</tool_call>"
    ]
  where
    formatToolDef :: ToolDefinition -> Text
    formatToolDef tool =
      T.unlines
        [ "Tool: " <> toolDefName tool
        , "Description: " <> toolDefDescription tool
        , "Parameters: " <> (TE.decodeUtf8 . BSL.toStrict . Aeson.encode $ toolDefParameters tool)
        ]
