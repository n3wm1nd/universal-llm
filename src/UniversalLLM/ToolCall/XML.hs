{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module UniversalLLM.ToolCall.XML
  ( -- * Types
    XMLToolCall(..)
  , XMLToolResult(..)
  , XMLArgPair(..)
    -- * Generic Tag Extraction
  , extractTags
  , extractAndRemoveTags
    -- * Tool Call Parsing
  , parseXMLToolCall
  , parseXMLToolCallTagged
  , extractXMLToolCalls
  , extractAndRemoveXMLToolCalls
    -- * Encoding
  , encodeXMLToolCall
  , encodeXMLToolCallTagged
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
import UniversalLLM.Core.Types (ToolCall(..), ToolResult(..), ToolDefinition(..))
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Text.Encoding as TE
import Data.Hashable (hash)

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
-- Parsing (XML text → Haskell values)
-- ============================================================================

-- | Extract tags: finds all <tag>...</tag> pairs and returns (cleaned_text, [contents])
extractTags :: Text -> Text -> (Text, [Text])
extractTags tagName input = go input [] []
  where
    openTag = "<" <> tagName <> ">"
    closeTag = "</" <> tagName <> ">"

    go txt beforeAcc contentsAcc =
      case T.breakOn openTag txt of
        (_, "") ->
          -- No more opening tags
          (T.concat (reverse (txt : beforeAcc)), reverse contentsAcc)
        (before, rest) ->
          let afterOpen = T.drop (T.length openTag) rest
          in case T.breakOn closeTag afterOpen of
               (_, "") ->
                 -- No closing tag, stop here
                 (T.concat (reverse (txt : beforeAcc)), reverse contentsAcc)
               (content, afterClose) ->
                 let remaining = T.drop (T.length closeTag) afterClose
                 in go remaining (before : beforeAcc) (content : contentsAcc)

-- | Extract all blocks for a specific tag from text
-- Returns: (list of full tag blocks, text with tags removed)
extractAndRemoveTags :: Text -> Text -> ([Text], Text)
extractAndRemoveTags tagName input =
  let (cleanedText, tagContents) = extractTags tagName input
      openTag = "<" <> tagName <> ">"
      closeTag = "</" <> tagName <> ">"
      -- Reconstruct full blocks with tags
      fullBlocks = map (\content -> openTag <> content <> closeTag) tagContents
  in (fullBlocks, cleanedText)

-- | Extract all XML tool call blocks from text
extractXMLToolCalls :: Text -> Text -> [Text]
extractXMLToolCalls tagName input = fst (extractAndRemoveXMLToolCalls tagName input)

-- | Extract XML tool calls AND return cleaned text (single pass)
-- Takes tag name as parameter to avoid hardcoding "tool_call" everywhere
-- Returns: (list of tool call blocks, text with tool calls removed)
extractAndRemoveXMLToolCalls :: Text -> Text -> ([Text], Text)
extractAndRemoveXMLToolCalls = extractAndRemoveTags

-- | Parse a single XML tool call block with custom tag
-- Format: <tag>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tag>
parseXMLToolCallTagged :: Text -> Text -> Maybe XMLToolCall
parseXMLToolCallTagged tagName block =
  let openTag = "<" <> tagName <> ">"
      closeTag = "</" <> tagName <> ">"

      -- Extract content between tags
      content = case T.breakOn openTag block of
                  (_, "") -> block  -- No opening tag, use as-is
                  (_, rest) ->
                    let afterOpen = T.drop (T.length openTag) rest
                    in case T.breakOn closeTag afterOpen of
                         (inner, _) -> inner

      -- Extract tool name: everything before first <
      (namePart, rest) = T.break (== '<') content
      name = T.strip namePart

      -- Extract arg_key and arg_value pairs
      (_, keys) = extractTags "arg_key" rest
      (_, values) = extractTags "arg_value" rest

      args = zipWith XMLArgPair keys values

  in if T.null name
       then Nothing
       else Just $ XMLToolCall name args

-- | Parse a single XML tool call block (default tag: "tool_call")
parseXMLToolCall :: Text -> Maybe XMLToolCall
parseXMLToolCall = parseXMLToolCallTagged "tool_call"

-- ============================================================================
-- Encoding (Haskell values → XML text)
-- ============================================================================

-- | Encode XMLToolCall to XML text with custom tag name
encodeXMLToolCallTagged :: Text -> XMLToolCall -> Text
encodeXMLToolCallTagged tagName (XMLToolCall name args) =
  let openTag = "<" <> tagName <> ">"
      closeTag = "</" <> tagName <> ">"
  in T.unlines $
       [ openTag <> name ]
       ++ map encodeArgPair args
       ++ [ closeTag ]
  where
    encodeArgPair (XMLArgPair k v) =
      "<arg_key>" <> k <> "</arg_key>\n" <>
      "<arg_value>" <> v <> "</arg_value>"

-- | Encode XMLToolCall to XML text (default tag: "tool_call")
encodeXMLToolCall :: XMLToolCall -> Text
encodeXMLToolCall = encodeXMLToolCallTagged "tool_call"

-- | Encode XMLToolResult to XML text
encodeXMLToolResult :: XMLToolResult -> Text
encodeXMLToolResult (XMLToolResult callId name output) =
  T.unlines
    [ "<tool_result>"
    , "<tool_call_id>" <> callId <> "</tool_call_id>"
    , "<tool_name>" <> name <> "</tool_name>"
    , "<result>" <> output <> "</result>"
    , "</tool_result>"
    ]

-- ============================================================================
-- Conversion Helpers
-- ============================================================================

-- | Convert XMLToolCall to ToolCall (generates deterministic ID from content)
xmlToolCallToToolCall :: XMLToolCall -> ToolCall
xmlToolCallToToolCall xmlCall@(XMLToolCall name args) =
  let -- Generate deterministic ID from hash of the tool call
      callId = "xml-" <> T.pack (show (abs (hash (encodeXMLToolCall xmlCall))))
      -- Convert args to JSON
      jsonParams = argsToJSON args
  in ToolCall callId name jsonParams
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
