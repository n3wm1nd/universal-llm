# Tool Calling in Universal-LLM

This library supports three different approaches for implementing tool calls, each suited to different use cases. The library is designed to make tools from ordinary Haskell functions with minimal boilerplate.

## Overview

Tool calling in this library involves:
1. **Tool Functions** - Regular Haskell functions that become tools
2. **Tool Definitions** - Pure metadata (`ToolDefinition`) sent to the LLM
3. **Tool Execution** - Running tools on `ToolCall` messages from the LLM

The three approaches differ in how you attach metadata (name, description, parameter names) to your functions.

---

## Approach A: ToolFunction Instance (Preferred)

**Use when:** You have a unique return type for your tool (or can create a newtype wrapper).

This is the cleanest approach - just implement `ToolFunction` on the return type and any function returning it automatically becomes a tool.

### Structure

```haskell
-- 1. Define a unique result type
data TimeResponse = TimeResponse
  { currentTime :: Text
  } deriving (Show, Eq)

instance HasCodec TimeResponse where
  codec = object "TimeResponse" $
    TimeResponse <$> requiredField "current_time" "Current time" .= currentTime

-- 2. Make it a ToolFunction - this is where the tool metadata lives
instance ToolFunction TimeResponse where
  toolFunctionName _ = "get_time"
  toolFunctionDescription _ = "Get the current system time"

-- 3. Implement your function normally - it's automatically a tool!
getTime :: IO TimeResponse
getTime = do
  now <- getCurrentTime
  let timeStr = T.pack $ formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S UTC" now
  return $ TimeResponse timeStr

-- 4. Use it directly
main = do
  let tools = [LLMTool getTime]  -- That's it!
```

### Multi-parameter Functions

The library automatically handles curried functions:

```haskell
-- Tool function with multiple parameters
searchWeb :: Text -> Int -> IO SearchResults
searchWeb query maxResults = do
  results <- callSearchAPI query maxResults
  return $ SearchResults results

-- Just wrap it
let tools = [LLMTool searchWeb]
```

Parameters get default names (`param_0`, `param_1`, etc.) unless you use Approach B.

### Advantages
- Cleanest code - no wrapper types
- Type-safe
- Return type documents what the tool does
- Works great with newtypes for unique tool results

### When to Use
- Tools with unique return types (e.g., `TimeResponse`, `WeatherData`, `SearchResults`)
- When you can newtype-wrap a common return type (e.g., `newtype TranslatedText = TranslatedText Text`)
- Pure functions or simple IO operations
- When you want minimal boilerplate

### Example: Newtype Wrapper

```haskell
-- Wrap a common type to make it unique
newtype TranslatedText = TranslatedText Text

instance HasCodec TranslatedText where
  codec = named "TranslatedText" $
    dimapCodec TranslatedText (\(TranslatedText t) -> t) codec

instance ToolFunction TranslatedText where
  toolFunctionName _ = "translate"
  toolFunctionDescription _ = "Translate text to English"

-- Function is now automatically a tool
translateToEnglish :: Text -> IO TranslatedText
translateToEnglish text = do
  result <- callTranslationAPI text "en"
  return $ TranslatedText result
```

---

## Approach B: mkTool / mkToolWithMeta

**Use when:** You want to give custom names to parameters, or the return type is shared across multiple tools.

Use `mkTool` or `mkToolWithMeta` to wrap a function with metadata.

### Structure with mkTool

```haskell
-- Basic tool with default parameter names
calculatorTool :: ToolWrapped (Text -> Int -> Int -> IO CalculatorResult) (Text, (Int, (Int, ())))
calculatorTool = mkTool "calculator" "Performs arithmetic operations" calculator

calculator :: Text -> Int -> Int -> IO CalculatorResult
calculator op a b = return $ CalculatorResult $ case op of
  "add" -> a + b
  "subtract" -> a - b
  "multiply" -> a * b
  "divide" -> a `div` b
```

### Structure with mkToolWithMeta

```haskell
-- Tool with custom parameter names and descriptions
searchTool :: ToolWrapped (Text -> Int -> IO SearchResults) (Text, (Int, ()))
searchTool = mkToolWithMeta "web_search" "Search the web" webSearch
               "query" "The search query"
               "max_results" "Maximum number of results to return"

webSearch :: Text -> Int -> IO SearchResults
webSearch query maxResults = do
  results <- callSearchAPI query maxResults
  return $ SearchResults results

-- Use it
let tools = [LLMTool searchTool]
```

### Vary-adic Syntax

`mkToolWithMeta` takes alternating name/description pairs for each parameter:

```haskell
-- 2 parameters = 4 extra arguments (2 pairs)
mkToolWithMeta "name" "description" function
  "param1" "description1"
  "param2" "description2"

-- 3 parameters = 6 extra arguments (3 pairs)
mkToolWithMeta "name" "description" function
  "param1" "description1"
  "param2" "description2"
  "param3" "description3"
```

All arguments must be on the same logical line (can wrap with proper indentation).

### Advantages
- Custom parameter names and descriptions
- Works with any function, any return type
- Good for tools where parameter names matter to the LLM

### When to Use
- Multiple tools share the same return type
- Parameter names need to be descriptive for the LLM
- You want fine-grained control over tool metadata

---

## Approach C: Manual ToolDefinition

**Use when:** You need complete control or want to avoid the type system entirely.

Skip the `Tool` typeclass and manually create everything.

### Structure

```haskell
-- 1. Manually create ToolDefinition
let toolDef = ToolDefinition
      { toolDefName = "custom_tool"
      , toolDefDescription = "Does something custom"
      , toolDefParameters = customJsonSchema  -- Hand-written JSON Schema
      }

-- 2. Manually handle ToolCall messages
handleToolCall :: ToolCall -> IO ToolResult
handleToolCall tc = case getToolCallName tc of
  "custom_tool" -> do
    -- Manually decode parameters from JSON
    case fromJSON (toolCallParameters tc) of
      Success params -> do
        result <- customImplementation params
        return $ ToolResult tc (Right $ toJSON result)
      Error err ->
        return $ ToolResult tc (Left $ T.pack err)
  _ -> return $ ToolResult tc (Left "Unknown tool")

-- 3. Use it
let configs = [Tools [toolDef], Temperature 0.7]
response <- queryLLM configs messages

-- 4. Handle responses manually
case response of
  [AssistantTool call] -> do
    result <- handleToolCall call
    -- Continue with ToolResultMsg result
```

### Advantages
- Maximum flexibility
- No typeclass magic
- Easy to understand
- Full control over JSON encoding/decoding

### Disadvantages
- No type safety for parameters/results
- Manual encoding/decoding boilerplate
- Runtime errors instead of compile-time checks

### When to Use
- Quick prototyping
- One-off custom tools
- When tool structure doesn't fit the typeclass model
- Gradual migration to typed tools
- Coming from dynamic languages

---

## Complete Example: Multi-Parameter Tool (Approach A + B)

```haskell
{-# LANGUAGE OverloadedStrings #-}

import UniversalLLM
import UniversalLLM.Tools
import Data.Time

-- 1. Define result type with ToolFunction instance
data SearchResults = SearchResults
  { results :: [Text]
  , count :: Int
  } deriving (Show, Eq)

instance HasCodec SearchResults where
  codec = object "SearchResults" $
    SearchResults
      <$> requiredField "results" "Search results" .= results
      <*> requiredField "count" "Number of results" .= count

instance ToolFunction SearchResults where
  toolFunctionName _ = "search_web"
  toolFunctionDescription _ = "Search the web"

-- 2. Implement the function
searchWeb :: Text -> Int -> IO SearchResults
searchWeb query maxResults = do
  -- Call search API
  apiResults <- callSearchAPI query maxResults
  return $ SearchResults apiResults (length apiResults)

-- 3. Wrap with custom parameter names (Approach B)
searchTool :: ToolWrapped (Text -> Int -> IO SearchResults) (Text, (Int, ()))
searchTool = mkToolWithMeta "search_web" "Search the web" searchWeb
               "query" "The search query string"
               "max_results" "Maximum number of results"

-- 4. Use in conversation
main :: IO ()
main = do
  let tools = [LLMTool searchTool]
      toolDefs = map llmToolToDefinition tools

      configs = [ Temperature 0.7
                , MaxTokens 500
                , Tools toolDefs
                ]

      messages = [UserText "Search for Haskell tutorials"]

  -- Send request
  response <- queryLLM @MyProvider @MyModel configs messages

  -- Handle tool calls
  case response of
    [AssistantTool call] -> do
      result <- executeToolCallFromList tools call
      -- Continue conversation with ToolResultMsg result
    [AssistantText text] ->
      putStrLn $ T.unpack text
```

---

## Choosing an Approach

| Approach | Use When | Type Safety | Boilerplate | Parameter Names |
|----------|----------|-------------|-------------|-----------------|
| **A: ToolFunction** | Unique return type | ✓ Full | Minimal | Default |
| **B: mkTool/Meta** | Custom param names | ✓ Full | Low | Custom |
| **C: Manual** | Need full control | ✗ Runtime | Medium* | Custom |

*Manual encoding/decoding work

**Recommendations:**
1. **Start with Approach A** (ToolFunction) - cleanest for most tools
2. **Use Approach B** (mkToolWithMeta) when parameter names matter to the LLM
3. **Use Approach C** (Manual) for quick prototyping or when you need full control

---

## Tool Execution Flow

Regardless of approach, the execution flow is:

1. **Define tools** → Create `[LLMTool m]` with implementations
2. **Extract definitions** → `map llmToolToDefinition tools` gives `[ToolDefinition]`
3. **Add to config** → `Tools toolDefs` as part of `[ModelConfig provider model]`
4. **Send request** → `toProviderRequest provider model configs messages`
5. **Receive tool calls** → LLM responds with `AssistantTool ToolCall` messages
6. **Execute tools** → `executeToolCallFromList tools call` dispatches to correct tool
7. **Return results** → Send `ToolResultMsg result` back to LLM

---

## Advanced: Polysemy Support

For Polysemy users, import `Runix.LLM.ToolInstances` to get orphan instances for `Sem r`:

```haskell
import Runix.LLM.ToolInstances ()  -- Enables Sem r support

-- Now Sem actions work as tools
loggingTool :: Members '[Logging] r => Text -> Text -> Sem r LogResult
loggingTool message level = do
  logMessage level message
  return $ LogResult True

-- Use directly with type application
let tools = [LLMTool (loggingTool @r)]
```

This allows Polysemy effects to integrate seamlessly with the tool system.

---

## Type System Details

The tool system uses:
- **Nested tuple encoding** for parameters: `(a, (b, (c, ())))`
- **Callable typeclass** for recursive function unwrapping
- **Tool typeclass** for tool metadata
- **ToolFunction typeclass** for return type metadata
- **ToolParameter typeclass** for parameter type information

You don't need to understand these internals to use the library, but they enable:
- Type-safe parameter passing
- Automatic JSON schema generation
- Compile-time verification of tool structure
- Support for arbitrary arity functions
