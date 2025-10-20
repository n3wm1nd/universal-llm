# Tool Calling Patterns in Universal-LLM

This library supports four different patterns for implementing tool calls, each suited to different use cases. All patterns use the same underlying type system, but differ in how you structure your tool types and implementations.

## Overview

Tool calling in this library involves:
1. **Tool Type** - A Haskell type representing the tool
2. **Tool Instance** - Implementation of the `Tool` typeclass
3. **Tool Definition** - Pure metadata (`ToolDefinition`) sent to the LLM
4. **Tool Execution** - Running tools on `ToolCall` messages from the LLM

The four patterns differ in what the tool type contains and when you provide the implementation.

---

## Pattern A: Static Tool Functions

**Use when:** Your tool implementation is fixed and has no configuration or external dependencies.

### Structure

```haskell
-- Tool type is empty - just a marker
data GetTime = GetTime deriving (Eq)

-- Tool implementation is in the typeclass instance
instance Tool GetTime IO where
  type ToolParams GetTime = GetTimeParams
  type ToolOutput GetTime = TimeResponse

  toolName _ = "get_time"
  toolDescription _ = "Get the current system time"

  -- Implementation directly in 'call'
  call _ params = do
    now <- getCurrentTime
    return $ TimeResponse (formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S" now)
```

### Advantages
- Simplest pattern
- No state to manage
- Easy to understand

### Example Use Case
- System utilities (time, random numbers)
- Pure functions
- Stateless operations

---

## Pattern B: Configured Tools

**Use when:** Tool behavior needs configuration that the LLM cannot change (like API keys, base URLs, preferences).

### Structure

```haskell
-- Tool type holds configuration
data WebSearch = WebSearch
  { searchEngine :: Text  -- "google", "bing", etc.
  , maxResults :: Int
  } deriving (Eq)

instance Tool WebSearch IO where
  type ToolParams WebSearch = SearchParams
  type ToolOutput WebSearch = SearchResults

  -- Configuration affects metadata
  toolName (WebSearch engine _) = "search_" <> engine
  toolDescription (WebSearch engine _) =
    "Search the web using " <> engine

  -- Configuration affects behavior
  call (WebSearch engine maxResults) params = do
    results <- searchAPI engine (query params) maxResults
    return $ SearchResults results
```

### Advantages
- Configuration is type-safe
- Same tool type, different configurations
- Configuration influences both description and behavior

### Example Use Cases
- API clients with different endpoints
- Tools with user preferences
- Multi-tenant scenarios

---

## Pattern C: Dependency Injection

**Use when:** You want to inject implementations (for testing, mocking, or partial application).

### Structure

```haskell
-- Tool type holds the implementation function
data DatabaseQuery m = DatabaseQuery (QueryParams -> m QueryResult)

-- Equality ignores the function (just check type)
instance Eq (DatabaseQuery m) where
  _ == _ = True

instance Tool (DatabaseQuery m) m where
  type ToolParams (DatabaseQuery m) = QueryParams
  type ToolOutput (DatabaseQuery m) = QueryResult

  toolName _ = "query_database"
  toolDescription _ = "Query the application database"

  -- Just call the injected function
  call (DatabaseQuery impl) params = impl params
```

### Usage

```haskell
-- Production: real database connection
productionQuery :: QueryParams -> IO QueryResult
productionQuery params = runQuery dbConnection params

let prodTool = DatabaseQuery productionQuery

-- Testing: mock implementation
mockQuery :: QueryParams -> IO QueryResult
mockQuery params = return $ MockResult [...]

let testTool = DatabaseQuery mockQuery

-- Both work with the same tool type!
tools = [LLMTool prodTool]  -- or [LLMTool testTool]
```

### Advantages
- Perfect for testing (inject mocks)
- Enables partial application
- Separates interface from implementation

### Example Use Cases
- Database access
- External API calls
- Any operation you want to mock in tests

---

## Pattern D: Manual Handling

**Use when:** You need complete control or want to avoid the `Tool` typeclass entirely.

### Structure

```haskell
-- Skip the Tool typeclass entirely
-- Manually create ToolDefinitions
let toolDef = ToolDefinition
      { toolDefName = "custom_tool"
      , toolDefDescription = "Does something custom"
      , toolDefParameters = customJsonSchema
      }

-- Manually handle ToolCall messages
handleToolCall :: ToolCall -> IO Value
handleToolCall tc = case toolCallName tc of
  "custom_tool" -> do
    -- Manually decode parameters
    let params = decode (toolCallParameters tc)
    -- Execute custom logic
    result <- customImplementation params
    -- Manually encode result
    return $ toJSON result
  _ -> error "Unknown tool"
```

### Advantages
- Maximum flexibility
- Straightforward and easy to understand
- No typeclass magic
- Full control over encoding/decoding

### Disadvantages
- No type safety for parameters/results
- Manual encoding/decoding boilerplate
- Runtime errors instead of compile-time checks

### Example Use Cases
- Quick prototyping
- One-off custom tools
- When tool structure doesn't fit the typeclass
- Gradual migration from dynamic to typed tools
- Coming from dynamic languages

---

## Complete Example: Pattern C (Dependency Injection)

Here's a full example from the library's test suite:

```haskell
{-# LANGUAGE OverloadedStrings #-}

import UniversalLLM
import Data.Time

-- 1. Define parameter and result types
data GetTimeParams = GetTimeParams
  { timezone :: Maybe Text
  } deriving (Show, Eq)

instance HasCodec GetTimeParams where
  codec = object "GetTimeParams" $
    GetTimeParams <$> optionalField "timezone" "Timezone" .= timezone

data TimeResponse = TimeResponse
  { currentTime :: Text
  } deriving (Show, Eq)

instance HasCodec TimeResponse where
  codec = object "TimeResponse" $
    TimeResponse <$> requiredField "current_time" "Current time" .= currentTime

-- 2. Define tool type (holds implementation)
data GetTime m = GetTime (GetTimeParams -> m TimeResponse)

instance Eq (GetTime m) where
  _ == _ = True

-- 3. Implement Tool typeclass
instance Tool (GetTime m) m where
  type ToolParams (GetTime m) = GetTimeParams
  type ToolOutput (GetTime m) = TimeResponse

  toolName _ = "get_time"
  toolDescription _ = "Get the current time"

  call (GetTime impl) params = impl params

-- 4. Create tool value with implementation
getTimeTool :: MonadIO m => GetTime m
getTimeTool = GetTime $ \params -> do
  now <- liftIO getCurrentTime
  let timeStr = formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S UTC" now
  return $ TimeResponse timeStr

-- 5. Use in conversation
main :: IO ()
main = do
  let tools = [LLMTool (getTimeTool @IO)]
      toolDefs = map llmToolToDefinition tools

      configs = [ Temperature 0.7
                , MaxTokens 100
                , Tools toolDefs
                ]

      messages = [UserText "What time is it?"]

  -- Send request with configs
  let request = toRequest OpenAI GPT4o configs messages

  -- ... get response from API ...

  -- Execute tool calls (single tool call per message)
  case response of
    AssistantTool call -> do
      result <- executeToolCall tools call
      -- Continue conversation with ToolResultMsg result...
    AssistantText text ->
      putStrLn text
```

---

## Choosing a Pattern

| Pattern | Use When | Type Safety | Boilerplate |
|---------|----------|-------------|-------------|
| **A: Static** | Simple, stateless tools | ✓ Full | Low |
| **B: Configured** | Need user preferences or API endpoints | ✓ Full | Medium |
| **C: Dependency Injection** | Want testability or partial application | ✓ Full | Medium |
| **D: Manual** | Quick prototyping or custom cases | ✗ Runtime | Low* |

*Low cognitive complexity but manual encoding/decoding work

**Recommendation:**
- Start with **Pattern A** for simple stateless tools
- Use **Pattern C** when you need testing or flexibility
- Use **Pattern D** for quick prototyping or when learning the library
- Use **Pattern B** for tools with configuration

---

## Tool Execution Flow

Regardless of pattern, the execution flow is:

1. **Define tools** → Create `[LLMTool m]` with implementations
2. **Extract definitions** → `map llmToolToDefinition tools` gives `[ToolDefinition]`
3. **Add to config** → `Tools toolDefs` as part of `[ModelConfig provider model]`
4. **Send request** → `toRequest provider model configs messages`
5. **Receive tool calls** → LLM responds with `AssistantTool ToolCall` (one tool call per message)
6. **Execute tools** → `executeToolCall tools call` dispatches to correct tool
7. **Return results** → Send `ToolResultMsg result` back to LLM

The separation of `[LLMTool m]` (executable, monad-specific) and `[ToolDefinition]` (pure metadata) allows the model to be monad-agnostic while tools can perform IO or other effects.

**Note:** Multiple tool calls from the LLM result in multiple `AssistantTool` messages, one per tool call. You handle each separately and return multiple `ToolResultMsg` messages back.

---

## Advanced: Combining Patterns

You can mix patterns in the same application:

```haskell
let tools =
      [ LLMTool GetTime                          -- Pattern A
      , LLMTool (WebSearch "google" 10)          -- Pattern B
      , LLMTool (DatabaseQuery prodQueryImpl)    -- Pattern C
      ]
```

All patterns work together because they all implement the same `Tool` typeclass!
