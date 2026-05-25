# Universal LLM

A type-safe Haskell library for encoding and decoding Large Language Model API protocols.

## What This Library Does

This library **encodes messages to provider-specific request formats** and **decodes provider responses back to messages**. It does NOT handle HTTP transport, retries, streaming, or rate limiting — that's your responsibility. Think of it as a type-safe protocol codec with compile-time capability constraints.

## Features

- **Pure Protocol Encoding/Decoding**: Transform messages to/from provider wire formats (JSON)
- **Multiple Providers**: OpenAI, Anthropic, and OpenAI-compatible providers (llama.cpp, Ollama, vLLM, OpenRouter, LiteLLM)
- **Compile-time Capability Checks**: Invalid capability combinations caught at compile time via GADTs
- **Composable Providers**: Build complex behavior by composing simple handlers via `chainProviders`
- **Tool Calling Support**: Type-safe tool definitions with automatic parameter validation
- **Provider-Agnostic Business Logic**: Write code once, run with any compatible provider/model
- **Vision Support**: Image inputs handled uniformly across providers
- **Reasoning Support**: Extended thinking / chain-of-thought across Anthropic and OpenAI-compatible models

## Design Principles

- **Protocol Layer Only**: You provide the bytes (HTTP/transport), we provide type-safe encoding/decoding
- **Compile-time Safety**: GADT-based messages ensure capability constraints are verified at compile time
- **Pure Functions**: All encoding/decoding logic is pure for easy testing and composition
- **Composable Architecture**: Build complex protocol behavior by composing simple transformations
- **No Universal Workarounds**: Provider quirks are opt-in per model, never silently applied to all

## Quick Example

```haskell
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}
{-# LANGUAGE TypeOperators     #-}
import UniversalLLM
import UniversalLLM.Models.Anthropic.Claude (ClaudeSonnet46(..))
import UniversalLLM.Providers.Anthropic (Anthropic(..))

-- Business logic polymorphic over any model with tools + reasoning
processQuery :: ( Provider m
                , HasTools m
                , HasReasoning m
                , SupportsMaxTokens (ProviderOf m)
                )
             => m -> Text -> IO ()
processQuery model query = do
  let configs  = [MaxTokens 2048, Tools myToolDefs, Reasoning True]
      messages = [UserText query]
      (state, req) = toProviderRequest route model configs () messages

  -- YOU handle HTTP transport here (not part of this library)
  resp <- yourHttpClient req

  case fromProviderResponse route model configs state resp of
    Left err   -> handleError err
    Right (_, msgs) -> handleMessages msgs

-- Use it with a concrete model/provider pair
main :: IO ()
main = processQuery (ClaudeSonnet46 `via` Anthropic) "What's the weather in London?"
```

## Core Concepts

### The Model Type

Models and providers are combined into a single unified type:

```haskell
data Model aiModel provider = Model aiModel provider

-- Nice infix syntax
type ClaudeOnAnthropic = ClaudeSonnet46 `Via` Anthropic

-- Value-level construction
let m = ClaudeSonnet46 `via` Anthropic
```

Capability classes (`HasTools`, `HasReasoning`, `HasVision`) are instanced on the combined
`Model aiModel provider` type, because capabilities often depend on both — a model may support
reasoning, but only when accessed through a provider that exposes it.

### Composable Providers

Each model's `Routing` instance assembles a composable provider chain from the handlers it needs:

```haskell
instance Routing (Model ClaudeSonnet46 Anthropic) where
  type RoutingState (Model ClaudeSonnet46 Anthropic) = (AnthropicReasoningState, ((), ((), ())))
  route = withReasoning
      `chainProviders` withTools
      `chainProviders` withVision
      `chainProviders` Anthropic.baseComposableProvider @(Model ClaudeSonnet46 Anthropic)
```

Only the handlers the model actually needs are composed. A text-only model pays no cost for
tool or reasoning handling. Provider quirks (system message reordering, OAuth magic prompts,
reasoning state tracking) are applied only where declared.

### Defining a Model

Models are defined by declaring the model type and instancing the capabilities it supports:

```haskell
data MyModel = MyModel deriving (Show, Eq)

instance ModelName (Model MyModel OpenRouter) where
  modelName (Model _ _) = "my-provider/my-model"

instance HasTools (Model MyModel OpenRouter) where
  withTools = OpenAI.openAITools

instance HasReasoning (Model MyModel OpenRouter) where
  withReasoning = OpenAI.openAIReasoning

instance Routing (Model MyModel OpenRouter) where
  type RoutingState (Model MyModel OpenRouter) = ((), ((), ()))
  route = withReasoning
      `chainProviders` withTools
      `chainProviders` OpenAI.baseComposableProvider @(Model MyModel OpenRouter)
```

The `Routing` instance is the complete specification of what wire-level transformations apply
to this model/provider combination. Nothing is implicit.

### Messages

Messages are a GADT parameterised by the combined model type `m`. Capability constraints are
encoded directly in the constructors:

```haskell
data Message m where
  UserText       :: Text                          -> Message m
  UserImage      :: HasVision m  => Text -> Text  -> Message m
  AssistantText  :: Text                          -> Message m
  AssistantTool  :: HasTools m   => ToolCall       -> Message m
  AssistantReasoning :: HasReasoning m => Text    -> Message m
  ToolResultMsg  :: HasTools m   => ToolResult     -> Message m
  SystemText     :: Text                          -> Message m
```

`UserImage` doesn't typecheck for a model without `HasVision`. `AssistantTool` doesn't
typecheck without `HasTools`. The capability check happens at the construction site, not at
runtime.

### Capabilities

```
SupportsX  — provider-level parameters (Temperature, MaxTokens, Seed, Stop)
HasX       — features requiring both model and provider support
```

| Class              | Constructor        | Notes                                      |
|--------------------|--------------------|--------------------------------------------|
| `HasTools`         | `Tools [ToolDefinition]` | Tool/function calling                |
| `HasReasoning`     | `Reasoning Bool`, `ReasoningEffort Text` | Extended thinking      |
| `HasVision`        | `UserImage`        | Image inputs                               |
| `HasJSON`          | —                  | Structured JSON output mode                |
| `SupportsMaxTokens`| `MaxTokens Int`    | Response length cap                        |
| `SupportsTemperature` | `Temperature Double` | Sampling temperature                  |

## Architecture

### Layers

1. **Protocols** (`UniversalLLM.Protocols.OpenAI`, `UniversalLLM.Protocols.Anthropic`)
   Wire format types — the actual JSON schemas sent over the network. Two protocols cover all
   supported providers: OpenAI-format and Anthropic-format.

2. **Providers** (`UniversalLLM.Providers.OpenAI`, `UniversalLLM.Providers.Anthropic`)
   Not implementations — **toolsets** of composable handlers. Each function (`openAITools`,
   `anthropicReasoning`, `baseComposableProvider`, etc.) is one composable unit. Models pick
   the ones they need.

3. **Models** (`UniversalLLM.Models.*`)
   Model types with their capability instances and `Routing` definitions. The library ships
   with definitions for Claude, GPT, Qwen, Gemini, GLM, Kimi, MiniMax, and others. Define
   your own the same way for any model not yet covered.

4. **Messages** (`Message m`)
   The universal, provider-agnostic message layer. Business logic works entirely at this level.

### Provider Modules Are Toolsets

`UniversalLLM.Providers.OpenAI` does not implement OpenAI. It provides reusable building
blocks — `openAITools`, `openAIReasoning`, `openAIJSON`, `baseComposableProvider` — that any
OpenAI-compatible model can compose. A llama.cpp model and an OpenRouter model both use
`openAITools` but may compose different base providers or additional quirk handlers.

This means provider-specific workarounds never silently apply to models that don't need them.
A Qwen model on llama.cpp that requires system messages to be reordered declares that handler
explicitly; a GPT model via OpenAI that doesn't need it simply doesn't include it.

## Tool Calling

See [TOOLCALLS.md](TOOLCALLS.md) for the full guide. Three approaches are available:

**Approach A — `ToolFunction` instance (preferred):**
```haskell
data WeatherResult = WeatherResult { temperature :: Int, condition :: Text }
  deriving (Show, Eq, Generic)

instance HasCodec WeatherResult where ...

instance ToolFunction WeatherResult where
  toolFunctionName _ = "get_weather"
  toolFunctionDescription _ = "Get current weather for a location"

getWeather :: Text -> IO WeatherResult
getWeather location = ...  -- automatically a tool

let tools = [LLMTool getWeather]
```

**Approach B — `mkToolWithMeta` for custom parameter names:**
```haskell
searchTool = mkToolWithMeta "search" "Search the web" searchFn
               "query"       "The search query"
               "max_results" "Maximum results to return"
```

**Approach C — manual `ToolDefinition`** for full control or quick prototyping.

## What You Need to Provide

This library handles **protocol encoding/decoding only**. You are responsible for:

- **HTTP Transport**: Making actual HTTP requests to LLM providers
- **Streaming**: SSE event handling if you want streaming responses
- **Error Handling**: Retries, exponential backoff, rate limiting
- **Observability**: Logging, metrics, tracing, cost tracking

The `examples/` directory contains simple HTTP transport implementations for reference.

## Examples

- **`examples/simple/`** — Minimal usage with a text-only model
- **`examples/claude/`** — Full tool-calling example with Anthropic
- **`examples/llama-cpp/`** — OpenAI-compatible API with a local llama.cpp server
- **`proxy/`** — Production proxy servers translating OpenAI requests to different backends

## Testing

```bash
cabal test      # Run all tests (uses cached API responses)
cabal build all # Build library, examples, and proxy servers
```

The test suite has three layers:

- **Protocol probes** — wire-level capability discovery against real APIs (cached). Tests what
  each model/provider accepts and rejects at the HTTP level, including negative probes that
  codify known constraints and workarounds.
- **Standard tests** — provider-agnostic functional tests through the composable provider
  abstraction. Identical test logic runs against every model; failures indicate a broken
  composable provider wiring.
- **Property tests** — round-trip and encoding correctness for protocol types.
