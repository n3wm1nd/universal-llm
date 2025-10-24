# Universal LLM

A type-safe Haskell library for encoding and decoding Large Language Model API protocols.

## What This Library Does

This library **encodes messages to provider-specific request formats** and **decodes provider responses back to messages**. It does NOT handle HTTP transport, retries, streaming, or rate limiting - that's your responsibility. Think of it as a type-safe protocol codec with compile-time capability constraints.

## Features

- **Pure Protocol Encoding/Decoding**: Transform messages to/from provider wire formats (JSON)
- **Multiple Providers**: OpenAI, Anthropic, and OpenAI-compatible providers (llama.cpp, Ollama, vLLM, OpenRouter, LiteLLM)
- **Compile-time Capability Checks**: Invalid capability combinations caught at compile time via GADTs
- **Composable Providers**: Build complex behavior by composing simple handlers via Monoid/Semigroup
- **Tool Calling Support**: Type-safe tool definitions with automatic parameter validation
- **Provider-Agnostic Business Logic**: Write code once, run with any compatible provider/model
- **External Model Definitions**: Specific models (gpt-4o, claude-sonnet-4.5, etc.) defined in separate packages to avoid version coupling

## Design Principles

- **Protocol Layer Only**: You provide the bytes (HTTP/transport), we provide type-safe encoding/decoding
- **Compile-time Safety**: GADT-based messages ensure capability constraints are verified at compile time
- **Pure Functions**: All encoding/decoding logic is pure for easy testing and composition
- **Composable Architecture**: Build complex protocol behavior by composing simple transformations

## Quick Example

```haskell
{-# LANGUAGE OverloadedStrings #-}
import UniversalLLM

-- Write business logic polymorphic over any provider/model with tools
processQuery :: (ProviderImplementation provider model, HasTools model provider, Monoid (ProviderRequest provider))
             => provider -> model -> [LLMTool IO] -> Text -> IO ()
processQuery provider model tools query = do
  let configs = [Temperature 0.7, MaxTokens 200, Tools toolDefs]
      messages = [UserText query]

  -- Encode to provider's wire format (pure function)
  let request = toProviderRequest provider model configs messages

  -- YOU handle HTTP transport here (not part of this library)
  response <- yourHttpClient request

  -- Decode response (pure function)
  let parsedMessages = fromProviderResponse provider model configs messages response

  -- Handle results...
  processMessages parsedMessages

-- Define your model locally (or import from external package)
data ClaudeSonnet45 = ClaudeSonnet45 deriving (Show, Eq)

instance ModelName Anthropic ClaudeSonnet45 where
  modelName _ = "claude-sonnet-4.5-20250514"

instance HasTools ClaudeSonnet45 Anthropic where
  toolsComposableProvider = Anthropic.toolsComposableProvider

instance ProviderImplementation Anthropic ClaudeSonnet45 where
  getComposableProvider = Anthropic.baseComposableProvider <> Anthropic.toolsComposableProvider

-- Select concrete provider/model at application entry point
main = do
  let provider = Anthropic
      model = ClaudeSonnet45
      tools = [LLMTool myTool]

  processQuery provider model tools "What's the weather?"
  -- Business logic (processQuery) works with ANY compatible provider/model
```

## Architecture

### Core Abstractions

1. **Protocols** (`UniversalLLM.Protocols.OpenAI`, `UniversalLLM.Protocols.Anthropic`): Wire format specifications - the actual JSON schemas for each provider's API

2. **Providers**: Phantom types representing LLM providers, plus **composable toolsets** for building protocol implementations. The library provides:

   **Provider Types:**
   - `OpenAI` - Official OpenAI API
   - `Anthropic` - Anthropic API
   - `OpenAICompatible`, `LlamaCpp`, `Ollama`, `VLLM`, `OpenRouter`, `LiteLLM` - OpenAI-compatible providers

   **Provider Modules** (`UniversalLLM.Providers.OpenAI`, `UniversalLLM.Providers.Anthropic`):
   - NOT specific provider implementations - they're **toolsets** for composing implementations
   - Contain reusable composable providers you mix-and-match based on your model's needs
   - Examples: `baseComposableProvider`, `toolsComposableProvider`, `ensureUserFirstProvider`, `withMagicSystemPrompt`
   - Use only what you need: if your OpenAI model doesn't need special handling, don't use those transformations
   - Provider-specific quirks (message ordering, unicode filtering, etc.) are opt-in transformations

3. **Models**: Model identifiers with capability instances. **Not provided by this library** - define them locally or import from external packages:
   - Library provides `SimpleModel` (text-only) and `FullFeaturedModel` (all capabilities) as examples only
   - Real applications define specific models (gpt-4o, claude-sonnet-4.5, etc.) with their required capabilities
   - This avoids version coupling between library and rapidly-evolving model releases

4. **Messages**: Universal message format parameterized by model and provider. GADT constructors enforce capability constraints at compile time

5. **Composable Providers**: Protocol transformations that compose via Monoid/Semigroup. Each provider implementation is built by composing handlers:
   ```haskell
   baseComposableProvider <> toolsComposableProvider <> jsonComposableProvider
   ```

### Message Translation

Messages are parameterized by `model` and `provider` but contain no provider-specific data - just phantom types for constraints. This means:
- Messages can be converted between compatible provider/model pairs
- Conversion is safe (just type coercion) when capabilities match
- Type system prevents invalid conversions (e.g., vision message to non-vision model)

### Defining Models

Models are **not** provided by this library. You define them in your application or import from external packages:

```haskell
-- In your application code
data GPT4o = GPT4o deriving (Show, Eq)

instance ModelName OpenAI GPT4o where
  modelName _ = "gpt-4o"

instance HasTools GPT4o OpenAI where
  toolsComposableProvider = OpenAI.toolsComposableProvider

instance HasJSON GPT4o OpenAI where
  jsonComposableProvider = OpenAI.jsonComposableProvider

instance ProviderImplementation OpenAI GPT4o where
  getComposableProvider =
    OpenAI.baseComposableProvider
    <> OpenAI.toolsComposableProvider
    <> OpenAI.jsonComposableProvider
```

**Provider modules are toolsets, not implementations:**

The `ProviderImplementation` instance above composes **only the transformations needed** for GPT-4o. For example:
- Anthropic's `ensureUserFirstProvider` enforces message ordering - but OpenAI doesn't need it, so we don't use it
- Anthropic's `withMagicSystemPrompt` handles Claude Code integration - irrelevant for standard OpenAI usage
- If a provider has unicode quirks, you'd add a `unicodeFilterProvider` - but only for models that need it

Different models using the same provider can compose different transformations:

```haskell
-- Minimal text-only model - just base encoding
instance ProviderImplementation OpenAI SimpleTextModel where
  getComposableProvider = OpenAI.baseComposableProvider

-- Full-featured model - compose all the transformations
instance ProviderImplementation OpenAI GPT4o where
  getComposableProvider =
    OpenAI.baseComposableProvider
    <> OpenAI.toolsComposableProvider
    <> OpenAI.jsonComposableProvider
    <> OpenAI.reasoningComposableProvider
```

**Why define models externally?**
- LLM models evolve rapidly (new versions, capabilities, deprecations)
- Decouples library updates from model changes
- Allows model-specific packages (e.g., `openai-models`, `anthropic-models`) maintained independently
- Users compose exactly the provider transformations their models need

### Extensibility

- **Add New Models**: Define a model type, implement `ModelName` and capability instances (`HasTools`, `HasJSON`, etc.)
- **Add New Providers**: Implement protocol types and composable provider handlers
- **Compose Behavior**: Build complex protocol handling by composing simple `ComposableProvider` instances
- **Custom Protocols**: Implement your own protocol for proprietary APIs
- **Avoid Orphan Instances**: Use the library's provider types (`OpenAI`, `LlamaCpp`, etc.) rather than defining your own

## What You Need to Provide

This library handles **protocol encoding/decoding only**. You are responsible for:

- **HTTP Transport**: Making actual HTTP requests to LLM providers
- **Streaming**: If you want streaming responses, handle that in your transport layer
- **Error Handling**: Retries, exponential backoff, rate limiting, fallback strategies
- **Observability**: Logging, metrics, tracing, cost tracking
- **Connection Management**: Connection pooling, timeouts, keep-alive

The `examples/` directory includes simple HTTP transport implementations for reference, but they're not part of the library - they're just to demonstrate usage.

## Examples

The library includes reference implementations demonstrating usage patterns:

- **`examples/simple/`** - Minimal example showing library design with `SimpleModel` and `FullFeaturedModel`
- **`examples/claude/`** - Full tool-calling example with Anthropic (defines `ClaudeSonnet45` model locally)
- **`examples/llama-cpp/`** - OpenAI-compatible API example using llama.cpp server
- **`proxy/`** - Production-quality proxy servers that translate OpenAI requests to different backends
  - Demonstrates local model definitions (`GPT4o`, `ClaudeSonnet45`)
  - Shows real-world usage patterns

All examples include simple HTTP transport code (not part of library) for reference.

## Available Capabilities

The library supports the following capabilities (when implemented by both model and provider):

- **`HasTools`** - Tool/function calling (OpenAI, Anthropic)
- **`HasJSON`** - Structured JSON output mode (OpenAI only)
- **`HasReasoning`** - Reasoning/chain-of-thought (OpenAI only)
- **`HasVision`** - Image/vision support (type defined, not yet implemented in any provider)

Models declare which capabilities they support via type class instances. The type system ensures you can only use capabilities that both the model and provider support.

## Testing

```bash
cabal test          # Run all tests
cabal build all     # Build library, examples, and proxy servers
```

The test suite includes:
- Unit tests for composable handler behavior
- Property tests for protocol encoding/decoding
- Integration tests (cached) demonstrating full request/response cycles