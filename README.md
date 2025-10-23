# Universal LLM

A type-safe Haskell library for encoding and decoding Large Language Model API protocols.

## What This Library Does

This library **encodes messages to provider-specific request formats** and **decodes provider responses back to messages**. It does NOT handle HTTP transport, retries, streaming, or rate limiting - that's your responsibility. Think of it as a type-safe protocol codec with compile-time capability constraints.

## Features

- **Pure Protocol Encoding/Decoding**: Transform messages to/from provider wire formats (JSON)
- **Multiple Providers**: OpenAI, Anthropic, and custom protocol support
- **Compile-time Capability Checks**: Invalid capability combinations caught at compile time via GADTs
- **Composable Providers**: Build complex behavior by composing simple handlers via Monoid/Semigroup
- **Tool Calling Support**: Type-safe tool definitions with automatic parameter validation
- **Provider-Agnostic Business Logic**: Write code once, run with any compatible provider/model

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

-- Select concrete provider/model at application entry point
main = do
  let provider = Anthropic
      model = Claude35Sonnet
      tools = [LLMTool myTool]

  processQuery provider model tools "What's the weather?"
  -- Business logic (processQuery) works with ANY compatible provider/model
```

## Architecture

### Core Abstractions

1. **Protocols** (`UniversalLLM.Protocols.OpenAI`, `UniversalLLM.Protocols.Anthropic`): Wire format specifications - the actual JSON schemas for each provider's API

2. **Providers** (`OpenAI`, `Anthropic`): Capability declarations + protocol implementations. Can be phantom types or carry provider-specific config

3. **Models** (`GPT4o`, `Claude35Sonnet`): Model identifiers with capability instances. Can be phantom types or carry model-specific config

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

### Extensibility

- **Add New Models**: Define a model type, implement `ModelName` and capability instances (`HasTools`, `HasVision`, etc.)
- **Add New Providers**: Implement protocol types and composable provider handlers
- **Compose Behavior**: Build complex protocol handling by composing simple `ComposableProvider` instances
- **Custom Protocols**: Implement your own protocol for proprietary APIs

Example: Adding tool support to a new model
```haskell
data MyModel = MyModel

instance ModelName OpenAI MyModel where
  modelName _ = "my-model-name"

instance HasTools MyModel OpenAI where
  toolsComposableProvider = OpenAI.toolsComposableProvider

instance ProviderImplementation OpenAI MyModel where
  getComposableProvider = baseComposableProvider <> toolsComposableProvider
```

## What You Need to Provide

This library handles **protocol encoding/decoding only**. You are responsible for:

- **HTTP Transport**: Making actual HTTP requests to LLM providers
- **Streaming**: If you want streaming responses, handle that in your transport layer
- **Error Handling**: Retries, exponential backoff, rate limiting, fallback strategies
- **Observability**: Logging, metrics, tracing, cost tracking
- **Connection Management**: Connection pooling, timeouts, keep-alive

The `examples/` directory includes simple HTTP transport implementations for reference, but they're not part of the library - they're just to demonstrate usage.

## See Also

- `examples/claude/` - Full tool-calling example with Anthropic
- `examples/llama-cpp/` - OpenAI-compatible API example (llama.cpp server)
- `test/` - Property tests and integration tests demonstrating the API