# Universal LLM

A type-safe, protocol-agnostic Haskell library for interacting with multiple Large Language Model APIs.

## Features

- **Type-Safe Translation**: Pure translation layer between universal message format and provider wire formats
- **Multiple Providers**: Support for OpenAI, Anthropic, and other LLM protocols
- **Compile-time Capability Checks**: Invalid capability combinations caught at compile time via GADTs
- **Protocol Separation**: Clean separation between wire formats (protocols) and capabilities (providers)
- **Tool Calling Support**: Type-safe tool definitions with automatic parameter validation

## Design Principles

- **Translation, Not Transport**: Library handles protocol translation; users handle HTTP/transport
- **Compile-time Safety**: GADT-based message types ensure capability constraints are checked at compile time
- **Pure Functions**: All translation logic is pure for easy testing and composition
- **Universal Message Format**: Messages can be converted between different providers/models

## Quick Example

```haskell
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
import UniversalLLM

-- Define your messages with capabilities enforced by types
let messages = [ UserText "What's the weather in Paris?"
               , AssistantTool weatherToolCall  -- Only valid if model HasTools
               , ToolResultMsg weatherResult
               , AssistantText "It's 22°C and sunny!"
               ]

-- Configure the request
let configs = [ Temperature 0.7
              , MaxTokens 200
              , Tools [weatherToolDef]  -- Only valid if provider HasTools
              ]

-- Translate to provider's wire format (pure function)
let request = toRequest OpenAI GPT4o configs messages

-- Send via your own HTTP transport...
-- Parse response (pure function)
let result = fromResponse @OpenAI @GPT4o response
```

## Architecture

The library separates three orthogonal concerns:

1. **Protocols** (`OpenAI`, `Anthropic`): Wire format specifications (JSON schemas)
2. **Providers** (phantom types or data): Capability declarations, optionally carrying provider-specific config
3. **Models** (phantom types or data): Model identifiers with name mapping, optionally carrying model-specific config

Generic/portable settings use `ModelConfig` (Temperature, MaxTokens, etc.), while provider-specific or model-specific behavior can use data fields.

Messages are universal and can be translated between any compatible provider/model combination. Capabilities are enforced at compile time via GADT constructors with type constraints.

### Extensibility

- **Model Extension**: Newtype-derive capabilities from existing models to create variations
- **Provider Wrapping**: Wrap `toRequest`/`fromResponse` to add behavior (caching, retry logic, emulation layers)
- **Custom Protocols**: Implement your own protocol instances for proprietary APIs

Example emulation via provider wrapping:
```haskell
-- Add seed emulation for providers that don't support it
data EmulatedProvider base = EmulatedProvider base

instance Provider base model => Provider (EmulatedProvider base) model where
  toRequest (EmulatedProvider base) model configs msgs =
    -- Filter out Seed config, handle it manually if needed
    toRequest base model (filter (not . isSeed) configs) msgs
```