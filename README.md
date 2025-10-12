# Universal LLM

A type-safe, capability-based Haskell library for interacting with multiple Large Language Model APIs.

## Features

- **Type-Safe Capabilities**: Compile-time validation of model and provider capabilities
- **Multiple Providers**: Support for OpenAI, Anthropic, and other LLM APIs
- **Capability Emulation**: Automatic emulation of missing features (JSON output, tool calls, etc.)
- **Pure Core Functions**: Separate request preparation from IO execution
- **Flexible Message System**: Composable message builders with capability constraints

## Design Principles

- **Compile-time Safety**: Invalid capability combinations result in compile errors, not runtime failures
- **Pure Transformations**: Core provider logic is pure functions for easy testing and composition
- **Capability-Based**: Declare what features you need, not which specific provider to use
- **Emulation Support**: Gracefully handle missing features through transparent emulation

## Quick Example

```haskell
{-# LANGUAGE OverloadedStrings #-}
import UniversalLLM

-- Function works with any provider that supports vision and JSON output
analyzeImage :: (Vision model provider, JSONOutput model provider)
             => provider
             -> ModelParams model
             -> Image
             -> IO AnalysisResult

-- Compose message capabilities as needed
request = WithJSONOutput analysisSchema $
          WithImages [image] $
          BasicUserRequest "Analyze this image"

-- Execute with any compatible provider
result <- callLLM provider params [request]
```

## Architecture

The library is built around three core concepts:

1. **Models as Types**: `GPT4o`, `Claude35`, etc. carry capability information
2. **Capability Constraints**: Functions declare required features via type constraints
3. **Pure Provider Functions**: Request preparation and response parsing are pure transformations

This ensures that capability mismatches are caught at compile time while maintaining flexibility through emulation and polymorphic interfaces.