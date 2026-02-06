{- |
Module: UniversalLLM.Models
Description: Central repository of production-ready model definitions

This module re-exports all production-ready model definitions from UniversalLLM.
Instead of defining models in each application, import them from here to use
tested and verified configurations.

= Quick Start

@
import UniversalLLM
import UniversalLLM.Models

-- Use Claude Sonnet 4.5
let model = Model ClaudeSonnet45 Anthropic
let provider = claudeSonnet45

-- Use GLM-4.5-Air via llama.cpp
let model = Model GLM45Air LlamaCpp
let provider = glm45AirLlamaCpp

-- Use Qwen 3 Coder
let model = Model Qwen3Coder LlamaCpp
let provider = qwen3Coder
@

= Model Organization

Models are organized by provider/vendor:

- "UniversalLLM.Models.Anthropic" - Claude models
- "UniversalLLM.Models.GLM" - Zhipu AI's GLM models
- "UniversalLLM.Models.Qwen" - Alibaba's Qwen models
- "UniversalLLM.Models.OpenRouter" - Models via OpenRouter

= Adding New Models

To add a new production model:

1. Create a module under @UniversalLLM.Models.@ for the vendor
2. Define the model type and instances
3. Add tests in @test/Models/@ directory
4. Re-export from this module

For application-specific models or experiments, define them locally in your app
rather than adding them here.
-}

module UniversalLLM.Models
  ( -- * Anthropic Models
    -- | Claude models from Anthropic
    module UniversalLLM.Models.Anthropic
    -- * GLM Models
    -- | GLM models from Zhipu AI
  , module UniversalLLM.Models.GLM
    -- * Qwen Models
    -- | Qwen models from Alibaba
  , module UniversalLLM.Models.Qwen
    -- * OpenRouter Models
    -- | Various models via OpenRouter
  , module UniversalLLM.Models.OpenRouter
  ) where

import UniversalLLM.Models.Anthropic
import UniversalLLM.Models.GLM
import UniversalLLM.Models.Qwen
import UniversalLLM.Models.OpenRouter
