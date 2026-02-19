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

Models are organized by vendor/family:

- "UniversalLLM.Models.Anthropic.Claude" - Anthropic's Claude models
- "UniversalLLM.Models.Google.Gemini"    - Google Gemini models
- "UniversalLLM.Models.Amazon.Nova"      - Amazon Nova models
- "UniversalLLM.Models.ZhipuAI.GLM"     - Zhipu AI's GLM models
- "UniversalLLM.Models.Alibaba.Qwen"    - Alibaba's Qwen models
- "UniversalLLM.Models.Moonshot.Kimi"   - Moonshot AI's Kimi models
- "UniversalLLM.Models.Minimax.M"       - MiniMax models
- "UniversalLLM.Models.OpenRouter"      - Parametric catch-alls for OpenRouter

= Adding New Models

To add a new production model:

1. Create a module under @UniversalLLM.Models.Vendor.Family@ (e.g. @Google.Gemini@)
2. Define the model type and instances
3. Add tests in @test/Models/Vendor/@ directory
4. Re-export from this module

For application-specific models or experiments, define them locally in your app
rather than adding them here.
-}

module UniversalLLM.Models
  ( -- * Anthropic Models
    -- | Claude models from Anthropic
    module UniversalLLM.Models.Anthropic.Claude
    -- * Google Models
    -- | Google Gemini models
  , module UniversalLLM.Models.Google.Gemini
    -- * Amazon Models
    -- | Amazon Nova models
  , module UniversalLLM.Models.Amazon.Nova
    -- * ZhipuAI Models
    -- | GLM models from Zhipu AI
  , module UniversalLLM.Models.ZhipuAI.GLM
    -- * Alibaba Models
    -- | Qwen models from Alibaba
  , module UniversalLLM.Models.Alibaba.Qwen
    -- * Moonshot Models
    -- | Kimi models from Moonshot AI
  , module UniversalLLM.Models.Moonshot.Kimi
    -- * MiniMax Models
    -- | MiniMax models
  , module UniversalLLM.Models.Minimax.M
    -- * OpenRouter Parametric Models
    -- | Catch-all parametric models for OpenRouter
  , module UniversalLLM.Models.OpenRouter
  ) where

import UniversalLLM.Models.Anthropic.Claude
import UniversalLLM.Models.Google.Gemini
import UniversalLLM.Models.Amazon.Nova
import UniversalLLM.Models.ZhipuAI.GLM
import UniversalLLM.Models.Alibaba.Qwen
import UniversalLLM.Models.Moonshot.Kimi
import UniversalLLM.Models.Minimax.M
import UniversalLLM.Models.OpenRouter
