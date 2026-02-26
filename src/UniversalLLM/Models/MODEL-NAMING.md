# Model Module Naming Convention

## Rule: `Vendor/Family.hs`

Every concrete model gets its own module named after its vendor and model family:

```
src/UniversalLLM/Models/
  Anthropic/Claude.hs       -- Claude Sonnet, Haiku, Opus, ...
  Google/Gemini.hs          -- Gemini Flash, Pro, ...
  Amazon/Nova.hs            -- Nova Lite, Pro, ...
  ZhipuAI/GLM.hs            -- GLM-4.5, GLM-4.6, GLM-5, ...
  Alibaba/Qwen.hs           -- Qwen3.5-122B, Qwen3-Coder, ...
  Moonshot/Kimi.hs          -- Kimi K2.5, ...
  Minimax/M.hs              -- M2.5, ...
  OpenRouter.hs             -- Parametric catch-alls only (Universal, UniversalXMLTools)
```

The test modules mirror this hierarchy exactly:

```
test/Models/
  Anthropic/Claude.hs
  Google/Gemini3Flash.hs    -- One file per model (or test group) within the family
  Amazon/Nova2Lite.hs
  ZhipuAI/GLM.hs
  Alibaba/Qwen3Coder.hs
  Alibaba/Qwen35.hs
  Moonshot/KimiK25.hs
  Minimax/MinimaxM25.hs
```

## Key principle: placement by identity, not access path

Even if a model is currently only reachable via OpenRouter, it goes in **its vendor's
module** — not in `OpenRouter.hs`. The access path (which provider/API you call) is
an implementation detail of the type class instances; the module hierarchy reflects
whose model it is.

## What goes in `OpenRouter.hs`

Only **parametric, non-vendor-specific** model types:

- `Universal` — text-only catch-all; any OpenRouter model name
- `UniversalXMLTools` — XML-based tool injection for models without native tool support

These are structural stand-ins for when you can't be bothered to define a proper type,
or when exploring a brand-new model. If a model earns proper production status, it
gets its own `Vendor/Family.hs` module.

## Adding a new model

1. Create (or extend) `src/UniversalLLM/Models/<Vendor>/<Family>.hs`
2. Declare the module as `UniversalLLM.Models.<Vendor>.<Family>`
3. Add the module to `exposed-modules` in `universal-llm.cabal`
4. Re-export from `UniversalLLM.Models` (the umbrella module)
5. Add tests in `test/Models/<Vendor>/<ModelOrGroup>.hs`
6. Add the test module to `other-modules` in `universal-llm.cabal`
7. Import and call from `test/Main.hs`

## Vendor names

Use the company/org name, not the API/product brand:

| Company    | Vendor dir   |
|------------|--------------|
| Anthropic  | `Anthropic/` |
| Google     | `Google/`    |
| Amazon     | `Amazon/`    |
| Zhipu AI   | `ZhipuAI/`   |
| Alibaba    | `Alibaba/`   |
| Moonshot AI| `Moonshot/`  |
| MiniMax    | `Minimax/`   |
