# Universal LLM Proxy

OpenAI-compatible HTTP proxy that translates requests through the universal message format to various LLM backends.

## Key Concept

All requests go through the universal `[Message]` format as an intermediate representation:

```
OpenAI Request → Universal Messages → Backend Request → Backend → Universal Messages → OpenAI Response
```

This architecture enables:
- **Provider translation**: Accept OpenAI format, proxy to Anthropic (or vice versa)
- **Feature emulation**: Add capabilities to models that don't natively support them
- **Request transformation**: Caching, filtering, routing based on model/capability

## Available Proxies

### 1. OpenAI Backend (`llm-proxy`)

Proxies OpenAI-format requests to OpenAI GPT-4o.

**Usage:**
```bash
export OPENAI_API_KEY=your-key-here
cabal run llm-proxy
```

Listens on: `http://localhost:8080`

### 2. Anthropic Backend (`llm-proxy-anthropic`)

Accepts OpenAI-format requests, translates to Anthropic Claude 3.5 Sonnet.

**Usage:**
```bash
export ANTHROPIC_OAUTH_TOKEN=your-token-here
cabal run llm-proxy-anthropic
```

Listens on: `http://localhost:8081`

## Testing

```bash
# Test OpenAI backend
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 50
  }'

# Test Anthropic backend (same request format!)
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

## Architecture Demonstration

The minimal code changes between `Main.hs` and `MainAnthropic.hs` prove the library's design:

**What Changed:**
1. Backend configuration (3 lines): `Anthropic` + `Claude35Sonnet` instead of `OpenAI` + `GPT4o`
2. HTTP transport (1 function): Call Anthropic API instead of OpenAI API
3. Environment variable name

**What Stayed the Same:**
- All business logic (request parsing, message conversion, response building)
- The `Proxy.OpenAICompat` module (completely reusable)
- HTTP server setup and routing

This demonstrates that the universal message abstraction successfully decouples protocol handling from provider-specific details.

## Current Limitations

- Only supports text messages (no tools, vision, JSON mode yet)
- No streaming support
- Basic error handling
- Single model per proxy instance

These are implementation limitations, not architectural ones. The design supports all these features.

## Extending

Adding new capabilities is straightforward:

1. **Add tool support**: Update `Proxy.OpenAICompat` to parse/encode tool messages
2. **Multi-model routing**: Add model name → provider/model mapping
3. **Feature emulation**: Intercept messages in universal format, add missing capabilities
4. **Caching**: Store universal messages instead of provider-specific formats

See `Proxy.OpenAICompat.hs` for the translation layer.
