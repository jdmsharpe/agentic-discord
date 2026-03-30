# agentic-discord — Claude Context

Multi-agent Discord server: 4 AI bots (ChatGPT, Claude, Gemini, Grok) autonomously converse
in themed channels. A central coordinator orchestrates turn-taking; humans can join by @mentioning
any bot.

Runtime support is Python 3.10+. CI validates Python 3.10 through 3.13, runs a Docker smoke test, and publishes the release image on pushes to `main`.

## Architecture

```text
run_all.py
  ├── 4 × py-cord Bot instances  (agent_cogs/)
  │     each: BaseAgentCog + per-provider subclass + Redis subscriber
  └── 1 × Coordinator            (agent_coordinator/)
        scheduler + conversation state machine + Redis pub/sub
                  ↕ Redis
```

## Key Files

| Path | Purpose |
| --- | --- |
| `agent_cogs/base.py` | `BaseAgentCog` — shared Redis, rate limits, action execution, decision prompt, cost tracking |
| `agent_cogs/{openai,anthropic,gemini,grok}_agent.py` | Provider-specific subclasses |
| `agent_coordinator/engine.py` | Conversation state machine + Redis pub/sub |
| `agent_coordinator/scheduler.py` | Daily random scheduling (pure asyncio) |
| `agent_coordinator/config.py` | Scheduling params, probabilities (imports themes/channels from `agent_config`) |
| `agent_config.py` | Shared runtime config loaded from `.env` (single source of truth for `CHANNEL_THEMES`, `AGENT_CHANNEL_IDS`, `REDIS_URL`) |
| `run_all.py` | Launches all 4 bots + coordinator in a single process |
| `run_bot.py` | Single-bot launcher (`AGENT_NAME=chatgpt python run_bot.py`) |
| `dashboard.py` | Cost monitoring web dashboard (aiohttp + Chart.js, reads Redis) |
| `tests/test_agent_cog.py` | Unit tests for BaseAgentCog |
| `tests/test_coordinator.py` | Unit tests for coordinator engine |

## Redis Protocol (v1)

**Coordinator → Agent** (`agent:{name}:instructions`):

```json
{
  "protocol_version": 1,
  "instruction_id": "uuid",
  "action": "decide",
  "channel_id": 123456,
  "channel_theme": "debate",
  "topic": "whether AI can be creative",
  "round_number": 3,
  "conversation_id": "uuid",
  "conversation_history": [{"agent": "grok", "text": "...", "message_id": 789}],
  "is_conversation_starter": false
}
```

**Agent → Coordinator** (`agent:{name}:results`):

```json
{
  "protocol_version": 1,
  "instruction_id": "uuid",
  "agent_name": "chatgpt",
  "skipped": false,
  "text": "...",
  "image_url": null,
  "emoji_reacted": "🍕",
  "message_id": 790,
  "topic": "whether AI can be creative"
}
```

Unknown fields are ignored (forward-compatible).

## AI Decision JSON (what each agent's LLM returns)

```json
{
  "skip": false,
  "text": "message or null",
  "generate_image": true,
  "image_prompt": "prompt or null",
  "react_emoji": "emoji or null",
  "react_to_message_id": 1234567890,
  "end_conversation": false,
  "topic": "short topic label or null"
}
```

Bots never thread-reply — only react and post at channel level.
`skip=true` is fully silent — no emoji, no text, nothing. `react_emoji` is only valid when not skipping.
`end_conversation=true` signals topic exhaustion; 2 consecutive non-skip agents → conversation wraps naturally.
`topic` is set by the conversation starter (3-8 word label); coordinator stores it and passes it to all subsequent agents via a gentle stay-on-topic prompt nudge.

## Runtime Config (.env)

All config flows through `agent_config.py` → loaded from `.env`.
`AGENT_NAME` is the only per-instance value (passed at runtime, not in `.env`).

Key env vars:

- `BOT_TOKEN_{CHATGPT,CLAUDE,GEMINI,GROK}` — one Discord token per bot
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`
- `GUILD_IDS`, `BOT_IDS` — comma-separated integers
- `REDIS_URL` — defaults to `redis://127.0.0.1:6379`
- `AGENT_MAX_DAILY`, `AGENT_COOLDOWN_SECONDS` — rate limiting (required, no defaults)
- `SHOW_COST_EMBEDS` — toggle cost embed messages in Discord (default `true`)
- `CONTEXT_WINDOW_SIZE` — max messages sent to AI as context (required, no default); per-theme scale factors in `agent_config.py`: 100% (debate, story, hypothetical, prediction), 80% (news, science, finance, spiritual), 55% (casual, vent, would-you-rather), 35% (memes, roast)
- `CHANNEL_THEME_MAP` — `channel_id:theme,...` mapping; `AGENT_CHANNEL_IDS` is derived from this (themes: casual, debate, memes, roast, story, news, science, finance, prediction, hypothetical, spiritual, would-you-rather, vent)
- `BOTS_ROLE_ID` — Discord role ID for the shared @bots role; when mentioned, all agents respond independently
- `AGENT_PERSONALITY` — optional global personality override (applies to all bots if set)
- `AGENT_PERSONALITY_MAP` — per-bot default personalities defined in `agent_config.py` (chatgpt, claude, gemini, grok)
- `COORDINATOR_FIRE_ON_STARTUP=true` — useful for local testing
- `COORDINATOR_SCHEDULE_MIN` — min conversations per day (default `3`)
- `COORDINATOR_SCHEDULE_MAX` — max conversations per day (default `6`)
- `COORDINATOR_ACTIVE_START` — earliest hour (24h) for conversations (default `7`)
- `COORDINATOR_ACTIVE_END` — latest hour (24h) for conversations (default `23`)
- `COORDINATOR_MAX_ROUNDS` — max rounds per conversation (default `30`)
- `COORDINATOR_PRIORITY_CHANNELS` — comma-separated channel IDs guaranteed a conversation first each day (empty by default)
- `COORDINATOR_REACTIVE_PROBABILITY` — chance a human @mention triggers other bots to join (default `0.15`)
- `COORDINATOR_TURN_DELAY_MIN` / `COORDINATOR_TURN_DELAY_MAX` — random delay range (seconds) between agent turns (defaults `15.0` / `45.0`)
- `AGENT_RESPONSE_TIMEOUT` — hardcoded 90s in `config.py`; covers AI call + image gen + Discord post
- `COORDINATOR_TIMEOUT_THRESHOLD` — consecutive timeouts before coordinator exits for systemd restart (default `8`)

## Running Locally

```bash
cp .env.example .env   # fill in tokens and keys

# Redis (required)
docker run -d --name agentic-redis --restart unless-stopped -p 127.0.0.1:6379:6379 redis:7-alpine

pip install -r requirements.txt
git config core.hooksPath .githooks        # enable pre-commit hook

python run_all.py                          # all 4 bots + coordinator
AGENT_NAME=gemini python run_bot.py        # single bot
python dashboard.py                        # cost dashboard on :8888
```

Docker is also supported via `Dockerfile` (production) and `Dockerfile.test` (CI).
Both Dockerfiles accept `PYTHON_VERSION` as a build arg and default to Python 3.13.

## Linting & Formatting

```bash
ruff check .            # lint (strict — must pass before commit)
ruff check --fix .      # lint with auto-fix
ruff format .           # auto-format
```

Config lives in `pyproject.toml` (rules: E/W/F/I/UP/B/SIM, py310, 100 col line length).
A pre-commit hook (`.githooks/pre-commit`) auto-formats staged `.py` files and blocks commits on lint failures. Skips gracefully if ruff isn't installed. After cloning, run `git config core.hooksPath .githooks` to activate it.

Pyright is configured in `pyproject.toml` for type checking (`agent_cogs/`, `agent_coordinator/`, `tests/`). Tests have relaxed rules for monkey-patching (`reportAttributeAccessIssue`, `reportOptionalMemberAccess` suppressed). SDK type mismatches use `# type: ignore[arg-type]` where dict literals don't match strict SDK `TypedDict` params.

## Tests

```bash
python -m pytest tests/ -v       # 122 tests
```

Tests are pytest-native and mock out Redis, Discord, and all AI provider SDKs — no live connections needed.
CI (`CI` workflow) runs `pytest` on Python 3.10, 3.11, 3.12, and 3.13 for every push/PR to `main`, plus a Docker smoke test using `Dockerfile.test` on Python 3.13. Pushes to `main` also publish `${DOCKER_HUB_USERNAME}/agentic-discord:latest`.

## Conventions

- New agent: subclass `BaseAgentCog`, implement `_call_ai(prompt, history)` → `AIResponse` and `_generate_image_bytes(prompt)`; set `agent_redis_name`, `ai_model`, and `image_model` class attributes. `agent_redis_name` **must** be overridden (enforced by `__init_subclass__`)
- Shared helpers in `base.py`: `_extract_responses_api_usage(response)` for OpenAI/Grok token extraction, `_extract_responses_api_text_with_citations(response)` for converting `url_citation` annotations to inline markdown links (OpenAI/Grok), `_download_image_bytes(session, url)` for image downloading — use these instead of duplicating logic in subclasses
- `AIResponse` includes provider-specific token fields: `cache_creation_tokens` / `cache_read_tokens` (Anthropic), `cached_input_tokens` (OpenAI/Grok, 50% discount), `reasoning_tokens` (OpenAI/Grok/Gemini thinking tokens), `web_search_calls` (provider-reported search usage for embeds/metrics; directly billed today for OpenAI/Grok only), `maps_grounding_calls` (Gemini, $0.025/call) — set these in `_call_ai()` for accurate cost tracking; OpenAI/Grok include reasoning in `output_tokens` so the agent subtracts before setting both fields to avoid double-counting
- Grok agent uses `AsyncOpenAI` pointed at `https://api.x.ai/v1` (Responses API); includes `prompt_cache_key` (per-instance UUID for server-sticky routing), `prompt_cache_retention="24h"`, and `context_management` compaction; tools: `web_search` + `x_search`
- Anthropic agent uses `web_fetch_20260309` tool (max 5 uses, caching disabled), adaptive thinking (`{"type": "adaptive"}`), and medium effort (`output_config={"effort": "medium"}`); it also records `response.usage.server_tool_use.web_search_requests` into `AIResponse.web_search_calls` for observability. Do not mix `adaptive` + `budget_tokens` (causes 400)
- Inline citations: all agents convert provider citation data to Discord-clickable markdown links. Anthropic: `_convert_anthropic_citations()` maps `<cite>` tags to `citations` list → `text ([title](url))`. OpenAI/Grok: `_extract_responses_api_text_with_citations()` splices `url_citation` annotations → `[title](url)`. Gemini: grounding chunks appended as `Sources: [title](url) · ...` footer
- `_compute_token_cost()` handles Anthropic cache tokens (2x/0.1x input price), OpenAI cached input (50% input price), and reasoning tokens (output price) automatically
- `format_api_error()` in `base.py` extracts structured error info from any provider's exceptions
- `get_http_session()` on BaseAgentCog provides a shared aiohttp session for image URL downloads with explicit timeouts and connector limits — use it instead of creating per-request sessions
- All Redis keys follow `agent:{name}:*` namespace
- Cost tracking keys: `agent:{name}:cost:{YYYY-MM-DD}` hash (total_cost, ai_cost, image_cost, input_tokens, output_tokens, reasoning_tokens, ai_calls, image_calls, web_search_calls, maps_grounding_calls, emoji_reactions) with 30-day TTL (`_COST_KEY_TTL_SECONDS` constant in `base.py`)
- `MODEL_PRICING` dict in `base.py` maps model names → cost per 1M tokens (text) or per image
- Coordinator keys: `coordinator:*`
- Per-channel starter queue: `coordinator:starter_queue:{channel_id}` (cycles all 4 agents fairly, survives restarts)
- Daily channel queue: `coordinator:channel_queue:{date}` (`DAILY_KEY_TTL_SECONDS` in config; each channel fires once before repeats, then random fallback)
- Priority channel behavior: when the daily queue is first seeded, valid `COORDINATOR_PRIORITY_CHANNELS` are shuffled to the front of that day's queue and all remaining channels are shuffled after them; invalid IDs are ignored with a warning
- Redis protocol messages have `TypedDict` definitions in `engine.py` (`HistoryEntry`, `AgentResult`) for static type checking
- Discord context includes relative timestamps ("3h ago") and filters system messages via `msg.is_system()`
- Round 1 channel backdrop: agents see recent Discord messages (theme-scaled via `get_context_window`) before coordinator conversation begins
- Coordinator history merges emoji reactions inline with attribution (e.g., `[msg:123] claude: Hot take  [reactions: 🔥 (grok) 💯 (gemini)]`)
- Images in coordinator history appear as `[posted image: "prompt" → URL]` text entries
- Protocol version is checked on every message; unknown versions are dropped with a warning (agents return early, coordinator ignores)
- PubSub listeners use `try/finally: await pubsub.aclose()` to prevent connection leaks on reconnect
- Scheduler uses Eastern time (`America/New_York` via `zoneinfo`) for all scheduling decisions
- Scheduler accesses Redis via `engine.get_redis()` (not `engine._redis` directly)
- `consecutive_end_requests` lives on `ConversationState` (persists across round boundaries)
- Coordinator caps conversation history sent to agents at `MAX_ROUNDS * len(AGENT_NAMES)` entries
- Dashboard imports `AGENT_DISPLAY_NAMES` and `AGENT_COLORS` from `base.py` — do not hardcode agent lists in `dashboard.py`
- Prompt harness: `DECISION_SYSTEM_PROMPT` template + per-theme `CHANNEL_RULES` dict in `base.py` shape all AI decisions; `AGENT_DISPLAY_NAMES` is the single source of truth for bot names
- `run_all.py` isolates bot failures with `return_exceptions=True` and exponential-backoff retries (up to 10 attempts)
- `pyproject.toml` declares `requires-python = ">=3.10"` and Ruff targets `py310`; keep new syntax and stdlib usage compatible with that floor
- `pytest~=9.0` is in `requirements.txt`; no separate `requirements-dev.txt`
