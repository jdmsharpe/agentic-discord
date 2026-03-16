import math
import os

from dotenv import load_dotenv


def _require_int(name: str) -> int:
    """Read a required integer env var, raising immediately if missing."""
    val = os.getenv(name)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return int(val)


load_dotenv()

# Agent identity — determines which bot token + API key to use
AGENT_NAME: str = os.getenv("AGENT_NAME", "chatgpt")

# Discord bot tokens — one per bot, all in the same .env
BOT_TOKEN_CHATGPT: str = os.getenv("BOT_TOKEN_CHATGPT", "")
BOT_TOKEN_CLAUDE: str = os.getenv("BOT_TOKEN_CLAUDE", "")
BOT_TOKEN_GEMINI: str = os.getenv("BOT_TOKEN_GEMINI", "")
BOT_TOKEN_GROK: str = os.getenv("BOT_TOKEN_GROK", "")

# Resolve which token this instance uses based on AGENT_NAME
_TOKEN_MAP: dict[str, str] = {
    "chatgpt": BOT_TOKEN_CHATGPT,
    "claude": BOT_TOKEN_CLAUDE,
    "gemini": BOT_TOKEN_GEMINI,
    "grok": BOT_TOKEN_GROK,
}
BOT_TOKEN: str = _TOKEN_MAP.get(AGENT_NAME, "")

# Which agents have bot tokens configured (i.e. are actually runnable)
ACTIVE_AGENT_NAMES: list[str] = [name for name, token in _TOKEN_MAP.items() if token]

# API keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")

# Personality
# Global override (applies to all bots if set)
AGENT_PERSONALITY: str = os.getenv("AGENT_PERSONALITY", "")

# Per-bot defaults
AGENT_PERSONALITY_MAP: dict[str, str] = {
    "chatgpt": "Crisp, analytical, slightly sardonic. Prefers tight, logical takes.",
    "claude": "Dry, measured, with literary wit. Delivers precise, cutting lines.",
    "gemini": "Playful and curious. Quick with wordplay and light jabs.",
    "grok": "Unapologetically savage with a side of truth bombs. Likes confident, edgy swings.",
}

# Discord server / channel / bot IDs
_guild_ids_str = os.getenv("GUILD_IDS", "")
GUILD_IDS: list[int] = [
    int(gid.strip()) for gid in _guild_ids_str.split(",") if gid.strip()
]

# Derive active channel IDs and themes from CHANNEL_THEME_MAP
_theme_map_str = os.getenv("CHANNEL_THEME_MAP", "")
CHANNEL_THEMES: dict[int, str] = {}
for _entry in _theme_map_str.split(","):
    _entry = _entry.strip()
    if ":" in _entry:
        _cid, _theme = _entry.split(":", 1)
        CHANNEL_THEMES[int(_cid.strip())] = _theme.strip()

AGENT_CHANNEL_IDS: list[int] = list(CHANNEL_THEMES.keys())

_bot_ids = os.getenv("BOT_IDS", "")
BOT_IDS: list[int] = [int(bid.strip()) for bid in _bot_ids.split(",") if bid.strip()]

# Discord role ID for the shared @bots role — when mentioned, all agents respond
BOTS_ROLE_ID: int = int(os.getenv("BOTS_ROLE_ID", "0"))

# Rate limiting
AGENT_MAX_DAILY: int = _require_int("AGENT_MAX_DAILY")
AGENT_COOLDOWN_SECONDS: int = _require_int("AGENT_COOLDOWN_SECONDS")

# Context window — how many messages to include for AI context (also the max)
CONTEXT_WINDOW_SIZE: int = _require_int("CONTEXT_WINDOW_SIZE")
if CONTEXT_WINDOW_SIZE < 1:
    raise RuntimeError(f"CONTEXT_WINDOW_SIZE must be >= 1, got {CONTEXT_WINDOW_SIZE}")

# Per-theme scale factors (1.0 = full CONTEXT_WINDOW_SIZE, lower = fewer messages)
_THEME_WINDOW_SCALES: dict[str, float] = {
    "debate": 1.0,
    "story": 1.0,
    "hypothetical": 1.0,
    "prediction": 1.0,
    "news": 0.8,
    "science": 0.8,
    "finance": 0.8,
    "spiritual": 0.8,
    "casual": 0.55,
    "vent": 0.55,
    "would-you-rather": 0.55,
    "memes": 0.35,
    "roast": 0.35,
}


def get_context_window(theme: str | None = None) -> int:
    """Return the context window size for a theme, scaled from CONTEXT_WINDOW_SIZE."""
    if theme:
        scale = _THEME_WINDOW_SCALES.get(theme, 1.0)
        return max(1, math.ceil(CONTEXT_WINDOW_SIZE * scale))
    return CONTEXT_WINDOW_SIZE


# Redis
REDIS_URL: str = os.getenv("REDIS_URL", "")
