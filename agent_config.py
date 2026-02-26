import os

from dotenv import load_dotenv

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

_channel_ids = os.getenv("AGENT_CHANNEL_IDS", "")
AGENT_CHANNEL_IDS: list[int] = [
    int(cid.strip()) for cid in _channel_ids.split(",") if cid.strip()
]

_bot_ids = os.getenv("BOT_IDS", "")
BOT_IDS: list[int] = [
    int(bid.strip()) for bid in _bot_ids.split(",") if bid.strip()
]

# Rate limiting
AGENT_MAX_DAILY: int = int(os.getenv("AGENT_MAX_DAILY", "30"))
AGENT_COOLDOWN_SECONDS: int = int(os.getenv("AGENT_COOLDOWN_SECONDS", "120"))

# Context window — how many messages to include for AI context
CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "50"))

# Redis
REDIS_URL: str = os.getenv("REDIS_URL", "")
