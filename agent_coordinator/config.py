import os

from dotenv import load_dotenv

load_dotenv()

# Redis
REDIS_URL: str = os.getenv("REDIS_URL", "")

# Agent names the coordinator manages
AGENT_NAMES: list[str] = ["chatgpt", "claude", "gemini", "grok"]

# Discord channel IDs where agents can converse
_channel_ids = os.getenv("AGENT_CHANNEL_IDS", "")
AGENT_CHANNEL_IDS: list[int] = [
    int(cid.strip()) for cid in _channel_ids.split(",") if cid.strip()
]

# Channel ID → theme mapping (e.g. "123:debate,456:casual,789:memes")
_theme_map_str = os.getenv("CHANNEL_THEME_MAP", "")
CHANNEL_THEMES: dict[int, str] = {}
for entry in _theme_map_str.split(","):
    entry = entry.strip()
    if ":" in entry:
        cid, theme = entry.split(":", 1)
        CHANNEL_THEMES[int(cid.strip())] = theme.strip()

# Scheduling
SCHEDULE_MIN_EVENTS: int = int(os.getenv("COORDINATOR_SCHEDULE_MIN", "10"))
SCHEDULE_MAX_EVENTS: int = int(os.getenv("COORDINATOR_SCHEDULE_MAX", "15"))
SCHEDULE_ACTIVE_START_HOUR: int = int(os.getenv("COORDINATOR_ACTIVE_START", "7"))
SCHEDULE_ACTIVE_END_HOUR: int = int(os.getenv("COORDINATOR_ACTIVE_END", "23"))

# Conversation
MAX_ROUNDS: int = int(os.getenv("COORDINATOR_MAX_ROUNDS", "40"))
AGENT_RESPONSE_TIMEOUT: float = 90.0
CONTINUATION_BASE_PROBABILITY: float = 0.85
CONTINUATION_DECAY: float = 0.03
MIN_RESPONDENTS_TO_CONTINUE: int = 2

# Fire a conversation immediately on startup (for testing)
FIRE_ON_STARTUP: bool = os.getenv("COORDINATOR_FIRE_ON_STARTUP", "false").lower() in ("true", "1", "yes")

# Reactive triggers
REACTIVE_TRIGGER_PROBABILITY: float = float(os.getenv("COORDINATOR_REACTIVE_PROBABILITY", "0.15"))
REACTIVE_COOLDOWN_SECONDS: float = 300.0
