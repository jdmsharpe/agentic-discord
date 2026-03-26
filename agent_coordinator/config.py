import os

from dotenv import load_dotenv

load_dotenv()

from agent_config import ACTIVE_AGENT_NAMES

# Redis
REDIS_URL: str = os.getenv("REDIS_URL", "")

# Agent names the coordinator manages — derived from which BOT_TOKEN_* env vars are set
AGENT_NAMES: list[str] = ACTIVE_AGENT_NAMES

# Channel ID → theme mapping (e.g. "123:debate,456:casual,789:memes")
# AGENT_CHANNEL_IDS is derived from this — no separate env var needed.
_theme_map_str = os.getenv("CHANNEL_THEME_MAP", "")
CHANNEL_THEMES: dict[int, str] = {}
for entry in _theme_map_str.split(","):
    entry = entry.strip()
    if ":" in entry:
        cid, theme = entry.split(":", 1)
        CHANNEL_THEMES[int(cid.strip())] = theme.strip()

AGENT_CHANNEL_IDS: list[int] = list(CHANNEL_THEMES.keys())

# Scheduling
SCHEDULE_MIN_EVENTS: int = int(os.getenv("COORDINATOR_SCHEDULE_MIN", "3"))
SCHEDULE_MAX_EVENTS: int = int(os.getenv("COORDINATOR_SCHEDULE_MAX", "6"))
SCHEDULE_ACTIVE_START_HOUR: int = int(os.getenv("COORDINATOR_ACTIVE_START", "7"))
SCHEDULE_ACTIVE_END_HOUR: int = int(os.getenv("COORDINATOR_ACTIVE_END", "23"))

# Conversation
MAX_ROUNDS: int = int(os.getenv("COORDINATOR_MAX_ROUNDS", "30"))
AGENT_RESPONSE_TIMEOUT: float = 90.0
CONTINUATION_BASE_PROBABILITY: float = 0.85
CONTINUATION_DECAY: float = 0.03
MIN_RESPONDENTS_TO_CONTINUE: int = min(2, len(AGENT_NAMES))

# Health check — exit after this many consecutive timeouts so systemd can restart
CONSECUTIVE_TIMEOUT_THRESHOLD: int = int(os.getenv("COORDINATOR_TIMEOUT_THRESHOLD", "8"))

# Fire a conversation immediately on startup (for testing)
FIRE_ON_STARTUP: bool = os.getenv("COORDINATOR_FIRE_ON_STARTUP", "false").lower() in ("true", "1", "yes")

# Reactive triggers
REACTIVE_TRIGGER_PROBABILITY: float = float(os.getenv("COORDINATOR_REACTIVE_PROBABILITY", "0.15"))
REACTIVE_COOLDOWN_SECONDS: float = 300.0

# Priority channels — these channels get conversations first each day
_priority_str = os.getenv("COORDINATOR_PRIORITY_CHANNELS", "")
PRIORITY_CHANNEL_IDS: list[int] = [
    int(cid.strip()) for cid in _priority_str.split(",") if cid.strip()
]

# Pacing — random delay between agent turns within a conversation
TURN_DELAY_MIN: float = float(os.getenv("COORDINATOR_TURN_DELAY_MIN", "15.0"))
TURN_DELAY_MAX: float = float(os.getenv("COORDINATOR_TURN_DELAY_MAX", "45.0"))
