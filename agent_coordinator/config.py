import os

from dotenv import load_dotenv

load_dotenv()

from agent_config import ACTIVE_AGENT_NAMES  # noqa: E402
from agent_config import AGENT_CHANNEL_IDS as AGENT_CHANNEL_IDS  # noqa: E402
from agent_config import CHANNEL_THEMES as CHANNEL_THEMES  # noqa: E402
from agent_config import REDIS_URL as REDIS_URL  # noqa: E402

# Agent names the coordinator manages — derived from which BOT_TOKEN_* env vars are set
AGENT_NAMES: list[str] = ACTIVE_AGENT_NAMES

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
FIRE_ON_STARTUP: bool = os.getenv("COORDINATOR_FIRE_ON_STARTUP", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Reactive triggers
REACTIVE_TRIGGER_PROBABILITY: float = float(os.getenv("COORDINATOR_REACTIVE_PROBABILITY", "0.15"))
REACTIVE_COOLDOWN_SECONDS: float = 300.0

# TTL for daily Redis keys (channel queue, schedule) — 48 hours
DAILY_KEY_TTL_SECONDS: int = 172_800

# Priority channels — these channels get conversations first each day
_priority_str = os.getenv("COORDINATOR_PRIORITY_CHANNELS", "")
PRIORITY_CHANNEL_IDS: list[int] = [
    int(cid.strip()) for cid in _priority_str.split(",") if cid.strip()
]

# Pacing — random delay between agent turns within a conversation
TURN_DELAY_MIN: float = float(os.getenv("COORDINATOR_TURN_DELAY_MIN", "15.0"))
TURN_DELAY_MAX: float = float(os.getenv("COORDINATOR_TURN_DELAY_MAX", "45.0"))
