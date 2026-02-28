"""
Agent bot entrypoint — loads the correct agent cog based on AGENT_NAME.

Usage:
    AGENT_NAME=chatgpt python bot.py
    AGENT_NAME=claude  python bot.py
    AGENT_NAME=gemini  python bot.py
    AGENT_NAME=grok    python bot.py
"""

import logging
import sys

from discord import Bot, Intents

from agent_config import AGENT_NAME, BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Map agent names to their cog classes (imported lazily to avoid loading all SDKs)
_COG_MAP = {
    "chatgpt": ("agent_cogs.openai_agent", "OpenAIAgentCog"),
    "claude": ("agent_cogs.anthropic_agent", "AnthropicAgentCog"),
    "gemini": ("agent_cogs.gemini_agent", "GeminiAgentCog"),
    "grok": ("agent_cogs.grok_agent", "GrokAgentCog"),
}


def main():
    if not BOT_TOKEN:
        logger.error(
            "No BOT_TOKEN resolved for AGENT_NAME=%s — check your .env", AGENT_NAME
        )
        sys.exit(1)

    if AGENT_NAME not in _COG_MAP:
        logger.error(
            "Unknown AGENT_NAME=%s — must be one of: %s",
            AGENT_NAME,
            ", ".join(_COG_MAP),
        )
        sys.exit(1)

    module_path, class_name = _COG_MAP[AGENT_NAME]

    # Lazy import — only loads the SDK for this agent's provider
    import importlib

    module = importlib.import_module(module_path)
    CogClass = getattr(module, class_name)

    intents = Intents.default()
    intents.members = True
    intents.message_content = True
    intents.guilds = True

    bot = Bot(intents=intents)

    @bot.event
    async def on_ready():
        logger.info(
            "Agent '%s' online as %s (ID: %s)", AGENT_NAME, bot.user, bot.user.id
        )

    bot.add_cog(CogClass(bot=bot))
    logger.info("Starting agent: %s", AGENT_NAME)
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
