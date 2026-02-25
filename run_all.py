"""Launch all 4 agent bots + coordinator in parallel. Ctrl+C stops all."""

import asyncio
import logging
import importlib
import signal
import sys

from discord import Bot, Intents

from agent_config import (
    BOT_TOKEN_CHATGPT,
    BOT_TOKEN_CLAUDE,
    BOT_TOKEN_GEMINI,
    BOT_TOKEN_GROK,
)

AGENTS = [
    ("chatgpt", BOT_TOKEN_CHATGPT, "agent_cogs.openai_agent", "OpenAIAgentCog"),
    ("claude", BOT_TOKEN_CLAUDE, "agent_cogs.anthropic_agent", "AnthropicAgentCog"),
    ("gemini", BOT_TOKEN_GEMINI, "agent_cogs.gemini_agent", "GeminiAgentCog"),
    ("grok", BOT_TOKEN_GROK, "agent_cogs.grok_agent", "GrokAgentCog"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _log(name: str, msg: str, *args):
    logger.info("[%s] " + msg, name, *args)


async def start_agent(name: str, token: str, module_path: str, class_name: str):
    if not token:
        _log(name, "Skipping — no BOT_TOKEN set")
        return

    module = importlib.import_module(module_path)
    CogClass = getattr(module, class_name)

    intents = Intents.default()
    intents.members = True
    intents.message_content = True
    intents.guilds = True

    bot = Bot(intents=intents)

    @bot.event
    async def on_ready(n=name):
        _log(n, "Online as %s (ID: %s)", bot.user, bot.user.id)

    bot.add_cog(CogClass(bot=bot))
    _log(name, "Starting...")
    await bot.start(token)


async def main():
    from agent_coordinator import start_coordinator

    tasks = [
        asyncio.create_task(start_agent(*agent))
        for agent in AGENTS
    ]
    tasks.append(asyncio.create_task(start_coordinator()))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down all agents")
