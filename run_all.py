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


MAX_RETRIES = 10
RETRY_BACKOFF_BASE = 5  # seconds; doubles each attempt up to ~2560s (~42min)


async def start_agent(name: str, token: str, module_path: str, class_name: str):
    if not token:
        _log(name, "Skipping — no BOT_TOKEN set")
        return

    module = importlib.import_module(module_path)
    CogClass = getattr(module, class_name)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
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
            return  # clean exit
        except asyncio.CancelledError:
            raise
        except Exception:
            delay = min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), 2560)
            logger.exception(
                "[%s] Failed to start (attempt %d/%d), retrying in %ds",
                name,
                attempt,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

    logger.error("[%s] Giving up after %d attempts", name, MAX_RETRIES)


async def main():
    from agent_coordinator import start_coordinator

    tasks = [asyncio.create_task(start_agent(*agent)) for agent in AGENTS]
    tasks.append(asyncio.create_task(start_coordinator()))

    # return_exceptions=True prevents one bot crash from killing the coordinator
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Task %d failed: %s", i, result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down all agents")
