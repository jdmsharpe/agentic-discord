"""Agent Coordinator — orchestrates AI bot conversations via Redis.

Usage:
    python -m agent_coordinator
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as aioredis

from .config import AGENT_NAMES, FIRE_ON_STARTUP, REDIS_URL
from .engine import ConversationEngine
from .scheduler import DailyScheduler

logger = logging.getLogger(__name__)


_BOT_READY_TIMEOUT = 60  # max seconds to wait for all bots


async def _wait_for_bots_ready(redis_client, timeout: float = _BOT_READY_TIMEOUT) -> None:
    """Poll Redis until every agent has set its ready key, or timeout."""
    ready_keys = [f"agent:{name}:ready" for name in AGENT_NAMES]
    deadline = asyncio.get_event_loop().time() + timeout

    while asyncio.get_event_loop().time() < deadline:
        results = await asyncio.gather(*(redis_client.exists(k) for k in ready_keys))
        missing = [AGENT_NAMES[i] for i, v in enumerate(results) if not v]
        if not missing:
            logger.info("All bots ready — proceeding with startup conversation")
            return
        await asyncio.sleep(1)

    logger.warning(
        "Timed out waiting for bots after %ds — missing: %s. Proceeding anyway.",
        timeout,
        missing,
    )


async def start_coordinator() -> None:
    """Main async entry point for the coordinator."""
    if not AGENT_NAMES:
        logger.error("No agents configured — set at least one BOT_TOKEN_* env var")
        return

    logger.info("Active agents: %s", AGENT_NAMES)

    if not REDIS_URL:
        logger.error("REDIS_URL is required for the coordinator")
        return

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

    engine = ConversationEngine(redis_client)
    scheduler = DailyScheduler(engine)

    await engine.start()
    logger.info("Coordinator connected to Redis at %s", REDIS_URL)
    await scheduler.start()

    if FIRE_ON_STARTUP:
        logger.info("FIRE_ON_STARTUP enabled — waiting for all bots to be ready")
        await _wait_for_bots_ready(redis_client)
        await scheduler._fire_conversation()

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await scheduler.stop()
        await engine.stop()
        await redis_client.aclose()
        logger.info("Coordinator shut down")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [coordinator] %(message)s",
    )
    try:
        asyncio.run(start_coordinator())
    except KeyboardInterrupt:
        logger.info("Coordinator interrupted")
