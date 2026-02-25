"""Agent Coordinator — orchestrates AI bot conversations via Redis.

Usage:
    python -m agent_coordinator
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as aioredis

from .config import FIRE_ON_STARTUP, REDIS_URL
from .engine import ConversationEngine
from .scheduler import DailyScheduler

logger = logging.getLogger(__name__)


async def start_coordinator() -> None:
    """Main async entry point for the coordinator."""
    if not REDIS_URL:
        logger.error("REDIS_URL is required for the coordinator")
        return

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Coordinator connected to Redis at %s", REDIS_URL)

    engine = ConversationEngine(redis_client)
    scheduler = DailyScheduler(engine)

    await engine.start()
    await scheduler.start()

    if FIRE_ON_STARTUP:
        logger.info("FIRE_ON_STARTUP enabled — triggering immediate conversation")
        # Delay to let bots connect and start Redis listeners
        await asyncio.sleep(10)
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
