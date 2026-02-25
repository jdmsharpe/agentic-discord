"""Daily scheduler — generates random conversation times and fires the engine."""

from __future__ import annotations

import asyncio
import datetime
import logging
import random

from .config import (
    AGENT_CHANNEL_IDS,
    CHANNEL_THEMES,
    SCHEDULE_ACTIVE_END_HOUR,
    SCHEDULE_ACTIVE_START_HOUR,
    SCHEDULE_MAX_EVENTS,
    SCHEDULE_MIN_EVENTS,
)
from .engine import ConversationEngine

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Generates random conversation times and sleeps between them."""

    def __init__(self, engine: ConversationEngine):
        self._engine = engine
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_forever(self) -> None:
        """Outer loop: generate today's schedule, execute it, repeat tomorrow."""
        while True:
            try:
                times = self._generate_todays_times()
                logger.info(
                    "Today's schedule: %d conversations at %s",
                    len(times),
                    [t.strftime("%H:%M") for t in times],
                )

                for scheduled_time in times:
                    await self._sleep_until(scheduled_time)
                    await self._fire_conversation()

                await self._sleep_until_midnight()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler error, retrying in 60s")
                await asyncio.sleep(60)

    def _generate_todays_times(self) -> list[datetime.datetime]:
        """Generate N random times spread across active hours via slot-based spread."""
        now = datetime.datetime.now()
        count = random.randint(SCHEDULE_MIN_EVENTS, SCHEDULE_MAX_EVENTS)

        start_minutes = SCHEDULE_ACTIVE_START_HOUR * 60
        end_minutes = SCHEDULE_ACTIVE_END_HOUR * 60
        total_window = end_minutes - start_minutes

        if total_window <= 0 or count <= 0:
            return []

        slot_size = total_window // count
        times = []
        for i in range(count):
            slot_start = start_minutes + (i * slot_size)
            slot_end = min(slot_start + slot_size - 1, end_minutes - 1)
            random_minute = random.randint(slot_start, slot_end)

            t = now.replace(
                hour=random_minute // 60,
                minute=random_minute % 60,
                second=random.randint(0, 59),
                microsecond=0,
            )
            times.append(t)

        # Only keep times that haven't passed yet
        times = [t for t in times if t > now]
        times.sort()
        return times

    async def _sleep_until(self, target: datetime.datetime) -> None:
        now = datetime.datetime.now()
        delta = (target - now).total_seconds()
        if delta > 0:
            logger.debug("Sleeping %.0fs until %s", delta, target.strftime("%H:%M:%S"))
            await asyncio.sleep(delta)

    async def _sleep_until_midnight(self) -> None:
        now = datetime.datetime.now()
        tomorrow = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        delta = (tomorrow - now).total_seconds() + random.uniform(1, 60)
        logger.info("Day complete — sleeping %.0fs until tomorrow", delta)
        await asyncio.sleep(delta)

    async def _fire_conversation(self) -> None:
        """Pick a random channel and start a conversation."""
        if not AGENT_CHANNEL_IDS:
            logger.warning("No AGENT_CHANNEL_IDS configured — skipping")
            return

        channel_id = random.choice(AGENT_CHANNEL_IDS)
        theme = CHANNEL_THEMES.get(channel_id, "casual")

        logger.info("Firing conversation in channel %s [%s]", channel_id, theme)
        try:
            await self._engine.run_conversation(channel_id, theme)
        except Exception:
            logger.exception("Scheduled conversation failed")
