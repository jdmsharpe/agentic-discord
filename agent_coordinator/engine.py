"""Conversation engine — manages turn-taking between agents via Redis pub/sub."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import TypedDict

from .config import (
    AGENT_CHANNEL_IDS,
    AGENT_NAMES,
    AGENT_RESPONSE_TIMEOUT,
    CHANNEL_THEMES,
    CONSECUTIVE_TIMEOUT_THRESHOLD,
    CONTINUATION_BASE_PROBABILITY,
    CONTINUATION_DECAY,
    DAILY_KEY_TTL_SECONDS,
    MAX_ROUNDS,
    MIN_RESPONDENTS_TO_CONTINUE,
    PRIORITY_CHANNEL_IDS,
    REACTIVE_COOLDOWN_SECONDS,
    REACTIVE_TRIGGER_PROBABILITY,
    TURN_DELAY_MAX,
    TURN_DELAY_MIN,
)

logger = logging.getLogger(__name__)


class HistoryEntry(TypedDict, total=False):
    """A single entry in the conversation history."""
    agent: str
    text: str
    message_id: int | None


class AgentResult(TypedDict, total=False):
    """Result dict returned by an agent via Redis."""
    protocol_version: int
    instruction_id: str
    agent_name: str
    skipped: bool
    text: str | None
    image_url: str | None
    image_prompt: str | None
    emoji_reacted: str | None
    react_to_message_id: int | None
    end_conversation: bool
    topic: str | None
    reason: str
    error: str


@dataclass
class ConversationState:
    """Tracks all state for a single conversation."""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel_id: int = 0
    channel_theme: str = "casual"
    topic: str = ""
    round_number: int = 0
    conversation_history: list[HistoryEntry] = field(default_factory=list)
    text_responses_this_round: int = 0
    total_skips_this_round: int = 0
    consecutive_end_requests: int = 0
    ended_naturally: bool = False


class ConversationEngine:
    """Manages turn-taking conversations between agents via Redis pub/sub."""

    def __init__(self, redis_client):
        self._redis = redis_client
        self._active_conversations: dict[int, ConversationState] = {}
        self._pending_responses: dict[str, asyncio.Future] = {}
        self._result_listener_task: asyncio.Task | None = None
        self._reactive_cooldowns: dict[int, float] = {}
        self._consecutive_timeouts: int = 0

    _LISTENER_MAX_BACKOFF = 30  # seconds

    def get_redis(self):
        """Public accessor for the Redis client — use instead of accessing _redis directly."""
        return self._redis

    async def start(self) -> None:
        await self._wait_for_redis()
        self._result_listener_task = asyncio.create_task(self._listen_for_results())

    async def stop(self) -> None:
        if self._result_listener_task and not self._result_listener_task.done():
            self._result_listener_task.cancel()
            try:
                await self._result_listener_task
            except asyncio.CancelledError:
                pass

    async def _wait_for_redis(self) -> None:
        """Block until Redis is reachable. Retries with backoff up to ~30s."""
        delay = 1
        while True:
            try:
                await self._redis.ping()
                logger.info("Redis health check passed")
                return
            except Exception:
                logger.warning("Redis not ready, retrying in %ds...", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._LISTENER_MAX_BACKOFF)

    # ------------------------------------------------------------------
    # Redis result listener
    # ------------------------------------------------------------------

    async def _listen_for_results(self) -> None:
        """Subscribe to all agent result channels and route responses.

        Automatically retries on connection failures with exponential backoff,
        making it resilient to Redis restarts and boot-time race conditions.
        """
        channels = [f"agent:{name}:results" for name in AGENT_NAMES]
        delay = 1

        while True:
            pubsub = self._redis.pubsub()
            try:
                await pubsub.subscribe(*channels)
                logger.info("Listening on result channels: %s", channels)
                delay = 1  # reset backoff on successful subscribe

                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    try:
                        data = json.loads(message["data"])
                        instruction_id = data.get("instruction_id")

                        if instruction_id and instruction_id in self._pending_responses:
                            future = self._pending_responses.pop(instruction_id)
                            if not future.done():
                                future.set_result(data)

                        if data.get("event") == "human_mention_response":
                            asyncio.create_task(self._maybe_trigger_reactive(data))
                    except Exception:
                        logger.exception("Error processing agent result")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Result listener disconnected, reconnecting in %ds...", delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._LISTENER_MAX_BACKOFF)
            finally:
                await pubsub.aclose()

    # ------------------------------------------------------------------
    # Starter rotation (Redis-backed, survives restarts)
    # ------------------------------------------------------------------

    async def _pop_starter(self, channel_id: int) -> str:
        """Return the next starter for this channel, cycling through all agents fairly.

        The queue is stored in Redis so the rotation persists across restarts.
        When the queue is empty a new shuffled cycle is pushed.
        """
        key = f"coordinator:starter_queue:{channel_id}"
        starter = await self._redis.lpop(key)
        if not starter:
            new_cycle = random.sample(AGENT_NAMES, len(AGENT_NAMES))
            await self._redis.rpush(key, *new_cycle)
            starter = await self._redis.lpop(key)
        return starter

    async def pop_channel_for_today(self) -> int:
        """Return the next channel to use, ensuring each channel fires once before any repeats.

        Uses a date-keyed Redis list so the queue resets automatically each day.
        Once all channels have had their first conversation, falls back to random choice.
        """
        today = datetime.date.today().isoformat()
        key = f"coordinator:channel_queue:{today}"

        # Warn about misconfigured priority channels (once per queue creation)
        invalid = [c for c in PRIORITY_CHANNEL_IDS if c not in AGENT_CHANNEL_IDS]
        if invalid:
            logger.warning(
                "Priority channels not in CHANNEL_THEME_MAP (ignored): %s", invalid
            )

        # Seed the queue on first access today — priority channels first, then the rest
        if not await self._redis.exists(key):
            valid_priority = [c for c in PRIORITY_CHANNEL_IDS if c in AGENT_CHANNEL_IDS]
            non_priority = [c for c in AGENT_CHANNEL_IDS if c not in valid_priority]
            ordered = random.sample(valid_priority, len(valid_priority)) + \
                      random.sample(non_priority, len(non_priority))
            await self._redis.rpush(key, *[str(c) for c in ordered])
            await self._redis.expire(key, DAILY_KEY_TTL_SECONDS)
            logger.info("Daily channel queue created for %s: %s", today, ordered)

        channel_id = await self._redis.lpop(key)
        if channel_id:
            return int(channel_id)

        # All channels have had a conversation today — pick freely
        choice = random.choice(AGENT_CHANNEL_IDS)
        logger.debug("All channels exhausted for today — random fallback: %s", choice)
        return choice

    # ------------------------------------------------------------------
    # Scheduled conversations
    # ------------------------------------------------------------------

    async def run_conversation(self, channel_id: int, channel_theme: str) -> None:
        """Run a full scheduled conversation from start to finish."""
        if channel_id in self._active_conversations:
            logger.info("Conversation already active in channel %s, skipping", channel_id)
            return

        state = ConversationState(
            channel_id=channel_id,
            channel_theme=channel_theme,
        )
        self._active_conversations[channel_id] = state

        logger.info(
            "Conversation %s started in channel %s [%s]",
            state.conversation_id,
            channel_id,
            channel_theme,
        )

        try:
            while state.round_number < MAX_ROUNDS:
                state.round_number += 1
                await self._run_round(state)

                if not self._should_continue(state):
                    break
        finally:
            del self._active_conversations[channel_id]
            logger.info(
                "Conversation %s ended — %d rounds, %d messages",
                state.conversation_id,
                state.round_number,
                len(state.conversation_history),
            )

    async def _run_round(self, state: ConversationState) -> None:
        """Execute one round: iterate through all agents in shuffled order."""
        agents = list(AGENT_NAMES)
        random.shuffle(agents)

        # Prevent back-to-back messages from the same bot across round boundaries.
        # Move the last visible speaker to the end so others get priority.
        if state.conversation_history:
            last_speaker = state.conversation_history[-1].get("agent")
            if last_speaker in agents:
                agents.remove(last_speaker)
                agents.append(last_speaker)

        state.text_responses_this_round = 0
        state.total_skips_this_round = 0

        is_first_round = state.round_number == 1

        if is_first_round:
            starter = await self._pop_starter(state.channel_id)
            agents = [starter] + [a for a in agents if a != starter]
            logger.info("Conversation starter for channel %s: %s", state.channel_id, starter)

        for i, agent_name in enumerate(agents):
            # First agent in the first round starts the conversation
            is_starter = is_first_round and i == 0
            result = await self._send_turn(state, agent_name, is_starter=is_starter)

            # Capture topic from the starter's response
            if is_starter and result.get("topic"):
                state.topic = result["topic"]
                logger.info("Conversation topic set: %s", state.topic)

            if result.get("skipped", True):
                state.total_skips_this_round += 1
            else:
                if result.get("text"):
                    state.text_responses_this_round += 1
                    state.conversation_history.append({
                        "agent": agent_name,
                        "text": result["text"],
                        "message_id": result.get("message_id"),
                    })
                if result.get("image_url"):
                    state.text_responses_this_round += 1
                    prompt = result.get("image_prompt", "image")
                    state.conversation_history.append({
                        "agent": agent_name,
                        "text": f'[posted image: "{prompt}" → {result["image_url"]}]',
                        "message_id": result.get("message_id"),
                    })
                if result.get("emoji_reacted"):
                    state.conversation_history.append({
                        "agent": agent_name,
                        "text": f'[reacted {result["emoji_reacted"]} to msg:{result.get("react_to_message_id", "?")}]',
                        "message_id": None,
                    })

                # Track consecutive end_conversation requests (skips don't affect counter)
                if result.get("end_conversation"):
                    state.consecutive_end_requests += 1
                    if state.consecutive_end_requests >= 2:
                        state.ended_naturally = True
                        logger.info(
                            "Conversation %s ending naturally — 2 consecutive end requests in round %d",
                            state.conversation_id,
                            state.round_number,
                        )
                        break
                else:
                    state.consecutive_end_requests = 0

            # Natural pacing between turns
            await asyncio.sleep(random.uniform(TURN_DELAY_MIN, TURN_DELAY_MAX))

    async def _send_turn(
        self, state: ConversationState, agent_name: str, is_starter: bool = False
    ) -> dict:
        """Send an instruction to one agent and await the result."""
        instruction_id = str(uuid.uuid4())
        instruction = {
            "protocol_version": 1,
            "instruction_id": instruction_id,
            "action": "decide",
            "channel_id": state.channel_id,
            "channel_theme": state.channel_theme,
            "topic": state.topic,
            "round_number": state.round_number,
            "conversation_id": state.conversation_id,
            "conversation_history": state.conversation_history[-MAX_ROUNDS * len(AGENT_NAMES):],
            "is_conversation_starter": is_starter,
        }

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_responses[instruction_id] = future

        try:
            await self._redis.publish(
                f"agent:{agent_name}:instructions",
                json.dumps(instruction),
            )
            logger.debug("Instruction %s sent to %s, awaiting result (timeout=%ss)", instruction_id, agent_name, AGENT_RESPONSE_TIMEOUT)
            result = await asyncio.wait_for(future, timeout=AGENT_RESPONSE_TIMEOUT)
            logger.info(
                "Agent %s responded: skipped=%s text=%s image=%s emoji=%s end_convo=%s",
                agent_name,
                result.get("skipped"),
                bool(result.get("text")),
                bool(result.get("image_url")),
                result.get("emoji_reacted"),
                result.get("end_conversation", False),
            )
            self._consecutive_timeouts = 0
            return result
        except asyncio.TimeoutError:
            self._consecutive_timeouts += 1
            logger.warning(
                "Agent %s timed out for instruction %s (consecutive: %d/%d)",
                agent_name, instruction_id,
                self._consecutive_timeouts, CONSECUTIVE_TIMEOUT_THRESHOLD,
            )
            self._pending_responses.pop(instruction_id, None)
            if self._consecutive_timeouts >= CONSECUTIVE_TIMEOUT_THRESHOLD:
                logger.critical(
                    "Hit %d consecutive timeouts — exiting for systemd restart",
                    self._consecutive_timeouts,
                )
                asyncio.get_running_loop().call_soon(
                    lambda: sys.exit(1)
                )
            return {"skipped": True, "reason": "timeout", "agent_name": agent_name}
        except Exception:
            logger.exception("Error sending turn to %s", agent_name)
            self._pending_responses.pop(instruction_id, None)
            return {"skipped": True, "reason": "error", "agent_name": agent_name}

    def _should_continue(self, state: ConversationState) -> bool:
        """Decide whether the conversation should continue to the next round."""
        if state.ended_naturally:
            return False

        if state.round_number >= MAX_ROUNDS:
            return False

        # Most agents skipped → natural disengagement
        if state.total_skips_this_round >= len(AGENT_NAMES) - 1:
            return False

        # Need minimum engagement to keep going
        if state.text_responses_this_round < MIN_RESPONDENTS_TO_CONTINUE:
            return False

        # Probabilistic continuation with decay per round
        probability = CONTINUATION_BASE_PROBABILITY - (CONTINUATION_DECAY * state.round_number)
        probability = max(probability, 0.1)
        return random.random() < probability

    # ------------------------------------------------------------------
    # Reactive triggers
    # ------------------------------------------------------------------

    async def _maybe_trigger_reactive(self, event_data: dict) -> None:
        """Potentially trigger a reactive round when a human @mentions a bot."""
        channel_id = event_data.get("channel_id")
        if not channel_id:
            return

        if channel_id in self._active_conversations:
            logger.debug("Reactive skipped — conversation already active in channel %s", channel_id)
            return

        now = time.time()
        cooldown_remaining = REACTIVE_COOLDOWN_SECONDS - (now - self._reactive_cooldowns.get(channel_id, 0))
        if cooldown_remaining > 0:
            logger.debug("Reactive skipped — cooldown %.0fs remaining in channel %s", cooldown_remaining, channel_id)
            return

        roll = random.random()
        if roll > REACTIVE_TRIGGER_PROBABILITY:
            logger.debug("Reactive skipped — probability check failed (%.2f > %.2f)", roll, REACTIVE_TRIGGER_PROBABILITY)
            return

        self._reactive_cooldowns[channel_id] = now
        triggering_agent = event_data.get("agent_name", "")
        other_agents = [a for a in AGENT_NAMES if a != triggering_agent]
        reactive_agents = random.sample(
            other_agents, k=min(random.randint(1, 2), len(other_agents))
        )

        logger.info(
            "Reactive trigger in channel %s by %s → pinging %s",
            channel_id,
            triggering_agent,
            reactive_agents,
        )

        theme = CHANNEL_THEMES.get(channel_id, "casual")
        state = ConversationState(channel_id=channel_id, channel_theme=theme)
        state.round_number = 1
        self._active_conversations[channel_id] = state

        try:
            for agent_name in reactive_agents:
                result = await self._send_turn(state, agent_name)
                if result.get("text"):
                    state.conversation_history.append({
                        "agent": agent_name,
                        "text": result["text"],
                        "message_id": result.get("message_id"),
                    })
                await asyncio.sleep(random.uniform(3.0, 8.0))
        finally:
            del self._active_conversations[channel_id]
