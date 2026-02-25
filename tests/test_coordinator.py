"""Tests for the agent coordinator."""

import asyncio
import datetime
import json
import random
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Patch coordinator config before imports
import sys
import types as stdlib_types

fake_config = stdlib_types.ModuleType("agent_coordinator.config")
fake_config.REDIS_URL = "redis://localhost:6379"
fake_config.AGENT_NAMES = ["chatgpt", "claude", "gemini", "grok"]
fake_config.AGENT_CHANNEL_IDS = [100, 200, 300]
fake_config.CHANNEL_THEMES = {100: "debate", 200: "casual", 300: "memes"}
fake_config.SCHEDULE_MIN_EVENTS = 3
fake_config.SCHEDULE_MAX_EVENTS = 5
fake_config.SCHEDULE_ACTIVE_START_HOUR = 9
fake_config.SCHEDULE_ACTIVE_END_HOUR = 23
fake_config.MAX_ROUNDS = 50
fake_config.AGENT_RESPONSE_TIMEOUT = 2.0  # Short timeout for tests
fake_config.CONTINUATION_BASE_PROBABILITY = 0.85
fake_config.CONTINUATION_DECAY = 0.03
fake_config.MIN_RESPONDENTS_TO_CONTINUE = 2
fake_config.REACTIVE_TRIGGER_PROBABILITY = 0.15
fake_config.REACTIVE_COOLDOWN_SECONDS = 300.0
fake_config.FIRE_ON_STARTUP = False
sys.modules["agent_coordinator.config"] = fake_config

from agent_coordinator.engine import ConversationEngine, ConversationState
from agent_coordinator.scheduler import DailyScheduler


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.scheduler = DailyScheduler(self.engine)

    def test_generate_todays_times_count(self):
        times = self.scheduler._generate_todays_times()
        self.assertLessEqual(len(times), 5)

    def test_generate_todays_times_sorted(self):
        times = self.scheduler._generate_todays_times()
        for i in range(len(times) - 1):
            self.assertLessEqual(times[i], times[i + 1])

    def test_generate_todays_times_within_active_hours(self):
        times = self.scheduler._generate_todays_times()
        for t in times:
            self.assertGreaterEqual(t.hour, 9)
            self.assertLess(t.hour, 23)

    def test_generate_todays_times_filters_past(self):
        times = self.scheduler._generate_todays_times()
        now = datetime.datetime.now()
        for t in times:
            self.assertGreater(t, now)


# ---------------------------------------------------------------------------
# Engine — should_continue tests
# ---------------------------------------------------------------------------


class TestShouldContinue(unittest.TestCase):
    def setUp(self):
        self.engine = ConversationEngine(MagicMock())

    def test_stops_at_max_rounds(self):
        state = ConversationState()
        state.round_number = 50
        state.text_responses_this_round = 4
        self.assertFalse(self.engine._should_continue(state))

    def test_stops_when_all_skip(self):
        state = ConversationState()
        state.round_number = 3
        state.total_skips_this_round = 3
        state.text_responses_this_round = 1
        self.assertFalse(self.engine._should_continue(state))

    def test_stops_when_not_enough_engagement(self):
        state = ConversationState()
        state.round_number = 3
        state.total_skips_this_round = 2
        state.text_responses_this_round = 1
        self.assertFalse(self.engine._should_continue(state))

    def test_continues_with_engagement(self):
        state = ConversationState()
        state.round_number = 1
        state.total_skips_this_round = 1
        state.text_responses_this_round = 3
        random.seed(42)
        result = self.engine._should_continue(state)
        self.assertTrue(result)

    def test_probability_decays_over_rounds(self):
        state = ConversationState()
        state.total_skips_this_round = 0
        state.text_responses_this_round = 4
        state.round_number = 25
        random.seed(0)
        self.assertFalse(self.engine._should_continue(state))


# ---------------------------------------------------------------------------
# Engine — send_turn tests
# ---------------------------------------------------------------------------


class TestSendTurn(unittest.TestCase):
    def setUp(self):
        self.mock_redis = MagicMock()
        self.mock_redis.publish = AsyncMock()
        self.engine = ConversationEngine(self.mock_redis)

    def test_publishes_correct_instruction_format(self):
        state = ConversationState(channel_id=100, channel_theme="debate")
        state.round_number = 1

        async def run():
            async def fake_publish(channel, data):
                instruction = json.loads(data)
                iid = instruction["instruction_id"]
                if iid in self.engine._pending_responses:
                    future = self.engine._pending_responses[iid]
                    if not future.done():
                        future.set_result({
                            "skipped": False,
                            "text": "Hello!",
                            "agent_name": "chatgpt",
                            "message_id": 999,
                        })

            self.mock_redis.publish = AsyncMock(side_effect=fake_publish)
            result = await self.engine._send_turn(state, "chatgpt")

            self.assertFalse(result["skipped"])
            self.assertEqual(result["text"], "Hello!")

            call_args = self.mock_redis.publish.call_args
            channel = call_args[0][0]
            instruction = json.loads(call_args[0][1])

            self.assertEqual(channel, "agent:chatgpt:instructions")
            self.assertEqual(instruction["protocol_version"], 1)
            self.assertEqual(instruction["action"], "decide")
            self.assertEqual(instruction["channel_id"], 100)
            self.assertFalse(instruction["is_conversation_starter"])

        asyncio.run(run())

    def test_starter_flag_passed_through(self):
        state = ConversationState(channel_id=100, channel_theme="debate")
        state.round_number = 1

        async def run():
            async def fake_publish(channel, data):
                instruction = json.loads(data)
                iid = instruction["instruction_id"]
                if iid in self.engine._pending_responses:
                    self.engine._pending_responses[iid].set_result({"skipped": False, "text": "yo"})

            self.mock_redis.publish = AsyncMock(side_effect=fake_publish)
            await self.engine._send_turn(state, "chatgpt", is_starter=True)

            call_args = self.mock_redis.publish.call_args
            instruction = json.loads(call_args[0][1])
            self.assertTrue(instruction["is_conversation_starter"])

        asyncio.run(run())

    def test_timeout_returns_skipped(self):
        state = ConversationState(channel_id=100)
        state.round_number = 1

        async def run():
            result = await self.engine._send_turn(state, "chatgpt")
            self.assertTrue(result["skipped"])
            self.assertEqual(result["reason"], "timeout")

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Engine — reactive trigger tests
# ---------------------------------------------------------------------------


class TestReactiveTrigger(unittest.TestCase):
    def setUp(self):
        self.mock_redis = MagicMock()
        self.mock_redis.publish = AsyncMock()
        self.engine = ConversationEngine(self.mock_redis)

    def test_skips_active_channel(self):
        self.engine._active_conversations[100] = ConversationState(channel_id=100)

        async def run():
            await self.engine._maybe_trigger_reactive({
                "channel_id": 100,
                "agent_name": "chatgpt",
            })
            self.mock_redis.publish.assert_not_called()

        asyncio.run(run())

    def test_skips_within_cooldown(self):
        self.engine._reactive_cooldowns[100] = time.time()

        async def run():
            await self.engine._maybe_trigger_reactive({
                "channel_id": 100,
                "agent_name": "chatgpt",
            })
            self.mock_redis.publish.assert_not_called()

        asyncio.run(run())

    def test_respects_probability(self):
        """With probability 0.15, random > 0.15 should skip."""
        random.seed(0)

        async def run():
            await self.engine._maybe_trigger_reactive({
                "channel_id": 100,
                "agent_name": "chatgpt",
            })
            self.mock_redis.publish.assert_not_called()

        asyncio.run(run())

    def test_excludes_triggering_agent(self):
        """When reactive fires, the triggering agent should not be included."""

        async def run():
            for seed in range(100):
                random.seed(seed)
                if random.random() <= 0.15:
                    random.seed(seed)
                    break
            else:
                self.skipTest("Could not find a seed that triggers reactive")

            called_agents = []

            async def mock_send(state, agent_name, is_starter=False):
                called_agents.append(agent_name)
                return {"skipped": True}

            self.engine._send_turn = mock_send

            await self.engine._maybe_trigger_reactive({
                "channel_id": 100,
                "agent_name": "chatgpt",
            })

            self.assertNotIn("chatgpt", called_agents)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Engine — full conversation flow
# ---------------------------------------------------------------------------


class TestRunRound(unittest.TestCase):
    def setUp(self):
        self.mock_redis = MagicMock()
        self.mock_redis.publish = AsyncMock()
        self.mock_redis.lpop = AsyncMock(return_value="chatgpt")
        self.mock_redis.rpush = AsyncMock()
        self.engine = ConversationEngine(self.mock_redis)

    def test_run_round_processes_all_agents(self):
        state = ConversationState(channel_id=100)
        state.round_number = 1

        agents_called = []

        async def mock_send(s, agent_name, is_starter=False):
            agents_called.append(agent_name)
            return {"skipped": False, "text": "hi", "message_id": 1}

        self.engine._send_turn = mock_send

        async def run():
            with patch("agent_coordinator.engine.asyncio.sleep", new_callable=AsyncMock):
                await self.engine._run_round(state)

            self.assertEqual(len(agents_called), 4)
            self.assertEqual(sorted(agents_called), sorted(fake_config.AGENT_NAMES))

        asyncio.run(run())

    def test_run_round_first_agent_is_starter_in_round_1(self):
        state = ConversationState(channel_id=100)
        state.round_number = 1

        starter_flags = []

        async def mock_send(s, agent_name, is_starter=False):
            starter_flags.append(is_starter)
            return {"skipped": False, "text": "hi", "message_id": 1}

        self.engine._send_turn = mock_send

        async def run():
            with patch("agent_coordinator.engine.asyncio.sleep", new_callable=AsyncMock):
                await self.engine._run_round(state)

            self.assertTrue(starter_flags[0])
            self.assertFalse(any(starter_flags[1:]))

        asyncio.run(run())

    def test_run_round_no_starter_in_round_2(self):
        state = ConversationState(channel_id=100)
        state.round_number = 2

        starter_flags = []

        async def mock_send(s, agent_name, is_starter=False):
            starter_flags.append(is_starter)
            return {"skipped": False, "text": "hi", "message_id": 1}

        self.engine._send_turn = mock_send

        async def run():
            with patch("agent_coordinator.engine.asyncio.sleep", new_callable=AsyncMock):
                await self.engine._run_round(state)

            self.assertFalse(any(starter_flags))

        asyncio.run(run())

    def test_run_round_counts_responses(self):
        state = ConversationState(channel_id=100)
        state.round_number = 1

        call_count = 0

        async def mock_send(s, agent_name, is_starter=False):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"skipped": False, "text": "response", "message_id": call_count}
            return {"skipped": True}

        self.engine._send_turn = mock_send

        async def run():
            with patch("agent_coordinator.engine.asyncio.sleep", new_callable=AsyncMock):
                await self.engine._run_round(state)

            self.assertEqual(state.text_responses_this_round, 2)
            self.assertEqual(state.total_skips_this_round, 2)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
