"""Tests for the agent cog base logic.

Uses a concrete mock subclass of BaseAgentCog to test shared behavior:
- Decision JSON parsing (valid, malformed, skip, unknown fields)
- Rate limiting (cooldown, daily cap)
- @mention detection
- Action execution (text, image, emoji)
- Coordinator result publishing
"""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Patch agent_config before importing base
import sys
import types as stdlib_types

# Create a fake agent_config module with test values
fake_config = stdlib_types.ModuleType("agent_config")
fake_config.AGENT_NAME = "test_bot"
fake_config.AGENT_PERSONALITY = "You are a test bot."
fake_config.AGENT_PERSONALITY_MAP = {
    "chatgpt": "Default GPT personality.",
}
fake_config.AGENT_CHANNEL_IDS = [100, 200]
fake_config.BOT_IDS = [900, 901, 902]
fake_config.AGENT_MAX_DAILY = 5
fake_config.AGENT_COOLDOWN_SECONDS = 60
fake_config.CONTEXT_WINDOW_SIZE = 50
fake_config.CHANNEL_THEMES = {100: "debate", 200: "memes"}
fake_config.BOTS_ROLE_ID = 55555
fake_config.REDIS_URL = ""
fake_config.SHOW_COST_EMBEDS = True


def _fake_get_context_window(theme=None):
    if theme:
        scales = {"debate": 1.0, "memes": 0.35}
        scale = scales.get(theme, 1.0)
        return round(fake_config.CONTEXT_WINDOW_SIZE * scale) or 1
    return fake_config.CONTEXT_WINDOW_SIZE


fake_config.get_context_window = _fake_get_context_window
sys.modules["agent_config"] = fake_config

from agent_cogs.base import (
    AIResponse,
    BaseAgentCog,
    _compute_token_cost,
    _parse_decision,
    _format_conversation_history,
    format_api_error,
)


async def _empty_async_iter():
    """Async iterator that yields nothing — used to mock channel.history()."""
    return
    yield  # noqa: makes this an async generator


class MockAgentCog(BaseAgentCog):
    """Concrete subclass for testing — overrides abstract methods with mocks."""

    agent_redis_name = "testbot"

    def __init__(self):
        # Don't call super().__init__ to avoid needing a real bot
        self.bot = MagicMock()
        self.bot.user = MagicMock()
        self.bot.user.id = 12345
        self.bot.user.__eq__ = lambda self, other: getattr(other, "id", None) == 12345
        self._redis = None
        self._listener_task = None
        self._http_session = None
        self.agent_display_name = "TestBot"
        self.other_agent_names = ["Clod Bot", "Google Bot", "Grok Bot"]
        self._last_response_time = {}
        self._daily_count = 0
        self._daily_reset_date = ""

        # Mock AI methods
        self.mock_ai_response = '{"skip": false, "text": "Hello!", "generate_image": false, "image_prompt": null, "react_emoji": null}'
        self.mock_image_bytes = b"fake_png_bytes"

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> AIResponse:
        return AIResponse(text=self.mock_ai_response, input_tokens=100, output_tokens=50)

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        return self.mock_image_bytes


# ---------------------------------------------------------------------------
# Decision parsing tests
# ---------------------------------------------------------------------------


class TestParseDecision(unittest.TestCase):
    def test_valid_json(self):
        raw = '{"skip": false, "text": "Hi there", "generate_image": false}'
        result = _parse_decision(raw)
        self.assertFalse(result["skip"])
        self.assertEqual(result["text"], "Hi there")

    def test_skip_true(self):
        raw = '{"skip": true}'
        result = _parse_decision(raw)
        self.assertTrue(result["skip"])

    def test_malformed_json_defaults_to_skip(self):
        raw = "not valid json at all"
        result = _parse_decision(raw)
        self.assertTrue(result["skip"])

    def test_markdown_fenced_json(self):
        raw = '```json\n{"skip": false, "text": "fenced"}\n```'
        result = _parse_decision(raw)
        self.assertFalse(result["skip"])
        self.assertEqual(result["text"], "fenced")

    def test_unknown_fields_ignored(self):
        raw = (
            '{"skip": false, "text": "hi", "generate_video": true, "tts_text": "hello"}'
        )
        result = _parse_decision(raw)
        self.assertFalse(result["skip"])
        self.assertEqual(result["text"], "hi")
        # Unknown fields are present in the dict but harmless
        self.assertTrue(result.get("generate_video"))

    def test_empty_string(self):
        result = _parse_decision("")
        self.assertTrue(result["skip"])

    def test_non_dict_json(self):
        result = _parse_decision("[1, 2, 3]")
        self.assertTrue(result["skip"])

    def test_null_text_field(self):
        raw = '{"skip": false, "text": null, "react_emoji": "😂"}'
        result = _parse_decision(raw)
        self.assertIsNone(result["text"])
        self.assertEqual(result["react_emoji"], "😂")


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting(unittest.TestCase):
    def setUp(self):
        self.cog = MockAgentCog()

    def test_first_request_allowed(self):
        self.assertTrue(self.cog._check_rate_limits(100))

    def test_cooldown_blocks_second_request(self):
        self.cog._record_response(100)
        self.assertFalse(self.cog._check_rate_limits(100))

    def test_different_channel_not_affected(self):
        self.cog._record_response(100)
        self.assertTrue(self.cog._check_rate_limits(200))

    def test_daily_cap_enforced(self):
        self.cog._daily_reset_date = time.strftime("%Y-%m-%d")
        self.cog._daily_count = 5
        self.assertFalse(self.cog._check_rate_limits(100))

    def test_daily_cap_resets_on_new_day(self):
        self.cog._daily_count = 5
        self.cog._daily_reset_date = "2020-01-01"  # Force stale date
        self.assertTrue(self.cog._check_rate_limits(100))
        self.assertEqual(self.cog._daily_count, 0)

    def test_cooldown_expires(self):
        self.cog._record_response(100)
        # Manually backdate the last response time
        self.cog._last_response_time[100] = time.time() - 120
        self.assertTrue(self.cog._check_rate_limits(100))


# ---------------------------------------------------------------------------
# @mention detection tests
# ---------------------------------------------------------------------------


class TestMentionDetection(unittest.TestCase):
    def setUp(self):
        self.cog = MockAgentCog()

    def _make_message(
        self, author_bot=False, mentions_bot=False, channel_id=100, role_mention_id=None
    ):
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = author_bot
        msg.author.id = 99999
        msg.author.__eq__ = lambda self, other: getattr(other, "id", None) == 99999
        msg.content = "Hey @TestBot what do you think?"
        msg.channel = MagicMock()
        msg.channel.id = channel_id
        msg.channel.name = "ai-general"
        msg.channel.history = AsyncMock(return_value=AsyncMock())
        msg.id = 54321

        if mentions_bot:
            msg.mentions = [self.cog.bot.user]
        else:
            msg.mentions = []

        if role_mention_id is not None:
            role = MagicMock()
            role.id = role_mention_id
            msg.role_mentions = [role]
        else:
            msg.role_mentions = []

        return msg

    def test_ignores_own_message(self):
        msg = self._make_message()
        msg.author = self.cog.bot.user
        # on_message should return early
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()
        # No crash = passed (the method returns early)

    def test_ignores_bot_messages(self):
        msg = self._make_message(author_bot=True, mentions_bot=True)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()

    def test_ignores_non_agent_channel(self):
        msg = self._make_message(channel_id=999, mentions_bot=True)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()

    def test_ignores_no_mention(self):
        msg = self._make_message(mentions_bot=False)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()

    def test_role_mention_triggers_response(self):
        """@bots role mention should trigger _decide_and_act."""
        msg = self._make_message(mentions_bot=False, role_mention_id=55555)
        self.cog._check_rate_limits = MagicMock(return_value=True)
        self.cog._decide_and_act = AsyncMock(
            return_value={"skipped": False, "message_id": 111}
        )
        # Provide async iterator for channel history
        msg.channel.history = MagicMock(return_value=_empty_async_iter())
        msg.guild = MagicMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()

        self.cog._decide_and_act.assert_called_once()
        # force_respond should be True
        _, kwargs = self.cog._decide_and_act.call_args
        self.assertTrue(kwargs.get("force_respond"))

    def test_role_mention_wrong_id_ignored(self):
        """A role mention with a different ID than BOTS_ROLE_ID is ignored."""
        msg = self._make_message(mentions_bot=False, role_mention_id=99999)
        self.cog._decide_and_act = AsyncMock()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()
        self.cog._decide_and_act.assert_not_called()

    def test_role_mention_skips_coordinator_notification(self):
        """Role mentions should NOT publish human_mention_response to coordinator."""
        msg = self._make_message(mentions_bot=False, role_mention_id=55555)
        self.cog._check_rate_limits = MagicMock(return_value=True)
        self.cog._decide_and_act = AsyncMock(
            return_value={"skipped": False, "message_id": 111}
        )
        self.cog._redis = AsyncMock()
        msg.channel.history = MagicMock(return_value=_empty_async_iter())
        msg.guild = MagicMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()

        self.cog._redis.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Action execution tests
# ---------------------------------------------------------------------------


class TestActionExecution(unittest.TestCase):
    def setUp(self):
        self.cog = MockAgentCog()

    def test_send_text(self):
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 111
        channel.send = AsyncMock(return_value=sent_msg)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.cog._send_text(channel, "hello"))
        loop.close()

        channel.send.assert_called_once_with("hello")
        self.assertEqual(result.id, 111)

    def test_send_text_truncates_long_messages(self):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=MagicMock(id=111))

        long_text = "x" * 3000
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog._send_text(channel, long_text))
        loop.close()

        sent_text = channel.send.call_args[0][0]
        self.assertLessEqual(len(sent_text), 2000)
        self.assertTrue(sent_text.endswith("..."))

    def test_generate_and_send_image(self):
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 222
        channel.send = AsyncMock(return_value=sent_msg)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._generate_and_send_image(channel, "a cool image")
        )
        loop.close()

        self.assertEqual(result.id, 222)
        channel.send.assert_called_once()

    def test_add_reaction(self):
        channel = MagicMock()
        message = MagicMock()
        message.add_reaction = AsyncMock()
        channel.fetch_message = AsyncMock(return_value=message)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.cog._add_reaction(channel, 12345, "😂"))
        loop.close()

        self.assertTrue(result)
        message.add_reaction.assert_called_once_with("😂")


# ---------------------------------------------------------------------------
# Decision + act integration tests
# ---------------------------------------------------------------------------


class TestDecideAndAct(unittest.TestCase):
    def setUp(self):
        self.cog = MockAgentCog()

    def test_text_response(self):
        self.cog.mock_ai_response = '{"skip": false, "text": "Hello world!"}'
        channel = MagicMock()
        sent = MagicMock(id=333)
        channel.send = AsyncMock(return_value=sent)
        channel.id = 100
        channel.name = "ai-general"

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(channel, "context", "topic", "ai-general")
        )
        loop.close()

        self.assertFalse(result["skipped"])
        self.assertEqual(result["text"], "Hello world!")
        self.assertEqual(result["message_id"], 333)

    def test_skip_decision(self):
        self.cog.mock_ai_response = '{"skip": true}'
        channel = MagicMock()
        channel.id = 100
        channel.name = "ai-general"

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(channel, "context", "topic", "ai-general")
        )
        loop.close()

        self.assertTrue(result["skipped"])

    def test_force_respond_overrides_skip(self):
        self.cog.mock_ai_response = '{"skip": true, "text": "Forced response"}'
        channel = MagicMock()
        sent = MagicMock(id=444)
        channel.send = AsyncMock(return_value=sent)
        channel.id = 100
        channel.name = "ai-general"

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(
                channel, "context", "topic", "ai-general", force_respond=True
            )
        )
        loop.close()

        # force_respond=True means skip is ignored, text should be sent
        self.assertFalse(result["skipped"])
        self.assertEqual(result["text"], "Forced response")

    def test_emoji_reaction(self):
        self.cog.mock_ai_response = '{"skip": false, "text": null, "react_emoji": "🔥"}'
        channel = MagicMock()
        channel.id = 100
        channel.name = "ai-general"
        target_msg = MagicMock()
        target_msg.add_reaction = AsyncMock()
        channel.fetch_message = AsyncMock(return_value=target_msg)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(
                channel, "context", "", "ai-general", react_to_message_id=555
            )
        )
        loop.close()

        self.assertEqual(result["emoji_reacted"], "🔥")

    def test_image_generation(self):
        self.cog.mock_ai_response = '{"skip": false, "text": null, "generate_image": true, "image_prompt": "a cat"}'
        channel = MagicMock()
        channel.id = 100
        channel.name = "ai-memes"
        sent = MagicMock(id=666)
        channel.send = AsyncMock(return_value=sent)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(channel, "context", "", "ai-memes")
        )
        loop.close()

        self.assertTrue(result.get("image_sent"))

    def test_end_conversation_passed_through(self):
        self.cog.mock_ai_response = (
            '{"skip": false, "text": "Good talk!", "end_conversation": true}'
        )
        channel = MagicMock()
        sent = MagicMock(id=333)
        channel.send = AsyncMock(return_value=sent)
        channel.id = 100
        channel.name = "ai-general"

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(channel, "context", "topic", "ai-general")
        )
        loop.close()

        self.assertTrue(result.get("end_conversation"))
        self.assertFalse(result["skipped"])

    def test_no_end_conversation_by_default(self):
        self.cog.mock_ai_response = '{"skip": false, "text": "Hello!"}'
        channel = MagicMock()
        sent = MagicMock(id=333)
        channel.send = AsyncMock(return_value=sent)
        channel.id = 100
        channel.name = "ai-general"

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(channel, "context", "topic", "ai-general")
        )
        loop.close()

        self.assertNotIn("end_conversation", result)

    def test_combo_text_and_emoji(self):
        self.cog.mock_ai_response = (
            '{"skip": false, "text": "Nice!", "react_emoji": "👍"}'
        )
        channel = MagicMock()
        channel.id = 100
        channel.name = "ai-general"
        sent = MagicMock(id=777)
        channel.send = AsyncMock(return_value=sent)
        target_msg = MagicMock()
        target_msg.add_reaction = AsyncMock()
        channel.fetch_message = AsyncMock(return_value=target_msg)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(
                channel, "context", "", "ai-general", react_to_message_id=888
            )
        )
        loop.close()

        self.assertEqual(result["text"], "Nice!")
        self.assertEqual(result["emoji_reacted"], "👍")
        self.assertFalse(result["skipped"])


# ---------------------------------------------------------------------------
# Conversation history formatting
# ---------------------------------------------------------------------------


class TestFormatConversationHistory(unittest.TestCase):
    def test_empty_history(self):
        result = _format_conversation_history([])
        self.assertIn("No messages yet", result)

    def test_formats_messages(self):
        messages = [
            {"agent": "grok", "text": "Hello"},
            {"agent": "claude", "text": "Hi there"},
        ]
        result = _format_conversation_history(messages)
        self.assertIn("Grok Bot: Hello", result)
        self.assertIn("Clod Bot: Hi there", result)

    def test_limits_to_context_window(self):
        messages = [
            {"agent": f"bot{i}", "text": f"msg{i}", "message_id": i} for i in range(75)
        ]
        result = _format_conversation_history(messages)
        # No theme → full CONTEXT_WINDOW_SIZE (50)
        self.assertNotIn("bot0:", result)
        self.assertIn("bot74:", result)
        self.assertIn("bot25:", result)

    def test_limits_to_theme_context_window(self):
        messages = [
            {"agent": f"bot{i}", "text": f"msg{i}", "message_id": i} for i in range(30)
        ]
        # "memes" theme → 0.35 * 50 = 18
        result = _format_conversation_history(messages, theme="memes")
        self.assertNotIn("bot0:", result)
        self.assertIn("bot29:", result)
        self.assertIn("bot12:", result)
        # bot11 should be outside the 18-message window
        self.assertNotIn("bot11:", result)

    def test_reactions_merged_inline(self):
        messages = [
            {"agent": "claude", "text": "Something edgy", "message_id": 123},
            {"agent": "grok", "text": "[reacted 💀 to msg:123]", "message_id": None},
            {"agent": "chatgpt", "text": "LOL", "message_id": 124},
        ]
        result = _format_conversation_history(messages)
        self.assertIn(
            "[msg:123] Clod Bot: Something edgy  [reactions: 💀 (Grok Bot)]", result
        )
        self.assertIn("[msg:124] GPT Bot: LOL", result)
        # Reaction line should NOT appear as a separate entry
        self.assertNotIn("grok:", result)  # no standalone "grok:" message line
        self.assertNotIn("[reacted", result)

    def test_multiple_reactions_on_same_message(self):
        messages = [
            {"agent": "claude", "text": "Hot take", "message_id": 200},
            {"agent": "grok", "text": "[reacted 🔥 to msg:200]", "message_id": None},
            {"agent": "gemini", "text": "[reacted 💯 to msg:200]", "message_id": None},
        ]
        result = _format_conversation_history(messages)
        self.assertIn("🔥 (Grok Bot)", result)
        self.assertIn("💯 (Google Bot)", result)
        self.assertNotIn("[reacted", result)

    def test_reaction_to_unknown_target_dropped(self):
        messages = [
            {"agent": "grok", "text": "[reacted 💀 to msg:999]", "message_id": None},
            {"agent": "claude", "text": "Hello", "message_id": 100},
        ]
        result = _format_conversation_history(messages)
        self.assertIn("[msg:100] Clod Bot: Hello", result)
        self.assertNotIn("💀", result)

    def test_reaction_with_unknown_id_skipped(self):
        messages = [
            {"agent": "grok", "text": "[reacted 💀 to msg:?]", "message_id": None},
            {"agent": "claude", "text": "Hello", "message_id": 100},
        ]
        result = _format_conversation_history(messages)
        self.assertNotIn("💀", result)
        self.assertNotIn("[reacted", result)

    def test_image_entries_pass_through(self):
        messages = [
            {
                "agent": "chatgpt",
                "text": '[posted image: "cat" → https://cdn.example.com/cat.png]',
                "message_id": 300,
            },
        ]
        result = _format_conversation_history(messages)
        self.assertIn('[posted image: "cat"', result)
        self.assertIn("https://cdn.example.com/cat.png", result)


# ---------------------------------------------------------------------------
# Coordinator instruction handling
# ---------------------------------------------------------------------------


class TestHandleInstruction(unittest.TestCase):
    def setUp(self):
        self.cog = MockAgentCog()
        self.cog.mock_ai_response = '{"skip": false, "text": "Coordinator response"}'
        self.cog._redis = MagicMock()
        self.cog._redis.publish = AsyncMock()

    def test_handles_decide_action(self):
        channel = MagicMock()
        channel.id = 100
        channel.name = "ai-general"
        sent = MagicMock(id=999)
        channel.send = AsyncMock(return_value=sent)
        self.cog.bot.get_channel = MagicMock(return_value=channel)

        instruction = {
            "protocol_version": 1,
            "instruction_id": "test-uuid",
            "action": "decide",
            "channel_id": 100,
            "topic": "Favorite programming languages",
            "conversation_history": [],
        }

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog._handle_instruction(instruction))
        loop.close()

        # Should have published a result
        self.cog._redis.publish.assert_called_once()
        call_args = self.cog._redis.publish.call_args
        result_payload = json.loads(call_args[0][1])
        self.assertEqual(result_payload["instruction_id"], "test-uuid")
        self.assertFalse(result_payload["skipped"])

    def test_ignores_unknown_action(self):
        instruction = {
            "protocol_version": 1,
            "action": "unknown_action",
            "channel_id": 100,
        }

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog._handle_instruction(instruction))
        loop.close()

        # No result published
        self.cog._redis.publish.assert_not_called()

    def test_rate_limited_instruction(self):
        # Exhaust daily cap (set today's date so it doesn't reset)
        self.cog._daily_reset_date = time.strftime("%Y-%m-%d")
        self.cog._daily_count = 5

        channel = MagicMock()
        channel.id = 100
        self.cog.bot.get_channel = MagicMock(return_value=channel)

        instruction = {
            "protocol_version": 1,
            "instruction_id": "rate-limited-uuid",
            "action": "decide",
            "channel_id": 100,
            "topic": "test",
            "conversation_history": [],
        }

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog._handle_instruction(instruction))
        loop.close()

        call_args = self.cog._redis.publish.call_args
        result_payload = json.loads(call_args[0][1])
        self.assertTrue(result_payload["skipped"])
        self.assertEqual(result_payload["reason"], "rate_limited")


# ---------------------------------------------------------------------------
# Redis listener retry tests
# ---------------------------------------------------------------------------


class TestListenerRetry(unittest.TestCase):
    def setUp(self):
        self.cog = MockAgentCog()
        self.cog._redis = MagicMock()

    def test_retries_on_subscribe_failure(self):
        """If pubsub.subscribe() fails, the listener retries instead of dying."""
        call_count = 0
        mock_pubsub = MagicMock()

        async def fake_subscribe(*channels):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Redis not ready")

        mock_pubsub.subscribe = AsyncMock(side_effect=fake_subscribe)

        async def fake_listen():
            yield {"type": "subscribe", "data": None}
            raise asyncio.CancelledError()

        mock_pubsub.listen = fake_listen
        self.cog._redis.pubsub = MagicMock(return_value=mock_pubsub)

        async def run():
            with patch("agent_cogs.base.asyncio.sleep", new_callable=AsyncMock):
                with self.assertRaises(asyncio.CancelledError):
                    await self.cog._listen_for_instructions()

            self.assertEqual(call_count, 2)

        asyncio.run(run())

    def test_cancelled_error_propagates(self):
        """CancelledError should not be caught by the retry loop."""
        mock_pubsub = MagicMock()
        mock_pubsub.subscribe = AsyncMock(side_effect=asyncio.CancelledError())
        self.cog._redis.pubsub = MagicMock(return_value=mock_pubsub)

        async def run():
            with self.assertRaises(asyncio.CancelledError):
                await self.cog._listen_for_instructions()

        asyncio.run(run())

    def test_backoff_increases_then_caps(self):
        """Delay should double each failure, capping at _LISTENER_MAX_BACKOFF."""
        call_count = 0
        mock_pubsub = MagicMock()
        sleep_values = []

        async def fake_subscribe(*channels):
            nonlocal call_count
            call_count += 1
            if call_count <= 6:
                raise ConnectionError("still down")
            # 7th call succeeds

        mock_pubsub.subscribe = AsyncMock(side_effect=fake_subscribe)

        async def fake_listen():
            yield {"type": "subscribe", "data": None}
            raise asyncio.CancelledError()

        mock_pubsub.listen = fake_listen
        self.cog._redis.pubsub = MagicMock(return_value=mock_pubsub)

        async def run():
            async def track_sleep(seconds):
                sleep_values.append(seconds)

            with patch("agent_cogs.base.asyncio.sleep", side_effect=track_sleep):
                with self.assertRaises(asyncio.CancelledError):
                    await self.cog._listen_for_instructions()

            # 6 failures → sleeps of 1, 2, 4, 8, 16, 30 (capped)
            self.assertEqual(sleep_values, [1, 2, 4, 8, 16, 30])

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Cost computation tests
# ---------------------------------------------------------------------------


class TestComputeTokenCost(unittest.TestCase):
    def test_basic_cost(self):
        # gpt-5.4-pro: input=3.00, output=12.00 per 1M tokens
        cost = _compute_token_cost("gpt-5.4-pro", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 15.00)

    def test_unknown_model_returns_zero(self):
        cost = _compute_token_cost("unknown-model", 1000, 500)
        self.assertAlmostEqual(cost, 0.0)

    def test_cache_creation_tokens(self):
        # claude-sonnet-4-6: input=3.00, output=15.00
        # cache_creation = 2x input price = 6.00 per 1M
        cost = _compute_token_cost(
            "claude-sonnet-4-6", 0, 0, cache_creation_tokens=1_000_000
        )
        self.assertAlmostEqual(cost, 6.00)

    def test_cache_read_tokens(self):
        # claude-sonnet-4-6: input=3.00
        # cache_read = 0.1x input price = 0.30 per 1M
        cost = _compute_token_cost(
            "claude-sonnet-4-6", 0, 0, cache_read_tokens=1_000_000
        )
        self.assertAlmostEqual(cost, 0.30)

    def test_reasoning_tokens_billed_at_output_rate(self):
        # grok-4.20: output=6.00 per 1M
        # reasoning tokens should add to output cost
        cost = _compute_token_cost(
            "grok-4.20-beta-latest-reasoning", 0, 0, reasoning_tokens=1_000_000
        )
        self.assertAlmostEqual(cost, 6.00)

    def test_openai_cached_input_tokens(self):
        # gpt-5.4-pro: input=3.00, output=12.00
        # 1M input tokens, 500k cached at 50% discount
        # cost = (500k * 3.00 + 500k * 1.50) / 1M = 1.50 + 0.75 = 2.25
        cost = _compute_token_cost(
            "gpt-5.4-pro", 1_000_000, 0, cached_input_tokens=500_000
        )
        self.assertAlmostEqual(cost, 2.25)

    def test_openai_all_cached(self):
        # gpt-5.4-pro: 1M input all cached → 1M * 3.00 * 0.5 / 1M = 1.50
        cost = _compute_token_cost(
            "gpt-5.4-pro", 1_000_000, 0, cached_input_tokens=1_000_000
        )
        self.assertAlmostEqual(cost, 1.50)

    def test_combined_tokens(self):
        # claude-sonnet-4-6: input=3.00, output=15.00
        cost = _compute_token_cost(
            "claude-sonnet-4-6",
            500_000,    # input: 500k * 3.00/1M = 1.50
            200_000,    # output: 200k * 15.00/1M = 3.00
            cache_creation_tokens=100_000,  # 100k * 3.00 * 2 / 1M = 0.60
            cache_read_tokens=1_000_000,    # 1M * 3.00 * 0.1 / 1M = 0.30
        )
        self.assertAlmostEqual(cost, 5.40)


# ---------------------------------------------------------------------------
# AIResponse dataclass tests
# ---------------------------------------------------------------------------


class TestAIResponse(unittest.TestCase):
    def test_defaults(self):
        r = AIResponse()
        self.assertEqual(r.text, "")
        self.assertEqual(r.input_tokens, 0)
        self.assertEqual(r.output_tokens, 0)
        self.assertEqual(r.cache_creation_tokens, 0)
        self.assertEqual(r.cache_read_tokens, 0)
        self.assertEqual(r.cached_input_tokens, 0)
        self.assertEqual(r.reasoning_tokens, 0)

    def test_provider_specific_fields(self):
        r = AIResponse(
            text="hi",
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=10,
            cache_read_tokens=80,
            reasoning_tokens=200,
        )
        self.assertEqual(r.cache_creation_tokens, 10)
        self.assertEqual(r.cache_read_tokens, 80)
        self.assertEqual(r.reasoning_tokens, 200)


# ---------------------------------------------------------------------------
# Error formatting tests
# ---------------------------------------------------------------------------


class TestFormatApiError(unittest.TestCase):
    def test_basic_exception(self):
        err = Exception("Something went wrong")
        result = format_api_error(err)
        self.assertIn("Something went wrong", result)

    def test_with_status_code(self):
        err = Exception("Rate limited")
        err.status_code = 429
        result = format_api_error(err)
        self.assertIn("429", result)
        self.assertIn("Rate limited", result)

    def test_with_message_attr(self):
        err = Exception()
        err.message = "Overloaded"
        result = format_api_error(err)
        self.assertIn("Overloaded", result)

    def test_openai_body_extraction(self):
        err = Exception("API error")
        err.body = {"error": {"type": "invalid_request", "code": "model_not_found"}}
        result = format_api_error(err)
        self.assertIn("invalid_request", result)
        self.assertIn("model_not_found", result)


if __name__ == "__main__":
    unittest.main()
