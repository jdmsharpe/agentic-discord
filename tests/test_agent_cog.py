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
fake_config.REDIS_URL = ""
sys.modules["agent_config"] = fake_config

from agent_cogs.base import BaseAgentCog, _parse_decision, _format_conversation_history


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
        self.agent_display_name = "TestBot"
        self.other_agent_names = ["Clod Bot", "Google Bot", "Grok Bot"]
        self._last_response_time = {}
        self._daily_count = 0
        self._daily_reset_date = ""

        # Mock AI methods
        self.mock_ai_response = '{"skip": false, "text": "Hello!", "generate_image": false, "image_prompt": null, "react_emoji": null}'
        self.mock_image_bytes = b"fake_png_bytes"

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> str:
        return self.mock_ai_response

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
        raw = '{"skip": false, "text": "hi", "generate_video": true, "tts_text": "hello"}'
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

    def _make_message(self, author_bot=False, mentions_bot=False, channel_id=100):
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
        result = loop.run_until_complete(
            self.cog._add_reaction(channel, 12345, "😂")
        )
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
        self.cog.mock_ai_response = (
            '{"skip": false, "text": null, "generate_image": true, "image_prompt": "a cat"}'
        )
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
        self.cog.mock_ai_response = '{"skip": false, "text": "Good talk!", "end_conversation": true}'
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
        self.cog.mock_ai_response = '{"skip": false, "text": "Nice!", "react_emoji": "👍"}'
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
        messages = [{"agent": f"bot{i}", "text": f"msg{i}", "message_id": i} for i in range(75)]
        result = _format_conversation_history(messages)
        # Should only contain the last CONTEXT_WINDOW_SIZE (50)
        self.assertNotIn("bot0:", result)
        self.assertIn("bot74:", result)
        self.assertIn("bot25:", result)

    def test_reactions_merged_inline(self):
        messages = [
            {"agent": "claude", "text": "Something edgy", "message_id": 123},
            {"agent": "grok", "text": "[reacted 💀 to msg:123]", "message_id": None},
            {"agent": "chatgpt", "text": "LOL", "message_id": 124},
        ]
        result = _format_conversation_history(messages)
        self.assertIn("[msg:123] Clod Bot: Something edgy  [reactions: 💀 (Grok Bot)]", result)
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
            {"agent": "chatgpt", "text": '[posted image: "cat" → https://cdn.example.com/cat.png]', "message_id": 300},
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


if __name__ == "__main__":
    unittest.main()
