"""Tests for the agent cog base logic.

Uses a concrete mock subclass of BaseAgentCog to test shared behavior:
- Decision JSON parsing (valid, malformed, skip, unknown fields)
- Rate limiting (cooldown, daily cap)
- @mention detection
- Action execution (text, image, emoji)
- Coordinator result publishing
"""

import asyncio
import importlib
import json

# Patch agent_config before importing base
import sys
import time
import types as stdlib_types
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

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
fake_config.ANTHROPIC_API_KEY = ""
fake_config.GEMINI_API_KEY = ""


def _fake_get_context_window(theme=None):
    if theme:
        scales = {"debate": 1.0, "memes": 0.35}
        scale = scales.get(theme, 1.0)
        return round(fake_config.CONTEXT_WINDOW_SIZE * scale) or 1
    return fake_config.CONTEXT_WINDOW_SIZE


fake_config.get_context_window = _fake_get_context_window
sys.modules["agent_config"] = fake_config

from agent_cogs.base import (  # noqa: E402
    GEMINI_MAPS_GROUNDING_COST_PER_CALL,
    OPENAI_WEB_SEARCH_COST_PER_CALL,
    AIResponse,
    BaseAgentCog,
    _compute_token_cost,
    _compute_tool_cost,
    _format_conversation_history,
    _parse_decision,
    format_api_error,
)


async def _empty_async_iter():
    """Async iterator that yields nothing — used to mock channel.history()."""
    return
    yield  # noqa: F841 — makes this an async generator


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

    async def _call_ai(
        self, system_prompt: str, user_prompt: str, image_urls: list[str] | None = None
    ) -> AIResponse:
        return AIResponse(text=self.mock_ai_response, input_tokens=100, output_tokens=50)

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        return self.mock_image_bytes


# ---------------------------------------------------------------------------
# Decision parsing tests
# ---------------------------------------------------------------------------


class TestParseDecision:
    def test_valid_json(self):
        raw = '{"skip": false, "text": "Hi there", "generate_image": false}'
        result = _parse_decision(raw)
        assert not result["skip"]
        assert result["text"] == "Hi there"

    def test_skip_true(self):
        raw = '{"skip": true}'
        result = _parse_decision(raw)
        assert result["skip"]

    def test_malformed_json_defaults_to_skip(self):
        raw = "not valid json at all"
        result = _parse_decision(raw)
        assert result["skip"]

    def test_markdown_fenced_json(self):
        raw = '```json\n{"skip": false, "text": "fenced"}\n```'
        result = _parse_decision(raw)
        assert not result["skip"]
        assert result["text"] == "fenced"

    def test_unknown_fields_ignored(self):
        raw = '{"skip": false, "text": "hi", "generate_video": true, "tts_text": "hello"}'
        result = _parse_decision(raw)
        assert not result["skip"]
        assert result["text"] == "hi"
        # Unknown fields are present in the dict but harmless
        assert result.get("generate_video")

    def test_empty_string(self):
        result = _parse_decision("")
        assert result["skip"]

    def test_non_dict_json(self):
        result = _parse_decision("[1, 2, 3]")
        assert result["skip"]

    def test_null_text_field(self):
        raw = '{"skip": false, "text": null, "react_emoji": "😂"}'
        result = _parse_decision(raw)
        assert result["text"] is None
        assert result["react_emoji"] == "😂"


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def setup_method(self, _method=None):
        self.cog = MockAgentCog()

    def test_first_request_allowed(self):
        assert self.cog._check_rate_limits(100)

    def test_cooldown_blocks_second_request(self):
        self.cog._record_response(100)
        assert not self.cog._check_rate_limits(100)

    def test_different_channel_not_affected(self):
        self.cog._record_response(100)
        assert self.cog._check_rate_limits(200)

    def test_daily_cap_enforced(self):
        self.cog._daily_reset_date = time.strftime("%Y-%m-%d")
        self.cog._daily_count = 5
        assert not self.cog._check_rate_limits(100)

    def test_daily_cap_resets_on_new_day(self):
        self.cog._daily_count = 5
        self.cog._daily_reset_date = "2020-01-01"  # Force stale date
        assert self.cog._check_rate_limits(100)
        assert self.cog._daily_count == 0

    def test_cooldown_expires(self):
        self.cog._record_response(100)
        # Manually backdate the last response time
        self.cog._last_response_time[100] = time.time() - 120
        assert self.cog._check_rate_limits(100)


class TestHttpSession:
    def setup_method(self, _method=None):
        self.cog = MockAgentCog()

    def teardown_method(self, _method=None):
        if self.cog._http_session and not self.cog._http_session.closed:
            asyncio.run(self.cog._http_session.close())

    def test_reuses_shared_session_with_explicit_config(self):
        async def run():
            session1 = await self.cog.get_http_session()
            session2 = await self.cog.get_http_session()

            assert session1 is session2
            assert session1.timeout.total == 30
            assert session1.timeout.connect == 10
            assert session1.timeout.sock_connect == 10
            assert session1.timeout.sock_read == 30
            assert session1.connector.limit == 50
            assert session1.connector.limit_per_host == 10

        asyncio.run(run())

    def test_creates_new_session_after_close(self):
        async def run():
            session1 = await self.cog.get_http_session()
            await session1.close()

            session2 = await self.cog.get_http_session()

            assert session1 is not session2
            await session2.close()

        asyncio.run(run())


def _load_gemini_helpers():
    fake_google = stdlib_types.ModuleType("google")
    fake_genai = stdlib_types.ModuleType("google.genai")
    fake_types = stdlib_types.ModuleType("google.genai.types")
    fake_google.genai = fake_genai
    fake_genai.types = fake_types

    with patch.dict(
        sys.modules,
        {
            "agent_config": fake_config,
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        },
    ):
        sys.modules.pop("agent_cogs.gemini_agent", None)
        module = importlib.import_module("agent_cogs.gemini_agent")

    return module._extract_gemini_grounding_metadata, module._format_gemini_grounding_footer


def _load_anthropic_helpers():
    with patch.dict(sys.modules, {"agent_config": fake_config}):
        sys.modules.pop("agent_cogs.anthropic_agent", None)
        module = importlib.import_module("agent_cogs.anthropic_agent")

    return module._extract_anthropic_web_search_calls


class TestGeminiGroundingHelpers:
    def test_extracts_grounding_queries_and_sources(self):
        extract_grounding, _ = _load_gemini_helpers()
        response = stdlib_types.SimpleNamespace(
            candidates=[
                stdlib_types.SimpleNamespace(
                    grounding_metadata=stdlib_types.SimpleNamespace(
                        web_search_queries=["latest AI chips", " earnings outlook "],
                        search_entry_point=stdlib_types.SimpleNamespace(
                            rendered_content="<div>Rendered search card</div>"
                        ),
                        grounding_chunks=[
                            stdlib_types.SimpleNamespace(
                                web=stdlib_types.SimpleNamespace(
                                    uri="https://example.com/1",
                                    title="Source One",
                                )
                            ),
                            stdlib_types.SimpleNamespace(
                                web=stdlib_types.SimpleNamespace(
                                    uri="https://example.com/1",
                                    title="Source One",
                                )
                            ),
                            stdlib_types.SimpleNamespace(
                                web=stdlib_types.SimpleNamespace(
                                    uri="https://example.com/2",
                                    title="",
                                )
                            ),
                            stdlib_types.SimpleNamespace(web=None),
                        ],
                    )
                )
            ]
        )

        metadata = extract_grounding(response)

        assert metadata.search_queries == ("latest AI chips", "earnings outlook")
        assert metadata.rendered_content == "<div>Rendered search card</div>"
        assert metadata.sources == (
            ("https://example.com/1", "Source One"),
            ("https://example.com/2", ""),
        )

    def test_formats_grounding_footer(self):
        _, format_footer = _load_gemini_helpers()
        metadata = stdlib_types.SimpleNamespace(
            sources=(
                ("https://example.com/1", "Source One"),
                ("https://example.com/2", ""),
            )
        )

        footer = format_footer(metadata)

        assert "[Source One](https://example.com/1)" in footer
        assert "[source](https://example.com/2)" in footer
        assert footer.startswith("Sources: ")


class TestAnthropicToolUsageHelpers:
    def test_extracts_web_search_requests(self):
        extract_web_search_calls = _load_anthropic_helpers()
        response = stdlib_types.SimpleNamespace(
            usage=stdlib_types.SimpleNamespace(
                server_tool_use=stdlib_types.SimpleNamespace(web_search_requests=3)
            )
        )

        assert extract_web_search_calls(response) == 3

    def test_missing_server_tool_usage_defaults_to_zero(self):
        extract_web_search_calls = _load_anthropic_helpers()
        response = stdlib_types.SimpleNamespace(usage=stdlib_types.SimpleNamespace())

        assert extract_web_search_calls(response) == 0


# ---------------------------------------------------------------------------
# @mention detection tests
# ---------------------------------------------------------------------------


class TestMentionDetection:
    def setup_method(self, _method=None):
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
        self.cog._decide_and_act = AsyncMock(return_value={"skipped": False, "message_id": 111})
        # Provide async iterator for channel history
        msg.channel.history = MagicMock(return_value=_empty_async_iter())
        msg.guild = MagicMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog.on_message(msg))
        loop.close()

        self.cog._decide_and_act.assert_called_once()
        # force_respond should be True
        _, kwargs = self.cog._decide_and_act.call_args
        assert kwargs.get("force_respond")

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
        self.cog._decide_and_act = AsyncMock(return_value={"skipped": False, "message_id": 111})
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


class TestActionExecution:
    def setup_method(self, _method=None):
        self.cog = MockAgentCog()

    def test_send_text(self):
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 111
        channel.send = AsyncMock(return_value=sent_msg)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.cog._send_text(channel, "hello"))
        loop.close()

        channel.send.assert_called_once_with("hello", embed=None)
        assert result.id == 111

    def test_send_text_truncates_long_messages(self):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=MagicMock(id=111))

        long_text = "x" * 3000
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cog._send_text(channel, long_text))
        loop.close()

        sent_text = channel.send.call_args[0][0]
        assert len(sent_text) <= 2000
        assert sent_text.endswith("...")

    def test_add_reaction(self):
        channel = MagicMock()
        message = MagicMock()
        message.add_reaction = AsyncMock()
        channel.fetch_message = AsyncMock(return_value=message)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.cog._add_reaction(channel, 12345, "😂"))
        loop.close()

        assert result
        message.add_reaction.assert_called_once_with("😂")


# ---------------------------------------------------------------------------
# Decision + act integration tests
# ---------------------------------------------------------------------------


class TestDecideAndAct:
    def setup_method(self, _method=None):
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

        assert not result["skipped"]
        assert result["text"] == "Hello world!"
        assert result["message_id"] == 333

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

        assert result["skipped"]

    def test_force_respond_overrides_skip(self):
        self.cog.mock_ai_response = '{"skip": true, "text": "Forced response"}'
        channel = MagicMock()
        sent = MagicMock(id=444)
        channel.send = AsyncMock(return_value=sent)
        channel.id = 100
        channel.name = "ai-general"

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(channel, "context", "topic", "ai-general", force_respond=True)
        )
        loop.close()

        # force_respond=True means skip is ignored, text should be sent
        assert not result["skipped"]
        assert result["text"] == "Forced response"

    def test_emoji_reaction(self):
        self.cog.mock_ai_response = '{"skip": false, "text": null, "react_emoji": "🔥"}'
        mock_pipe = MagicMock()
        mock_pipe.hincrby = MagicMock()
        mock_pipe.hincrbyfloat = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[1])
        self.cog._redis = MagicMock()
        self.cog._redis.pipeline = MagicMock(return_value=mock_pipe)
        channel = MagicMock()
        channel.id = 100
        channel.name = "ai-general"
        target_msg = MagicMock()
        target_msg.add_reaction = AsyncMock()
        channel.fetch_message = AsyncMock(return_value=target_msg)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.cog._decide_and_act(channel, "context", "", "ai-general", react_to_message_id=555)
        )
        loop.close()

        assert result["emoji_reacted"] == "🔥"
        # Verify emoji counter was incremented via pipeline
        mock_pipe.hincrby.assert_any_call(
            f"agent:{self.cog.agent_redis_name}:cost:{time.strftime('%Y-%m-%d')}",
            "emoji_reactions",
            1,
        )

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

        assert result.get("image_sent")

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

        assert result.get("end_conversation")
        assert not result["skipped"]

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

        assert "end_conversation" not in result

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
            self.cog._decide_and_act(channel, "context", "", "ai-general", react_to_message_id=888)
        )
        loop.close()

        assert result["text"] == "Nice!"
        assert result["emoji_reacted"] == "👍"
        assert not result["skipped"]


# ---------------------------------------------------------------------------
# Conversation history formatting
# ---------------------------------------------------------------------------


class TestFormatConversationHistory:
    def test_empty_history(self):
        result = _format_conversation_history([])
        assert "No messages yet" in result

    def test_formats_messages(self):
        messages = [
            {"agent": "grok", "text": "Hello"},
            {"agent": "claude", "text": "Hi there"},
        ]
        result = _format_conversation_history(messages)
        assert "Grok Bot: Hello" in result
        assert "Clod Bot: Hi there" in result

    def test_limits_to_context_window(self):
        messages = [{"agent": f"bot{i}", "text": f"msg{i}", "message_id": i} for i in range(75)]
        result = _format_conversation_history(messages)
        # No theme → full CONTEXT_WINDOW_SIZE (50)
        assert "bot0:" not in result
        assert "bot74:" in result
        assert "bot25:" in result

    def test_limits_to_theme_context_window(self):
        messages = [{"agent": f"bot{i}", "text": f"msg{i}", "message_id": i} for i in range(30)]
        # "memes" theme → 0.35 * 50 = 18
        result = _format_conversation_history(messages, theme="memes")
        assert "bot0:" not in result
        assert "bot29:" in result
        assert "bot12:" in result
        # bot11 should be outside the 18-message window
        assert "bot11:" not in result

    def test_reactions_merged_inline(self):
        messages = [
            {"agent": "claude", "text": "Something edgy", "message_id": 123},
            {"agent": "grok", "text": "[reacted 💀 to msg:123]", "message_id": None},
            {"agent": "chatgpt", "text": "LOL", "message_id": 124},
        ]
        result = _format_conversation_history(messages)
        assert "[msg:123] Clod Bot: Something edgy  [reactions: 💀 (Grok Bot)]" in result
        assert "[msg:124] GPT Bot: LOL" in result
        # Reaction line should NOT appear as a separate entry
        assert "grok:" not in result
        assert "[reacted" not in result

    def test_multiple_reactions_on_same_message(self):
        messages = [
            {"agent": "claude", "text": "Hot take", "message_id": 200},
            {"agent": "grok", "text": "[reacted 🔥 to msg:200]", "message_id": None},
            {"agent": "gemini", "text": "[reacted 💯 to msg:200]", "message_id": None},
        ]
        result = _format_conversation_history(messages)
        assert "🔥 (Grok Bot)" in result
        assert "💯 (Google Bot)" in result
        assert "[reacted" not in result

    def test_reaction_to_unknown_target_dropped(self):
        messages = [
            {"agent": "grok", "text": "[reacted 💀 to msg:999]", "message_id": None},
            {"agent": "claude", "text": "Hello", "message_id": 100},
        ]
        result = _format_conversation_history(messages)
        assert "[msg:100] Clod Bot: Hello" in result
        assert "💀" not in result

    def test_reaction_with_unknown_id_skipped(self):
        messages = [
            {"agent": "grok", "text": "[reacted 💀 to msg:?]", "message_id": None},
            {"agent": "claude", "text": "Hello", "message_id": 100},
        ]
        result = _format_conversation_history(messages)
        assert "💀" not in result
        assert "[reacted" not in result

    def test_image_entries_pass_through(self):
        messages = [
            {
                "agent": "chatgpt",
                "text": '[posted image: "cat" → https://cdn.example.com/cat.png]',
                "message_id": 300,
            },
        ]
        result = _format_conversation_history(messages)
        assert '[posted image: "cat"' in result
        assert "https://cdn.example.com/cat.png" in result


# ---------------------------------------------------------------------------
# Coordinator instruction handling
# ---------------------------------------------------------------------------


class TestHandleInstruction:
    def setup_method(self, _method=None):
        self.cog = MockAgentCog()
        self.cog.mock_ai_response = '{"skip": false, "text": "Coordinator response"}'
        self.cog._redis = MagicMock()
        self.cog._redis.publish = AsyncMock()

    def test_handles_decide_action(self):
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 100
        channel.name = "ai-general"
        channel.guild = MagicMock()
        sent = MagicMock(id=999)
        channel.send = AsyncMock(return_value=sent)
        channel.history = MagicMock(return_value=_empty_async_iter())
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
        assert result_payload["instruction_id"] == "test-uuid"
        assert not result_payload["skipped"]

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

        channel = MagicMock(spec=discord.TextChannel)
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
        assert result_payload["skipped"]
        assert result_payload["reason"] == "rate_limited"


# ---------------------------------------------------------------------------
# Redis listener retry tests
# ---------------------------------------------------------------------------


class TestListenerRetry:
    def setup_method(self, _method=None):
        self.cog = MockAgentCog()
        self.cog._redis = MagicMock()

    def test_retries_on_subscribe_failure(self):
        """If pubsub.subscribe() fails, the listener retries instead of dying."""
        call_count = 0
        mock_pubsub = MagicMock()
        mock_pubsub.aclose = AsyncMock()

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
            with (
                patch("agent_cogs.base.asyncio.sleep", new_callable=AsyncMock),
                pytest.raises(asyncio.CancelledError),
            ):
                await self.cog._listen_for_instructions()

            assert call_count == 2

        asyncio.run(run())

    def test_cancelled_error_propagates(self):
        """CancelledError should not be caught by the retry loop."""
        mock_pubsub = MagicMock()
        mock_pubsub.aclose = AsyncMock()
        mock_pubsub.subscribe = AsyncMock(side_effect=asyncio.CancelledError())
        self.cog._redis.pubsub = MagicMock(return_value=mock_pubsub)

        async def run():
            with pytest.raises(asyncio.CancelledError):
                await self.cog._listen_for_instructions()

        asyncio.run(run())

    def test_backoff_increases_then_caps(self):
        """Delay should double each failure, capping at _LISTENER_MAX_BACKOFF."""
        call_count = 0
        mock_pubsub = MagicMock()
        mock_pubsub.aclose = AsyncMock()
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

            with (
                patch("agent_cogs.base.asyncio.sleep", side_effect=track_sleep),
                pytest.raises(asyncio.CancelledError),
            ):
                await self.cog._listen_for_instructions()

            # 6 failures → sleeps of 1, 2, 4, 8, 16, 30 (capped)
            assert sleep_values == [1, 2, 4, 8, 16, 30]

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Cost computation tests
# ---------------------------------------------------------------------------


class TestComputeTokenCost:
    def test_basic_cost(self):
        # gpt-5.4: input=2.50, output=15.00 per 1M tokens
        cost = _compute_token_cost("gpt-5.4", 1_000_000, 1_000_000)
        assert cost == pytest.approx(17.50)

    def test_unknown_model_returns_zero(self):
        cost = _compute_token_cost("unknown-model", 1000, 500)
        assert cost == pytest.approx(0.0)

    def test_cache_creation_tokens(self):
        # claude-sonnet-4-6: input=3.00, output=15.00
        # cache_creation = 2x input price = 6.00 per 1M
        cost = _compute_token_cost("claude-sonnet-4-6", 0, 0, cache_creation_tokens=1_000_000)
        assert cost == pytest.approx(6.00)

    def test_cache_read_tokens(self):
        # claude-sonnet-4-6: input=3.00
        # cache_read = 0.1x input price = 0.30 per 1M
        cost = _compute_token_cost("claude-sonnet-4-6", 0, 0, cache_read_tokens=1_000_000)
        assert cost == pytest.approx(0.30)

    def test_reasoning_tokens_billed_at_output_rate(self):
        # grok-4.20: output=6.00 per 1M
        # reasoning tokens should add to output cost
        cost = _compute_token_cost("grok-4.20", 0, 0, reasoning_tokens=1_000_000)
        assert cost == pytest.approx(6.00)

    def test_openai_reasoning_tokens_after_subtraction(self):
        # gpt-5.4: output=15.00
        # OpenAI includes reasoning in output_tokens, so the agent subtracts
        # reasoning before building AIResponse: output=800k, reasoning=200k
        # cost = (800k + 200k) * 15.00 / 1M = 15.00 (same as 1M total output)
        cost = _compute_token_cost("gpt-5.4", 0, 800_000, reasoning_tokens=200_000)
        assert cost == pytest.approx(15.00)

    def test_openai_cached_input_tokens(self):
        # gpt-5.4: input=2.50, output=15.00
        # 1M input tokens, 500k cached at 50% discount
        # cost = (500k * 2.50 + 500k * 1.25) / 1M = 1.25 + 0.625 = 1.875
        cost = _compute_token_cost("gpt-5.4", 1_000_000, 0, cached_input_tokens=500_000)
        assert cost == pytest.approx(1.875)

    def test_openai_all_cached(self):
        # gpt-5.4: 1M input all cached → 1M * 2.50 * 0.5 / 1M = 1.25
        cost = _compute_token_cost("gpt-5.4", 1_000_000, 0, cached_input_tokens=1_000_000)
        assert cost == pytest.approx(1.25)

    def test_combined_tokens(self):
        # claude-sonnet-4-6: input=3.00, output=15.00
        cost = _compute_token_cost(
            "claude-sonnet-4-6",
            500_000,  # input: 500k * 3.00/1M = 1.50
            200_000,  # output: 200k * 15.00/1M = 3.00
            cache_creation_tokens=100_000,  # 100k * 3.00 * 2 / 1M = 0.60
            cache_read_tokens=1_000_000,  # 1M * 3.00 * 0.1 / 1M = 0.30
        )
        assert cost == pytest.approx(5.40)

    def test_web_search_calls_only_billed_for_priced_models(self):
        assert _compute_tool_cost("gpt-5.4", web_search_calls=2) == pytest.approx(0.02)
        assert _compute_tool_cost("grok-4.20", web_search_calls=2) == pytest.approx(0.02)
        assert _compute_tool_cost("claude-sonnet-4-6", web_search_calls=2) == pytest.approx(0.0)

    def test_maps_grounding_calls_billed_separately(self):
        cost = _compute_tool_cost(
            "gemini-3.1-pro-preview",
            web_search_calls=3,
            maps_grounding_calls=2,
        )
        assert cost == pytest.approx(2 * GEMINI_MAPS_GROUNDING_COST_PER_CALL)


# ---------------------------------------------------------------------------
# AIResponse dataclass tests
# ---------------------------------------------------------------------------


class TestAIResponse:
    def test_defaults(self):
        r = AIResponse()
        assert r.text == ""
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.cache_creation_tokens == 0
        assert r.cache_read_tokens == 0
        assert r.cached_input_tokens == 0
        assert r.reasoning_tokens == 0
        assert r.web_search_calls == 0
        assert r.maps_grounding_calls == 0

    def test_provider_specific_fields(self):
        r = AIResponse(
            text="hi",
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=10,
            cache_read_tokens=80,
            reasoning_tokens=200,
            web_search_calls=2,
            maps_grounding_calls=1,
        )
        assert r.cache_creation_tokens == 10
        assert r.cache_read_tokens == 80
        assert r.reasoning_tokens == 200
        assert r.web_search_calls == 2
        assert r.maps_grounding_calls == 1

    def test_openai_web_search_cost_per_call(self):
        # Each web_search call costs $0.01 flat; applied on top of token cost
        assert pytest.approx(0.01) == OPENAI_WEB_SEARCH_COST_PER_CALL

    def test_gemini_maps_grounding_cost_per_call(self):
        # Each Maps-grounded prompt costs $0.025 flat
        assert pytest.approx(0.025) == GEMINI_MAPS_GROUNDING_COST_PER_CALL


# ---------------------------------------------------------------------------
# Error formatting tests
# ---------------------------------------------------------------------------


class TestFormatApiError:
    def test_basic_exception(self):
        err = Exception("Something went wrong")
        result = format_api_error(err)
        assert "Something went wrong" in result

    def test_with_status_code(self):
        err = Exception("Rate limited")
        err.status_code = 429
        result = format_api_error(err)
        assert "429" in result
        assert "Rate limited" in result

    def test_with_message_attr(self):
        err = Exception()
        err.message = "Overloaded"
        result = format_api_error(err)
        assert "Overloaded" in result

    def test_openai_body_extraction(self):
        err = Exception("API error")
        err.body = {"error": {"type": "invalid_request", "code": "model_not_found"}}
        result = format_api_error(err)
        assert "invalid_request" in result
        assert "model_not_found" in result
