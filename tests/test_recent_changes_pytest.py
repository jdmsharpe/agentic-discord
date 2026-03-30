"""Pytest coverage for recent queue, transport, and metrics changes."""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_fake_agent_config() -> types.ModuleType:
    module = types.ModuleType("agent_config")
    module.AGENT_NAME = "chatgpt"
    module.AGENT_PERSONALITY = ""
    module.AGENT_PERSONALITY_MAP = {"chatgpt": "Default GPT personality."}
    module.AGENT_CHANNEL_IDS = [100, 200, 300]
    module.BOT_IDS = [900, 901, 902]
    module.AGENT_MAX_DAILY = 5
    module.AGENT_COOLDOWN_SECONDS = 60
    module.CONTEXT_WINDOW_SIZE = 50
    module.CHANNEL_THEMES = {100: "debate", 200: "casual", 300: "memes"}
    module.BOTS_ROLE_ID = 55555
    module.REDIS_URL = ""
    module.SHOW_COST_EMBEDS = True
    module.GEMINI_API_KEY = ""
    module.ANTHROPIC_API_KEY = ""
    module.get_context_window = lambda theme=None: 50
    return module


def _make_fake_coordinator_config() -> types.ModuleType:
    module = types.ModuleType("agent_coordinator.config")
    module.REDIS_URL = "redis://localhost:6379"
    module.AGENT_NAMES = ["chatgpt", "claude", "gemini", "grok"]
    module.AGENT_CHANNEL_IDS = [100, 200, 300]
    module.CHANNEL_THEMES = {100: "debate", 200: "casual", 300: "memes"}
    module.SCHEDULE_MIN_EVENTS = 3
    module.SCHEDULE_MAX_EVENTS = 6
    module.SCHEDULE_ACTIVE_START_HOUR = 7
    module.SCHEDULE_ACTIVE_END_HOUR = 23
    module.MAX_ROUNDS = 30
    module.AGENT_RESPONSE_TIMEOUT = 2.0
    module.CONTINUATION_BASE_PROBABILITY = 0.85
    module.CONTINUATION_DECAY = 0.03
    module.MIN_RESPONDENTS_TO_CONTINUE = 2
    module.REACTIVE_TRIGGER_PROBABILITY = 0.15
    module.REACTIVE_COOLDOWN_SECONDS = 300.0
    module.CONSECUTIVE_TIMEOUT_THRESHOLD = 3
    module.PRIORITY_CHANNEL_IDS = [100]
    module.FIRE_ON_STARTUP = False
    module.TURN_DELAY_MIN = 0.0
    module.TURN_DELAY_MAX = 0.0
    module.DAILY_KEY_TTL_SECONDS = 172_800
    return module


@pytest.fixture
def base_module(monkeypatch: pytest.MonkeyPatch):
    fake_config = _make_fake_agent_config()
    previous_base = sys.modules.get("agent_cogs.base")

    monkeypatch.setitem(sys.modules, "agent_config", fake_config)
    sys.modules.pop("agent_cogs.base", None)
    module = importlib.import_module("agent_cogs.base")

    yield module

    if previous_base is not None:
        sys.modules["agent_cogs.base"] = previous_base
    else:
        sys.modules.pop("agent_cogs.base", None)


@pytest.fixture
def engine_module(monkeypatch: pytest.MonkeyPatch):
    fake_agent_config = _make_fake_agent_config()
    fake_coord_config = _make_fake_coordinator_config()
    previous_engine = sys.modules.get("agent_coordinator.engine")

    monkeypatch.setitem(sys.modules, "agent_config", fake_agent_config)
    monkeypatch.setitem(sys.modules, "agent_coordinator.config", fake_coord_config)
    sys.modules.pop("agent_coordinator.engine", None)
    module = importlib.import_module("agent_coordinator.engine")

    yield module

    if previous_engine is not None:
        sys.modules["agent_coordinator.engine"] = previous_engine
    else:
        sys.modules.pop("agent_coordinator.engine", None)


@pytest.fixture
def claude_cog(base_module):
    class ClaudeMetricsCog(base_module.BaseAgentCog):
        agent_redis_name = "claude"
        ai_model = "claude-sonnet-4-6"
        image_model = ""

        def __init__(self):
            super().__init__(MagicMock())

        async def _call_ai(self, system_prompt, user_prompt, image_urls=None):
            return base_module.AIResponse()

        async def _generate_image_bytes(self, prompt):
            return None

    return ClaudeMetricsCog()


@pytest.fixture
def gemini_cog(base_module):
    class GeminiMetricsCog(base_module.BaseAgentCog):
        agent_redis_name = "gemini"
        ai_model = "gemini-3.1-pro-preview"
        image_model = "gemini-3.1-flash-image-preview"

        def __init__(self):
            super().__init__(MagicMock())

        async def _call_ai(self, system_prompt, user_prompt, image_urls=None):
            return base_module.AIResponse()

        async def _generate_image_bytes(self, prompt):
            return None

    return GeminiMetricsCog()


class _FakeResponse:
    def __init__(self, status: int, data: bytes = b"", content_type: str | None = None):
        self.status = status
        self._data = data
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._data


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse | None = None, exc: Exception | None = None):
        self._response = response
        self._exc = exc

    def get(self, _url: str) -> _FakeRequestContext:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return _FakeRequestContext(self._response)


class _FakePipeline:
    def __init__(self, result: float):
        self.calls: list[tuple[str, tuple]] = []
        self._result = result

    def hincrbyfloat(self, *args):
        self.calls.append(("hincrbyfloat", args))
        return self

    def hincrby(self, *args):
        self.calls.append(("hincrby", args))
        return self

    def expire(self, *args):
        self.calls.append(("expire", args))
        return self

    async def execute(self):
        return [self._result]


def test_pop_channel_for_today_seeds_priority_channels_first(
    engine_module, monkeypatch: pytest.MonkeyPatch
):
    mock_redis = MagicMock()
    mock_redis.exists = AsyncMock(return_value=False)
    mock_redis.lpop = AsyncMock(return_value="200")
    mock_redis.rpush = AsyncMock()
    mock_redis.expire = AsyncMock()
    warning = MagicMock()

    monkeypatch.setattr(engine_module, "AGENT_CHANNEL_IDS", [100, 200, 300], raising=False)
    monkeypatch.setattr(engine_module, "PRIORITY_CHANNEL_IDS", [200, 999], raising=False)
    monkeypatch.setattr(engine_module.random, "sample", lambda seq, _k: list(seq))
    monkeypatch.setattr(engine_module.logger, "warning", warning)

    engine = engine_module.ConversationEngine(mock_redis)
    result = asyncio.run(engine.pop_channel_for_today())

    assert result == 200
    assert mock_redis.rpush.await_count == 1
    assert mock_redis.rpush.call_args.args[1:] == ("200", "100", "300")
    mock_redis.expire.assert_awaited_once()
    warning.assert_called_once()


def test_pop_channel_for_today_falls_back_to_random_when_queue_exhausted(
    engine_module, monkeypatch: pytest.MonkeyPatch
):
    mock_redis = MagicMock()
    mock_redis.exists = AsyncMock(return_value=True)
    mock_redis.lpop = AsyncMock(return_value=None)

    monkeypatch.setattr(engine_module, "AGENT_CHANNEL_IDS", [100, 200, 300], raising=False)
    monkeypatch.setattr(engine_module.random, "choice", lambda seq: seq[-1])

    engine = engine_module.ConversationEngine(mock_redis)
    result = asyncio.run(engine.pop_channel_for_today())

    assert result == 300
    assert not mock_redis.rpush.called


def test_download_image_bytes_returns_none_for_http_errors(base_module):
    session = _FakeSession(response=_FakeResponse(status=404, content_type="text/html"))

    result = asyncio.run(base_module._download_image_bytes(session, "https://example.com/img.png"))

    assert result is None


def test_download_image_bytes_returns_none_on_timeout(base_module):
    session = _FakeSession(exc=asyncio.TimeoutError())

    result = asyncio.run(base_module._download_image_bytes(session, "https://example.com/img.png"))

    assert result is None


def test_download_image_bytes_normalizes_content_type(base_module):
    session = _FakeSession(
        response=_FakeResponse(
            status=200,
            data=b"fake-image",
            content_type="image/webp; charset=utf-8",
        )
    )

    result = asyncio.run(base_module._download_image_bytes(session, "https://example.com/img.webp"))

    assert result == (b"fake-image", "image/webp")


def test_build_cost_embed_includes_anthropic_web_search_metric(claude_cog):
    embed = claude_cog._build_cost_embed(
        ai_cost=0.42,
        image_cost=0.0,
        input_tokens=1_200,
        output_tokens=345,
        daily_total=1.23,
        thinking_used=True,
        web_search_calls=3,
        text_generated=True,
    )

    assert embed.footer.text is not None
    assert "web search ×3" in embed.footer.text
    assert "(w/ thinking)" in embed.footer.text


def test_build_cost_embed_includes_gemini_maps_metric(gemini_cog):
    embed = gemini_cog._build_cost_embed(
        ai_cost=0.18,
        image_cost=0.0,
        input_tokens=800,
        output_tokens=210,
        daily_total=0.91,
        maps_grounding_calls=2,
        text_generated=True,
    )

    assert embed.footer.text is not None
    assert "maps ×2" in embed.footer.text


def test_accumulate_cost_persists_tool_metrics(gemini_cog):
    today = time.strftime("%Y-%m-%d")
    key = f"agent:{gemini_cog.agent_redis_name}:cost:{today}"
    pipeline = _FakePipeline(result=4.2)
    gemini_cog._redis = MagicMock()
    gemini_cog._redis.pipeline.return_value = pipeline

    daily_total = asyncio.run(
        gemini_cog._accumulate_cost(
            ai_cost=0.4,
            image_cost=0.1,
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=25,
            image_generated=True,
            web_search_calls=3,
            maps_grounding_calls=2,
        )
    )

    assert daily_total == pytest.approx(4.2)
    assert ("hincrby", (key, "web_search_calls", 3)) in pipeline.calls
    assert ("hincrby", (key, "maps_grounding_calls", 2)) in pipeline.calls
    assert ("hincrby", (key, "image_calls", 1)) in pipeline.calls
