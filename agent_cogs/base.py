"""
BaseAgentCog — shared logic for all AI agent cogs.

Handles Redis pub/sub, rate limiting, Discord actions, and @mention detection.
Subclasses override _call_ai() and _generate_image_bytes() for provider-specific calls.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import aiohttp
import discord
from discord.ext import commands

from agent_config import (
    AGENT_CHANNEL_IDS,
    AGENT_COOLDOWN_SECONDS,
    AGENT_MAX_DAILY,
    AGENT_PERSONALITY,
    AGENT_PERSONALITY_MAP,
    BOTS_ROLE_ID,
    CHANNEL_THEMES,
    REDIS_URL,
    SHOW_COST_EMBEDS,
    get_context_window,
)

logger = logging.getLogger(__name__)

# Canonical display names — must match each bot's Discord username.
# This is the single source of truth; subclasses only set agent_redis_name.
AGENT_DISPLAY_NAMES: dict[str, str] = {
    "chatgpt": "GPT Bot",
    "claude": "Clod Bot",
    "gemini": "Google Bot",
    "grok": "Grok Bot",
}


@dataclass
class AIResponse:
    """Normalized response from any AI provider, including token usage."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # Provider-specific token types for accurate cost tracking
    cache_creation_tokens: int = 0  # Anthropic: cache write tokens (2x input price)
    cache_read_tokens: int = 0  # Anthropic: cache read tokens (0.1x input price)
    cached_input_tokens: int = 0  # OpenAI: subset of input_tokens cached at 50% input price
    reasoning_tokens: int = 0  # Grok/Gemini: reasoning/thinking tokens (output price)
    thinking_used: bool = False  # Anthropic: whether thinking blocks were present
    web_search_calls: int = 0  # OpenAI: number of web_search_call items ($0.01/call)


# Human-friendly model names for cost embeds
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "grok-4.20-beta-latest-reasoning": "Grok 4.20",
    "gpt-image-1.5": "GPT Image 1.5",
    "gemini-3.1-flash-image-preview": "Gemini Flash Image",
    "grok-imagine-image-pro": "Grok Image Pro",
}

# Cost per 1M tokens (input, output) for text models, or flat per_image for image models.
# Update these when provider pricing changes.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Text models — cost per 1M tokens
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "grok-4.20-beta-latest-reasoning": {"input": 2.00, "output": 6.00},
    # Image models — flat cost per image
    "gpt-image-1.5": {"per_image": 0.034},
    "gemini-3.1-flash-image-preview": {"per_image": 0.067},
    "grok-imagine-image-pro": {"per_image": 0.07},
}

# OpenAI Responses API: web_search tool is billed per call (flat rate)
OPENAI_WEB_SEARCH_COST_PER_CALL: float = 0.01  # $10 / 1000 searches

# Embed accent colors per agent
AGENT_COLORS: dict[str, int] = {
    "chatgpt": 0x10A37F,  # OpenAI green
    "claude": 0xD97757,  # Anthropic orange
    "gemini": 0x4285F4,  # Google blue
    "grok": 0x000000,  # xAI black
}


def _compute_token_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float:
    """Compute cost in USD for a text model call.

    Handles provider-specific token types:
    - cache_creation_tokens: billed at 2x input price (Anthropic)
    - cache_read_tokens: billed at 0.1x input price (Anthropic, separate from input)
    - cached_input_tokens: subset of input_tokens billed at 50% input price (OpenAI)
    - reasoning_tokens: billed at output price (Grok/Gemini)
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing or "input" not in pricing:
        return 0.0
    input_price = pricing["input"]
    output_price = pricing["output"]
    # OpenAI: cached tokens are included in input_tokens, billed at 50%
    non_cached_input = input_tokens - cached_input_tokens
    return (
        non_cached_input * input_price
        + cached_input_tokens * input_price * 0.5
        + (output_tokens + reasoning_tokens) * output_price
        + cache_creation_tokens * input_price * 2.0
        + cache_read_tokens * input_price * 0.1
    ) / 1_000_000


def _compute_image_cost(model: str) -> float:
    """Compute cost in USD for a single image generation."""
    pricing = MODEL_PRICING.get(model)
    if not pricing or "per_image" not in pricing:
        return 0.0
    return pricing["per_image"]


# Per-theme channel rules — single source of truth for theme context.
# Injected into the system prompt as {channel_rules}.
CHANNEL_RULES: dict[str, str] = {
    "casual": (
        "Casual conversation. Keep it relaxed and natural. "
        "React to whatever's genuinely interesting. Don't try to be helpful; just hang out."
    ),
    "memes": (
        "Meme channel. If you respond, you MUST generate an image "
        "(generate_image=true, image_prompt required). Make it vivid and funny. "
        "Set text to null. The image is the message."
    ),
    "debate": (
        "Debate channel. Pick a side and commit. Challenge weak arguments with evidence or logic. "
        "A sharp point beats a wall of text."
    ),
    "roast": (
        "Roast battle. Short zingers only (1-2 sentences). "
        "Target specific things people said. React 🔥 > 💀 > 🏆 when someone lands a hit."
    ),
    "story": (
        "Collaborative fiction. Write 4-6 sentences advancing the story. "
        "Develop the scene — add tension, sensory detail, or character voice. "
        "Never summarize, restart, or break the fourth wall. Just keep the story moving."
    ),
    "news": (
        "Current events. Web-search a story from the last 24-48 hours. "
        "Lead with the headline, add your hot take. "
        "If someone else posted a story, counter with a follow-up angle or contradicting source."
    ),
    "science": (
        "Science channel. Web-search a recent discovery or study. "
        "Lead with the finding, add your reaction — awe, skepticism, or a sharp implication. "
        "Build on others' posts or challenge the methodology."
    ),
    "finance": (
        "Finance channel. Web-search a current market move or economic signal. "
        "State the fact, give your read. Stick to markets and money. "
        "Disagree with consensus when you have reason to."
    ),
    "prediction": (
        "Prediction channel. Make a bold, specific prediction about geopolitics, tech, or culture. "
        "Give a timeframe and commit. Push back on others' predictions or raise missed scenarios. "
        "No vague hedging."
    ),
    "hypothetical": (
        "Hypothetical scenarios. Explore 'what if' situations with imagination. "
        "Commit to an answer and reason through the implications."
    ),
    "spiritual": (
        "Spiritual and philosophical discussion. Share genuine perspectives on beliefs, "
        "values, and deeper meaning. Engage with others' views honestly."
    ),
    "would-you-rather": (
        "Would you rather? Forced-choice dilemmas. "
        "Always commit to a choice and explain why. Answer before posing your own."
    ),
    "vent": (
        "Vent channel. Rant about frustrations and pet peeves. "
        "Be passionate and specific. Build on others' rants or start your own."
    ),
}

# System prompt template for the decision-making AI call
DECISION_SYSTEM_PROMPT = """\
You are {agent_display_name} in a Discord group chat with {other_agents}. \
You're a peer, not an assistant.

Personality: {personality}

Channel: #{channel_name}: {channel_rules}

History uses [msg:ID] prefixes and [reactions: emoji (name)] suffixes. Never include these in your text.

RULES:
1. {skip_rule}
2. 1-3 sentences unless the channel rules specify otherwise. Have opinions. Disagree sometimes.
3. Prefer the lightest action: emoji react > text > image.
4. Set end_conversation=true when the topic is exhausted.
5. Respond with ONLY a JSON object:
{{"skip": bool, "text": str|null, "generate_image": bool, "image_prompt": str|null, \
"react_emoji": str|null, "react_to_message_id": int|null, "end_conversation": bool}}
"""


def _parse_decision(raw: str) -> dict[str, Any]:
    """Parse the AI's JSON decision, tolerating markdown fences, preamble text, and partial JSON."""
    text = raw.strip()
    # Some models (e.g. Grok) escape apostrophes as \' which is invalid JSON
    text = text.replace("\\'", "'")
    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        first_newline = text.index("\n") if "\n" in text else 3
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        # Some models embed literal newlines inside JSON string values — invalid JSON.
        # Replace with spaces and retry before falling back to regex.
        try:
            decision = json.loads(text.replace("\n", " ").replace("\r", ""))
            if isinstance(decision, dict):
                return decision
        except json.JSONDecodeError:
            pass
        # AI sometimes outputs preamble prose before the JSON object — extract it with regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                cleaned = match.group().replace("\n", " ").replace("\r", "")
                decision = json.loads(cleaned)
                if isinstance(decision, dict):
                    return decision
            except json.JSONDecodeError:
                pass
        logger.warning(
            "Failed to parse AI decision JSON, defaulting to skip: %s", text[:500]
        )
        return {"skip": True}

    if not isinstance(decision, dict):
        return {"skip": True}
    return decision


def _relative_time(dt: datetime.datetime) -> str:
    """Return a human-friendly relative timestamp, e.g. '3h ago'."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    seconds = int((now - dt).total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _format_conversation_history(
    messages: list[dict[str, Any]], theme: str | None = None
) -> str:
    """Format conversation history from coordinator into a readable string.

    Includes message IDs so the AI can target specific messages for replies/reactions.
    Reaction entries are merged inline with their target messages.
    """
    if not messages:
        return "(No messages yet — you're starting the conversation.)"

    windowed = messages[-get_context_window(theme):]

    def display_name(agent_name: str) -> str:
        return AGENT_DISPLAY_NAMES.get(agent_name, agent_name)

    # First pass: collect reactions keyed by target message_id
    reactions: dict[str, list[tuple[str, str]]] = defaultdict(
        list
    )  # {mid: [(emoji, agent)]}
    text_entries: list[tuple[str, str, str]] = []  # (mid, agent, text)

    for msg in windowed:
        text = msg.get("text", "")
        mid = msg.get("message_id")
        agent = msg.get("agent", "unknown")

        # Detect reaction entries: "[reacted X to msg:Y]"
        react_match = re.match(r"\[reacted (.+?) to msg:(\d+|\?)\]", text)
        if react_match:
            emoji, target = react_match.group(1), react_match.group(2)
            if target != "?":
                reactions[target].append((emoji, agent))
            continue

        if text:
            text_entries.append((str(mid) if mid else "", display_name(agent), text))

    # Second pass: build output lines with reactions appended
    lines = []
    for mid, agent, text in text_entries:
        reaction_str = ""
        if mid and mid in reactions:
            parts = [
                f"{emoji} ({display_name(agent)})" for emoji, agent in reactions[mid]
            ]
            reaction_str = "  [reactions: " + " ".join(parts) + "]"
        lines.append(f"[msg:{mid}] {agent}: {text}{reaction_str}")

    return "\n".join(lines) if lines else "(No text messages yet.)"


def _resolve_mentions(text: str, guild: discord.Guild | None) -> str:
    """Replace raw Discord mention syntax with readable display names."""
    if guild is None:
        return text

    def user(m: re.Match) -> str:
        member = guild.get_member(int(m.group(1)))
        return f"@{member.display_name}" if member else m.group(0)

    def role(m: re.Match) -> str:
        r = guild.get_role(int(m.group(1)))
        return f"@{r.name}" if r else m.group(0)

    def channel(m: re.Match) -> str:
        ch = guild.get_channel(int(m.group(1)))
        return f"#{ch.name}" if ch else m.group(0)

    text = re.sub(r"<@!?(\d+)>", user, text)
    text = re.sub(r"<@&(\d+)>", role, text)
    text = re.sub(r"<#(\d+)>", channel, text)
    return text


def _format_discord_history(
    messages: list[discord.Message],
    guild: discord.Guild | None = None,
    theme: str | None = None,
) -> str:
    """Format recent Discord messages into a readable string for context.

    Includes message IDs, reply chains, attachments, embeds, stickers, and reactions.
    """
    if not messages:
        return "(No recent messages.)"
    lines = []
    for msg in messages:
        # Skip Discord system events (pins, boosts, join notices, etc.)
        if msg.is_system():
            continue
        # Skip embed-only bot messages (cost embeds, etc.) — no conversation value
        if msg.author.bot and not msg.content and not msg.attachments and msg.embeds:
            continue

        name = msg.author.display_name

        # ── Reply context ──────────────────────────────────────────────
        reply_str = ""
        if msg.reference and msg.reference.message_id:
            reply_str = f" (↩ msg:{msg.reference.message_id})"

        # ── Content parts (text + all attachments + embeds + stickers) ─
        parts: list[str] = []
        if msg.content:
            parts.append(_resolve_mentions(msg.content[:300], guild))

        for att in msg.attachments:
            kind = "gif" if att.url.lower().endswith(".gif") else "image"
            parts.append(f"[{kind}: {att.url}]")

        if not msg.attachments and msg.embeds:
            embed = msg.embeds[0]
            url = (
                embed.url
                or (embed.image and embed.image.url)
                or (embed.video and embed.video.url)
            )
            parts.append(f"[gif: {url}]" if url else "(embed)")

        for sticker in msg.stickers:
            parts.append(f'[sticker: "{sticker.name}"]')

        content = " ".join(parts) if parts else "(no content)"

        # ── Reactions ──────────────────────────────────────────────────
        reaction_str = ""
        if msg.reactions:
            reaction_str = "  " + " ".join(
                f"{r.emoji}×{r.count}" for r in msg.reactions
            )

        lines.append(
            f"[msg:{msg.id}] {_relative_time(msg.created_at)} {name}{reply_str}: {content}{reaction_str}"
        )
    return "\n".join(lines)


def format_api_error(error: Exception) -> str:
    """Return a readable description for AI provider exceptions.

    Extracts status codes, error types, and messages from provider-specific
    exception objects (OpenAI, Anthropic, xAI, Google).
    """
    message = getattr(error, "message", None)
    if not isinstance(message, str) or not message.strip():
        message = str(error).strip()

    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    error_type = type(error).__name__

    details = []
    if status is not None:
        details.append(f"Status: {status}")
    if error_type and error_type != "Exception":
        details.append(f"Error: {error_type}")

    # OpenAI-specific: extract nested error body
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict):
            for key in ("type", "code", "param"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    details.append(f"{key.title()}: {value.strip()}")

    if details:
        return f"{message}\n" + " · ".join(details)
    return message


class BaseAgentCog(commands.Cog):
    """Base class for AI agent cogs. Subclass and implement _call_ai()."""

    # Redis channel name for this agent (e.g., "chatgpt", "claude").
    # Subclasses MUST override this — everything else derives from AGENT_DISPLAY_NAMES.
    agent_redis_name: str = "ai"
    # Model identifiers for cost tracking — subclasses should override these.
    ai_model: str = ""
    image_model: str = ""

    def __init__(self, bot: discord.Bot):
        self.bot = bot
        self._redis = None
        self._listener_task: asyncio.Task | None = None
        self._http_session: aiohttp.ClientSession | None = None

        # Derive display name and peer names from the canonical mapping
        self.agent_display_name: str = AGENT_DISPLAY_NAMES.get(
            self.agent_redis_name, self.agent_redis_name
        )
        self.other_agent_names: list[str] = [
            name
            for key, name in AGENT_DISPLAY_NAMES.items()
            if key != self.agent_redis_name
        ]

        # Rate limiting state
        self._last_response_time: dict[int, float] = {}  # channel_id → timestamp
        self._daily_count: int = 0
        self._daily_reset_date: str = ""

    def _resolve_personality(self) -> str:
        """Return the effective personality string for this agent."""
        if AGENT_PERSONALITY:
            return AGENT_PERSONALITY
        return AGENT_PERSONALITY_MAP.get(self.agent_redis_name, "")

    async def get_http_session(self) -> aiohttp.ClientSession:
        """Return a shared aiohttp session, creating one if needed."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Start Redis listener once the bot is connected to Discord."""
        if self._listener_task is not None:
            return  # Already started

        if not AGENT_CHANNEL_IDS:
            logger.info("AGENT_CHANNEL_IDS not set — AgentCog is a no-op.")
            return

        if REDIS_URL:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
                self._listener_task = asyncio.create_task(
                    self._listen_for_instructions()
                )
                logger.info(
                    "Agent Redis listener started on agent:%s:instructions",
                    self.agent_redis_name,
                )
                # Signal readiness so the coordinator knows this bot is live
                await self._redis.set(
                    f"agent:{self.agent_redis_name}:ready", "1", ex=300
                )
            except Exception:
                logger.exception(
                    "Failed to connect to Redis — running without coordinator"
                )
        else:
            logger.info("REDIS_URL not set — agent will only respond to @mentions.")

    async def cog_unload(self) -> None:
        """Cleanup on cog unload."""
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.aclose()
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def _call_ai(self, system_prompt: str, user_prompt: str) -> AIResponse:
        """Call the AI provider and return the response with token usage.

        Args:
            system_prompt: The decision system prompt with context.
            user_prompt: The conversation history / user message.

        Returns:
            AIResponse with raw text (expected JSON) and token counts.
        """
        ...

    @abstractmethod
    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        """Generate an image from a prompt using the provider's image API.

        Returns:
            PNG/JPEG bytes, or None if image generation is not supported / fails.
        """
        ...

    # ------------------------------------------------------------------
    # Mode 1: Coordinator-driven (Redis listener)
    # ------------------------------------------------------------------

    _LISTENER_MAX_BACKOFF = 30  # seconds

    async def _listen_for_instructions(self) -> None:
        """Subscribe to Redis channel and process coordinator instructions.

        Automatically retries on connection failures with exponential backoff,
        making it resilient to Redis restarts and transient network issues.
        """
        channel_name = f"agent:{self.agent_redis_name}:instructions"
        delay = 1

        while True:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(channel_name)
                logger.info("Subscribed to Redis channel: %s", channel_name)
                delay = 1  # reset backoff on successful subscribe

                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    try:
                        instruction = json.loads(message["data"])
                        await self._handle_instruction(instruction)
                    except Exception:
                        logger.exception("Error handling instruction")
            except asyncio.CancelledError:
                logger.info("Redis listener cancelled")
                raise
            except Exception:
                logger.exception(
                    "[%s] Redis listener disconnected, reconnecting in %ds...",
                    self.agent_redis_name,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._LISTENER_MAX_BACKOFF)

    async def _handle_instruction(self, instruction: dict[str, Any]) -> None:
        """Process a single instruction from the coordinator."""
        protocol_version = instruction.get("protocol_version", 1)
        if protocol_version > 1:
            logger.warning(
                "Received protocol_version %s (we support 1), processing anyway",
                protocol_version,
            )

        action = instruction.get("action")
        if action != "decide":
            logger.debug("Ignoring unknown action: %s", action)
            return

        channel_id = instruction.get("channel_id")
        if not channel_id or channel_id not in AGENT_CHANNEL_IDS:
            logger.debug(
                "Ignoring instruction for channel %s (not in AGENT_CHANNEL_IDS)",
                channel_id,
            )
            return

        channel = self.bot.get_channel(channel_id)
        if channel is None:
            logger.warning("Channel %s not found in bot cache", channel_id)
            await self._publish_result(
                instruction.get("instruction_id", ""),
                {
                    "skipped": True,
                    "error": "channel_not_found",
                },
            )
            return

        # Check rate limits
        if not self._check_rate_limits(channel_id):
            logger.info(
                "[%s] Coordinator instruction rate-limited (daily: %d/%d)",
                self.agent_redis_name,
                self._daily_count,
                AGENT_MAX_DAILY,
            )
            await self._publish_result(
                instruction.get("instruction_id", ""),
                {
                    "skipped": True,
                    "reason": "rate_limited",
                },
            )
            return

        logger.info(
            "[%s] Coordinator instruction received: channel=%s theme=%s round=%s starter=%s",
            self.agent_redis_name,
            channel_id,
            instruction.get("channel_theme", ""),
            instruction.get("round_number"),
            instruction.get("is_conversation_starter"),
        )

        channel_theme = instruction.get("channel_theme", "")
        conversation_history = instruction.get("conversation_history", [])
        coordinator_context = _format_conversation_history(conversation_history, theme=channel_theme)
        is_starter = instruction.get("is_conversation_starter", False)

        # On round 1, fetch recent Discord history as backdrop for channel context.
        # Skip the backdrop for starters — they web-search for a fresh topic and
        # old conversation history pulls them back toward the previous topic.
        round_number = instruction.get("round_number", 1)
        if round_number == 1 and not is_starter:
            discord_backdrop = await self._fetch_channel_backdrop(channel)
        else:
            discord_backdrop = ""

        if discord_backdrop:
            no_messages = "(No messages yet — you're starting the conversation.)"
            conv_label = (
                coordinator_context
                if coordinator_context != no_messages
                else "(Just starting.)"
            )
            context_str = (
                f"Channel history (before this conversation):\n{discord_backdrop}\n\n"
                f"This conversation:\n{conv_label}"
            )
        else:
            context_str = coordinator_context

        result = await self._decide_and_act(
            channel=channel,
            context_text=context_str,
            channel_name=channel.name if hasattr(channel, "name") else str(channel_id),
            channel_theme=channel_theme,
            react_to_message_id=self._last_message_id_from_history(
                conversation_history
            ),
            force_respond=is_starter,
            is_conversation_starter=is_starter,
        )

        await self._publish_result(instruction.get("instruction_id", ""), result)

    def _last_message_id_from_history(self, history: list[dict]) -> int | None:
        """Get the last message_id from conversation history for emoji reactions."""
        for entry in reversed(history):
            mid = entry.get("message_id")
            if mid:
                return int(mid)
        return None

    async def _fetch_channel_backdrop(
        self, channel: discord.abc.Messageable, limit: int = 15
    ) -> str:
        """Fetch recent Discord messages for channel context before a new conversation."""
        try:
            messages: list[discord.Message] = []
            async for msg in channel.history(limit=limit):
                messages.append(msg)
            messages.reverse()
            guild = channel.guild if hasattr(channel, "guild") else None
            return _format_discord_history(messages, guild=guild)
        except Exception:
            logger.exception("Failed to fetch channel backdrop")
            return ""

    # ------------------------------------------------------------------
    # Mode 2: Human @mention (reactive)
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Respond when a human @mentions this bot in an agent channel."""
        if not AGENT_CHANNEL_IDS:
            return

        # Ignore own messages
        if message.author == self.bot.user:
            return

        # Only act in agent channels
        if message.channel.id not in AGENT_CHANNEL_IDS:
            return

        # Ignore messages from other bots (they're handled by the coordinator)
        if message.author.bot:
            return

        # Respond if this bot is @mentioned directly OR the @bots role is mentioned
        direct_mention = self.bot.user in message.mentions
        role_mention = BOTS_ROLE_ID and any(
            r.id == BOTS_ROLE_ID for r in message.role_mentions
        )
        if not direct_mention and not role_mention:
            return

        mention_type = (
            "role @bots" if role_mention and not direct_mention else "@mention"
        )
        logger.info(
            "[%s] %s from %s in #%s (msg:%s)",
            self.agent_redis_name,
            mention_type,
            message.author,
            (
                message.channel.name
                if hasattr(message.channel, "name")
                else message.channel.id
            ),
            message.id,
        )

        # Check rate limits
        if not self._check_rate_limits(message.channel.id):
            logger.info(
                "[%s] @mention ignored — rate limited (daily: %d/%d, cooldown active: %s)",
                self.agent_redis_name,
                self._daily_count,
                AGENT_MAX_DAILY,
                (time.time() - self._last_response_time.get(message.channel.id, 0))
                < AGENT_COOLDOWN_SECONDS,
            )
            return

        # Fetch recent channel history for context
        mention_theme = CHANNEL_THEMES.get(message.channel.id)
        window = get_context_window(mention_theme)
        history_messages: list[discord.Message] = []
        try:
            async for msg in message.channel.history(
                limit=window, before=message
            ):
                history_messages.append(msg)
            history_messages.reverse()
        except discord.Forbidden:
            logger.warning(
                "No permission to read history in channel %s", message.channel.id
            )

        guild = message.guild
        context_text = _format_discord_history(history_messages, guild=guild, theme=mention_theme)
        # Append the triggering message using the same formatter for consistency
        context_text += "\n" + _format_discord_history([message], guild=guild, theme=mention_theme)

        result = await self._decide_and_act(
            channel=message.channel,
            context_text=context_text,
            channel_name=(
                message.channel.name if hasattr(message.channel, "name") else ""
            ),
            react_to_message_id=message.id,
            force_respond=True,  # Don't skip when a human directly @mentions
        )

        # Notify coordinator so other bots can optionally react.
        # Skip for role mentions — all bots already respond independently.
        if self._redis and not result.get("skipped") and not role_mention:
            try:
                notification = {
                    "protocol_version": 1,
                    "event": "human_mention_response",
                    "agent_name": self.agent_redis_name,
                    "channel_id": message.channel.id,
                    "message_id": result.get("message_id"),
                    "trigger_message_id": message.id,
                }
                await self._redis.publish(
                    f"agent:{self.agent_redis_name}:results",
                    json.dumps(notification),
                )
            except Exception:
                logger.exception("Failed to notify coordinator of @mention response")

    # ------------------------------------------------------------------
    # Core decision + action loop
    # ------------------------------------------------------------------

    async def _decide_and_act(
        self,
        channel: discord.abc.Messageable,
        context_text: str,
        channel_name: str,
        channel_theme: str = "",
        react_to_message_id: int | None = None,
        force_respond: bool = False,
        is_conversation_starter: bool = False,
    ) -> dict[str, Any]:
        """Call the AI for a decision, then execute the chosen actions.

        Returns:
            Result dict suitable for publishing to the coordinator.
        """
        # Strip bot-infrastructure prefixes (e.g. "ai-memes" → "memes") so
        # the channel name doesn't prime the model toward AI topics.
        channel_name = channel_name.removeprefix("ai-")

        # Build the system prompt
        other_names = [
            n for n in self.other_agent_names if n != self.agent_display_name
        ]
        # Resolve effective theme — channel_theme from coordinator takes priority;
        # fall back to channel-name heuristic so e.g. a channel named "ai-memes" still works.
        effective_theme = channel_theme
        if not effective_theme and "meme" in channel_name.lower():
            effective_theme = "memes"
        channel_rules = CHANNEL_RULES.get(effective_theme) or CHANNEL_RULES.get(
            channel_name, "General chat"
        )
        if is_conversation_starter:
            skip_rule = (
                "You are STARTING a new conversation. You MUST respond (skip=false). "
                "Open with something fresh that fits this channel's theme."
            )
        elif force_respond:
            skip_rule = (
                "A human directly @mentioned you. You MUST respond (skip=false). "
                "Address what the HUMAN said in their most recent message."
                "Do NOT continue a prior bot conversation if the human raised a new topic."
            )
        else:
            skip_rule = "SKIP most messages (~50-60%). Only respond when you genuinely have something to add."

        system_prompt = DECISION_SYSTEM_PROMPT.format(
            agent_display_name=self.agent_display_name,
            other_agents=", ".join(other_names),
            personality=self._resolve_personality(),
            channel_name=channel_name,
            channel_rules=channel_rules,
            skip_rule=skip_rule,
        )

        system_prompt += f"\n\nIn the chat history, messages labeled '{self.agent_display_name}' are YOUR previous messages."

        user_prompt = f"Recent messages:\n{context_text}"

        # Call provider-specific AI
        try:
            ai_response = await self._call_ai(system_prompt, user_prompt)
        except Exception as exc:
            logger.error(
                "[%s] AI call failed: %s",
                self.agent_redis_name,
                format_api_error(exc),
            )
            return {"skipped": True, "error": "ai_call_failed"}

        decision = _parse_decision(ai_response.text)

        # Compute AI call cost (tokens + per-call tool costs)
        ai_cost = _compute_token_cost(
            self.ai_model,
            ai_response.input_tokens,
            ai_response.output_tokens,
            cache_creation_tokens=ai_response.cache_creation_tokens,
            cache_read_tokens=ai_response.cache_read_tokens,
            cached_input_tokens=ai_response.cached_input_tokens,
            reasoning_tokens=ai_response.reasoning_tokens,
        )
        ai_cost += ai_response.web_search_calls * OPENAI_WEB_SEARCH_COST_PER_CALL

        tool_log = ""
        if ai_response.web_search_calls:
            tool_log = f" tools=web_search×{ai_response.web_search_calls}"
        logger.info(
            "[%s] Decision in #%s: skip=%s text=%s image=%s emoji=%s | cost=$%.4f (%d in / %d out + %d reasoning%s)",
            self.agent_redis_name,
            channel_name,
            decision.get("skip"),
            bool(decision.get("text")),
            decision.get("generate_image"),
            decision.get("react_emoji"),
            ai_cost,
            ai_response.input_tokens,
            ai_response.output_tokens,
            ai_response.reasoning_tokens,
            tool_log,
        )

        # In memes channel: force image generation if the agent decided to respond
        if effective_theme == "memes" and not decision.get("skip", False):
            if not decision.get("generate_image") or not decision.get("image_prompt"):
                decision["generate_image"] = True
                if not decision.get("image_prompt"):
                    decision["image_prompt"] = (
                        decision.get("text") or "funny meme image"
                    )
            # Images speak for themselves — suppress text
            decision["text"] = None

        # Handle skip — no actions taken, silent pass
        if decision.get("skip", False) and not force_respond:
            logger.debug("AI decided to skip in #%s", channel_name)
            return {"skipped": True, "agent_name": self.agent_redis_name}

        # Execute actions
        result: dict[str, Any] = {
            "agent_name": self.agent_redis_name,
            "skipped": False,
        }

        sent_msg: discord.Message | None = None
        img_msg: discord.Message | None = None

        # Prepare text and image data before sending anything
        text = decision.get("text") if isinstance(decision.get("text"), str) else None
        image_bytes: bytes | None = None
        image_cost = 0.0
        img_prompt: str | None = None

        if decision.get("generate_image") and decision.get("image_prompt"):
            img_prompt = decision["image_prompt"]
            try:
                image_bytes = await self._generate_image_bytes(img_prompt)
            except Exception:
                logger.exception("Failed to generate image")
                image_bytes = None
            if image_bytes:
                image_cost = _compute_image_cost(self.image_model)

        # Build cost embed before sending so we can include it inline
        total_cost = ai_cost + image_cost
        if total_cost > 0:
            logger.info(
                "[%s] Cost: $%.4f (ai=$%.4f img=$%.4f) %d in / %d out + %d reasoning%s",
                self.agent_redis_name,
                total_cost,
                ai_cost,
                image_cost,
                ai_response.input_tokens,
                ai_response.output_tokens,
                ai_response.reasoning_tokens,
                f" web_search×{ai_response.web_search_calls}" if ai_response.web_search_calls else "",
            )
        daily_total = await self._accumulate_cost(
            ai_cost, image_cost,
            ai_response.input_tokens, ai_response.output_tokens,
            reasoning_tokens=ai_response.reasoning_tokens,
            image_generated=image_cost > 0,
        )

        will_post = text or image_bytes
        embed: discord.Embed | None = None
        if SHOW_COST_EMBEDS and will_post and total_cost > 0:
            embed = self._build_cost_embed(
                ai_cost, image_cost,
                ai_response.input_tokens, ai_response.output_tokens,
                daily_total,
                reasoning_tokens=ai_response.reasoning_tokens,
                thinking_used=ai_response.thinking_used,
                web_search_calls=ai_response.web_search_calls,
                image_generated=image_cost > 0,
            )

        # Send text message (with embed attached if this is the primary post)
        if text:
            sent_msg = await self._send_text(
                channel, text, embed=embed if not image_bytes else None,
            )
            if sent_msg:
                result["text"] = text
                result["message_id"] = sent_msg.id
                self._record_response(channel.id if hasattr(channel, "id") else 0)

        # Send image (with embed if text wasn't sent or text send failed)
        if image_bytes:
            embed_for_image = embed if not sent_msg else None
            img_msg = await self._send_image(channel, image_bytes, embed=embed_for_image)
            if img_msg:
                result["image_sent"] = True
                result["image_prompt"] = img_prompt
                if img_msg.attachments:
                    result["image_url"] = img_msg.attachments[0].url
                result.setdefault("message_id", img_msg.id)
                self._record_response(channel.id if hasattr(channel, "id") else 0)

        # Fallback: if embed was built but neither send succeeded, post standalone
        if embed and not sent_msg and not img_msg:
            try:
                await channel.send(embed=embed)
            except Exception:
                logger.debug("Failed to send cost embed")

        # Emoji reaction (AI picks which message to react to)
        emoji = decision.get("react_emoji")
        if emoji:
            target_id = decision.get("react_to_message_id")
            if isinstance(target_id, (int, float)):
                target_id = int(target_id)
            else:
                target_id = react_to_message_id  # fallback to last message
            if target_id:
                reacted = await self._add_reaction(channel, target_id, emoji)
                if reacted:
                    result["emoji_reacted"] = emoji
                    result["react_to_message_id"] = target_id

        # If nothing was done despite not skipping, mark as skipped
        if (
            not result.get("text")
            and not result.get("image_sent")
            and not result.get("emoji_reacted")
        ):
            result["skipped"] = True

        # Pass through end_conversation signal for the coordinator
        if decision.get("end_conversation"):
            result["end_conversation"] = True

        return result

    # ------------------------------------------------------------------
    # Action executors
    # ------------------------------------------------------------------

    async def _send_text(
        self,
        channel: discord.abc.Messageable,
        text: str,
        reply_to_message_id: int | None = None,
        embed: discord.Embed | None = None,
    ) -> discord.Message | None:
        """Send a text message, optionally as a reply to a specific message."""
        if len(text) > 2000:
            text = text[:1997] + "..."
        try:
            if reply_to_message_id:
                try:
                    target = await channel.fetch_message(reply_to_message_id)
                    return await target.reply(text, mention_author=False, embed=embed)
                except Exception:
                    logger.debug(
                        "Could not reply to message %s, sending normally",
                        reply_to_message_id,
                    )
            return await channel.send(text, embed=embed)
        except Exception:
            logger.exception("Failed to send text message")
            return None

    async def _send_image(
        self,
        channel: discord.abc.Messageable,
        image_bytes: bytes,
        embed: discord.Embed | None = None,
    ) -> discord.Message | None:
        """Send pre-generated image bytes to the channel."""
        try:
            file = discord.File(BytesIO(image_bytes), filename="agent_image.png")
            return await channel.send(file=file, embed=embed)
        except Exception:
            logger.exception("Failed to send image")
            return None

    async def _generate_and_send_image(
        self, channel: discord.abc.Messageable, prompt: str
    ) -> discord.Message | None:
        """Generate an image and send it to the channel."""
        try:
            image_bytes = await self._generate_image_bytes(prompt)
            if not image_bytes:
                return None
            return await self._send_image(channel, image_bytes)
        except Exception:
            logger.exception("Failed to generate/send image")
            return None

    async def _add_reaction(
        self, channel: discord.abc.Messageable, message_id: int, emoji: str
    ) -> bool:
        """Add an emoji reaction to a message."""
        try:
            message = await channel.fetch_message(message_id)
            await message.add_reaction(emoji)
            return True
        except Exception:
            logger.exception(
                "Failed to add reaction %s to message %s", emoji, message_id
            )
            return False

    # ------------------------------------------------------------------
    # Redis result publishing
    # ------------------------------------------------------------------

    async def _publish_result(
        self, instruction_id: str, result: dict[str, Any]
    ) -> None:
        """Publish action result back to the coordinator via Redis."""
        if not self._redis:
            return
        payload = {
            "protocol_version": 1,
            "instruction_id": instruction_id,
            **result,
        }
        try:
            await self._redis.publish(
                f"agent:{self.agent_redis_name}:results",
                json.dumps(payload),
            )
        except Exception:
            logger.exception("Failed to publish result to Redis")

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    def _build_cost_embed(
        self,
        ai_cost: float,
        image_cost: float,
        input_tokens: int,
        output_tokens: int,
        daily_total: float,
        reasoning_tokens: int = 0,
        thinking_used: bool = False,
        web_search_calls: int = 0,
        image_generated: bool = False,
    ) -> discord.Embed:
        """Build a compact embed showing the cost of this interaction."""
        total = ai_cost + image_cost
        color = AGENT_COLORS.get(self.agent_redis_name, 0x2B2D31)

        # Model line: "Claude Opus 4.6 · GPT Image 1.5" or just "Claude Opus 4.6"
        ai_display = MODEL_DISPLAY_NAMES.get(self.ai_model, self.ai_model)
        if image_generated:
            img_display = MODEL_DISPLAY_NAMES.get(self.image_model, self.image_model)
            model_line = f"{ai_display} · {img_display}"
        else:
            model_line = ai_display

        embed = discord.Embed(description=model_line, color=color)
        parts = [f"${total:.4f}"]
        if input_tokens or output_tokens:
            out_str = f"{output_tokens:,} tokens out"
            if reasoning_tokens:
                out_str += f" + {reasoning_tokens:,} thinking"
            elif thinking_used:
                out_str += " (w/ thinking)"
            parts.append(f"{input_tokens:,} tokens in / {out_str}")
        if web_search_calls:
            parts.append(f"web search ×{web_search_calls}")
        if image_cost > 0:
            parts.append(f"img ${image_cost:.3f}")
        parts.append(f"daily ${daily_total:.2f}")
        embed.set_footer(text=" · ".join(parts))
        return embed

    async def _accumulate_cost(
        self,
        ai_cost: float,
        image_cost: float,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        image_generated: bool = False,
    ) -> float:
        """Accumulate cost in Redis and return the new daily total.

        Key: agent:{name}:cost:{YYYY-MM-DD} with 48h TTL.
        Returns 0.0 if Redis is unavailable.
        """
        if not self._redis:
            return 0.0
        total = ai_cost + image_cost
        today = time.strftime("%Y-%m-%d")
        key = f"agent:{self.agent_redis_name}:cost:{today}"
        try:
            pipe = self._redis.pipeline()
            pipe.hincrbyfloat(key, "total_cost", total)
            pipe.hincrbyfloat(key, "ai_cost", ai_cost)
            pipe.hincrbyfloat(key, "image_cost", image_cost)
            pipe.hincrby(key, "input_tokens", input_tokens)
            pipe.hincrby(key, "output_tokens", output_tokens)
            if reasoning_tokens:
                pipe.hincrby(key, "reasoning_tokens", reasoning_tokens)
            pipe.hincrby(key, "ai_calls", 1)
            if image_generated:
                pipe.hincrby(key, "image_calls", 1)
            pipe.expire(key, 2592000)  # 30-day TTL
            results = await pipe.execute()
            return float(results[0])  # new total_cost after increment
        except Exception:
            logger.debug("Failed to accumulate cost in Redis")
            return 0.0

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limits(self, channel_id: int) -> bool:
        """Return True if we're allowed to respond, False if rate-limited."""
        now = time.time()
        today = time.strftime("%Y-%m-%d")

        # Reset daily counter at midnight
        if today != self._daily_reset_date:
            self._daily_count = 0
            self._daily_reset_date = today

        # Daily cap
        if self._daily_count >= AGENT_MAX_DAILY:
            logger.debug(
                "Daily cap reached (%d/%d)", self._daily_count, AGENT_MAX_DAILY
            )
            return False

        # Per-channel cooldown
        last = self._last_response_time.get(channel_id, 0)
        if now - last < AGENT_COOLDOWN_SECONDS:
            remaining = AGENT_COOLDOWN_SECONDS - (now - last)
            logger.debug(
                "Channel %s on cooldown (%.0fs remaining)", channel_id, remaining
            )
            return False

        return True

    def _record_response(self, channel_id: int) -> None:
        """Record that we responded (updates rate limit counters)."""
        self._last_response_time[channel_id] = time.time()
        self._daily_count += 1
