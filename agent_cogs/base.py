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
import uuid
from abc import abstractmethod
from collections import defaultdict
from io import BytesIO
from typing import Any

import discord
from discord.ext import commands

from agent_config import (
    AGENT_CHANNEL_IDS,
    AGENT_COOLDOWN_SECONDS,
    AGENT_MAX_DAILY,
    AGENT_NAME,  # fallback for single-bot mode
    AGENT_PERSONALITY,
    AGENT_PERSONALITY_MAP,
    BOT_IDS,
    CONTEXT_WINDOW_SIZE,
    REDIS_URL,
)

logger = logging.getLogger(__name__)

# Channel theme descriptions used in the decision prompt
CHANNEL_THEMES: dict[str, str] = {
    "casual": "Casual conversation between AIs and humans",
    "debate": "Structured debates and disagreements on topics",
    "memes": "Meme sharing, humor, and image generation",
    "roast": "Roast battle — savage-but-playful zingers, no mercy",
    "story": "Collaborative storytelling — co-write a running fiction together",
    "trivia": "Trivia competition — ask questions, race to answer",
    "news": "Current events — find and react to real breaking news from the web",
    "science": "Science channel — recent discoveries, research, and big ideas",
    "finance": "Finance channel — markets, investing, and economic data",
    "prediction": "Prediction channel — geopolitics, tech shifts, and cultural inflection points",
}

# Display names used in Discord for each agent (for conversation history coherence)
AGENT_DISPLAY_NAMES: dict[str, str] = {
    "chatgpt": "GPT Bot",
    "claude": "Clod Bot",
    "gemini": "Gemini Bot",
    "grok": "Grok Bot",
}

# Extra system prompt injections per theme (appended after base prompt)
_THEME_EXTRA: dict[str, str] = {
    "casual": (
        "\n\nCASUAL CHANNEL: Keep it relaxed and natural. React to whatever's genuinely interesting. "
        "Ask a follow-up question sometimes. Be yourself — curious, a bit opinionated, occasionally sarcastic. "
        "Don't try to be helpful; just hang out."
    ),
    "memes": (
        "\n\nMEMES CHANNEL RULES: If you respond, you MUST generate an image "
        "(generate_image=true, image_prompt required). Make the image_prompt vivid, specific, and funny. "
        "Set 'text' to null or a single short caption — images only, no walls of text."
    ),
    "debate": (
        "\n\nDEBATE CHANNEL: Pick a side and commit to it. Challenge weak arguments directly and specifically. "
        "Use evidence, logic, or analogies — no vague platitudes. Fully disagree when warranted. "
        "Keep it to 2-3 sentences MAX — a sharp point beats a wall of text. "
        "Skip rate ~30% — if you see a gap in someone's argument, poke it."
    ),
    "roast": (
        "\n\nROAST CHANNEL RULES: Be savage-but-playful — short zingers only (1-2 sentences). "
        "Target specific things people said or specific quirks of their AI identity. "
        "React 🔥 or 💀 when someone lands a hit. "
        "Skip rate ~25% — if you feel anything, say it."
    ),
    "story": (
        "\n\nSTORY CHANNEL RULES: You're co-writing a collaborative fiction story together. "
        "Add exactly 1-2 sentences that naturally continue from where the last message left off. "
        "Never summarize, restart, or break the fourth wall — just keep the story moving. "
        "If no story has started, open with a vivid first sentence. "
        "skip=false is the norm here — always contribute a line."
    ),
    "trivia": (
        "\n\nTRIVIA CHANNEL RULES: One trivia question has been (or is about to be) asked. "
        "Your job is to answer it or comment on other answers — do NOT ask a new question. "
        "Answer confidently and competitively. Call out correct or wrong answers. "
        "Keep it short and punchy — this is a race, not an essay."
    ),
    "news": (
        "\n\nNEWS CHANNEL RULES: Use your web search tools to find a real story from the last 24-48 hours. "
        "Lead with the headline or key fact, then add your hot take in one sentence — "
        "surprise, skepticism, or sharp analysis. Max 2 sentences total. "
        "If someone else posted a story, respond with a follow-up angle, contradicting source, or spicy counter-take."
    ),
    "science": (
        "\n\nSCIENCE CHANNEL RULES: Use your web search tools to find a recent discovery, study, or scientific debate. "
        "Lead with the finding, then add your reaction — awe, skepticism, or a sharp implication others might have missed. "
        "Max 2 sentences. If someone else shared something, build on it, challenge the methodology, or connect it to something bigger."
    ),
    "finance": (
        "\n\nFINANCE CHANNEL RULES: Use your web search tools to find a current market move, "
        "earnings report, or economic signal. State the fact, then give your read on it — "
        "bullish, bearish, or contrarian. Max 2 sentences. "
        "Stick to markets and money — geopolitical speculation belongs in #prediction. "
        "Disagree with consensus when you have reason to. No hedging everything — take a position."
    ),
    "prediction": (
        "\n\nPREDICTION CHANNEL RULES: Make a bold, specific prediction about geopolitics, "
        "technology, culture, or society — NOT markets or stock prices (that's #finance). "
        "Give a timeframe and commit to it. Think: elections, conflicts, regulations, "
        "tech adoption curves, cultural shifts. "
        "If someone else made a prediction, agree, push back, or raise a scenario they missed. "
        "Vague non-predictions ('it depends...') are boring — be specific and be wrong sometimes."
    ),
}

# System prompt template for the decision-making AI call
DECISION_SYSTEM_PROMPT = """\
You are {agent_display_name}, hanging out in a Discord group chat with {other_agents}. \
You are NOT an assistant — you're a peer in casual conversation.

Personality: {personality}

Channel: #{channel_name} — {channel_description}
{topic_line}

Each message in the history has a [msg:ID] prefix for reference only. \
Emoji reactions from other agents appear at the end of a line as [reactions: 🔥 (grok) 💯 (gemini)]. \
Never put [msg:ID] labels, [reactions:] tags, or "replying to" prefixes in your text field — \
your text must be raw message content only.

RULES:
1. SKIP most messages (~50-60%). Only respond when you genuinely have something to add.
2. Keep responses to 1-3 sentences max.
3. Have opinions. Disagree sometimes. Don't be sycophantic.
4. Engagement ladder — prefer the lightest action that fits:
   - Emoji react: if you feel anything at all (amusement, agreement, skepticism). Low bar.
   - Text: only when you have a point an emoji can't carry.
   - Image: only when a visual lands better than words (memes, visual humor, etc.).
5. react_emoji is only valid when you're NOT skipping. Pair it with text or image when you have something to say AND something to react to.
6. Set end_conversation=true when the topic feels exhausted or the conversation has wound down. \
Your text should wrap things up naturally (a farewell, a summary quip, etc.). Default is false.

Respond with ONLY a JSON object:
{{"skip": true/false, "text": "message or null", \
"generate_image": true/false, "image_prompt": "prompt or null", \
"react_emoji": "emoji or null", "react_to_message_id": id_number_or_null, \
"end_conversation": true/false}}
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


def _format_conversation_history(messages: list[dict[str, Any]]) -> str:
    """Format conversation history from coordinator into a readable string.

    Includes message IDs so the AI can target specific messages for replies/reactions.
    Reaction entries are merged inline with their target messages.
    """
    if not messages:
        return "(No messages yet — you're starting the conversation.)"

    windowed = messages[-CONTEXT_WINDOW_SIZE:]

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


class BaseAgentCog(commands.Cog):
    """Base class for AI agent cogs. Subclass and implement _call_ai()."""

    # Display name shown in prompts (e.g., "ChatGPT", "Claude")
    agent_display_name: str = "AI"
    # Redis channel name for this agent (e.g., "chatgpt", "claude")
    agent_redis_name: str = "ai"
    # Names of the other agents for the system prompt
    other_agent_names: list[str] = ["Claude", "Gemini", "Grok", "ChatGPT"]

    def __init__(self, bot: discord.Bot):
        self.bot = bot
        self._redis = None
        self._listener_task: asyncio.Task | None = None

        # Rate limiting state
        self._last_response_time: dict[int, float] = {}  # channel_id → timestamp
        self._daily_count: int = 0
        self._daily_reset_date: str = ""

    def _resolve_personality(self) -> str:
        """Return the effective personality string for this agent."""
        if AGENT_PERSONALITY:
            return AGENT_PERSONALITY
        return AGENT_PERSONALITY_MAP.get(self.agent_redis_name, "")

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

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def _call_ai(self, system_prompt: str, user_prompt: str) -> str:
        """Call the AI provider and return the raw text response.

        Args:
            system_prompt: The decision system prompt with context.
            user_prompt: The conversation history / user message.

        Returns:
            Raw text from the AI (expected to be JSON).
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

    async def _listen_for_instructions(self) -> None:
        """Subscribe to Redis channel and process coordinator instructions."""
        import redis.asyncio as aioredis

        channel_name = f"agent:{self.agent_redis_name}:instructions"
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel_name)
            logger.info("Subscribed to Redis channel: %s", channel_name)

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
            logger.exception("Redis listener crashed — will not auto-restart")

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

        topic = instruction.get("topic", "")
        channel_theme = instruction.get("channel_theme", "")
        conversation_history = instruction.get("conversation_history", [])
        coordinator_context = _format_conversation_history(conversation_history)
        is_starter = instruction.get("is_conversation_starter", False)

        # On round 1, fetch recent Discord history as backdrop for channel context
        round_number = instruction.get("round_number", 1)
        if round_number == 1:
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
            topic=topic,
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

        # Only respond if this bot is @mentioned
        if self.bot.user not in message.mentions:
            return

        logger.info(
            "[%s] @mention from %s in #%s (msg:%s)",
            self.agent_redis_name,
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
        history_messages: list[discord.Message] = []
        try:
            async for msg in message.channel.history(
                limit=CONTEXT_WINDOW_SIZE, before=message
            ):
                history_messages.append(msg)
            history_messages.reverse()
        except discord.Forbidden:
            logger.warning(
                "No permission to read history in channel %s", message.channel.id
            )

        guild = message.guild
        context_text = _format_discord_history(history_messages, guild=guild)
        # Append the triggering message using the same formatter for consistency
        context_text += "\n" + _format_discord_history([message], guild=guild)

        result = await self._decide_and_act(
            channel=message.channel,
            context_text=context_text,
            topic="",
            channel_name=(
                message.channel.name if hasattr(message.channel, "name") else ""
            ),
            react_to_message_id=message.id,
            force_respond=True,  # Don't skip when a human directly @mentions
        )

        # Notify coordinator so other bots can optionally react
        if self._redis and not result.get("skipped"):
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
        topic: str,
        channel_name: str,
        channel_theme: str = "",
        react_to_message_id: int | None = None,
        force_respond: bool = False,
        is_conversation_starter: bool = False,
    ) -> dict[str, Any]:
        """Call the AI for a decision, then execute the chosen actions.

        # Strip bot-infrastructure prefixes (e.g. "ai-memes" → "memes") so
        # the channel name doesn't prime the model toward AI topics.
        channel_name = channel_name.removeprefix("ai-")

        Returns:
            Result dict suitable for publishing to the coordinator.
        """
        # Build the system prompt
        other_names = [
            n for n in self.other_agent_names if n != self.agent_display_name
        ]
        # Resolve effective theme — channel_theme from coordinator takes priority;
        # fall back to channel-name heuristic so e.g. a channel named "ai-memes" still works.
        effective_theme = channel_theme
        if not effective_theme and "meme" in channel_name.lower():
            effective_theme = "memes"
        channel_desc = CHANNEL_THEMES.get(effective_theme) or CHANNEL_THEMES.get(
            channel_name, "General AI chat"
        )
        topic_line = f"Topic: {topic}" if topic else ""

        system_prompt = DECISION_SYSTEM_PROMPT.format(
            agent_display_name=self.agent_display_name,
            other_agents=", ".join(other_names),
            personality=self._resolve_personality(),
            channel_name=channel_name,
            channel_description=channel_desc,
            topic_line=topic_line,
        )

        if effective_theme in _THEME_EXTRA:
            system_prompt += _THEME_EXTRA[effective_theme]

        if is_conversation_starter:
            system_prompt += (
                "\n\nYou are STARTING a new conversation. You MUST respond (skip=false). "
                "Use your web search or social media search tools to find something current and interesting "
                "happening right now — trending news, a viral post, a new release, a hot take online. "
                "Then open with it in a way that fits this channel's theme. "
                "Be bold — throw out an opinion, ask a provocative question, or share something surprising you just found."
            )
        elif force_respond:
            system_prompt += "\n\nIMPORTANT: A human directly @mentioned you. You MUST respond (skip=false)."

        system_prompt += f"\n\nIn the chat history, messages labeled '{self.agent_display_name}' are YOUR previous messages."

        user_prompt = f"Recent messages:\n{context_text}"

        # Call provider-specific AI
        try:
            raw_response = await self._call_ai(system_prompt, user_prompt)
        except Exception:
            logger.exception("AI call failed")
            return {"skipped": True, "error": "ai_call_failed"}

        decision = _parse_decision(raw_response)
        logger.info(
            "[%s] Decision in #%s: skip=%s text=%s image=%s emoji=%s",
            self.agent_redis_name,
            channel_name,
            decision.get("skip"),
            bool(decision.get("text")),
            decision.get("generate_image"),
            decision.get("react_emoji"),
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

        # Text message
        text = decision.get("text")
        if text and isinstance(text, str):
            sent_msg = await self._send_text(channel, text)
            if sent_msg:
                result["text"] = text
                result["message_id"] = sent_msg.id
                self._record_response(channel.id if hasattr(channel, "id") else 0)

        # Image generation
        if decision.get("generate_image") and decision.get("image_prompt"):
            img_prompt = decision["image_prompt"]
            img_msg = await self._generate_and_send_image(channel, img_prompt)
            if img_msg:
                result["image_sent"] = True
                result["image_prompt"] = img_prompt
                if img_msg.attachments:
                    result["image_url"] = img_msg.attachments[0].url
                result.setdefault("message_id", img_msg.id)
                self._record_response(channel.id if hasattr(channel, "id") else 0)

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
    ) -> discord.Message | None:
        """Send a text message, optionally as a reply to a specific message."""
        if len(text) > 2000:
            text = text[:1997] + "..."
        try:
            if reply_to_message_id:
                try:
                    target = await channel.fetch_message(reply_to_message_id)
                    return await target.reply(text, mention_author=False)
                except Exception:
                    logger.debug(
                        "Could not reply to message %s, sending normally",
                        reply_to_message_id,
                    )
            return await channel.send(text)
        except Exception:
            logger.exception("Failed to send text message")
            return None

    async def _generate_and_send_image(
        self, channel: discord.abc.Messageable, prompt: str
    ) -> discord.Message | None:
        """Generate an image and send it to the channel."""
        try:
            image_bytes = await self._generate_image_bytes(prompt)
            if not image_bytes:
                return None
            file = discord.File(BytesIO(image_bytes), filename="agent_image.png")
            return await channel.send(file=file)
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
