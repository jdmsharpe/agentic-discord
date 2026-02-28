"""Anthropic (Claude) agent cog — uses AsyncAnthropic for decision-making."""

from __future__ import annotations

import logging
import re

import discord
from anthropic import AsyncAnthropic

from agent_config import ANTHROPIC_API_KEY
from .base import BaseAgentCog

logger = logging.getLogger(__name__)


class AnthropicAgentCog(BaseAgentCog):
    agent_redis_name = "claude"

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not ANTHROPIC_API_KEY:
            logger.warning(
                "ANTHROPIC_API_KEY not set — AnthropicAgentCog will not function"
            )
        self._client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.messages.create(
            model="claude-opus-4-6",
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
                {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5},
            ],
        )
        # Extract text from content blocks, stripping web search citation tags
        parts = []
        for block in response.content:
            if block.type == "text":
                clean = re.sub(r"</?cite[^>]*>", "", block.text)
                parts.append(clean)
        return "\n".join(parts) if parts else ""

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        # Claude does not support image generation natively
        return None
