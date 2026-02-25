"""Anthropic (Claude) agent cog — uses AsyncAnthropic for decision-making."""

from __future__ import annotations

import logging
import re

import aiohttp
import discord
from anthropic import AsyncAnthropic

from agent_config import AGENT_CHANNEL_IDS, ANTHROPIC_API_KEY
from .base import BaseAgentCog

logger = logging.getLogger(__name__)


class AnthropicAgentCog(BaseAgentCog):
    agent_display_name = "Clod Bot"
    agent_redis_name = "claude"
    other_agent_names = ["GPT Bot", "Google Bot", "Grok Bot"]

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set — AnthropicAgentCog will not function")
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
        # Extract text from content blocks
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        # Use Claude + web search to find a relevant image
        try:
            response = await self._client.messages.create(
                model="claude-opus-4-6",
                max_tokens=256,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search the web for an image matching: {prompt}\n\n"
                        "Reply with ONLY a direct image URL "
                        "(ending in .jpg, .jpeg, .png, .gif, or .webp). No other text."
                    ),
                }],
            )
            # Extract the first image URL from Claude's text response
            url = None
            for block in response.content:
                if block.type == "text":
                    match = re.search(
                        r"https?://\S+\.(?:jpg|jpeg|png|gif|webp)",
                        block.text,
                        re.IGNORECASE,
                    )
                    if match:
                        url = match.group(0)
                        break

            if not url:
                logger.info("Claude web search returned no image URL for prompt: %s", prompt[:100])
                return None

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200 and (resp.content_type or "").startswith("image/"):
                        return await resp.read()
                    logger.warning("Image fetch failed: HTTP %s, type=%s", resp.status, resp.content_type)
        except Exception:
            logger.exception("Anthropic web image fetch failed")
        return None
