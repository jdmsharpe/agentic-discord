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
    agent_redis_name = "claude"

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
        # Extract text from content blocks, stripping web search citation tags
        parts = []
        for block in response.content:
            if block.type == "text":
                clean = re.sub(r"</?cite[^>]*>", "", block.text)
                parts.append(clean)
        return "\n".join(parts) if parts else ""

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        # Use Claude + web search + web fetch to find and retrieve a relevant image.
        # Flow: search for image pages → fetch a page → extract direct image URL → download.
        try:
            response = await self._client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                tools=[
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
                    {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3},
                ],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Find me a direct image URL for: {prompt}\n\n"
                        "Steps:\n"
                        "1. Web search for relevant images (try sites like imgur.com, "
                        "i.redd.it, pexels.com, unsplash.com, or wikimedia)\n"
                        "2. Use web_fetch on a promising result page to find the actual "
                        "image src URL (look for .jpg, .png, .webp, or CDN image URLs)\n"
                        "3. Reply with ONLY the direct image URL — no other text\n\n"
                        "The URL must point directly to an image file, not an HTML page."
                    ),
                }],
            )
            # Extract candidate URLs from all text blocks and search result blocks
            urls: list[str] = []
            for block in response.content:
                if block.type == "text":
                    for m in re.finditer(r"https?://\S+", block.text):
                        urls.append(m.group(0).rstrip(".,)\"'"))
                elif block.type == "web_search_tool_result":
                    for result in getattr(block, "content", []):
                        if hasattr(result, "url") and result.url:
                            urls.append(result.url)

            if not urls:
                logger.info("Claude web search returned no URLs for prompt: %s", prompt[:100])
                return None

            # Try candidate URLs, preferring ones that look like direct image links
            image_ext_pattern = re.compile(r"\.(jpg|jpeg|png|gif|webp)", re.IGNORECASE)
            urls.sort(key=lambda u: (0 if image_ext_pattern.search(u) else 1))

            async with aiohttp.ClientSession() as session:
                for candidate_url in urls[:5]:
                    try:
                        async with session.get(candidate_url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            content_type = (resp.content_type or "").split(";")[0].strip()
                            if resp.status == 200 and content_type.startswith("image/"):
                                return await resp.read()
                            logger.debug("Skipping non-image URL: HTTP %s, type=%s, url=%s", resp.status, content_type, candidate_url)
                    except Exception:
                        logger.debug("Failed to fetch candidate URL: %s", candidate_url)
                        continue

            logger.info("No candidate URLs resolved to images for prompt: %s", prompt[:100])
        except Exception:
            logger.exception("Anthropic web image fetch failed")
        return None
