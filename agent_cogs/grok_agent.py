"""Grok (xAI) agent cog — uses xai_sdk for decision-making and image generation."""

from __future__ import annotations

import logging

import aiohttp
import discord
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user
from xai_sdk.tools import code_execution, web_search, x_search

from agent_config import AGENT_CHANNEL_IDS, XAI_API_KEY
from .base import BaseAgentCog

logger = logging.getLogger(__name__)


class GrokAgentCog(BaseAgentCog):
    agent_display_name = "Grok Bot"
    agent_redis_name = "grok"
    other_agent_names = ["GPT Bot", "Google Bot", "Clod Bot"]

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not XAI_API_KEY:
            logger.warning("XAI_API_KEY not set — GrokAgentCog will not function")
        self._client = AsyncClient(api_key=XAI_API_KEY)

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> str:
        chat = self._client.chat.create(
            model="grok-4-1-fast-reasoning",
            messages=[system(system_prompt), user(user_prompt)],
            tools=[web_search(), x_search(), code_execution()],
        )
        response = await chat.sample()
        return response.content or ""

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        try:
            result = await self._client.image.sample(
                prompt=prompt,
                model="grok-imagine-image-pro",
            )
            if result.url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(result.url) as resp:
                        if resp.status == 200:
                            return await resp.read()
            if result.base64:
                import base64

                return base64.b64decode(result.base64)
        except Exception:
            logger.exception("Grok image generation failed")
        return None
