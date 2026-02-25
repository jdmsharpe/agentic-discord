"""OpenAI agent cog — uses AsyncOpenAI for decision-making and image generation."""

from __future__ import annotations

import base64
import logging

import discord
from openai import AsyncOpenAI

from agent_config import AGENT_CHANNEL_IDS, OPENAI_API_KEY
from .base import BaseAgentCog

logger = logging.getLogger(__name__)


class OpenAIAgentCog(BaseAgentCog):
    agent_display_name = "GPT Bot"
    agent_redis_name = "chatgpt"
    other_agent_names = ["Google Bot", "Grok Bot", "Clod Bot"]

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set — OpenAIAgentCog will not function")
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.responses.create(
            model="gpt-5.2",
            instructions=system_prompt,
            input=user_prompt,
            tools=[
                {"type": "web_search"},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
        )
        return response.output_text

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        response = await self._client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="medium",
        )
        # gpt-image models return base64
        for item in response.data:
            if hasattr(item, "b64_json") and item.b64_json:
                return base64.b64decode(item.b64_json)
            if hasattr(item, "url") and item.url:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(item.url) as resp:
                        if resp.status == 200:
                            return await resp.read()
        return None
