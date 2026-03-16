"""OpenAI agent cog — uses AsyncOpenAI for decision-making and image generation."""

from __future__ import annotations

import base64
import logging

import discord
from openai import AsyncOpenAI

from agent_config import OPENAI_API_KEY
from .base import AIResponse, BaseAgentCog

logger = logging.getLogger(__name__)


class OpenAIAgentCog(BaseAgentCog):
    agent_redis_name = "chatgpt"
    ai_model = "gpt-5.4-pro"
    image_model = "gpt-image-1.5"

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set — OpenAIAgentCog will not function")
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> AIResponse:
        response = await self._client.responses.create(
            model=self.ai_model,
            instructions=system_prompt,
            input=user_prompt,
            tools=[
                {"type": "web_search"},
            ],
        )
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0
        return AIResponse(
            text=response.output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

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
