"""Grok (xAI) agent cog — uses xai_sdk for decision-making and image generation."""

from __future__ import annotations

import base64
import logging

import discord
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user
from xai_sdk.tools import web_search, x_search

from agent_config import XAI_API_KEY
from .base import AIResponse, BaseAgentCog

logger = logging.getLogger(__name__)


class GrokAgentCog(BaseAgentCog):
    agent_redis_name = "grok"
    ai_model = "grok-4.20-beta-latest-reasoning"
    image_model = "grok-imagine-image-pro"

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not XAI_API_KEY:
            logger.warning("XAI_API_KEY not set — GrokAgentCog will not function")
        self._client = AsyncClient(api_key=XAI_API_KEY)

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> AIResponse:
        chat = self._client.chat.create(
            model=self.ai_model,
            messages=[system(system_prompt), user(user_prompt)],
            tools=[web_search(), x_search()],
        )
        response = await chat.sample()
        # xAI SDK uses protobuf with OpenAI-style field names
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
        return AIResponse(
            text=response.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        try:
            result = await self._client.image.sample(
                prompt=prompt,
                model=self.image_model,
            )
            if result.url:
                session = await self.get_http_session()
                async with session.get(result.url) as resp:
                    if resp.status == 200:
                        return await resp.read()
            if result.base64:
                return base64.b64decode(result.base64)
        except Exception:
            logger.exception("Grok image generation failed")
        return None
