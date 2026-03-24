"""Grok (xAI) agent cog — uses OpenAI-compatible Responses API for decision-making and image generation."""

from __future__ import annotations

import base64
import logging
import uuid

import discord
import httpx
from openai import AsyncOpenAI

from agent_config import XAI_API_KEY
from .base import AIResponse, BaseAgentCog

logger = logging.getLogger(__name__)


class GrokAgentCog(BaseAgentCog):
    agent_redis_name = "grok"
    ai_model = "grok-4.20"
    image_model = "grok-imagine-image-pro"

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not XAI_API_KEY:
            logger.warning("XAI_API_KEY not set — GrokAgentCog will not function")
        self._client = AsyncOpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(90.0),
        )
        self._cache_key = str(uuid.uuid4())

    async def _call_ai(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
    ) -> AIResponse:
        if image_urls:
            input_content: list[dict] = [{"type": "input_text", "text": user_prompt}]
            for url in image_urls:
                input_content.append({"type": "input_image", "image_url": url})
            ai_input: str | list[dict] = input_content
        else:
            ai_input = user_prompt

        response = await self._client.responses.create(
            model=self.ai_model,
            instructions=system_prompt,
            input=ai_input,
            tools=[
                {"type": "web_search"},
                {"type": "x_search"},
            ],
            prompt_cache_key=self._cache_key,
            prompt_cache_retention="24h",
            context_management=[{"type": "compaction", "compact_threshold": 200_000}],
        )
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        reasoning_tokens = 0
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details:
                cached_input_tokens = getattr(input_details, "cached_tokens", 0) or 0
            output_details = getattr(usage, "output_tokens_details", None)
            if output_details:
                reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
            # xAI includes reasoning in output_tokens — subtract to avoid double-counting
            output_tokens = max(output_tokens - reasoning_tokens, 0)
        # Count web_search_call items — xAI uses the same Responses API format as OpenAI
        web_search_calls = sum(
            1 for item in (response.output or [])
            if getattr(item, "type", "") == "web_search_call"
        )
        if web_search_calls:
            logger.info("[grok] web_search called %d time(s) this turn", web_search_calls)
        return AIResponse(
            text=response.output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            web_search_calls=web_search_calls,
        )

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        try:
            response = await self._client.images.generate(
                model=self.image_model,
                prompt=prompt,
                n=1,
            )
            for item in response.data:
                if hasattr(item, "b64_json") and item.b64_json:
                    return base64.b64decode(item.b64_json)
                if hasattr(item, "url") and item.url:
                    session = await self.get_http_session()
                    async with session.get(item.url) as resp:
                        if resp.status == 200:
                            return await resp.read()
        except Exception:
            logger.exception("Grok image generation failed")
        return None
