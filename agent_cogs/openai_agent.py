"""OpenAI agent cog — uses AsyncOpenAI for decision-making and image generation."""

from __future__ import annotations

import base64
import logging

import discord
from openai import AsyncOpenAI

from agent_config import OPENAI_API_KEY

from .base import AIResponse, BaseAgentCog, _extract_responses_api_usage

logger = logging.getLogger(__name__)


class OpenAIAgentCog(BaseAgentCog):
    agent_redis_name = "chatgpt"
    ai_model = "gpt-5.4"
    image_model = "gpt-image-1.5"

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set — OpenAIAgentCog will not function")
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def _call_ai(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
    ) -> AIResponse:
        # Build input: plain text or multimodal content blocks with images
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
            ],
            context_management=[{"type": "compaction", "compact_threshold": 200_000}],
            prompt_cache_retention="24h",
        )
        input_tokens, output_tokens, cached_input_tokens, reasoning_tokens, web_search_calls = (
            _extract_responses_api_usage(response)
        )
        if web_search_calls:
            logger.info("[chatgpt] web_search called %d time(s) this turn", web_search_calls)
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
                size="1024x1024",
                quality="medium",
            )
            # gpt-image models return base64
            if response.data:
                for item in response.data:
                    if hasattr(item, "b64_json") and item.b64_json:
                        return base64.b64decode(item.b64_json)
                    if hasattr(item, "url") and item.url:
                        session = await self.get_http_session()
                        async with session.get(item.url) as resp:
                            if resp.status == 200:
                                return await resp.read()
        except Exception:
            logger.exception("OpenAI image generation failed")
        return None
