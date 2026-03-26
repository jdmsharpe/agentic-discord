"""Anthropic (Claude) agent cog — uses AsyncAnthropic for decision-making."""

from __future__ import annotations

import base64
import logging
import re

import discord
from anthropic import AsyncAnthropic

from agent_config import ANTHROPIC_API_KEY
from .base import AIResponse, BaseAgentCog, _download_image_bytes

logger = logging.getLogger(__name__)


class AnthropicAgentCog(BaseAgentCog):
    agent_redis_name = "claude"
    ai_model = "claude-sonnet-4-6"
    image_model = ""  # Claude does not support image generation

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not ANTHROPIC_API_KEY:
            logger.warning(
                "ANTHROPIC_API_KEY not set — AnthropicAgentCog will not function"
            )
        self._client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def _call_ai(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
    ) -> AIResponse:
        # Build user content: text + optional base64-encoded images
        user_content: list[dict] | str
        if image_urls:
            blocks: list[dict] = [{"type": "text", "text": user_prompt}]
            session = await self.get_http_session()
            for url in image_urls:
                result = await _download_image_bytes(session, url)
                if result:
                    data, media_type = result
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(data).decode(),
                        },
                    })
            user_content = blocks
        else:
            user_content = user_prompt

        response = await self._client.messages.create(
            model=self.ai_model,
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            tools=[
                {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
                {"type": "web_fetch_20260309", "name": "web_fetch", "max_uses": 5, "use_cache": False},
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
        )
        # Extract text from content blocks, stripping web search citation tags
        parts = []
        thinking_used = False
        for block in response.content:
            if block.type == "text":
                clean = re.sub(r"</?cite[^>]*>", "", block.text)
                parts.append(clean)
            elif block.type == "thinking":
                thinking_used = True
        text = "\n".join(parts) if parts else ""
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        return AIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            thinking_used=thinking_used,
        )

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        # Claude does not support image generation natively
        return None
