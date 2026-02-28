"""Gemini agent cog — uses google-genai for decision-making and image generation."""

from __future__ import annotations

import logging

import discord
from google import genai
from google.genai import types

from agent_config import GEMINI_API_KEY
from .base import BaseAgentCog

logger = logging.getLogger(__name__)


class GeminiAgentCog(BaseAgentCog):
    agent_redis_name = "gemini"

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — GeminiAgentCog will not function")
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.aio.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[
                    {"google_search": {}},
                    {"url_context": {}},
                ],
            ),
        )
        return response.text or ""

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        try:
            response = await self._client.aio.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            # Gemini native image models return inline image data in parts
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        return part.inline_data.data
        except Exception:
            logger.exception("Gemini image generation failed")
        return None
