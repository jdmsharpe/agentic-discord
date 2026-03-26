"""Gemini agent cog — uses google-genai for decision-making and image generation."""

from __future__ import annotations

import logging
from typing import Any

import discord
from google import genai
from google.genai import types

from agent_config import GEMINI_API_KEY
from .base import AIResponse, BaseAgentCog, _download_image_bytes

logger = logging.getLogger(__name__)

# Tools that are only supported on specific Gemini model families.
# Models not listed are assumed to NOT support the tool.
_TOOL_MODEL_COMPATIBILITY: dict[str, set[str]] = {
    "url_context": {
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    },
}

# All tools this agent can request, subject to model compatibility.
_ALL_TOOLS: list[dict[str, Any]] = [
    {"google_search": {}},
    {"url_context": {}},
]


def _filter_tools_for_model(model: str, tools: list[dict]) -> list[dict]:
    """Return only the tools compatible with the given model."""
    supported = []
    for tool in tools:
        tool_name = next(iter(tool), None)
        compat_set = _TOOL_MODEL_COMPATIBILITY.get(tool_name)
        if compat_set is not None and model not in compat_set:
            logger.debug("Tool %s not supported on %s, skipping", tool_name, model)
            continue
        supported.append(tool)
    return supported


class GeminiAgentCog(BaseAgentCog):
    agent_redis_name = "gemini"
    ai_model = "gemini-3.1-pro-preview"
    image_model = "gemini-3.1-flash-image-preview"

    def __init__(self, bot: discord.Bot):
        super().__init__(bot)
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — GeminiAgentCog will not function")
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    async def _call_ai(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
    ) -> AIResponse:
        tools = _filter_tools_for_model(self.ai_model, _ALL_TOOLS)
        # Build parts: text + optional inline image data
        parts: list[dict[str, Any]] = [{"text": user_prompt}]
        if image_urls:
            session = await self.get_http_session()
            for url in image_urls:
                result = await _download_image_bytes(session, url)
                if result:
                    data, mime = result
                    parts.append({"inline_data": {"mime_type": mime, "data": data}})
        response = await self._client.aio.models.generate_content(
            model=self.ai_model,
            contents=[{"role": "user", "parts": parts}],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools,
            ),
        )
        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            thinking_tokens = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
        return AIResponse(
            text=response.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=thinking_tokens,
        )

    async def _generate_image_bytes(self, prompt: str) -> bytes | None:
        try:
            response = await self._client.aio.models.generate_content(
                model=self.image_model,
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
