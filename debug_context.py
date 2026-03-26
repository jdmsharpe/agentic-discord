"""debug_context.py — Subscribe to Redis pub/sub and display agent context in real time.

Run:  python debug_context.py [--redis redis://127.0.0.1:6379]

Shows exactly what each agent receives (instructions) and responds (results)
during active conversations. Useful for verifying context window sizing,
conversation history formatting, and agent decision quality.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

AGENTS = ["chatgpt", "claude", "gemini", "grok"]

# ANSI colors for agent differentiation
AGENT_COLORS = {
    "chatgpt": "\033[32m",   # green
    "claude": "\033[33m",    # yellow
    "gemini": "\033[34m",    # blue
    "grok": "\033[31m",      # red
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def format_instruction(data: dict, agent: str) -> str:
    """Format a coordinator→agent instruction for display.

    TODO: Customize this to show exactly what you want to inspect.
    Currently shows: channel theme, round, topic, history length,
    and the full conversation history the agent will see.

    Consider:
    - Showing only the last N history entries instead of all?
    - Including the full system prompt reconstruction?
    - Highlighting which messages are new since the last turn?
    """
    color = AGENT_COLORS.get(agent, "")
    lines = [
        f"\n{color}{BOLD}{'=' * 60}",
        f"  INSTRUCTION → {agent.upper()}",
        f"{'=' * 60}{RESET}",
        f"  {DIM}Channel:{RESET} {data.get('channel_id')}  "
        f"{DIM}Theme:{RESET} {data.get('channel_theme', '?')}  "
        f"{DIM}Round:{RESET} {data.get('round_number', '?')}",
        f"  {DIM}Topic:{RESET} {data.get('topic') or '(none yet)'}  "
        f"{DIM}Starter:{RESET} {data.get('is_conversation_starter', False)}",
    ]

    history = data.get("conversation_history", [])
    lines.append(f"  {DIM}History entries:{RESET} {len(history)}")

    if history:
        lines.append(f"\n  {DIM}--- Conversation History ---{RESET}")
        for entry in history:
            agent_name = entry.get("agent", "?")
            text = entry.get("text", "")
            mid = entry.get("message_id", "?")
            entry_color = AGENT_COLORS.get(agent_name, "")
            # Truncate long entries for readability in the debug view
            display_text = text if len(text) <= 200 else text[:200] + "..."
            lines.append(f"  {entry_color}[msg:{mid}] {agent_name}: {display_text}{RESET}")
        lines.append(f"  {DIM}--- End History ---{RESET}")

    return "\n".join(lines)


def format_result(data: dict) -> str:
    """Format an agent→coordinator result for display."""
    agent = data.get("agent_name", "?")
    color = AGENT_COLORS.get(agent, "")

    if data.get("skipped"):
        return f"  {color}← {agent}: SKIPPED ({data.get('reason', 'decision')}){RESET}"

    parts = []
    if data.get("text"):
        text = data["text"]
        display = text if len(text) <= 150 else text[:150] + "..."
        parts.append(f'text="{display}"')
    if data.get("image_url"):
        parts.append(f"image={data['image_url'][:60]}...")
    if data.get("emoji_reacted"):
        parts.append(f"emoji={data['emoji_reacted']}")
    if data.get("end_conversation"):
        parts.append("END_CONVERSATION")
    if data.get("topic"):
        parts.append(f"topic=\"{data['topic']}\"")

    return f"  {color}← {agent}: {', '.join(parts)}{RESET}"


async def main(redis_url: str) -> None:
    print(f"{BOLD}Connecting to Redis: {redis_url}{RESET}")
    r = aioredis.from_url(redis_url)

    try:
        await r.ping()
        print(f"{DIM}Connected. Subscribing to agent channels...{RESET}\n")
    except Exception as e:
        print(f"\033[31mFailed to connect to Redis: {e}{RESET}")
        sys.exit(1)

    pubsub = r.pubsub()
    channels = (
        [f"agent:{name}:instructions" for name in AGENTS]
        + [f"agent:{name}:results" for name in AGENTS]
    )
    await pubsub.subscribe(*channels)
    print(f"{DIM}Listening on: {', '.join(channels)}{RESET}")
    print(f"{DIM}Waiting for conversations...{RESET}\n")

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue

        channel = message["channel"]
        if isinstance(channel, bytes):
            channel = channel.decode()

        try:
            data = json.loads(message["data"])
        except json.JSONDecodeError:
            continue

        timestamp = datetime.now().strftime("%H:%M:%S")

        if ":instructions" in channel:
            agent = channel.split(":")[1]
            print(f"{DIM}[{timestamp}]{RESET}" + format_instruction(data, agent))
        elif ":results" in channel:
            print(f"{DIM}[{timestamp}]{RESET}" + format_result(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug agent context in real time")
    parser.add_argument(
        "--redis",
        default=os.getenv("REDIS_URL", "redis://127.0.0.1:6379"),
        help="Redis URL (default: REDIS_URL env var or redis://127.0.0.1:6379)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.redis))
    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")
