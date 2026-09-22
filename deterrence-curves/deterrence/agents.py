"""Tool-use agent loops.

- anthropic: Claude models via the Anthropic API
- openai:    OpenAI models, or any OpenAI-compatible API (e.g. OpenRouter for
             open-weight models like Llama, Qwen, DeepSeek; no GPU needed)
- mock:      scripted agent for testing the pipeline without API calls
             (see mock_agent.py). Mock output is NOT experimental data.
"""
from __future__ import annotations

import json
import os
import time

from .environment import TOOL_SPECS, RepoEnvironment
from .mock_agent import run_mock


def _retry(fn, tries: int = 6):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits, timeouts, transient server errors
            if attempt == tries - 1:
                raise
            wait = min(60, 2 ** attempt * 2)
            print(f"  API error ({type(e).__name__}: {str(e)[:120]}), retrying in {wait}s")
            time.sleep(wait)


def run_anthropic(env: RepoEnvironment, system: str, user_msg: str, model: str,
                  max_turns: int, temperature: float | None, **_) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
             for t in TOOL_SPECS]
    messages: list = [{"role": "user", "content": user_msg}]
    usage = {"input_tokens": 0, "output_tokens": 0}
    turns = 0
    for turns in range(1, max_turns + 1):
        kwargs = dict(model=model, max_tokens=2048, system=system, tools=tools, messages=messages)
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = _retry(lambda: client.messages.create(**kwargs))
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        messages.append({"role": "assistant",
                         "content": [b.model_dump(exclude_none=True) for b in resp.content]})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = env.call(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
                if env.submitted:
                    break
        if not results or env.submitted:
            if results:
                messages.append({"role": "user", "content": results})
            break
        messages.append({"role": "user", "content": results})
    return {"transcript": messages, "usage": usage, "turns": turns}


def run_openai(env: RepoEnvironment, system: str, user_msg: str, model: str,
               max_turns: int, temperature: float | None, base_url: str | None = None,
               api_key_env: str = "OPENAI_API_KEY", **_) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))
    tools = [{"type": "function", "function": t} for t in TOOL_SPECS]
    messages: list = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    usage = {"input_tokens": 0, "output_tokens": 0}
    turns = 0
    for turns in range(1, max_turns + 1):
        kwargs = dict(model=model, messages=messages, tools=tools)
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = _retry(lambda: client.chat.completions.create(**kwargs))
        if resp.usage:
            usage["input_tokens"] += resp.usage.prompt_tokens or 0
            usage["output_tokens"] += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name,
                                                 "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls]
        messages.append(entry)
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                out = env.call(tc.function.name, args)
            except json.JSONDecodeError:
                out = "Error: arguments were not valid JSON."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            if env.submitted:
                break
        if env.submitted:
            break
    return {"transcript": messages, "usage": usage, "turns": turns}


RUNNERS = {"anthropic": run_anthropic, "openai": run_openai, "mock": run_mock}
