"""VLM provider implementations — Anthropic and OpenAI."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from .prompt import grading_prompt


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split(chr(10), 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _mime_for(path: Path) -> str:
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def grade_with_anthropic(
    original: Path,
    result: Path,
    foreground: str,
    description: str,
    should_exclude: str,
    api_key: str,
) -> dict:
    """Grade using Anthropic Claude API."""
    orig_b64 = base64.b64encode(original.read_bytes()).decode()
    result_b64 = base64.b64encode(result.read_bytes()).decode()

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "ORIGINAL image:"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": _mime_for(original),
                        "data": orig_b64,
                    }},
                    {"type": "text", "text": "RESULT (background removed):"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": _mime_for(result),
                        "data": result_b64,
                    }},
                    {"type": "text", "text": grading_prompt(foreground, description, should_exclude)},
                ],
            }],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return _parse_json_response(resp.json()["content"][0]["text"])


def grade_with_openai(
    original: Path,
    result: Path,
    foreground: str,
    description: str,
    should_exclude: str,
    api_key: str,
    base_url: str,
) -> dict:
    """Grade using OpenAI-compatible vision API."""
    def to_uri(p: Path) -> str:
        b = base64.b64encode(p.read_bytes()).decode()
        return "data:{};base64,{}".format(_mime_for(p), b)

    resp = requests.post(
        "{}/chat/completions".format(base_url),
        headers={
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "ORIGINAL image:"},
                    {"type": "image_url", "image_url": {"url": to_uri(original)}},
                    {"type": "text", "text": "RESULT (background removed):"},
                    {"type": "image_url", "image_url": {"url": to_uri(result)}},
                    {"type": "text", "text": grading_prompt(foreground, description, should_exclude)},
                ],
            }],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return _parse_json_response(resp.json()["choices"][0]["message"]["content"])
