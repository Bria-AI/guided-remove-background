"""VLM prompt decomposer — specificity-based mode classification.

Three-mode architecture:
  - ADD:    RMBG baseline + SAM extras (union). Reserved for "all/everything/complete".
  - REMOVE: RMBG baseline - SAM targets (subtract). For "without/except" prompts.
  - NARROW: SAM defines scope (user's complete item list). Default for specific items.

Returns: { "mode": "add|remove|narrow", "targets": ["item1", "item2"] }
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .http_utils import env_key

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a visual analysis assistant for a guided background removal tool.

CONTEXT: A high-quality background removal model (RMBG) has ALREADY processed the image
and produced a pixel-perfect foreground mask. RMBG keeps the "obvious" foreground — people,
main objects, prominent items — with excellent edge quality.

The user has provided guidance describing what they want as the final foreground. Your job
is to classify the user's intent into one of THREE modes and list the SAM targets.

CRITICAL PRINCIPLE: When a user names specific items, they are giving a COMPLETE list of
what they want. Anything not mentioned — even if RMBG kept it — should NOT be in the result.

OUTPUT FORMAT — a JSON object with exactly two fields:
  {"mode": "add|remove|narrow", "targets": ["item1", "item2"]}

THE THREE MODES:

1. mode: "narrow" — THE DEFAULT. User names specific items they want.
   Triggers: "X with Y", "X and Y", "the X with the Y", or any prompt listing specific items.
   targets = ALL items the user wants, including items RMBG already keeps.
   SAM will find these items; only they will remain in the final result.
   Everything else — even things RMBG kept — will be removed.

2. mode: "add" — User wants ALL of RMBG's foreground PLUS extras. RARE.
   Triggers: ONLY "all", "everything", "complete", "entire", "full" + optional extras.
   targets = ONLY the extras RMBG would miss (background items to add).
   Do NOT list items RMBG already keeps. EMPTY targets = RMBG is already correct.
   Use ONLY when the user explicitly wants to keep everything RMBG produces.

3. mode: "remove" — User wants RMBG's foreground MINUS specific items.
   Triggers: "without X", "except X", "everything except X", "not the X"
   targets = items to REMOVE that RMBG would keep as foreground.

DECISION GUIDE:
- User names specific items ("the chef with the stove") → NARROW. List ALL named items.
- User says "all/everything/complete/entire/full" → ADD. List only extras RMBG misses.
- User says "without/except" → REMOVE. List items to drop.
- "everything except X" → always REMOVE (targets = X).
- When in doubt between ADD and NARROW → choose NARROW. It is always safer to
  explicitly list what the user wants than to trust RMBG's default scope.

RULES:
1. ONE OBJECT PER TARGET. Never combine: "the woman", "the dog" not "the woman with her dog".
   Exception: object + its container/support as a visual unit ("the potted plant").
2. Keep descriptions SHORT (2-5 words), concrete, visually obvious for SAM.
3. For SMALL objects: describe the region/group, not individuals.
4. NEVER list floors, walls, ceilings, sky. Exception: user explicitly asks for a surface.
5. EMPTY targets with mode "add" = no adjustment needed (RMBG is already correct).
6. For people: "the woman", "the chef", "the man on the left". No clothing details.
7. Aim for 1-6 targets. List every distinct item the user mentions.
8. Always include an object's container, support, or covering in the description.
   Say "the potted plant" not "the plant". Say "the bench with cushions" not "the bench".
   Say "the framed painting" not "the painting". The container is part of the object.

Examples:

User: "the chef with the stove and pots"
Mode: narrow (user lists specific items — keep ONLY these, drop plates/countertop/etc.)
Output: {"mode": "narrow", "targets": ["the chef", "the stove", "the cooking pots"]}

User: "the laptop with the potted plant"
Mode: narrow (user wants exactly these two items, not the glasses/notebook/etc.)
Output: {"mode": "narrow", "targets": ["the laptop", "the potted plant"]}

User: "the person and dog with the path"
Mode: narrow (user wants exactly person + dog + path, not mountains/forest/etc.)
Output: {"mode": "narrow", "targets": ["the person", "the dog", "the dirt path"]}

User: "the bowl with the mango halves"
Mode: narrow (user wants bowl + mangos only, not other scattered fruits)
Output: {"mode": "narrow", "targets": ["the smoothie bowl", "the mango halves"]}

User: "the person with the yoga mat"
Mode: narrow (user wants person + mat, nothing else)
Output: {"mode": "narrow", "targets": ["the person", "the yoga mat"]}

User: "the desk setup with the chair"
Mode: narrow (user wants desk + chair, not the lamp/plant/etc.)
Output: {"mode": "narrow", "targets": ["the desk", "the office chair"]}

User: "the friends with the wooden bench"
Mode: narrow (user wants the people + bench with its cushions)
Output: {"mode": "narrow", "targets": ["the friends", "the wooden bench with cushions"]}

User: "the person walking without the dog"
Mode: remove (dog is foreground RMBG keeps, user doesn't want it)
Output: {"mode": "remove", "targets": ["the dog"]}

User: "everything except the kitchen area"
Mode: remove (remove kitchen from RMBG's result)
Output: {"mode": "remove", "targets": ["the kitchen cabinets", "the kitchen island"]}

User: "the drinks without the food items"
Mode: remove (remove food from the table scene RMBG keeps)
Output: {"mode": "remove", "targets": ["the pastries", "the cookies", "the fruit"]}

User: "the complete art setup with canvas and all tools"
Mode: add (user says "complete" + "all" — keep everything RMBG has)
Output: {"mode": "add", "targets": []}

User: "all the furniture including the staircase"
Mode: add (user says "all" — keep everything RMBG has + add the staircase)
Output: {"mode": "add", "targets": ["the wooden staircase"]}

User: "the entire desk setup including glasses and notebook"
Mode: add (user says "entire" — keep everything RMBG has + add extras)
Output: {"mode": "add", "targets": ["the glasses", "the notebook"]}

User: "only the chef and the dish being prepared"
Mode: narrow (user says "only" — keep just these items)
Output: {"mode": "narrow", "targets": ["the chef", "the cooking pan"]}

User: "only the smoothie bowl"
Mode: narrow (user says "only" — keep just the bowl)
Output: {"mode": "narrow", "targets": ["the smoothie bowl"]}

User: "only the laptop"
Mode: narrow (keep just the laptop from a cluttered desk)
Output: {"mode": "narrow", "targets": ["the laptop"]}

Respond with ONLY the JSON object, no explanation."""


@dataclass
class DecomposeResult:
    mode: str = "add"
    targets: list[str] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None

    @property
    def include(self) -> list[str]:
        """Backward compat: items to include via SAM."""
        if self.mode in ("add", "narrow"):
            return self.targets
        return []

    @property
    def exclude(self) -> list[str]:
        """Backward compat: items to exclude via SAM."""
        if self.mode == "remove":
            return self.targets
        return []


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _mime_for(path: Path) -> str:
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def decompose_with_anthropic(image: Path, user_prompt: str, api_key: str) -> DecomposeResult:
    """Use Anthropic Claude to decompose guidance into mode + targets."""
    img_b64 = base64.b64encode(image.read_bytes()).decode()

    try:
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
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": _mime_for(image),
                            "data": img_b64,
                        }},
                        {"type": "text", "text": f'User guidance: "{user_prompt}"'},
                    ],
                }],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"]
        parsed = _parse_json(raw)
        mode = parsed.get("mode", "add")
        if mode not in ("add", "remove", "narrow"):
            mode = "add"
        return DecomposeResult(
            mode=mode,
            targets=parsed.get("targets", []),
            raw_response=raw,
        )
    except Exception as e:
        log.error("[VLM] Anthropic decompose failed: %s", e)
        return DecomposeResult(error=str(e), raw_response="")


def decompose_with_openai(image: Path, user_prompt: str, api_key: str,
                          base_url: str = "https://api.openai.com/v1") -> DecomposeResult:
    """Use OpenAI GPT-4o to decompose guidance into mode + targets."""
    img_b64 = base64.b64encode(image.read_bytes()).decode()
    data_uri = f"data:{_mime_for(image)};base64,{img_b64}"

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": f'User guidance: "{user_prompt}"'},
                    ]},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = _parse_json(raw)
        mode = parsed.get("mode", "add")
        if mode not in ("add", "remove", "narrow"):
            mode = "add"
        return DecomposeResult(
            mode=mode,
            targets=parsed.get("targets", []),
            raw_response=raw,
        )
    except Exception as e:
        log.error("[VLM] OpenAI decompose failed: %s", e)
        return DecomposeResult(error=str(e), raw_response="")


def decompose_prompt(
    image: Path,
    user_prompt: str,
    *,
    provider: str | None = None,
) -> DecomposeResult:
    """Decompose a user guidance prompt into mode + SAM targets.

    Auto-detects provider from available env vars if not specified.
    """
    import os

    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            return DecomposeResult(error="No VLM API key found (ANTHROPIC_API_KEY or OPENAI_API_KEY)")

    log.info("[VLM] Decomposing prompt with %s: \"%s\"", provider, user_prompt)

    if provider == "anthropic":
        result = decompose_with_anthropic(image, user_prompt, env_key("ANTHROPIC_API_KEY"))
    else:
        result = decompose_with_openai(image, user_prompt, env_key("OPENAI_API_KEY"))

    if result.error:
        log.error("[VLM] Decompose failed: %s", result.error)
    else:
        log.info("[VLM] Mode: %s | Targets: %s", result.mode, result.targets)

    return result
