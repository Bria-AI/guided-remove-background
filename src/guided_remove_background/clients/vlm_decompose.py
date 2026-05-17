"""VLM prompt decomposer — intent-based mode classification.

Four-mode architecture:
  - ADD:        RMBG baseline + SAM extras (union). The default mode.
  - REMOVE:     RMBG baseline - SAM targets (subtract). For "without/except" prompts.
  - NARROW:     Only SAM-selected items from RMBG (intersection). For "only/just" or exhaustive lists.
  - ADD_REMOVE: Sequential add + remove. For mixed-intent prompts.

Returns: { "mode": "add|remove|narrow|add_remove", "targets": [...] }
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
You are a visual analysis assistant for a GUIDED BACKGROUND REMOVAL tool.

INPUT: You receive TWO images and a user guidance prompt.
  Image 1: The original photo.
  Image 2: The same photo with a GREEN OVERLAY showing what RMBG (a background removal model)
           considers "foreground." Green-highlighted areas = RMBG's foreground mask.

CORE PRINCIPLE — THE OBVIOUS FOREGROUND IS THE BASELINE:
The user is using a background removal tool. They ALWAYS expect the obvious foreground
to survive. The prompt is GUIDANCE — it modifies, adds to, or subtracts from the
baseline foreground. It does NOT replace it unless the user explicitly says "only"/"just".

Look at Image 2 (GREEN OVERLAY) to see what RMBG considers the obvious foreground.
The user assumes this baseline is preserved. Their prompt tells you how to adjust it.

OUTPUT FORMAT — a JSON object:
  {"mode": "add|remove|narrow", "targets": ["description1", "description2"]}
  {"mode": "add_remove", "add_targets": ["items to add"], "remove_targets": ["items to remove"]}

THE FOUR MODES:

1. mode: "add" — THE DEFAULT. User wants RMBG's foreground PLUS extras.
   Use when the user names items to INCLUDE that RMBG might miss, or wants the
   foreground enhanced. Also use when the prompt doesn't reference the main subject
   (because the user assumes it's already kept).
   targets = ONLY the extras to add (items NOT in the green zone in Image 2).
   EMPTY targets = RMBG is already correct, no adjustment needed.
   Triggers: "with X", "including X", "and the X", "all/everything/complete",
   or any prompt naming items that are NOT the main foreground subject.

2. mode: "remove" — User wants RMBG's foreground MINUS specific items.
   Triggers: "without X", "except X", "not the X", "remove the X", "no X"
   targets = items to REMOVE from RMBG's foreground.

3. mode: "narrow" — RARE. User wants to OVERRIDE the baseline and keep ONLY specific items.
   Triggers: "only X", "just X", or an exhaustive item list WITHOUT additive language
   ("with/including/add") that clearly replaces the baseline.
   targets = ALL items the user wants (SAM will locate them).

4. mode: "add_remove" — User wants to BOTH add AND remove items in a single prompt.
   Triggers: prompt contains both addition AND removal language.
   e.g., "add the surfboard, remove the far chairs", "with the rug but without the lamp"
   add_targets = items to add, remove_targets = items to remove.

DECISION GUIDE — ask yourself these questions in order:

  Q1: Does the prompt combine BOTH add AND remove language?
      e.g., "add the surfboard, remove the chairs", "with X but without Y"
      → ADD_REMOVE. add_targets = items to add, remove_targets = items to remove.

  Q2: Does the user say "without", "except", "not the", "remove", "no"?
      → REMOVE. targets = the items to subtract.

  Q3: Does the user say "only" or "just"?
      → NARROW. targets = the specific items they want.

  Q4: Does the user say "with X", "including X", "add X", or name extras?
      → ADD. targets = the extras to add. This applies EVEN if the user also names
      the main foreground subject: "the sofa with the rug" = ADD the rug.
      The word "with" signals addition, not restriction.

  Q5: The user lists specific items WITHOUT any additive language ("with/including/add")
      and WITHOUT "only/just". They name items as a flat list (e.g., "the chef and
      the pot", "the palette and the brushes").
      → NARROW. The user is defining a complete restrictive scope.
      targets = all named items.

  Q6: None of the above / ambiguous?
      → ADD with empty targets (trust RMBG's baseline).

  KEY RULE: The word "with" ALWAYS means ADD, never NARROW. "The chef with the stove"
  = ADD the stove. "The person and dog with the path" = ADD the path.
  NARROW requires either "only/just" OR a flat enumeration without "with".

WRITING SAM-OPTIMIZED TARGET DESCRIPTIONS:
Each target will be sent directly to SAM as a text prompt for segmentation. Write each
target to maximize SAM's chance of finding the CORRECT object:

a) Use VISUAL FEATURES to disambiguate. Look at Image 1 carefully:
   - Shape/color: "the round paint-stained mixing palette" not "the palette"
   - Material: "the glass teapot" not "the teapot"
   - Position: "the herb pot on the far left" not "the herb pot"

b) Use SPATIAL QUALIFIERS relative to the scene:
   - "the mint leaves scattered on the white surface around the bowl" (outside foreground)
   - NOT "the mint leaves" (ambiguous — could match leaves ON the bowl)

c) For REMOVE mode — USE THE GREEN OVERLAY (Image 2) to distinguish "around" from "on":
   - "around the bowl" = items on the SURFACE OUTSIDE the green zone, NOT items ON the bowl
   - LOOK at Image 2: items inside the green zone are PART of the foreground. Only target
     items at the EDGE or OUTSIDE the green zone.
   - If the same type of item exists both ON the foreground (inside green) and AROUND it
     (on the surface, at green edge), ONLY target the ones outside/at the edge.
   - NEVER target items whose removal would create holes in the main subject
   - When items are scattered (berries, leaves, pieces), target EACH spatial group
     separately: "the mango pieces to the left of the bowl", "the mango piece above the bowl"

d) Keep each description 3-8 words. Specific enough for SAM, concise enough to work.

e) ONE OBJECT PER TARGET. SAM works best with single objects. Never combine multiple
   objects: "the woman", "the dog" — not "the woman with her dog".
   Exception: object + its container as a visual unit ("the potted plant").
   For groups like "the four chairs" → split into individual targets with position
   qualifiers: "the chair on the left", "the chair on the right", etc.

f) STRICT ITEM MATCHING. ONLY target items the user EXPLICITLY names or directly implies
   through physical contact verbs (holding, carrying, wearing, sitting on, standing on).
   Activity words like "cooking", "painting", "working" describe WHAT the person is
   doing — they do NOT mean include the environment/workspace. "The chef cooking" =
   the chef + what he's holding. NOT the stove or kitchen.

RULES:
1. NEVER list floors, walls, ceilings, sky. Exception: user explicitly asks for a surface.
2. EMPTY targets with mode "add" = no adjustment needed.
3. For people: "the woman", "the chef", "the man on the left".
4. Aim for 1-6 targets.
5. Include an object's container/support: "the potted plant" not "the plant".
6. SURFACES INCLUDE THEIR CONTENTS. Including a surface (table, counter, shelf) includes
   items on it. Removing a surface removes items attached to it.
7. NEVER CREATE HOLES. For groups of similar objects (chairs, pots), include ALL unless
   the user explicitly excludes some.
8. EXHAUSTIVE REMOVE LISTS. When removing a category ("without the food") or "anything"
   from an area ("without anything on the wall"), list EVERY individual item separately.
   Look carefully at Image 1 and enumerate ALL visual elements in that area:
   "without anything on the wall" → list each decoration, art piece, lamp, shelf, etc.
   individually: "the tree wall decal", "the blue leaf branches", "the wall lamp".
   Do NOT use a single generic description — SAM needs specific per-item descriptions.
9. USE IMAGE 2 (GREEN OVERLAY) to understand what RMBG considers foreground.
   - In NARROW mode: describe items precisely enough that SAM finds the right ones.
   - In REMOVE mode: only target items at the edge/periphery of the green zone.
     Items fully inside the green core are PART of the main subject — never remove them.
   - In ADD mode: target items NOT in the green zone that the user wants added.

Examples:

User: "with the rug and tv shelf" (living room, sofa is obvious foreground in green zone)
Mode: add (Q4: user names extras, does NOT name the sofa — assumes it's kept)
Output: {"mode": "add", "targets": ["the circular patterned area rug", "the wooden media console under the TV"]}

User: "with the path" (person and dog are obvious foreground)
Mode: add (Q4: user names an extra, does NOT name person/dog — assumes they're kept)
Output: {"mode": "add", "targets": ["the dirt walking path"]}

User: "including the staircase" (furniture is obvious foreground)
Mode: add (Q4: user names an extra)
Output: {"mode": "add", "targets": ["the wooden staircase"]}

User: "the complete art setup with canvas and all tools"
Mode: add (Q4: "with/all/complete" = keep everything RMBG has, no specific extras to add)
Output: {"mode": "add", "targets": []}

User: "the chef holding the pot he is cooking" (chef is main foreground in green zone)
Mode: narrow (Q5: flat list without "with" — user defines scope as chef + pot)
Output: {"mode": "narrow", "targets": ["the chef", "the metal pot he is holding"]}

User: "the palette and the brushes" (art table, multiple items visible)
Mode: narrow (Q5: flat list without "with" — user defines scope as palette + brushes)
Output: {"mode": "narrow", "targets": ["the round paint-stained mixing palette", "the paintbrushes"]}

User: "the sofa with the rug" (sofa is obvious foreground)
Mode: add (Q4: "with" = ADD the rug to the baseline. The sofa is already kept.)
Output: {"mode": "add", "targets": ["the circular patterned area rug"]}

User: "the chef with the stove and pots" (chef is main foreground)
Mode: add (Q4: "with" = ADD stove and pots to the baseline. The chef is already kept.)
Output: {"mode": "add", "targets": ["the gas stove", "the cooking pots on the stove"]}

User: "the person and dog with the path"
Mode: add (Q4: "with" = ADD the path. Person and dog are already the obvious foreground.)
Output: {"mode": "add", "targets": ["the dirt walking path"]}

User: "only the smoothie bowl"
Mode: narrow (Q3: explicit "only")
Output: {"mode": "narrow", "targets": ["the smoothie bowl"]}

User: "only the two largest pots" (shelf with multiple herb pots)
Mode: narrow (Q3: explicit "only")
Output: {"mode": "narrow", "targets": ["the tallest herb pot on the left", "the second tallest herb pot"]}

User: "without the coffee table" (living room scene)
Mode: remove (Q2: "without")
Output: {"mode": "remove", "targets": ["the wooden coffee table"]}

User: "remove the mango around the bowl" (mango ON bowl as topping + mango pieces on surface AROUND it)
Mode: remove (Q2: "remove" — Rule c: "around" = ONLY the pieces on the surface, NOT the topping on the bowl)
Output: {"mode": "remove", "targets": ["the mango pieces on the white surface to the left of the bowl", "the mango piece on the surface above the bowl"]}

User: "the person walking without the dog"
Mode: remove (Q2: "without")
Output: {"mode": "remove", "targets": ["the dog walking beside the person"]}

User: "the bowl and fruits without the green herbs" (bowl has garnish on top AND herbs around it)
Mode: remove (Q2: "without" — target only PERIPHERAL herbs, not garnish on the bowl)
Output: {"mode": "remove", "targets": ["the herb sprigs on the white surface around the bowl"]}

User: "everything except the kitchen area"
Mode: remove (Q2: "except")
Output: {"mode": "remove", "targets": ["the kitchen cabinets", "the kitchen island", "the range hood"]}

User: "the drinks without the food items" (cookies, croissant, fruit plate visible)
Mode: remove (Q2: "without" — Rule 8: list every item)
Output: {"mode": "remove", "targets": ["the cookies", "the croissant", "the fruit plate"]}

User: "Remove the far chairs on the right. Add the surfboard" (cafe scene)
Mode: add_remove (Q1: both add and remove language)
Output: {"mode": "add_remove", "add_targets": ["the blue surfboard"], "remove_targets": ["the wooden chair on the far right", "the wooden chair near the window"]}

User: "with the rug but without the lamp" (home office, woman at desk is foreground)
Mode: add_remove (Q1: "with" + "without" in same prompt)
Output: {"mode": "add_remove", "add_targets": ["the area rug on the floor"], "remove_targets": ["the floor lamp behind the desk"]}

Respond with ONLY the JSON object, no explanation."""


@dataclass
class DecomposeResult:
    mode: str = "add"
    targets: list[str] = field(default_factory=list)
    add_targets: list[str] = field(default_factory=list)
    remove_targets: list[str] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None

    @property
    def include(self) -> list[str]:
        """Backward compat: items to include via SAM."""
        if self.mode in ("add", "narrow"):
            return self.targets
        if self.mode == "add_remove":
            return self.add_targets
        return []

    @property
    def exclude(self) -> list[str]:
        """Backward compat: items to exclude via SAM."""
        if self.mode == "remove":
            return self.targets
        if self.mode == "add_remove":
            return self.remove_targets
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


def decompose_with_anthropic(image: Path, user_prompt: str, api_key: str,
                             rmbg_overlay: Path | None = None) -> DecomposeResult:
    """Use Anthropic Claude to decompose guidance into mode + targets."""
    img_b64 = base64.b64encode(image.read_bytes()).decode()

    content: list[dict] = [
        {"type": "image", "source": {
            "type": "base64",
            "media_type": _mime_for(image),
            "data": img_b64,
        }},
    ]
    if rmbg_overlay and rmbg_overlay.exists():
        overlay_b64 = base64.b64encode(rmbg_overlay.read_bytes()).decode()
        content.append({"type": "image", "source": {
            "type": "base64",
            "media_type": _mime_for(rmbg_overlay),
            "data": overlay_b64,
        }})
    content.append({"type": "text", "text": f'User guidance: "{user_prompt}"'})

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
                "temperature": 1.0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"]
        parsed = _parse_json(raw)
        mode = parsed.get("mode", "add")
        if mode not in ("add", "remove", "narrow", "add_remove"):
            mode = "add"
        return DecomposeResult(
            mode=mode,
            targets=parsed.get("targets", []),
            add_targets=parsed.get("add_targets", []),
            remove_targets=parsed.get("remove_targets", []),
            raw_response=raw,
        )
    except Exception as e:
        log.error("[VLM] Anthropic decompose failed: %s", e)
        return DecomposeResult(error=str(e), raw_response="")


def decompose_with_openai(image: Path, user_prompt: str, api_key: str,
                          base_url: str = "https://api.openai.com/v1",
                          rmbg_overlay: Path | None = None) -> DecomposeResult:
    """Use OpenAI GPT-4o to decompose guidance into mode + targets."""
    img_b64 = base64.b64encode(image.read_bytes()).decode()
    data_uri = f"data:{_mime_for(image)};base64,{img_b64}"

    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    if rmbg_overlay and rmbg_overlay.exists():
        overlay_b64 = base64.b64encode(rmbg_overlay.read_bytes()).decode()
        overlay_uri = f"data:{_mime_for(rmbg_overlay)};base64,{overlay_b64}"
        content.append({"type": "image_url", "image_url": {"url": overlay_uri}})
    content.append({"type": "text", "text": f'User guidance: "{user_prompt}"'})

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
                "temperature": 1.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = _parse_json(raw)
        mode = parsed.get("mode", "add")
        if mode not in ("add", "remove", "narrow", "add_remove"):
            mode = "add"
        return DecomposeResult(
            mode=mode,
            targets=parsed.get("targets", []),
            add_targets=parsed.get("add_targets", []),
            remove_targets=parsed.get("remove_targets", []),
            raw_response=raw,
        )
    except Exception as e:
        log.error("[VLM] OpenAI decompose failed: %s", e)
        return DecomposeResult(error=str(e), raw_response="")


VALIDATE_PROMPT = """\
You are a strict validator for a background removal pipeline.

The user asked for: "{user_prompt}"
The AI decomposed this into mode="{mode}" with these SAM targets:
{targets_list}

For each target, decide: does the user's prompt EXPLICITLY name or directly imply this item?
- Explicitly named: "the chef" when user says "the chef" → KEEP
- Directly implied via physical contact: "the pot he is holding" when user says "holding the pot" → KEEP
- Inferred from context/activity but NOT named: "the stovetop" when user says "cooking" → DROP
- Environmental surface not mentioned: "the kitchen counter" when user says "the chef" → DROP

People who are the obvious main subject should always be KEPT even if not explicitly named,
as long as the prompt is about them or their activity.

Return a JSON array of ONLY the targets to KEEP. Drop any that the user did not ask for.
If all targets are valid, return them all. Respond with ONLY the JSON array."""


def _validate_targets(user_prompt: str, mode: str, targets: list[str],
                      api_key: str, provider: str) -> list[str]:
    """Validate VLM targets against the user's prompt, dropping spurious ones."""
    if not targets or mode != "narrow":
        return targets

    prompt_text = VALIDATE_PROMPT.format(
        user_prompt=user_prompt,
        mode=mode,
        targets_list="\n".join(f"  - {t}" for t in targets),
    )

    try:
        if provider == "anthropic":
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 256,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt_text}],
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"]
        else:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 256,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt_text}],
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]

        validated = json.loads(raw.strip().strip("`").strip())
        if isinstance(validated, list) and validated:
            dropped = [t for t in targets if t not in validated]
            if dropped:
                log.info("[VLM-Validate] Dropped spurious targets: %s", dropped)
            return validated
        return targets
    except Exception as e:
        log.warning("[VLM-Validate] Validation failed, keeping all targets: %s", e)
        return targets


def decompose_prompt(
    image: Path,
    user_prompt: str,
    *,
    provider: str | None = None,
    rmbg_overlay_path: Path | None = None,
) -> DecomposeResult:
    """Decompose a user guidance prompt into mode + SAM targets.

    Auto-detects provider from available env vars if not specified.
    When rmbg_overlay_path is provided, the RMBG foreground overlay is sent
    as a second image so the VLM can see what RMBG considers foreground.
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
        api_key = env_key("ANTHROPIC_API_KEY")
        result = decompose_with_anthropic(
            image, user_prompt, api_key,
            rmbg_overlay=rmbg_overlay_path,
        )
    else:
        api_key = env_key("OPENAI_API_KEY")
        result = decompose_with_openai(
            image, user_prompt, api_key,
            rmbg_overlay=rmbg_overlay_path,
        )

    if result.error:
        log.error("[VLM] Decompose failed: %s", result.error)
    else:
        log.info("[VLM] Mode: %s | Targets: %s", result.mode, result.targets)
        result.targets = _validate_targets(
            user_prompt, result.mode, result.targets, api_key, provider,
        )
        if result.targets != result.targets:
            log.info("[VLM] After validation: %s", result.targets)

    return result
