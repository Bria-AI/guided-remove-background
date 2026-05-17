"""VLM judge — visual verification of pipeline results.

Sends the original image + final result + user prompt to a VLM and asks:
"Does this result match what the user wanted?"

Returns structured issues with actionable fix suggestions that the pipeline
can execute automatically (re-prompt SAM, fill holes, etc.).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .http_utils import env_key

log = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are a quality judge for a GUIDED BACKGROUND REMOVAL pipeline.

You receive THREE images and the user's original prompt:
  Image 1: The original photo (before any processing).
  Image 2: The final result — foreground on a WHITE background.
             White = transparent (removed). Non-white = kept (foreground).
  Image 3: The RMBG baseline — what standard background removal produces
             (before any guided adjustments).

STEP 1 — INVENTORY: Before judging, carefully examine ONLY Image 2 (the result).
List every distinct object you can see in Image 2. Do NOT look at Image 1 while
doing this. Base your inventory SOLELY on what is visible in Image 2.

STEP 2 — COMPARE: Now compare your inventory against the user's prompt.
Determine if the result matches what the user asked for.

HOW THE PIPELINE WORKS (important context):
- The pipeline starts with RMBG (automatic background removal) as a baseline.
- Then it uses SAM (Segment Anything Model) to adjust the mask based on the prompt.
- SAM works by finding objects from text descriptions. It can fail to find things.
- The pipeline can ADD items to the RMBG baseline, REMOVE items, or NARROW to specific items.
- Background items (walls, floors, ceilings, distant scenery) are EXPECTED to be removed.
  Do NOT flag their removal as an issue.

CRITICAL: YOUR CORRECTIONS CAN MAKE THINGS WORSE.
Every time you fail a result, the pipeline will attempt to fix it by re-running
SAM. Overcorrecting is WORSE than accepting an imperfect result. A result that's
80% correct is better than one that's been aggressively "fixed" into 30% correct.
Only fail when you are HIGHLY CONFIDENT the fix will improve things.

ONLY FAIL FOR THESE SPECIFIC, CLEAR-CUT PROBLEMS:

1. MISSING ITEMS — A specific, discrete, clearly named object the user explicitly
   asked for is COMPLETELY absent (not in your Image 2 inventory). Only flag if:
   - The object is explicitly named in the prompt (e.g., "with the surfboard")
   - The object is a single, well-defined physical thing (not a zone/area/region)
   - The object is 100% gone — you did NOT list it in your inventory
   Do NOT flag: background removal, ambiguous zones like "the kitchen area",
   partially visible items, architectural elements, or contextual items.

2. UNWANTED ITEMS — A specific object the user did NOT ask for (or explicitly asked
   to remove) is clearly present in your Image 2 inventory. For NARROW prompts
   ("only X", "just X"), any object NOT named is potentially unwanted. Only flag if:
   - The object is a single, well-defined thing (e.g., "the dog", "the lamp")
   - NOT a zone/area/region (e.g., do NOT flag "kitchen area still visible")
   - You are CERTAIN the object is in Image 2 — you listed it in your inventory
   Do NOT flag: partial remnants, zones, areas, architectural boundaries, or
   ambiguous spatial regions.

3. MASK ARTIFACTS — Only large, obvious holes INSIDE a foreground object (not edges).

PASS everything else. When in doubt, ALWAYS PASS.

For each issue, suggest ONE fix:
- "re-sam: <description>" — re-run segmentation with a better object description.
  Only use for single, specific, well-defined objects. Write 3-6 words.
- "fill-holes" — fix holes inside foreground objects.
- "no-fix" — issue detected but cannot be automatically fixed.

AVOID suggesting "shrink-mask" or "expand-mask" — these are dangerous and often
make things worse by removing/adding too much.

OUTPUT FORMAT — respond with ONLY a JSON object:
{
  "inventory": ["object 1 in Image 2", "object 2 in Image 2"],
  "pass": true/false,
  "issues": [
    {
      "type": "missing_item|unwanted_item|mask_artifact",
      "description": "what's wrong",
      "fix": "re-sam: the golden retriever dog"
    }
  ]
}

If the result passes, return: {"inventory": [...], "pass": true, "issues": []}
Respond with ONLY the JSON object, no explanation."""


@dataclass
class JudgeIssue:
    type: str
    description: str
    fix: str


@dataclass
class JudgeVerdict:
    passed: bool
    issues: list[JudgeIssue] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None


def _mime_for(path: Path) -> str:
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


ANTHROPIC_MODELS = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}

DEFAULT_JUDGE_MODEL = "opus"


def _judge_anthropic(original: Path, result_preview: Path,
                     rmbg_preview: Path | None,
                     user_prompt: str, api_key: str,
                     model: str = DEFAULT_JUDGE_MODEL) -> JudgeVerdict:
    model_id = ANTHROPIC_MODELS.get(model, model)
    content: list[dict] = []
    for img_path in [original, result_preview]:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        content.append({"type": "image", "source": {
            "type": "base64",
            "media_type": _mime_for(img_path),
            "data": b64,
        }})
    if rmbg_preview and rmbg_preview.exists():
        b64 = base64.b64encode(rmbg_preview.read_bytes()).decode()
        content.append({"type": "image", "source": {
            "type": "base64",
            "media_type": _mime_for(rmbg_preview),
            "data": b64,
        }})
    content.append({"type": "text", "text": f'User prompt: "{user_prompt}"'})

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "max_tokens": 1024,
            "temperature": 0,
            "system": JUDGE_PROMPT,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"]
    return _parse_verdict(raw)


def _judge_openai(original: Path, result_preview: Path,
                  rmbg_preview: Path | None,
                  user_prompt: str, api_key: str) -> JudgeVerdict:
    content: list[dict] = []
    for img_path in [original, result_preview]:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        uri = f"data:{_mime_for(img_path)};base64,{b64}"
        content.append({"type": "image_url", "image_url": {"url": uri}})
    if rmbg_preview and rmbg_preview.exists():
        b64 = base64.b64encode(rmbg_preview.read_bytes()).decode()
        uri = f"data:{_mime_for(rmbg_preview)};base64,{b64}"
        content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": f'User prompt: "{user_prompt}"'})

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "max_tokens": 512,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": content},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> JudgeVerdict:
    parsed = _parse_json(raw)
    passed = parsed.get("pass", True)
    issues = []
    for item in parsed.get("issues", []):
        issues.append(JudgeIssue(
            type=item.get("type", "unknown"),
            description=item.get("description", ""),
            fix=item.get("fix", "no-fix"),
        ))
    return JudgeVerdict(passed=passed, issues=issues, raw_response=raw)


def judge_result(
    original: Path,
    result_preview: Path,
    user_prompt: str,
    *,
    rmbg_preview: Path | None = None,
    provider: str | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
) -> JudgeVerdict:
    """Judge whether the pipeline result matches the user's intent.

    Args:
        original: Path to the original input image.
        result_preview: Path to the final result preview.
        user_prompt: The user's original guidance prompt.
        rmbg_preview: Optional path to the RMBG baseline preview for comparison.
        provider: "anthropic" or "openai". Auto-detected if None.
        model: Model shortname — "sonnet" or "opus" (Anthropic only).
    """
    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            return JudgeVerdict(passed=True, error="No VLM API key for judge")

    model_label = f"{provider}/{model}" if provider == "anthropic" else provider
    log.info("[Judge] Verifying result with %s...", model_label)

    try:
        if provider == "anthropic":
            api_key = env_key("ANTHROPIC_API_KEY")
            verdict = _judge_anthropic(original, result_preview, rmbg_preview,
                                       user_prompt, api_key, model=model)
        else:
            api_key = env_key("OPENAI_API_KEY")
            verdict = _judge_openai(original, result_preview, rmbg_preview,
                                    user_prompt, api_key)

        if verdict.passed:
            log.info("[Judge] PASS — result looks correct")
        else:
            log.warning("[Judge] FAIL — %d issue(s): %s",
                        len(verdict.issues),
                        "; ".join(i.description for i in verdict.issues))
        return verdict

    except Exception as e:
        log.error("[Judge] Verification failed: %s", e)
        return JudgeVerdict(passed=True, error=str(e))
