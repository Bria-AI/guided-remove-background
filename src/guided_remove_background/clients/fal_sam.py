"""SAM 3.1 client via Fal.ai — text-prompted object segmentation."""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import fal_client
import numpy as np
from PIL import Image

from .http_utils import env_key, http_get

log = logging.getLogger(__name__)

SAM_FAL_MODEL = "fal-ai/sam-3-1/image"


@dataclass
class SamHit:
    prompt: str
    mask: np.ndarray | None
    score: float
    error: str | None = None


def _sam_one(fal_url: str, prompt: str) -> SamHit:
    """Run SAM on a single prompt."""
    try:
        result = fal_client.subscribe(
            SAM_FAL_MODEL,
            arguments={
                "image_url": fal_url,
                "prompt": prompt,
                "apply_mask": False,
                "include_scores": True,
            },
        )
    except Exception as e:
        return SamHit(prompt=prompt, mask=None, score=0.0, error=str(e))

    masks = result.get("masks", [])
    scores = result.get("scores", [])
    if not masks:
        return SamHit(prompt=prompt, mask=None, score=0.0, error="no mask returned")

    score = float(scores[0]) if scores else 0.0
    try:
        r = http_get(masks[0]["url"], timeout=60)
        arr = np.array(Image.open(io.BytesIO(r.content)).convert("L")) > 128
    except Exception as e:
        return SamHit(prompt=prompt, mask=None, score=score, error=f"download failed: {e}")

    return SamHit(prompt=prompt, mask=arr, score=score)


def call_sam(
    image_path: Path,
    prompts: list[str],
    *,
    min_score: float = 0.0,
) -> tuple[np.ndarray | None, dict[str, float], dict[str, np.ndarray]]:
    """Run SAM 3.1 per prompt in parallel.

    Returns (union_mask, per-prompt scores, per-prompt individual masks).
    """
    import os
    os.environ["FAL_KEY"] = env_key("FAL_KEY")

    log.info("[SAM] Uploading %s to Fal ...", image_path.name)
    try:
        fal_url = fal_client.upload_file(str(image_path))
    except Exception as e:
        log.error("[SAM] Upload failed (account issue?): %s", e)
        return None, {p: 0.0 for p in prompts}, {}

    log.info("[SAM] Running %d prompt(s): %s", len(prompts), prompts)
    hits: list[SamHit] = []
    workers = min(4, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sam_one, fal_url, p): p for p in prompts}
        for f in as_completed(futs):
            hits.append(f.result())

    union: np.ndarray | None = None
    scores: dict[str, float] = {}
    individual: dict[str, np.ndarray] = {}
    for h in hits:
        scores[h.prompt] = h.score
        if h.error:
            log.warning("[SAM] '%s' failed: %s", h.prompt, h.error)
            continue
        if h.score < min_score:
            log.warning("[SAM] '%s' score %.3f < threshold %.2f, skipping", h.prompt, h.score, min_score)
            continue
        log.info("[SAM] '%s' score=%.3f accepted", h.prompt, h.score)
        individual[h.prompt] = h.mask
        union = h.mask if union is None else (union | h.mask)

    if union is None:
        log.warning("[SAM] No usable masks from %d prompt(s)", len(prompts))
    return union, scores, individual
