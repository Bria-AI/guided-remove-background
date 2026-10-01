"""Candidate F1: keep the SAM self-correction retry, drop Opus as the trigger.

Same RMBG -> VLM decompose -> SAM -> mode-specific merge chain as the production
pipeline (`remove_bg.py`), but the judge verification loop (`_verify_and_correct`,
one or more Opus calls per image) is replaced with a single deterministic retry
right after Step 3, before the mode-specific merge runs:

  retry a target's SAM call (with a simplified prompt) if either
    - SAM found nothing for it (missing from `individual`), or
    - SAM's own confidence score for it is below `low_conf_threshold`

This tests whether most of what the judge loop was buying (~20pts of pass rate)
can be had from SAM's own signal, without any Opus round-trip in the request path.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes

from ..clients.bria_rmbg import call_rmbg
from ..clients.fal_sam import call_sam
from ..clients.vlm_decompose import decompose_prompt, DecomposeResult
from ..processing.debug import rmbg_overlay
from ..processing.output import save_preview, save_result
from ..remove_bg import _build_alpha

log = logging.getLogger(__name__)


@dataclass
class DeterministicRetryResult:
    output_path: Path
    preview_path: Path
    elapsed_s: float
    sam_scores: dict[str, float] = field(default_factory=dict)
    vlm_decompose: DecomposeResult | None = None
    retried_targets: list[str] = field(default_factory=list)
    error: str | None = None


def _simplify(prompt: str) -> str | None:
    words = prompt.split()[-2:]
    simplified = " ".join(words) if words else None
    return simplified if simplified and simplified != prompt else None


def remove_bg_deterministic_retry(
    image: Path,
    prompts: list[str],
    output: Path,
    *,
    dilation: int = 3,
    edge_band_px: int = 0,
    blur: float = 1.0,
    min_score: float = 0.0,
    low_conf_threshold: float = 0.5,
    vlm_provider: str | None = None,
) -> DeterministicRetryResult:
    t0 = time.monotonic()
    orig = np.array(Image.open(image).convert("RGBA"))
    h, w = orig.shape[:2]
    if edge_band_px <= 0:
        edge_band_px = max(4, min(12, int(max(h, w) * 0.005)))

    rmbg_arr = call_rmbg(image)
    rmbg_alpha = rmbg_arr[:, :, 3]
    rmbg_mask = rmbg_alpha > 128

    user_prompt = " ".join(prompts)
    overlay_img = rmbg_overlay(orig, rmbg_mask)
    overlay_tmp = Path(tempfile.mktemp(suffix=".jpg"))
    overlay_img.convert("RGB").save(overlay_tmp, quality=90)
    try:
        vlm_result = decompose_prompt(
            image, user_prompt, provider=vlm_provider, rmbg_overlay_path=overlay_tmp,
        )
    finally:
        overlay_tmp.unlink(missing_ok=True)

    if vlm_result.error:
        log.warning("VLM decompose failed; returning RMBG baseline")
        orig[:, :, 3] = rmbg_alpha
        elapsed = time.monotonic() - t0
        out_path = save_result(orig, output)
        prev_path = save_preview(orig, output)
        return DeterministicRetryResult(
            output_path=out_path, preview_path=prev_path, elapsed_s=elapsed,
            vlm_decompose=vlm_result, error="vlm_decompose_failed",
        )

    vlm_mode = None
    targets: list[str] = []
    if vlm_result.mode == "add_remove":
        all_targets = vlm_result.add_targets + vlm_result.remove_targets
        if all_targets:
            vlm_mode = "add_remove"
            targets = all_targets
    elif vlm_result.targets:
        vlm_mode = vlm_result.mode
        targets = vlm_result.targets

    if not vlm_mode or not targets:
        orig[:, :, 3] = rmbg_alpha
        elapsed = time.monotonic() - t0
        out_path = save_result(orig, output)
        prev_path = save_preview(orig, output)
        return DeterministicRetryResult(
            output_path=out_path, preview_path=prev_path, elapsed_s=elapsed,
            vlm_decompose=vlm_result,
        )

    # --- Step 3: SAM segmentation of targets ---
    sam_mask, scores, individual = call_sam(image, targets, min_score=min_score)

    # --- Deterministic retry (replaces the judge loop as the trigger) ---
    retried: list[str] = []
    to_retry = [
        t for t in targets
        if t not in individual or scores.get(t, 0.0) < low_conf_threshold
    ]
    simplifications = {t: _simplify(t) for t in to_retry}
    simplifications = {t: s for t, s in simplifications.items() if s}
    if simplifications:
        retry_prompts = list(dict.fromkeys(simplifications.values()))
        log.info("[DetRetry] Low-confidence/missing targets %s -> retrying with %s",
                 list(simplifications), retry_prompts)
        _, retry_scores, retry_individual = call_sam(image, retry_prompts, min_score=min_score)
        for target, simplified in simplifications.items():
            new_score = retry_scores.get(simplified)
            if new_score is None:
                continue
            old_score = scores.get(target, 0.0)
            if simplified in retry_individual and new_score > old_score:
                individual[target] = retry_individual[simplified]
                scores[target] = new_score
                retried.append(target)
                log.info("[DetRetry] '%s' improved %.3f -> %.3f via '%s'",
                         target, old_score, new_score, simplified)

    if individual:
        sam_mask = None
        for m in individual.values():
            sam_mask = m if sam_mask is None else (sam_mask | m)

    sam_scores = dict(scores)

    # --- Step 4: mode-specific mask logic (identical to remove_bg.py, judge-free) ---
    if sam_mask is None:
        log.warning("SAM found nothing; returning RMBG baseline")
        final_mask = rmbg_mask.copy()

    elif vlm_mode == "add":
        expanded = binary_dilation(sam_mask, iterations=dilation)
        merged = rmbg_mask | expanded
        final_mask = binary_fill_holes(merged)

    elif vlm_mode == "remove":
        rmbg_core = binary_erosion(rmbg_mask, iterations=max(8, edge_band_px))
        rmbg_total = max(int(rmbg_mask.sum()), 1)
        remove_mask = np.zeros_like(rmbg_mask)
        for prompt, obj_mask in individual.items():
            selector = binary_dilation(obj_mask, iterations=max(dilation, 3))
            obj_rmbg = selector & rmbg_mask
            obj_size_ratio = int(obj_rmbg.sum()) / rmbg_total
            is_small_interior = (
                obj_size_ratio < 0.05
                and int((selector & rmbg_core).sum()) / max(int(selector.sum()), 1) > 0.9
            )
            if is_small_interior:
                continue
            remove_mask |= selector
        final_mask = rmbg_mask & ~remove_mask

    elif vlm_mode == "narrow":
        final_mask = np.zeros_like(rmbg_mask)
        for prompt, obj_mask in individual.items():
            selector = binary_dilation(obj_mask, iterations=dilation)
            obj_rmbg = rmbg_mask & selector
            rmbg_coverage = int(obj_rmbg.sum()) / max(int(selector.sum()), 1)
            if rmbg_coverage > 0.15:
                padded = binary_dilation(obj_rmbg, iterations=3)
                filled = binary_fill_holes(padded)
                filled &= rmbg_mask
                final_mask |= filled
            else:
                final_mask |= selector

    elif vlm_mode == "add_remove":
        add_prompts = set(vlm_result.add_targets)
        remove_prompts = set(vlm_result.remove_targets)

        add_mask = np.zeros_like(rmbg_mask)
        for prompt, obj_mask in individual.items():
            if prompt in add_prompts:
                add_mask |= binary_dilation(obj_mask, iterations=dilation)

        rmbg_core = binary_erosion(rmbg_mask, iterations=max(8, edge_band_px))
        rmbg_total = max(int(rmbg_mask.sum()), 1)
        remove_mask = np.zeros_like(rmbg_mask)
        for prompt, obj_mask in individual.items():
            if prompt in remove_prompts:
                selector = binary_dilation(obj_mask, iterations=max(dilation, 3))
                obj_rmbg = selector & rmbg_mask
                obj_size_ratio = int(obj_rmbg.sum()) / rmbg_total
                is_small_interior = (
                    obj_size_ratio < 0.05
                    and int((selector & rmbg_core).sum()) / max(int(selector.sum()), 1) > 0.9
                )
                if is_small_interior:
                    continue
                remove_mask |= selector

        merged = (rmbg_mask | add_mask) & ~remove_mask
        final_mask = binary_fill_holes(merged)

    else:
        final_mask = rmbg_mask.copy()

    final_alpha = _build_alpha(final_mask, rmbg_alpha, edge_band_px, blur, vlm_mode)
    orig[:, :, 3] = final_alpha

    elapsed = time.monotonic() - t0
    out_path = save_result(orig, output)
    prev_path = save_preview(orig, output)

    return DeterministicRetryResult(
        output_path=out_path,
        preview_path=prev_path,
        elapsed_s=elapsed,
        sam_scores=sam_scores,
        vlm_decompose=vlm_result,
        retried_targets=retried,
    )
