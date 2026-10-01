"""Candidates B and D from AG-184: RMBG 2.0 alpha unioned with SAM's text-prompted
mask, with no VLM decomposition step and no judge/self-correction loop.

This is the "naive baseline" — the raw prompt is sent to SAM as-is (no mode/target
reasoning), and the result is always `final = RMBG ∪ SAM(prompt)`. It is expected to
fail exclude/narrow-style prompts by design (a union can only add pixels, never
subtract them) — that's a deliberate finding for the comparison, not a bug here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes

from ..clients.bria_rmbg import call_rmbg
from ..clients.fal_sam import call_sam
from ..processing.output import save_preview, save_result
from ..remove_bg import _build_alpha

log = logging.getLogger(__name__)


@dataclass
class RmbgSamDirectResult:
    output_path: Path
    preview_path: Path
    elapsed_s: float
    sam_scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def rmbg_sam_direct(
    image: Path,
    prompts: list[str],
    output: Path,
    *,
    dilation: int = 3,
    edge_band_px: int = 6,
    blur: float = 1.0,
    min_score: float = 0.0,
) -> RmbgSamDirectResult:
    """RMBG baseline unioned with SAM called directly on the case's raw prompt text(s) —
    no VLM decomposition, so each prompt phrasing is sent to SAM as-is and unioned."""
    t0 = time.monotonic()
    orig = np.array(Image.open(image).convert("RGBA"))

    rmbg_arr = call_rmbg(image)
    rmbg_alpha = rmbg_arr[:, :, 3]
    rmbg_mask = rmbg_alpha > 128

    sam_mask, sam_scores, _ = call_sam(image, prompts, min_score=min_score)

    if sam_mask is None:
        log.warning("[rmbg_sam_direct] SAM found nothing for %r; returning RMBG baseline", prompts)
        final_mask = rmbg_mask.copy()
    else:
        expanded = binary_dilation(sam_mask, iterations=dilation)
        merged = rmbg_mask | expanded
        final_mask = binary_fill_holes(merged)

    final_alpha = _build_alpha(final_mask, rmbg_alpha, edge_band_px, blur, vlm_mode="add")
    orig[:, :, 3] = final_alpha

    elapsed = time.monotonic() - t0
    out_path = save_result(orig, output)
    prev_path = save_preview(orig, output)

    return RmbgSamDirectResult(
        output_path=out_path,
        preview_path=prev_path,
        elapsed_s=elapsed,
        sam_scores=sam_scores,
    )
