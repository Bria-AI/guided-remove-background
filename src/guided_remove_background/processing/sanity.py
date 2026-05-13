"""Pipeline sanity guards — catch RMBG/SAM disagreements before final output.

Two ratio-based guards:
  1. Agreement check: RMBG zeroed out too much of SAM's foreground → SAM-only fallback
  2. Bloat check: final alpha leaked far beyond SAM mask → constrain to SAM
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt

log = logging.getLogger(__name__)


def _feathered_sam_alpha(sam_mask: np.ndarray, blur: float = 1.5) -> np.ndarray:
    """Soft alpha from SAM mask with distance-based feathering at edges."""
    alpha = (sam_mask.astype(np.uint8) * 255)
    if blur > 0:
        alpha = np.array(
            Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=blur))
        )
    return alpha


def check_rmbg_sam_agreement(
    final_alpha: np.ndarray,
    sam_mask: np.ndarray,
    *,
    min_keep_ratio: float = 0.7,
    blur: float = 1.5,
) -> tuple[np.ndarray, str, float]:
    """If edge refinement dropped too much of SAM's foreground, RMBG disagrees.

    Returns (alpha, mode, ratio) where mode is "keep" or "sam-fallback".
    """
    sam_pixels = int(sam_mask.sum())
    if sam_pixels == 0:
        return final_alpha, "keep", 1.0

    kept = int(((final_alpha > 0) & sam_mask).sum())
    ratio = kept / sam_pixels

    if ratio >= min_keep_ratio:
        log.info(
            "[Sanity] RMBG-SAM agreement: %.1f%% kept (>= %.0f%%) — OK",
            ratio * 100, min_keep_ratio * 100,
        )
        return final_alpha, "keep", ratio

    log.warning(
        "[Sanity] RMBG-SAM agreement: only %.1f%% kept (< %.0f%%) — falling back to SAM alpha",
        ratio * 100, min_keep_ratio * 100,
    )
    return _feathered_sam_alpha(sam_mask, blur=blur), "sam-fallback", ratio


def check_bloat(
    final_alpha: np.ndarray,
    sam_mask: np.ndarray,
    *,
    max_bloat: float = 1.5,
    blur: float = 1.5,
) -> tuple[np.ndarray, str, float]:
    """If final alpha is much larger than SAM mask, RMBG leaked beyond request.

    Returns (alpha, mode, bloat_ratio) where mode is "keep" or "sam-constrained".
    """
    sam_px = int(sam_mask.sum())
    if sam_px == 0:
        return final_alpha, "keep", 0.0

    final_px = int((final_alpha > 0).sum())
    bloat = final_px / sam_px

    if bloat <= max_bloat:
        log.info(
            "[Sanity] Bloat check: %.2fx (<=%.2f) — OK",
            bloat, max_bloat,
        )
        return final_alpha, "keep", bloat

    log.warning(
        "[Sanity] Bloat check: %.2fx (>%.2f) — constraining to SAM mask",
        bloat, max_bloat,
    )
    return _feathered_sam_alpha(sam_mask, blur=blur), "sam-constrained", bloat
