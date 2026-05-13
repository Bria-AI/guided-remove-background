"""Edge-band refinement — the core combination algorithm.

Hybrid approach:
  - SAM mask interior (beyond edge band): alpha = 255 (solid)
  - SAM mask edge band where RMBG has alpha: blend with RMBG's precise alpha
  - SAM mask edge band where RMBG has NO alpha: use gaussian-feathered SAM edge
  - Outside SAM mask: alpha = 0

Edge band is adaptive: for small masks, band_px is reduced so erosion
never consumes more than ~30% of the mask's effective diameter.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_erosion, distance_transform_edt


def _adaptive_band(sam_mask: np.ndarray, band_px: int) -> int:
    """Scale band_px down for small masks to prevent erosion from destroying them."""
    mask_pixels = int(sam_mask.sum())
    if mask_pixels <= 0:
        return band_px
    approx_diameter = int(np.sqrt(mask_pixels))
    max_band = max(2, approx_diameter // 4)
    return min(band_px, max_band)


def refine_edges(
    sam_mask: np.ndarray,
    rmbg_alpha: np.ndarray,
    band_px: int = 8,
    blur: float = 1.0,
) -> np.ndarray:
    """Build final alpha from SAM mask + RMBG alpha using hybrid edge-band blending.

    Uses RMBG's precise edge alpha where RMBG overlaps the SAM mask's edge band.
    Falls back to distance-based feathering where RMBG has no information.
    Band width is automatically reduced for small masks.
    """
    band_px = _adaptive_band(sam_mask, band_px)

    eroded = binary_erosion(sam_mask, iterations=band_px)
    edge_band = sam_mask & ~eroded

    alpha = np.zeros(sam_mask.shape, dtype=np.uint8)
    alpha[eroded] = 255

    rmbg_has_data = rmbg_alpha > 10

    rmbg_zone = edge_band & rmbg_has_data
    alpha[rmbg_zone] = rmbg_alpha[rmbg_zone]

    feather_zone = edge_band & ~rmbg_has_data
    if np.any(feather_zone):
        outside = ~sam_mask
        dist = distance_transform_edt(~outside)
        max_dist = float(band_px) if band_px > 0 else 1.0
        feather = np.clip(dist / max_dist, 0, 1)
        alpha[feather_zone] = (feather[feather_zone] * 255).astype(np.uint8)

    if blur > 0:
        alpha = np.array(
            Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=blur))
        )

    alpha[~sam_mask & ~edge_band] = 0

    return alpha
