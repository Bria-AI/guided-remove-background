"""Morphological mask cleanup — dilate and keep relevant components."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, label


def fill_mask(
    mask: np.ndarray,
    dilation: int = 5,
    min_component_ratio: float = 0.01,
    fill_holes: bool = False,
) -> np.ndarray:
    """Dilate and keep all components that overlap with the original mask.

    When fill_holes=False (default for guided mode), only dilates and keeps
    overlapping components — never fills gaps BETWEEN separate SAM blobs.
    When fill_holes=True, fills holes within each individual connected component
    of the original mask before merging, but still won't fill inter-object gaps.
    """
    if dilation > 0:
        closed = binary_dilation(mask, iterations=dilation)
    else:
        closed = mask.copy()

    if fill_holes:
        # Per-component hole filling: fill holes within each connected component
        # of the ORIGINAL mask, never between separate components
        orig_lbl, orig_n = label(mask)
        for i in range(1, orig_n + 1):
            component_region = orig_lbl == i
            filled_component = binary_fill_holes(component_region)
            new_pixels = filled_component & ~component_region
            if np.any(new_pixels):
                closed = closed | new_pixels

    lbl, n = label(closed)
    if n <= 1:
        return closed

    keep = np.zeros_like(closed)
    for i in range(1, n + 1):
        component = lbl == i
        overlap = np.sum(component & mask)
        area = np.sum(component)
        if area > 0 and (overlap / area) >= min_component_ratio:
            keep |= component

    if not np.any(keep):
        return closed

    return keep
