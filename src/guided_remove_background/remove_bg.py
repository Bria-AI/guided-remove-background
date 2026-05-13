"""Guided background removal orchestrator — specificity-based architecture.

RMBG-first: Bria's RMBG provides the pixel-perfect baseline.
SAM adjusts it in one of three modes based on VLM classification:

  ADD:    final = RMBG ∪ SAM_targets   (add items RMBG missed)
  REMOVE: final = RMBG − SAM_targets   (remove items user doesn't want)
  NARROW: final = SAM_targets          (keep exactly what user listed)

Alpha: RMBG's precise alpha wherever it has data; feathered for SAM-only zones.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from . import MODES
from .clients.bria_rmbg import call_rmbg
from .clients.fal_sam import call_sam
from .clients.vlm_decompose import decompose_prompt, DecomposeResult
from .processing.debug import StepRecorder
from .processing.output import save_result, save_preview

log = logging.getLogger(__name__)


@dataclass
class RemoveBgResult:
    output_path: Path
    preview_path: Path
    mode: str
    elapsed_s: float
    sam_scores: dict[str, float] = field(default_factory=dict)
    vlm_decompose: DecomposeResult | None = None
    step_images: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _feather_mask(mask: np.ndarray, edge_px: int = 6, blur: float = 1.0) -> np.ndarray:
    """Convert a binary mask to a soft alpha channel with feathered edges."""
    eroded = binary_erosion(mask, iterations=max(2, edge_px // 2))
    alpha = np.zeros(mask.shape, dtype=np.uint8)
    alpha[eroded] = 255

    edge_band = mask & ~eroded
    if np.any(edge_band):
        dist = distance_transform_edt(mask)
        max_dist = float(edge_px) if edge_px > 0 else 1.0
        feather = np.clip(dist / max_dist, 0, 1)
        alpha[edge_band] = (feather[edge_band] * 255).astype(np.uint8)

    if blur > 0:
        alpha = np.array(
            Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=blur))
        )
    alpha[~mask] = 0
    return alpha


def _build_alpha(final_mask: np.ndarray, rmbg_alpha: np.ndarray,
                 edge_band_px: int, blur: float, vlm_mode: str) -> np.ndarray:
    """Build final alpha channel using RMBG edges where available."""
    h, w = final_mask.shape
    final_alpha = np.zeros((h, w), dtype=np.uint8)

    if vlm_mode == "remove":
        final_alpha[final_mask] = rmbg_alpha[final_mask]
    else:
        rmbg_zone = final_mask & (rmbg_alpha > 0)
        final_alpha[rmbg_zone] = rmbg_alpha[rmbg_zone]

        sam_only_zone = final_mask & (rmbg_alpha == 0)
        if np.any(sam_only_zone):
            sam_feathered = _feather_mask(final_mask, edge_px=edge_band_px, blur=blur)
            final_alpha[sam_only_zone] = sam_feathered[sam_only_zone]
            log.info("[Alpha] Feathered %d SAM-only pixels", int(sam_only_zone.sum()))

    final_alpha[~final_mask] = 0
    return final_alpha


def remove_bg(
    image: Path,
    prompts: list[str],
    output: Path,
    *,
    mode: str = "guided",
    edge_band_px: int = 0,
    dilation: int = 3,
    blur: float = 1.0,
    min_score: float = 0.0,
    vlm_provider: str | None = None,
    save_steps: bool = False,
) -> RemoveBgResult:
    """Remove background from an image, guided by user prompts."""
    if mode not in MODES:
        sys.exit(f"Unknown mode '{mode}'. Choose from: {MODES}")

    t0 = time.monotonic()
    orig = np.array(Image.open(image).convert("RGBA"))
    sam_scores: dict[str, float] = {}
    vlm_result: DecomposeResult | None = None

    steps_dir = output.parent / "steps" / output.stem if save_steps else None
    recorder = StepRecorder(steps_dir)

    if mode == "rmbg-only":
        rmbg_arr = call_rmbg(image)
        result = rmbg_arr

    else:  # guided
        h, w = orig.shape[:2]
        if edge_band_px <= 0:
            edge_band_px = max(4, min(12, int(max(h, w) * 0.005)))

        # --- Step 1: RMBG baseline (the foundation) ---
        rmbg_arr = call_rmbg(image)
        rmbg_alpha = rmbg_arr[:, :, 3]
        rmbg_mask = rmbg_alpha > 128
        recorder.save_rmbg(rmbg_arr, orig)

        # --- Step 2: VLM decomposition → mode + targets ---
        user_prompt = " ".join(prompts)
        vlm_result = decompose_prompt(image, user_prompt, provider=vlm_provider)

        if vlm_result.error:
            log.warning("VLM decompose failed; returning RMBG baseline")
            result = rmbg_arr
        elif not vlm_result.targets:
            log.info("VLM returned empty targets; RMBG baseline is already correct")
            result = rmbg_arr
        else:
            vlm_mode = vlm_result.mode
            targets = vlm_result.targets
            recorder.save_vlm_three_mode(vlm_mode, targets)

            # --- Step 3: SAM segmentation of targets ---
            sam_mask, scores, individual = call_sam(
                image, targets, min_score=min_score
            )
            if vlm_mode == "remove":
                sam_scores.update({f"excl:{k}": v for k, v in scores.items()})
            else:
                sam_scores.update(scores)

            sam_prefix = {
                "add": "step3_sam_add",
                "remove": "step3_sam_remove",
                "narrow": "step3_sam_narrow",
            }.get(vlm_mode, "step3_sam")
            recorder.save_sam_masks(orig, individual, prefix=sam_prefix)

            # --- Step 4: Apply mode-specific mask logic ---
            if sam_mask is None:
                log.warning("SAM found nothing; returning RMBG baseline")
                final_mask = rmbg_mask.copy()
            elif vlm_mode == "add":
                expanded = binary_dilation(sam_mask, iterations=dilation)
                final_mask = rmbg_mask | expanded
                n_added = int(expanded.sum()) - int((expanded & rmbg_mask).sum())
                log.info("[%s] Added %d pixels beyond RMBG", vlm_mode.upper(), max(0, n_added))

            elif vlm_mode == "remove":
                final_mask = rmbg_mask & ~sam_mask
                n_removed = int(rmbg_mask.sum()) - int(final_mask.sum())
                log.info("[%s] Removed %d pixels from RMBG", vlm_mode.upper(), max(0, n_removed))

            elif vlm_mode == "narrow":
                expanded = binary_dilation(sam_mask, iterations=max(dilation, 5))
                final_mask = expanded
                n_from_rmbg = int((expanded & rmbg_mask).sum())
                n_sam_only = int((expanded & ~rmbg_mask).sum())
                log.info("[%s] Scope: %d RMBG pixels kept, %d SAM-only pixels added",
                         vlm_mode.upper(), n_from_rmbg, n_sam_only)
            else:
                final_mask = rmbg_mask.copy()

            recorder.save_combined_mask(orig, final_mask)

            # --- Step 5: Build final alpha ---
            final_alpha = _build_alpha(final_mask, rmbg_alpha, edge_band_px, blur, vlm_mode)
            recorder.save_edge_refined(orig, final_alpha)

            # --- Step 6: Final composite ---
            orig[:, :, 3] = final_alpha
            result = orig
            recorder.save_final(result)

    elapsed = time.monotonic() - t0
    out_path = save_result(result, output)
    prev_path = save_preview(result, output)

    log.info("Done in %.1fs [%s]", elapsed, mode)
    return RemoveBgResult(
        output_path=out_path,
        preview_path=prev_path,
        mode=mode,
        elapsed_s=elapsed,
        sam_scores=sam_scores,
        vlm_decompose=vlm_result,
        step_images=recorder.paths,
        metadata={
            "image": str(image),
            "prompts": prompts,
            "dilation": dilation,
            "blur": blur,
            "edge_band_px": edge_band_px,
            "vlm_mode": vlm_result.mode if vlm_result else None,
        },
    )
