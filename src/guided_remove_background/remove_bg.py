"""Guided background removal orchestrator.

RMBG-first: Bria's RMBG provides the pixel-perfect baseline mask and alpha.
SAM is a *selector* — it points at which RMBG regions to keep or drop,
but never contributes pixels to the final output.

  ADD:    final = RMBG ∪ SAM_targets   (add items RMBG missed)
  REMOVE: final = RMBG − SAM_targets   (subtract items user doesn't want)
  NARROW: final = RMBG ∩ SAM_targets   (keep only RMBG pixels SAM selected)

Alpha always comes from RMBG for pixel-perfect edges on every object.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, distance_transform_edt

from . import MODES
from .clients.bria_rmbg import call_rmbg
from .clients.fal_sam import call_sam
from .clients.vlm_decompose import decompose_prompt, DecomposeResult
from .clients.vlm_judge import judge_result, JudgeVerdict
from .processing.debug import StepRecorder, rmbg_overlay
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
    judge_verdicts: list[JudgeVerdict] = field(default_factory=list)


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

        if vlm_mode in ("add", "add_remove") and np.any(sam_only_zone):
            rmbg_dilated = binary_dilation(rmbg_zone, iterations=2)
            sam_dilated = binary_dilation(sam_only_zone, iterations=2)
            seam_gap = rmbg_dilated & sam_dilated & final_mask & (final_alpha < 128)
            if np.any(seam_gap):
                final_alpha[seam_gap] = 255
                log.info("[Alpha] Bridged %d seam-gap pixels between RMBG and SAM",
                         int(seam_gap.sum()))

    final_alpha[~final_mask] = 0
    return final_alpha


def _verify_and_correct(
    *,
    image: Path,
    result: np.ndarray,
    final_mask: np.ndarray,
    rmbg_mask: np.ndarray,
    rmbg_alpha: np.ndarray,
    user_prompt: str,
    vlm_mode: str,
    edge_band_px: int,
    blur: float,
    dilation: int,
    min_score: float,
    max_retries: int,
    recorder: StepRecorder,
    provider: str | None,
    output: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[JudgeVerdict]]:
    """Run judge verification loop with corrections. Returns updated result."""
    verdicts: list[JudgeVerdict] = []
    rmbg_preview_path = None
    if recorder.enabled and "step1_rmbg" in recorder.paths:
        rmbg_preview_path = Path(recorder.paths["step1_rmbg"])

    current_alpha = result[:, :, 3].copy()
    pre_judge_mask_size = int(final_mask.sum())
    mask_floor = max(int(pre_judge_mask_size * 0.20), 1000)

    snapshot_result = result.copy()
    snapshot_mask = final_mask.copy()
    snapshot_alpha = current_alpha.copy()
    corrections_applied = False

    for attempt in range(max_retries + 1):
        preview_tmp = Path(tempfile.mktemp(suffix=".png"))
        preview_path = save_preview(result, preview_tmp)
        try:
            verdict = judge_result(
                image, preview_path, user_prompt,
                rmbg_preview=rmbg_preview_path,
                provider=provider,
            )
        finally:
            preview_tmp.unlink(missing_ok=True)
            preview_path.unlink(missing_ok=True)

        verdicts.append(verdict)
        recorder.save_judge(attempt, verdict)

        if verdict.passed or verdict.error:
            break
        if attempt >= max_retries:
            if corrections_applied:
                log.warning("[Judge] Max retries (%d) reached — reverting to pre-judge result",
                            max_retries)
                result = snapshot_result
                final_mask = snapshot_mask
                current_alpha = snapshot_alpha
                recorder.save_final(result)
            else:
                log.warning("[Judge] Max retries (%d) reached, no corrections were applied",
                            max_retries)
            break

        log.info("[Judge] Attempting corrections (retry %d/%d)...", attempt + 1, max_retries)
        mask_changed = False

        for issue in verdict.issues:
            fix = issue.fix

            if fix.startswith("re-sam:"):
                new_prompt = fix.split(":", 1)[1].strip()
                log.info("[Judge] Re-running SAM: '%s' (issue: %s)", new_prompt, issue.type)
                new_mask, _, new_individual = call_sam(
                    image, [new_prompt], min_score=min_score
                )
                if new_mask is not None:
                    if issue.type == "missing_item":
                        expanded = binary_dilation(new_mask, iterations=dilation)
                        final_mask = final_mask | expanded
                        final_mask = binary_fill_holes(final_mask)
                        mask_changed = True
                        log.info("[Judge] Added %d px for missing item", int(expanded.sum()))
                    elif issue.type == "unwanted_item":
                        expanded = binary_dilation(new_mask, iterations=max(dilation, 3))
                        candidate = final_mask & ~expanded
                        if int(candidate.sum()) < mask_floor:
                            log.warning("[Judge] Skipping removal of '%s': would reduce mask "
                                        "below safety floor (%d px < %d)",
                                        new_prompt, int(candidate.sum()), mask_floor)
                        else:
                            final_mask = candidate
                            mask_changed = True
                            log.info("[Judge] Removed %d px for unwanted item", int(expanded.sum()))
                else:
                    log.warning("[Judge] SAM found nothing for '%s'", new_prompt)

            elif fix == "fill-holes":
                log.info("[Judge] Applying hole-filling")
                filled = binary_fill_holes(final_mask)
                n_filled = int(filled.sum()) - int(final_mask.sum())
                if n_filled > 0:
                    final_mask = filled
                    mask_changed = True
                    log.info("[Judge] Filled %d hole pixels", n_filled)

            elif fix.startswith("expand-mask:") or fix.startswith("shrink-mask:"):
                log.info("[Judge] Skipping %s (disabled for safety)", fix.split(":")[0])

            elif fix == "no-fix":
                log.info("[Judge] Issue acknowledged (no automated fix): %s", issue.description)

        if mask_changed:
            corrections_applied = True
            recorder.save_combined_mask(
                np.array(Image.open(image).convert("RGBA")), final_mask,
            )
            current_alpha = _build_alpha(final_mask, rmbg_alpha, edge_band_px, blur, vlm_mode)
            orig_fresh = np.array(Image.open(image).convert("RGBA"))
            orig_fresh[:, :, 3] = current_alpha
            result = orig_fresh
            recorder.save_final(result)
            log.info("[Judge] Rebuilt result after corrections (mask: %d px, floor: %d px)",
                     int(final_mask.sum()), mask_floor)
        else:
            log.info("[Judge] No actionable fixes applied, stopping loop")
            break

    return result, final_mask, current_alpha, verdicts


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
    verify: bool = True,
    max_retries: int = 2,
) -> RemoveBgResult:
    """Remove background from an image, guided by user prompts."""
    if mode not in MODES:
        sys.exit(f"Unknown mode '{mode}'. Choose from: {MODES}")

    t0 = time.monotonic()
    orig = np.array(Image.open(image).convert("RGBA"))
    sam_scores: dict[str, float] = {}
    vlm_result: DecomposeResult | None = None
    judge_verdicts: list[JudgeVerdict] = []

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
        overlay_img = rmbg_overlay(orig, rmbg_mask)
        overlay_tmp = Path(tempfile.mktemp(suffix=".jpg"))
        overlay_img.convert("RGB").save(overlay_tmp, quality=90)
        try:
            vlm_result = decompose_prompt(
                image, user_prompt, provider=vlm_provider,
                rmbg_overlay_path=overlay_tmp,
            )
        finally:
            overlay_tmp.unlink(missing_ok=True)

        vlm_mode = None
        targets = []
        if vlm_result.error:
            log.warning("VLM decompose failed; returning RMBG baseline")
            result = rmbg_arr
        elif vlm_result.mode == "add_remove":
            all_targets = vlm_result.add_targets + vlm_result.remove_targets
            if not all_targets:
                log.info("VLM returned empty add_remove targets; RMBG baseline is correct")
                result = rmbg_arr
            else:
                vlm_mode = "add_remove"
                targets = all_targets
        elif not vlm_result.targets:
            log.info("VLM returned empty targets; RMBG baseline is already correct")
            result = rmbg_arr
        else:
            vlm_mode = vlm_result.mode
            targets = vlm_result.targets

        if vlm_mode and targets:
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

            # --- Step 3b: Retry SAM with simplified prompts if ADD mode found nothing ---
            if sam_mask is None and vlm_mode == "add" and targets:
                simplified = [t.split()[-2:] for t in targets]
                simplified = [" ".join(words) for words in simplified if words]
                if simplified != targets:
                    log.info("[SAM] Retrying ADD targets with simplified prompts: %s", simplified)
                    sam_mask, retry_scores, retry_individual = call_sam(
                        image, simplified, min_score=min_score
                    )
                    scores.update(retry_scores)
                    sam_scores.update(retry_scores)
                    individual.update(retry_individual)

            # --- Step 4: Apply mode-specific mask logic ---
            if sam_mask is None:
                log.warning("SAM found nothing; returning RMBG baseline")
                final_mask = rmbg_mask.copy()
            elif vlm_mode == "add":
                expanded = binary_dilation(sam_mask, iterations=dilation)
                merged = rmbg_mask | expanded
                final_mask = binary_fill_holes(merged)
                n_filled = int(final_mask.sum()) - int(merged.sum())
                n_added = int(expanded.sum()) - int((expanded & rmbg_mask).sum())
                log.info("[%s] Added %d pixels beyond RMBG, filled %d holes at seams",
                         vlm_mode.upper(), max(0, n_added), max(0, n_filled))

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
                        log.warning("[REMOVE] Skipping '%s': small item (%.1f%% of foreground) "
                                    "deep inside core — likely part of main subject",
                                    prompt, obj_size_ratio * 100)
                        continue
                    remove_mask |= selector
                    log.debug("[REMOVE] '%s': %d px to remove (%.1f%% of foreground)",
                              prompt, int(obj_rmbg.sum()), obj_size_ratio * 100)
                final_mask = rmbg_mask & ~remove_mask
                n_removed = int(rmbg_mask.sum()) - int(final_mask.sum())
                log.info("[%s] Removed %d pixels from RMBG via %d targets",
                         vlm_mode.upper(), max(0, n_removed), len(individual))

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
                        n_filled = int(filled.sum()) - int(obj_rmbg.sum())
                        log.debug("[NARROW] '%s': %d RMBG px (%.0f%% coverage, %d holes filled)",
                                  prompt, int(obj_rmbg.sum()), rmbg_coverage * 100, n_filled)
                    else:
                        final_mask |= selector
                        log.debug("[NARROW] '%s': SAM-direct (%d px, %.0f%% RMBG coverage)",
                                  prompt, int(selector.sum()), rmbg_coverage * 100)
                log.info("[%s] Selected %d pixels via %d SAM selectors",
                         vlm_mode.upper(), int(final_mask.sum()), len(individual))

            elif vlm_mode == "add_remove":
                add_prompts = set(vlm_result.add_targets)
                remove_prompts = set(vlm_result.remove_targets)

                add_mask = np.zeros_like(rmbg_mask)
                for prompt, obj_mask in individual.items():
                    if prompt in add_prompts:
                        expanded = binary_dilation(obj_mask, iterations=dilation)
                        add_mask |= expanded

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
                            log.warning("[ADD_REMOVE] Skipping removal of '%s': small interior item",
                                        prompt)
                            continue
                        remove_mask |= selector

                merged = (rmbg_mask | add_mask) & ~remove_mask
                final_mask = binary_fill_holes(merged)
                n_added = int(add_mask.sum()) - int((add_mask & rmbg_mask).sum())
                n_removed = int(rmbg_mask.sum()) - int((rmbg_mask & ~remove_mask).sum())
                log.info("[ADD_REMOVE] Added %d px, removed %d px", max(0, n_added), max(0, n_removed))

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

            # --- Step 7: Judge verification loop ---
            if verify and vlm_mode:
                result, final_mask, _, judge_verdicts = _verify_and_correct(
                    image=image,
                    result=result,
                    final_mask=final_mask,
                    rmbg_mask=rmbg_mask,
                    rmbg_alpha=rmbg_alpha,
                    user_prompt=user_prompt,
                    vlm_mode=vlm_mode,
                    edge_band_px=edge_band_px,
                    blur=blur,
                    dilation=dilation,
                    min_score=min_score,
                    max_retries=max_retries,
                    recorder=recorder,
                    provider=vlm_provider,
                    output=output,
                )

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
        judge_verdicts=judge_verdicts,
    )
