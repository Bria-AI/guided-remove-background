"""Debug visualization — render intermediate pipeline steps as images."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

OVERLAY_COLORS = [
    (34, 197, 94),    # green
    (59, 130, 246),   # blue
    (249, 115, 22),   # orange
    (168, 85, 247),   # purple
    (236, 72, 153),   # pink
    (20, 184, 166),   # teal
    (245, 158, 11),   # amber
    (239, 68, 68),    # red
]


class StepRecorder:
    """Saves intermediate pipeline step images to a directory."""

    def __init__(self, output_dir: Path | None):
        self.output_dir = output_dir
        self.paths: dict[str, str] = {}
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def _save(self, name: str, img: Image.Image) -> Path | None:
        if not self.enabled:
            return None
        path = self.output_dir / f"{name}.jpg"
        img.convert("RGB").save(path, quality=90)
        self.paths[name] = str(path)
        log.debug("[Step] Saved %s -> %s", name, path)
        return path

    def save_rmbg(self, rmbg_arr: np.ndarray, original: np.ndarray) -> None:
        """Step 1: RMBG baseline (the foundation) on checkerboard."""
        img = _checker_composite(rmbg_arr)
        self._save("step1_rmbg", img)

    def save_vlm(self, add_prompts: list[str], remove_prompts: list[str]) -> None:
        """Step 2: VLM decomposition (legacy two-list format)."""
        if not self.enabled:
            return
        path = self.output_dir / "step2_vlm.json"
        path.write_text(json.dumps({"add_to_rmbg": add_prompts, "remove_from_rmbg": remove_prompts}, indent=2))
        self.paths["step2_vlm"] = str(path)

    def save_vlm_three_mode(self, mode: str, targets: list[str]) -> None:
        """Step 2: VLM three-mode decomposition."""
        if not self.enabled:
            return
        path = self.output_dir / "step2_vlm.json"
        path.write_text(json.dumps({"mode": mode, "targets": targets}, indent=2))
        self.paths["step2_vlm"] = str(path)

    def save_sam_masks(
        self, original: np.ndarray, masks: dict[str, np.ndarray], prefix: str = "step3_sam_include"
    ) -> None:
        """Step 3: SAM masks overlaid on original, each prompt in a different color."""
        if not masks:
            return
        base = Image.fromarray(original[:, :, :3])
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

        for i, (prompt, mask) in enumerate(masks.items()):
            color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
            mask_u8 = (mask.astype(np.uint8) * 255)
            mask_pil = Image.fromarray(mask_u8, "L")
            if mask_pil.size != base.size:
                mask_pil = mask_pil.resize(base.size, Image.NEAREST)
            fill = Image.new("RGBA", base.size, color + (0,))
            fill.putalpha(mask_pil.point(lambda v: 130 if v > 0 else 0))
            overlay = Image.alpha_composite(overlay, fill)

        result = Image.alpha_composite(base.convert("RGBA"), overlay)
        self._save(prefix, result)

    def save_combined_mask(self, original: np.ndarray, mask: np.ndarray) -> None:
        """Step 4: Final adjusted mask as green overlay."""
        result = rmbg_overlay(original, mask)
        self._save("step4_combined_mask", result)

    def save_edge_refined(self, original: np.ndarray, alpha: np.ndarray) -> None:
        """Step 5: Final alpha (RMBG edges preserved, SAM adds feathered)."""
        rgba = original.copy()
        rgba[:, :, 3] = alpha
        img = _checker_composite(rgba)
        self._save("step5_edge_refined", img)

    def save_final(self, rgba: np.ndarray) -> None:
        """Step 6: Final result on checkerboard."""
        img = _checker_composite(rgba)
        self._save("step6_final", img)

    def save_judge(self, attempt: int, verdict) -> None:
        """Step 7: Judge verification verdict."""
        if not self.enabled:
            return
        key = f"step7_judge_{attempt}"
        data = {
            "attempt": attempt,
            "passed": verdict.passed,
            "error": verdict.error,
            "issues": [
                {"type": i.type, "description": i.description, "fix": i.fix}
                for i in verdict.issues
            ],
        }
        path = self.output_dir / f"{key}.json"
        path.write_text(json.dumps(data, indent=2))
        self.paths[key] = str(path)
        log.debug("[Step] Saved judge verdict %d -> %s", attempt, path)


def rmbg_overlay(original: np.ndarray, rmbg_mask: np.ndarray) -> Image.Image:
    """Generate a green-highlighted overlay showing RMBG's foreground regions.

    Used to give the VLM visual context about what RMBG considers foreground.
    """
    base = Image.fromarray(original[:, :, :3]).convert("RGBA")
    mask_u8 = (rmbg_mask.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_u8, "L")
    if mask_pil.size != base.size:
        mask_pil = mask_pil.resize(base.size, Image.NEAREST)
    fill = Image.new("RGBA", base.size, (34, 197, 94, 0))
    fill.putalpha(mask_pil.point(lambda v: 140 if v > 0 else 0))
    return Image.alpha_composite(base, fill)


def _checker_composite(rgba: np.ndarray) -> Image.Image:
    """Composite RGBA onto a checkerboard background."""
    h, w = rgba.shape[:2]
    cell = 30
    yy, xx = np.indices((h, w))
    pattern = ((xx // cell + yy // cell) % 2 == 0)
    checker = np.where(pattern[..., None], 240, 200).astype(np.uint8)
    checker = np.broadcast_to(checker, (h, w, 3)).copy()
    bg = Image.fromarray(checker)
    fg = Image.fromarray(rgba, "RGBA")
    bg.paste(fg, mask=fg.split()[3])
    return bg
