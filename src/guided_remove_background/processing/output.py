"""Output helpers — save background-removed PNGs and checkerboard previews."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def save_result(rgba: np.ndarray, output_path: Path) -> Path:
    """Save RGBA array as PNG and return the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output_path)
    log.info("[Save] Result -> %s", output_path)
    return output_path


def save_preview(rgba: np.ndarray, output_path: Path) -> Path:
    """Save a checkerboard-background preview as JPG."""
    h, w = rgba.shape[:2]
    cell = 40
    yy, xx = np.indices((h, w))
    pattern = ((xx // cell + yy // cell) % 2 == 0)
    checker = np.where(pattern[..., None], 255, 180).astype(np.uint8)
    checker = np.broadcast_to(checker, (h, w, 3)).copy()
    bg = Image.fromarray(checker)
    fg = Image.fromarray(rgba, "RGBA")
    bg.paste(fg, mask=fg.split()[3])

    preview_path = output_path.with_name(output_path.stem + "_preview.jpg")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(preview_path, quality=92)
    log.info("[Save] Preview -> %s", preview_path)
    return preview_path
