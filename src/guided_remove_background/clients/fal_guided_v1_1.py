"""Bria guided-remove-background v1.1 client via Fal.ai -- the real shipped pipeline
(Bria-AI/bria-everywhere's ``GuidedRemoveBackground``: RMBG baseline, Gemini intent
decompose, per-target SAM, mode merge, per-added-object crop+RMBG-vs-SAM refinement),
served by the ``object_extraction`` fal app's ``/guided-remove-background`` endpoint.

Mirrors ``fal_extract_object.py``'s call shape exactly, except the model id is NOT a
fixed constant: during development it's whatever ephemeral app-id a `fal run` dev
session is currently serving at (e.g. "bria/ab2ed5d0-.../guided-remove-background"),
not a stable published path -- set it via the ``GUIDED_V1_1_FAL_MODEL`` env var
rather than hardcoding it here. Once a real `fal deploy` exists, this constant can be
replaced with the permanent model id.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import fal_client
import numpy as np
from PIL import Image

from .http_utils import env_key, http_get

log = logging.getLogger(__name__)


def call_guided_v1_1(
    image_path: Path,
    instruction: str,
    *,
    autocrop: bool = False,
) -> np.ndarray | None:
    """Run one guided-remove-background v1.1 call. Returns an RGBA numpy array, or None on failure."""
    import os
    os.environ["FAL_KEY"] = env_key("FAL_KEY")
    model = env_key("GUIDED_V1_1_FAL_MODEL")

    log.info("[GuidedV1.1] Uploading %s to Fal ...", image_path.name)
    try:
        image_url = fal_client.upload_file(str(image_path))
    except Exception as e:
        log.error("[GuidedV1.1] Upload failed (account issue?): %s", e)
        return None

    log.info("[GuidedV1.1] model=%r instruction=%r", model, instruction)
    try:
        result = fal_client.subscribe(
            model,
            arguments={
                "image_url": image_url,
                "instruction": instruction,
                "autocrop": autocrop,
            },
        )
    except Exception as e:
        log.error("[GuidedV1.1] call failed: %s", e)
        return None

    image = result.get("image")
    if not image:
        log.error("[GuidedV1.1] returned no image")
        return None

    try:
        r = http_get(image["url"], timeout=60)
        return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    except Exception as e:
        log.error("[GuidedV1.1] Download failed: %s", e)
        return None
