"""Bria object-extraction client via Fal.ai — single-object cutout from a
natural-language prompt, with a `remove_background` toggle:

  remove_background=False (default): alpha comes straight from the SAM
    segmentation mask used to find the object.
  remove_background=True: alpha is refined by an RMBG pass over the cutout.

https://fal.ai/models/bria/object-extraction/extract-object
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

EXTRACT_OBJECT_MODEL = "bria/object-extraction/extract-object"


def call_extract_object(
    image_path: Path,
    prompt: str,
    *,
    remove_background: bool = False,
    autocrop: bool = False,
) -> np.ndarray | None:
    """Run one extract-object call. Returns an RGBA numpy array, or None on failure."""
    import os
    os.environ["FAL_KEY"] = env_key("FAL_KEY")

    log.info("[ExtractObject] Uploading %s to Fal ...", image_path.name)
    try:
        image_url = fal_client.upload_file(str(image_path))
    except Exception as e:
        log.error("[ExtractObject] Upload failed (account issue?): %s", e)
        return None

    log.info("[ExtractObject] prompt=%r remove_background=%s", prompt, remove_background)
    try:
        result = fal_client.subscribe(
            EXTRACT_OBJECT_MODEL,
            arguments={
                "image_url": image_url,
                "prompt": prompt,
                "remove_background": remove_background,
                "autocrop": autocrop,
            },
        )
    except Exception as e:
        log.error("[ExtractObject] call failed: %s", e)
        return None

    image = result.get("image")
    if not image:
        log.error("[ExtractObject] returned no image")
        return None

    try:
        r = http_get(image["url"], timeout=60)
        return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    except Exception as e:
        log.error("[ExtractObject] Download failed: %s", e)
        return None
