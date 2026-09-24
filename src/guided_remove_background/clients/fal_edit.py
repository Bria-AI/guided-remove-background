"""Single-call instruction-based edit models via Fal.ai — candidates C1-C3 in AG-184.

Each of these models takes one image + one plain-language instruction and returns one
edited image. Confirmed via a spike (see AG-184 notes): all three currently return an
opaque RGB PNG with a flat/repainted background, not a transparent cutout, and each
model expects a different request shape.
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

FIBO_EDIT_1_5 = "bria/fibo-edit-1.5/edit"
NANO_BANANA_2_EDIT = "fal-ai/nano-banana-2/edit"
GPT_IMAGE_2_EDIT = "openai/gpt-image-2/edit"

EDIT_MODELS = (FIBO_EDIT_1_5, NANO_BANANA_2_EDIT, GPT_IMAGE_2_EDIT)


def _arguments_for(model: str, image_url: str, instruction: str) -> dict:
    if model == FIBO_EDIT_1_5:
        return {"image_url": image_url, "instruction": instruction}
    # nano-banana-2 and gpt-image-2 both take a list of image URLs + "prompt"
    return {"image_urls": [image_url], "prompt": instruction}


def call_edit_model(
    image_path: Path,
    instruction: str,
    *,
    model: str,
) -> np.ndarray | None:
    """Run a single edit-model call. Returns an RGB (no alpha) numpy array, or None on failure."""
    import os
    os.environ["FAL_KEY"] = env_key("FAL_KEY")

    log.info("[Edit] Uploading %s to Fal ...", image_path.name)
    try:
        image_url = fal_client.upload_file(str(image_path))
    except Exception as e:
        log.error("[Edit] Upload failed (account issue?): %s", e)
        return None

    log.info("[Edit] Calling %s with instruction: %r", model, instruction)
    try:
        result = fal_client.subscribe(model, arguments=_arguments_for(model, image_url, instruction))
    except Exception as e:
        log.error("[Edit] %s call failed: %s", model, e)
        return None

    images = result.get("images") or []
    if not images:
        log.error("[Edit] %s returned no images", model)
        return None

    try:
        r = http_get(images[0]["url"], timeout=60)
        return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
    except Exception as e:
        log.error("[Edit] Download failed: %s", e)
        return None
