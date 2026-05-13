"""Bria RMBG-2.0 API client — background removal."""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .http_utils import (
    CONTENT_TYPES,
    HTTP_BACKOFF,
    HTTP_RETRIES,
    POLL_INTERVAL,
    POLL_TIMEOUT,
    env_key,
    http_get,
)

log = logging.getLogger(__name__)

BRIA_RMBG_URL = "https://engine.int.bria-api.com/v2/image/edit/remove_background"


def _bria_headers() -> dict[str, str]:
    return {
        "api_token": env_key("BRIA_API_KEY"),
        "Content-Type": "application/json",
        "User-Agent": "GuidedRemoveBackground/0.1.0",
    }


def _poll_bria(status_url: str, headers: dict) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        r = http_get(status_url, timeout=30, headers=headers)
        data = r.json()
        status = data.get("status", "")
        if status == "COMPLETED":
            return data
        if status == "FAILED":
            raise RuntimeError(f"Bria job failed: {data}")
    raise TimeoutError("Bria polling timed out")


def call_rmbg(image_path: Path) -> np.ndarray:
    """Run Bria RMBG-2.0. Returns RGBA numpy array."""
    log.info("[RMBG] Uploading %s ...", image_path.name)
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    ct = CONTENT_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    headers = _bria_headers()

    for attempt in range(HTTP_RETRIES + 1):
        try:
            resp = requests.post(
                BRIA_RMBG_URL,
                json={"image": f"data:{ct};base64,{b64}"},
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException:
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_BACKOFF ** attempt)
            else:
                raise

    if "status_url" in data:
        data = _poll_bria(data["status_url"], headers)

    result = data.get("result", data)
    img_url = result.get("image_url") or result.get("url")
    if not img_url:
        raise ValueError(f"No image URL in RMBG response: {data}")

    log.info("[RMBG] Downloading result ...")
    r = http_get(img_url, timeout=60)
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
