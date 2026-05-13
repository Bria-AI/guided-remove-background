"""Shared HTTP helpers, retry logic, and environment utilities."""

from __future__ import annotations

import os
import sys
import time

import requests

POLL_INTERVAL = 2.0
POLL_TIMEOUT = 120
HTTP_RETRIES = 2
HTTP_BACKOFF = 2.0

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
SUPPORTED_EXTS = set(CONTENT_TYPES)


def env_key(name: str) -> str:
    """Read a required environment variable or exit."""
    val = os.environ.get(name)
    if not val:
        sys.exit(f"Missing env var {name}. Set it in .env or environment.")
    return val


def http_get(url: str, *, timeout: int = 60, headers: dict | None = None) -> requests.Response:
    """GET with automatic retries."""
    last: Exception | None = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_BACKOFF ** attempt)
    raise last  # type: ignore[misc]
