"""Download curated multi-object photos from Pexels (direct links, no API key).

Run once to populate benchmark/images/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

from data.catalog import CATALOG

IMAGES_DIR = Path(__file__).parent / "images"

MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0  # doubles each attempt


def _download_with_retry(url: str) -> bytes:
    """Fetch url, retrying on transient errors (Pexels' resize CDN 503s intermittently)."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def fetch_all(*, force: bool = False) -> list[Path]:
    """Download all catalog images. Returns list of saved paths."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for filename, url, category, desc in CATALOG:
        dest = IMAGES_DIR / filename
        if dest.exists() and not force:
            print(f"  [skip] {filename} (exists)")
            downloaded.append(dest)
            continue

        print(f"  [download] {filename} ({category}: {desc})")
        try:
            dest.write_bytes(_download_with_retry(url))
            downloaded.append(dest)
        except Exception as e:
            print(f"  [FAILED] {filename}: {e}", file=sys.stderr)

    print(f"\n{len(downloaded)}/{len(CATALOG)} images ready in {IMAGES_DIR}")
    return downloaded


if __name__ == "__main__":
    force = "--force" in sys.argv
    fetch_all(force=force)
