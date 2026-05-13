"""Download curated multi-object photos from Pexels (direct links, no API key).

Run once to populate benchmark/images/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

from data.catalog import CATALOG

IMAGES_DIR = Path(__file__).parent / "images"


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
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            dest.write_bytes(r.content)
            downloaded.append(dest)
        except Exception as e:
            print(f"  [FAILED] {filename}: {e}", file=sys.stderr)

    print(f"\n{len(downloaded)}/{len(CATALOG)} images ready in {IMAGES_DIR}")
    return downloaded


if __name__ == "__main__":
    force = "--force" in sys.argv
    fetch_all(force=force)
