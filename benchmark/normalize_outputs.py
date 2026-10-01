"""Downscale every system's benchmark outputs to one size, for side-by-side review and judging.

Each system writes at its own native size (fibo ~832px, nano-banana ~843px, gpt-image-2 and
the Bria chain at the input's 1280px). This writes one copy per output at a fixed long edge,
keeping the alpha channel when the system produced one and leaving RGB outputs as RGB, so a
flat painted background stays visible as exactly that. Cases a system failed get no file; the
failure stays recorded in that system's run_meta file.

Usage:
  uv run python benchmark/normalize_outputs.py --results-dir benchmark/results/oct-2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

BENCHMARK_DIR = Path(__file__).parent
LONG_EDGE = 1024

# system id -> run_meta file written by runner.py / candidate_runner.py
SYSTEMS = {
    "bria_a2": "run_meta_noverify.json",
    "rmbg_only": "run_meta.json",
    "c1_fibo": "run_meta_edit_fibo.json",
    "c2_nanobanana2": "run_meta_edit_nanobanana2.json",
    "c3_gptimage2": "run_meta_edit_gptimage2.json",
    "g2_extract_object_rmbg_on": "run_meta_extract_object_rmbg_on.json",
}


def normalize(src: Path, dst: Path) -> tuple[int, int]:
    im = Image.open(src)
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info else im.convert("RGB")
    scale = LONG_EDGE / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    return im.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    out_root = results_dir / "normalized"

    for system, meta_name in SYSTEMS.items():
        meta_path = results_dir / meta_name
        if not meta_path.exists():
            print(f"[{system}] no {meta_name}, skipped")
            continue
        rows = json.loads(meta_path.read_text(encoding="utf-8"))["results"]
        written = failed = 0
        for r in rows:
            if r.get("error"):
                failed += 1
                continue
            dst = out_root / system / f"{Path(r['image']).stem}__{r['foreground']}.png"
            normalize(BENCHMARK_DIR / r["output_png"], dst)
            written += 1
        print(f"[{system}] {written} normalized, {failed} hard fails kept as missing -> {out_root / system}")


if __name__ == "__main__":
    main()
