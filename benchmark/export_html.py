"""Export benchmark results as a single self-contained HTML file.

All images are embedded as base64 data URIs so the file can be opened
in any browser without a server.

Usage:
  uv run python benchmark/export_html.py
  uv run python benchmark/export_html.py -o benchmark_report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
from datetime import datetime
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
IMAGES_DIR = BENCHMARK_DIR / "images"


def img_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def collect_image_paths(meta: dict, *, include_steps: bool = False) -> set[str]:
    """Walk run_meta.json and collect image paths — prefer JPG previews over PNGs."""
    paths: set[str] = set()
    for result in meta.get("results", []):
        if result.get("image"):
            paths.add("images/" + result["image"])
        if result.get("preview_jpg"):
            paths.add(result["preview_jpg"])
        elif result.get("output_png"):
            paths.add(result["output_png"])
        if include_steps:
            for step_path in result.get("step_images", {}).values():
                if step_path:
                    paths.add(step_path)
        for run in result.get("runs", []):
            if run.get("preview_jpg"):
                paths.add(run["preview_jpg"])
            elif run.get("output_png"):
                paths.add(run["output_png"])
            if include_steps:
                for step_path in run.get("step_images", {}).values():
                    if step_path:
                        paths.add(step_path)
    return paths


def build_image_map(paths: set[str]) -> dict[str, str]:
    """Convert file paths to base64 data URIs."""
    img_map: dict[str, str] = {}
    for rel in sorted(paths):
        full = BENCHMARK_DIR / rel
        if not full.exists():
            for prefix in ("results/", ""):
                alt = BENCHMARK_DIR / prefix / rel
                if alt.exists():
                    full = alt
                    break
        data_uri = img_to_data_uri(full)
        if data_uri:
            img_map[rel] = data_uri
            print(f"  embedded: {rel} ({full.stat().st_size // 1024}KB)")
        else:
            print(f"  MISSING:  {rel}")
    return img_map


def read_live_html() -> str:
    return (BENCHMARK_DIR / "live.html").read_text()


def generate(meta: dict, img_map: dict[str, str]) -> str:
    live_html = read_live_html()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    style_start = live_html.index("<style>")
    style_end = live_html.index("</style>") + len("</style>")
    css_block = live_html[style_start:style_end]

    js_from_live = live_html[
        live_html.index("<script>") + len("<script>") : live_html.index("</script>")
    ]

    # Extract all HTML between </style>\n</head>\n<body> and <script> —
    # this captures the full body including modal, step overlay, etc.
    body_match = re.search(
        r"<body>\s*(.+?)\s*<script>", live_html, re.DOTALL
    )
    if not body_match:
        sys.exit("Could not extract body from live.html")
    body_html = body_match.group(1)

    # Replace the live header with a static report header
    body_html = re.sub(
        r'<div style="display:flex.*?</div>\s*'
        r'<p class="subtitle">.*?</p>',
        f'<h1>Guided Remove Background — Benchmark Report</h1>\n'
        f'  <p class="subtitle">Generated {timestamp} · '
        f'{meta.get("completed", 0)} jobs · '
        f'{meta.get("succeeded", 0)} succeeded · '
        f'{meta.get("failed", 0)} failed</p>',
        body_html,
        flags=re.DOTALL,
    )

    meta_json = json.dumps(meta, default=str)
    img_map_json = json.dumps(img_map)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Guided Remove Background — Benchmark Report ({timestamp})</title>
{css_block}
</head>
<body>
{body_html}

<script>
const EMBEDDED_META = {meta_json};
const IMG_MAP = {img_map_json};

function resolveImg(src) {{
  if (!src) return '';
  if (IMG_MAP[src]) return IMG_MAP[src];
  for (const [key, val] of Object.entries(IMG_MAP)) {{
    if (key.endsWith(src) || src.endsWith(key)) return val;
  }}
  return src;
}}

{js_from_live}

// Override fixPaths to resolve embedded base64 images
const _origFixPaths = fixPaths;
fixPaths = function(r) {{
  r = _origFixPaths(r);
  for (const key of ['output_png', 'preview_jpg']) {{
    if (r[key]) r[key] = resolveImg(r[key]);
  }}
  if (r.step_images) {{
    for (const [k, v] of Object.entries(r.step_images)) {{
      r.step_images[k] = resolveImg(v);
    }}
  }}
  if (r.runs) {{
    for (const run of r.runs) {{
      for (const key of ['output_png', 'preview_jpg']) {{
        if (run[key]) run[key] = resolveImg(run[key]);
      }}
      if (run.step_images) {{
        for (const [k, v] of Object.entries(run.step_images)) {{
          run.step_images[k] = resolveImg(v);
        }}
      }}
    }}
  }}
  return r;
}};

// Resolve images inside modal after it renders
const _origOpenModal = openModal;
openModal = function(caseId) {{
  _origOpenModal(caseId);
  document.querySelectorAll('#modalContent img, #stepOverlayContent img').forEach(img => {{
    const src = img.getAttribute('src');
    if (src && !src.startsWith('data:')) {{
      const resolved = resolveImg(src);
      if (resolved) img.src = resolved;
    }}
  }});
}};

// Resolve images inside step overlay after it renders
const _origOpenStep = openStep;
openStep = function(idx) {{
  _origOpenStep(idx);
  document.querySelectorAll('#stepOverlayContent img').forEach(img => {{
    const src = img.getAttribute('src');
    if (src && !src.startsWith('data:')) {{
      const resolved = resolveImg(src);
      if (resolved) img.src = resolved;
    }}
  }});
}};

// Override poll to use embedded data instead of fetching
poll = async function() {{
  const data = EMBEDDED_META;
  document.getElementById('completed').textContent = data.completed || 0;
  document.getElementById('total').textContent = data.total_cases || '?';
  document.getElementById('succeeded').textContent = data.succeeded || 0;
  document.getElementById('failed').textContent = data.failed || 0;
  document.getElementById('elapsed').textContent = (data.total_elapsed_s || 0).toFixed(1) + 's';
  document.getElementById('progressFill').style.width = '100%';
  isDone = true;
  document.getElementById('pulseIndicator').classList.add('done');
  allResults = (data.results || []).map(fixPaths);
  buildFilterChips();
  renderCards();
}};

// Disable feedback in exported report
loadFeedback = async function() {{}};
sendVote = async function() {{}};

init();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark as self-contained HTML")
    parser.add_argument("-o", "--output", default=str(BENCHMARK_DIR / "benchmark_report.html"),
                        help="Output path (default: benchmark/benchmark_report.html)")
    parser.add_argument("--with-steps", action="store_true",
                        help="Include pipeline step images (much larger file)")
    args = parser.parse_args()

    meta_path = RESULTS_DIR / "run_meta.json"
    if not meta_path.exists():
        sys.exit(f"No results found at {meta_path}. Run the benchmark first.")

    print("Loading results...")
    meta = json.loads(meta_path.read_text())

    for r in meta.get("results", []):
        if r.get("preview_jpg"):
            r["output_png"] = r["preview_jpg"]
        for run in r.get("runs", []):
            if run.get("preview_jpg"):
                run["output_png"] = run["preview_jpg"]

    print(f"  {meta.get('completed', 0)} jobs, {len(meta.get('results', []))} cases")

    print("Collecting images...")
    paths = collect_image_paths(meta, include_steps=args.with_steps)
    print(f"  {len(paths)} unique image paths")

    print("Embedding images as base64...")
    img_map = build_image_map(paths)
    print(f"  {len(img_map)} images embedded")

    print("Generating HTML...")
    html = generate(meta, img_map)

    out = Path(args.output)
    out.write_text(html)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nDone! {out} ({size_mb:.1f} MB)")
    print(f"Open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
