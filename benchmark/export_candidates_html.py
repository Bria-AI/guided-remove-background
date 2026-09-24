"""Export candidates.html (the AG-184 multi-candidate comparison dashboard) as one
self-contained HTML file — all run_meta/judge_uniform JSON and images embedded, so it
can be opened in any browser or emailed/shared without needing the feedback_server.

Usage:
  uv run python benchmark/export_candidates_html.py
  uv run python benchmark/export_candidates_html.py -o candidates_report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
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


def find_candidate_files_and_ids(candidates_html: str) -> tuple[list[str], list[str]]:
    """Pull the { id: '...', file: 'results/....json' } pairs straight out of the
    CANDIDATES array in candidates.html, so this stays in sync automatically."""
    ids = re.findall(r"id:\s*'([^']+)'", candidates_html)
    files = re.findall(r"file:\s*'results/([^']+)'", candidates_html)
    return ids, files


def main() -> None:
    parser = argparse.ArgumentParser(description="Export candidates.html as a self-contained HTML file")
    parser.add_argument("-o", "--output", default=str(BENCHMARK_DIR / "candidates_report.html"))
    args = parser.parse_args()

    src_path = BENCHMARK_DIR / "candidates.html"
    html = src_path.read_text(encoding="utf-8")

    ids, run_meta_files = find_candidate_files_and_ids(html)
    print(f"Found {len(ids)} candidates in candidates.html: {ids}")

    # 1. Embed every candidate's run_meta_*.json + judge_uniform_<id>.json (if present).
    embedded_files: dict[str, dict] = {}
    all_results: list[dict] = []
    for fname in set(run_meta_files):
        path = RESULTS_DIR / fname
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            embedded_files[f"results/{fname}"] = data
            all_results.extend(data.get("results", []))
            print(f"  embedded run_meta: results/{fname} ({len(data.get('results', []))} records)")
        else:
            print(f"  MISSING run_meta: results/{fname}")

    for cid in ids:
        judge_path = RESULTS_DIR / f"judge_uniform_{cid}.json"
        if judge_path.exists():
            data = json.loads(judge_path.read_text(encoding="utf-8"))
            embedded_files[f"results/judge_uniform_{cid}.json"] = data
            print(f"  embedded judge:    results/judge_uniform_{cid}.json ({len(data.get('results', []))} verdicts)")

    # 2. Collect every image path referenced by any record (original + result) and embed as base64.
    image_paths: set[str] = set()
    for r in all_results:
        if r.get("image"):
            image_paths.add("images/" + r["image"])
        if r.get("preview_jpg"):
            image_paths.add(r["preview_jpg"])
        elif r.get("output_png"):
            image_paths.add(r["output_png"])

    print(f"\nEmbedding {len(image_paths)} images as base64 ...")
    img_map: dict[str, str] = {}
    for rel in sorted(image_paths):
        full = BENCHMARK_DIR / rel
        data_uri = img_to_data_uri(full)
        if data_uri:
            img_map[rel] = data_uri
        else:
            print(f"  MISSING image: {rel}")
    print(f"  {len(img_map)}/{len(image_paths)} images embedded")

    # 3. Inject a fetch() shim (serves embedded_files instead of hitting the network)
    #    and an image-resolving pass after each render, without touching candidates.html's
    #    own rendering logic at all.
    injected = f"""
<script>
const EMBEDDED_FILES = {json.dumps(embedded_files, default=str)};
const IMG_MAP = {json.dumps(img_map)};

const _origFetch = window.fetch.bind(window);
window.fetch = async function(url, opts) {{
  const clean = String(url).split('?')[0];
  if (Object.prototype.hasOwnProperty.call(EMBEDDED_FILES, clean)) {{
    return new Response(JSON.stringify(EMBEDDED_FILES[clean]), {{status: 200, headers: {{'Content-Type': 'application/json'}}}});
  }}
  return new Response('Not found (offline export)', {{status: 404}});
}};

function resolveImg(src) {{
  if (!src || src.startsWith('data:')) return src;
  if (IMG_MAP[src]) return IMG_MAP[src];
  for (const [key, val] of Object.entries(IMG_MAP)) {{
    if (key.endsWith(src) || src.endsWith(key)) return val;
  }}
  return src;
}}

function resolveAllImages() {{
  document.querySelectorAll('img').forEach(img => {{
    const src = img.getAttribute('src');
    if (src && !src.startsWith('data:')) {{
      const resolved = resolveImg(src);
      if (resolved && resolved !== src) img.src = resolved;
    }}
  }});
}}
</script>
"""

    # Wrap renderCards/openModal so every image swap happens right after they run,
    # without editing candidates.html's own JS at all.
    hook = """
<script>
const _origRenderCards = renderCards;
renderCards = function(...args) {
  const r = _origRenderCards.apply(this, args);
  resolveAllImages();
  return r;
};
const _origOpenModal = openModal;
openModal = function(...args) {
  const r = _origOpenModal.apply(this, args);
  resolveAllImages();
  return r;
};
</script>
"""

    # Insert the fetch shim BEFORE candidates.html's own <script> (so window.fetch is
    # already patched when its init() runs), and the render hook AFTER it (so
    # renderCards/openModal exist to be wrapped).
    first_script = html.index("<script>")
    last_script_close = html.rindex("</script>")

    out_html = (
        html[:first_script]
        + injected
        + html[first_script:last_script_close + len("</script>")]
        + hook
        + html[last_script_close + len("</script>"):]
    )

    timestamp_title = re.sub(
        r"<title>.*?</title>",
        "<title>Guided Remove Background — Candidate Comparison (exported)</title>",
        out_html,
        count=1,
    )

    out = Path(args.output)
    out.write_text(timestamp_title, encoding="utf-8")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nDone! {out} ({size_mb:.1f} MB)")
    print(f"Open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
