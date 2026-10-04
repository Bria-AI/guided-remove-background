"""Build a browser page comparing all 6 AG-194 systems side by side, one row per case.

Two modes:
  - Default: references images by relative path (normalized/<system>/..., ../images/...).
    Small and fast, but only opens correctly from inside the results folder, and isn't
    shareable on its own (needs the (gitignored) images alongside it).
  - --embed: base64-embeds every image into one self-contained file — same pattern as
    export_candidates_html.py's candidates_report.html. Large (~hundreds of MB at 354+
    images) but opens anywhere and is what you'd actually hand someone to look at the
    results, matching benchmark_report.html / gallery.html, both gitignored exports
    that were always shared directly rather than committed.

Usage:
  uv run python benchmark/build_review_html.py --results-dir benchmark/results/oct-2026
  uv run python benchmark/build_review_html.py --results-dir benchmark/results/oct-2026 --embed
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
CASES_CSV = BENCHMARK_DIR / "data" / "cases.csv"

SYSTEMS = [
    ("bria_a2", "Bria (judge off)"),
    ("rmbg_only", "RMBG baseline"),
    ("c1_fibo", "FIBO-Edit-1.5"),
    ("c2_nanobanana2", "Nano Banana 2"),
    ("c3_gptimage2", "GPT Image 2"),
    ("g2_extract_object_rmbg_on", "extract-object"),
]

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AG-194 benchmark review — oct-2026</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; background: #0f1115; color: #e6e6e6; margin: 0; }
  header { position: sticky; top: 0; background: #16181f; padding: 12px 20px; border-bottom: 1px solid #2a2d36; z-index: 10; }
  header h1 { font-size: 16px; margin: 0 0 6px; }
  header .controls { display: flex; gap: 10px; align-items: center; font-size: 13px; flex-wrap: wrap; }
  select, input { background: #1e2129; color: #e6e6e6; border: 1px solid #3a3d46; border-radius: 4px; padding: 4px 8px; }
  .case { border-bottom: 1px solid #2a2d36; padding: 14px 20px; }
  .case-head { font-size: 13px; color: #9aa0ad; margin-bottom: 8px; }
  .case-head b { color: #e6e6e6; }
  .tag { display: inline-block; background: #2a2d36; border-radius: 3px; padding: 1px 6px; margin-right: 6px; font-size: 11px; }
  .row { display: flex; gap: 10px; overflow-x: auto; }
  .cell { flex: 0 0 170px; text-align: center; }
  .cell img { width: 170px; height: 170px; object-fit: contain; background:
    repeating-conic-gradient(#20232c 0% 25%, #15171d 0% 50%) 50% / 16px 16px; border-radius: 4px; border: 1px solid #2a2d36; }
  .cell .lbl { font-size: 11px; color: #9aa0ad; margin-top: 3px; }
  .missing { color: #c44; font-size: 11px; line-height: 170px; }
  #count { color: #9aa0ad; }
</style>
</head>
<body>
<header>
  <h1>AG-194 benchmark review — oct-2026 (59 cases × 6 systems)</h1>
  <div class="controls">
    <label>Scenario: <select id="scenarioFilter"><option value="">all</option></select></label>
    <label>Find: <input id="search" placeholder="image or foreground"></label>
    <span id="count"></span>
  </div>
</header>
<div id="cases"></div>
<script>
const CASES = __CASES_JSON__;
const SYSTEMS = __SYSTEMS_JSON__;

function uniqueSorted(key) { return [...new Set(CASES.map(c => c[key]))].sort(); }
function populate(sel, values) {
  for (const v of values) { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }
}
populate(document.getElementById('scenarioFilter'), uniqueSorted('scenario'));

function caseHtml(c) {
  const cells = [`<div class="cell"><img src="${c.input_src}" loading="lazy"><div class="lbl">INPUT</div></div>`];
  for (const [id, label] of SYSTEMS) {
    const src = c.outputs[id];
    cells.push(src
      ? `<div class="cell"><img src="${src}" loading="lazy"><div class="lbl">${label}</div></div>`
      : `<div class="cell"><div class="missing">no output</div><div class="lbl">${label}</div></div>`);
  }
  return `<div class="case" data-scenario="${c.scenario}" data-search="${c.image} ${c.foreground}">
    <div class="case-head">
      <span class="tag">${c.scenario}</span><span class="tag">${c.difficulty}</span>
      <b>${c.image} / ${c.foreground}</b> — "${c.prompts}"${c.should_exclude ? ` <i>(exclude: ${c.should_exclude})</i>` : ''}
    </div>
    <div class="row">${cells.join('')}</div>
  </div>`;
}

const container = document.getElementById('cases');
function render() {
  const scn = document.getElementById('scenarioFilter').value;
  const q = document.getElementById('search').value.trim().toLowerCase();
  const filtered = CASES.filter(c =>
    (!scn || c.scenario === scn) &&
    (!q || c.image.toLowerCase().includes(q) || c.foreground.toLowerCase().includes(q)));
  container.innerHTML = filtered.map(caseHtml).join('');
  document.getElementById('count').textContent = filtered.length + ' / ' + CASES.length + ' cases';
}
document.getElementById('scenarioFilter').onchange = render;
document.getElementById('search').oninput = render;
render();
</script>
</body>
</html>
"""


def to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--embed", action="store_true",
                         help="Base64-embed every image into one shareable, self-contained file")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    images_dir = results_dir.parent.parent / "images"
    out_path = results_dir / ("review_shareable.html" if args.embed else "review.html")

    with open(CASES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cases = []
    embedded = 0
    for r in rows:
        stem = Path(r["image"]).stem
        fg = r["foreground"]
        outputs = {}
        for system_id, _ in SYSTEMS:
            candidate = results_dir / "normalized" / system_id / f"{stem}__{fg}.png"
            if not candidate.exists():
                continue
            if args.embed:
                outputs[system_id] = to_data_uri(candidate)
                embedded += 1
            else:
                outputs[system_id] = f"normalized/{system_id}/{stem}__{fg}.png"
        input_path = images_dir / r["image"]
        if args.embed:
            input_src = to_data_uri(input_path) if input_path.exists() else ""
            embedded += 1 if input_path.exists() else 0
        else:
            input_src = f"../../images/{r['image']}"
        cases.append({
            "image": r["image"], "foreground": fg,
            "scenario": r["scenario"], "difficulty": r["difficulty"],
            "prompts": r["prompts"], "should_exclude": r["should_exclude"],
            "input_src": input_src,
            "outputs": outputs,
        })

    html = (TEMPLATE
            .replace("__CASES_JSON__", json.dumps(cases))
            .replace("__SYSTEMS_JSON__", json.dumps(SYSTEMS)))
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({len(cases)} cases, {embedded} images embedded, {size_mb:.1f} MB)")
    print(f"Open: file://{out_path}")


if __name__ == "__main__":
    main()
