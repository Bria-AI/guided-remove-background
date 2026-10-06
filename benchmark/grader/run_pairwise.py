"""AG-195 — pairwise judge: Bria against each competitor, one case at a time.

Shows Claude the ORIGINAL photo plus two anonymous results (sides shuffled per
comparison so the model can't learn "A is always Bria"), and asks for a verdict —
tie allowed. Run once per (competitor, case); also runs Bria against the plain
no-guidance baseline (rmbg_only), per the ticket.

Reads outputs from an AG-194 results dir (normalized/<system>/<stem>__<fg>.png —
fixed size, alpha preserved where real). Writes one row per comparison to
pairwise_results.json in that same dir, plus a win-rate table (overall, by
scenario, and by benchmark_class) to pairwise_summary.json and printed to stdout.

Usage:
  uv run python benchmark/grader/run_pairwise.py --results-dir benchmark/results/oct-2026
  uv run python benchmark/grader/run_pairwise.py --results-dir benchmark/results/oct-2026 --limit 5
  uv run python benchmark/grader/run_pairwise.py --results-dir benchmark/results/oct-2026 --resume
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from pairwise_prompt import PROMPT_VERSION, SYSTEM, VERDICT_SCHEMA, pairwise_prompt  # noqa: E402

log = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
CASES_CSV = BENCHMARK_DIR / "data" / "cases.csv"
IMAGES_DIR = BENCHMARK_DIR / "images"
MODEL = "claude-opus-5-5"

BRIA_SYSTEM = "h1_guided_v1_1"
BRIA_LABEL = "Bria v1.1 (Gemini intent, in-house SAM 3 + RMBG, fal)"
# competitor_id -> label used in the summary table. rmbg_only is "Bria against
# plain remove background" per the ticket; the other four are AG-194's competitors.
COMPETITORS = {
    "rmbg_only": "Plain remove background (no guidance)",
    "c1_fibo": "FIBO-Edit-1.5",
    "c2_nanobanana2": "Nano Banana 2",
    "c3_gptimage2": "GPT Image 2",
    "g2_extract_object_rmbg_on": "extract-object",
}


def load_cases() -> list[dict]:
    with open(CASES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


CHECKER_CELL = 16


def _checkerboard(size: tuple[int, int]) -> Image.Image:
    w, h = size
    board = Image.new("RGB", size, (255, 255, 255))
    px = board.load()
    for y in range(h):
        for x in range(w):
            if ((x // CHECKER_CELL) + (y // CHECKER_CELL)) % 2 == 0:
                px[x, y] = (222, 222, 222)
    return board


def image_block(path: Path) -> tuple[dict, bool]:
    """Base64 image block for the Messages API. Claude's vision input flattens
    PNG alpha onto solid white before the model ever sees it (verified directly —
    a transparent/opaque test image came back as "no checkerboard, uniform white"
    regardless of the real alpha channel). So real transparency has to be made
    visible ourselves: composite RGBA onto a checkerboard here, same as the AG-178
    deck's thumbnails, instead of relying on the prompt's checkerboard instruction
    to mean anything for an untouched alpha PNG.

    Also returns whether the source image actually has real alpha, so the caller
    can tell the grader the fact as text -- a competitor that paints its own fake
    checkerboard into flat RGB (Nano Banana 2 did this) is otherwise indistinguishable
    from a real cutout once both are composited onto the same checkerboard pixels."""
    im = Image.open(path)
    has_real_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    if has_real_alpha:
        im = im.convert("RGBA")
        bg = _checkerboard(im.size)
        bg.paste(im, (0, 0), im)
        im = bg
    else:
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(buf.getvalue()).decode()},
    }
    return block, has_real_alpha


def judge_one(client: anthropic.Anthropic, original: Path, result_a: Path, result_b: Path,
              instruction: str, should_exclude: str, scenario: str) -> dict:
    original_block, _ = image_block(original)
    block_a, a_real_alpha = image_block(result_a)
    block_b, b_real_alpha = image_block(result_b)
    content = [
        {"type": "text", "text": "ORIGINAL photo:"},
        original_block,
        {"type": "text", "text": "RESULT A:"},
        block_a,
        {"type": "text", "text": "RESULT B:"},
        block_b,
        {"type": "text", "text": pairwise_prompt(instruction, should_exclude, scenario,
                                                   a_real_alpha, b_real_alpha)},
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    verdict = json.loads(text)
    verdict["_usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return verdict


def run_comparison(client: anthropic.Anthropic, case: dict, competitor_id: str,
                    bria_dir: Path, competitor_dir: Path) -> dict:
    stem = Path(case["image"]).stem
    fg = case["foreground"]
    fname = f"{stem}__{fg}.png"
    bria_path = bria_dir / fname
    comp_path = competitor_dir / fname
    base_row = {
        "image": case["image"], "foreground": fg, "benchmark_class": case["benchmark_class"],
        "scenario": case["scenario"], "competitor": competitor_id,
    }
    if not bria_path.exists() or not comp_path.exists():
        return {**base_row, "error": f"missing output: bria={bria_path.exists()} competitor={comp_path.exists()}"}

    # Seeded per (competitor, case) so a --resume re-run reproduces the same side assignment.
    bria_is_a = random.Random(f"{competitor_id}:{stem}:{fg}").random() < 0.5
    result_a, result_b = (bria_path, comp_path) if bria_is_a else (comp_path, bria_path)

    try:
        verdict = judge_one(client, IMAGES_DIR / case["image"], result_a, result_b,
                             case["prompts"], case["should_exclude"], case["scenario"])
    except Exception as e:
        return {**base_row, "error": str(e)}

    def resolve(side_winner: str) -> str:
        if side_winner == "tie":
            return "tie"
        bria_side = "A" if bria_is_a else "B"
        return "bria" if side_winner == bria_side else "competitor"

    return {
        **base_row, "error": None, "bria_is_a": bria_is_a,
        "reasoning": verdict["reasoning"],
        "selection_winner": resolve(verdict["selection_winner"]),
        "fidelity_winner": resolve(verdict["fidelity_winner"]),
        "overall_winner": resolve(verdict["overall_winner"]),
        "usage": verdict["_usage"],
    }


def summarize(rows: list[dict]) -> dict:
    """Win rate (ties counted as half, matching the Product Holding one-pager's
    convention) per competitor: overall, plus a breakdown by scenario (include/
    exclude/narrow) and by benchmark_class (people/product/interior/multi_object),
    since every row already carries both fields."""
    by_competitor: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_competitor[r["competitor"]].append(r)

    def table_for(rows_subset: list[dict]) -> dict:
        n = len(rows_subset)
        if n == 0:
            return {"n": 0}
        c = Counter(r["overall_winner"] for r in rows_subset)
        bria_wins, ties, comp_wins = c["bria"], c["tie"], c["competitor"]
        win_rate = (bria_wins + 0.5 * ties) / n
        return {
            "n": n, "bria_wins": bria_wins, "ties": ties, "competitor_wins": comp_wins,
            "bria_win_rate_ties_half": round(win_rate, 3),
            "tie_share": round(ties / n, 3),
        }

    def breakdowns(comp_rows: list[dict]) -> dict:
        return {
            "overall": table_for(comp_rows),
            "by_scenario": {
                s: table_for([r for r in comp_rows if r["scenario"] == s])
                for s in sorted({r["scenario"] for r in comp_rows})
            },
            "by_class": {
                c: table_for([r for r in comp_rows if r["benchmark_class"] == c])
                for c in sorted({r["benchmark_class"] for r in comp_rows})
            },
        }

    return {
        "bria_candidate": BRIA_SYSTEM,
        "bria_label": BRIA_LABEL,
        "competitors": {
            comp_id: {"label": COMPETITORS[comp_id], **breakdowns(comp_rows)}
            for comp_id, comp_rows in by_competitor.items()
        },
    }


def print_table(summary: dict) -> None:
    def row(label: str, o: dict, indent: str = "") -> None:
        if o["n"] == 0:
            return
        print(f"{indent}{label:<40} {o['n']:>4} {o['bria_win_rate_ties_half']*100:>9.1f}% "
              f"{o['tie_share']*100:>6.1f}% {o['bria_wins']:>5} {o['ties']:>5} {o['competitor_wins']:>5}")

    print(f"\nBria candidate: {summary['bria_label']}  (system id: {summary['bria_candidate']})")
    header = f"\n{'COMPETITOR':<40} {'N':>4} {'BRIA WIN%':>10} {'TIE%':>7} {'BRIA':>5} {'TIE':>5} {'COMP':>5}"
    print(header)
    for comp_id, data in summary["competitors"].items():
        row(data["label"], data["overall"])

    for by_key, title in (("by_scenario", "BY SCENARIO"), ("by_class", "BY BENCHMARK CLASS")):
        print(f"\n--- {title} ---")
        for comp_id, data in summary["competitors"].items():
            if not data[by_key]:
                continue
            print(f"\n{data['label']}:")
            print(header)
            for key, o in data[by_key].items():
                row(key, o, indent="  ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                         help="AG-194 results dir, e.g. benchmark/results/oct-2026")
    parser.add_argument("--limit", type=int, default=None, help="Only run this many cases (smoke test)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Skip comparisons already in pairwise_results.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    load_dotenv(BENCHMARK_DIR.parent / ".env", override=True)
    client = anthropic.Anthropic()

    results_dir = args.results_dir.resolve()
    normalized = results_dir / "normalized"
    bria_dir = normalized / BRIA_SYSTEM
    if not bria_dir.exists():
        sys.exit(f"No {bria_dir} — run benchmark/normalize_outputs.py first.")

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    jobs = [(case, comp_id) for case in cases for comp_id in COMPETITORS]
    out_path = results_dir / "pairwise_results.json"
    rows: list[dict] = []
    if args.resume and out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8"))["results"]
        done = {(r["image"], r["foreground"], r["competitor"]) for r in rows}
        jobs = [(c, comp) for c, comp in jobs if (c["image"], c["foreground"], comp) not in done]
        log.info("Resuming: %d comparison(s) already done, %d to run", len(rows), len(jobs))

    def _flush(done: bool) -> None:
        out_path.write_text(json.dumps({
            "prompt_version": PROMPT_VERSION, "model": MODEL, "done": done,
            "total": len(rows), "results": rows,
        }, indent=2, default=str), encoding="utf-8")

    t0 = time.monotonic()
    log.info("Running %d comparison(s) across %d case(s) x %d competitor(s) ...",
              len(jobs), len(cases), len(COMPETITORS))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_comparison, client, case, comp_id, bria_dir, normalized / comp_id): (case, comp_id)
            for case, comp_id in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            case, comp_id = futures[fut]
            row = fut.result()
            rows.append(row)
            _flush(done=False)
            status = row["overall_winner"] if not row.get("error") else f"ERR: {row['error'][:60]}"
            log.info("  [%d/%d] %s/%s vs %s -> %s", i, len(jobs), case["image"], case["foreground"], comp_id, status)

    _flush(done=True)
    elapsed = time.monotonic() - t0
    errors = sum(1 for r in rows if r.get("error"))
    log.info("Done: %d comparisons, %d errors, %.1fs. Raw -> %s", len(rows), errors, elapsed, out_path)

    summary = summarize(rows)
    summary_path = results_dir / "pairwise_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Summary -> %s", summary_path)
    print_table(summary)


if __name__ == "__main__":
    main()
