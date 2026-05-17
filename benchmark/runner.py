"""Benchmark runner — batch-process all cases across guided and baseline modes.

Outputs:
  benchmark/results/{mode}/{image_stem}__{foreground}.png
  benchmark/results/run_meta.json

Usage:
  uv run python benchmark/runner.py
  uv run python benchmark/runner.py --mode guided
  uv run python benchmark/runner.py --filter living
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from guided_remove_background import MODES
from guided_remove_background.remove_bg import remove_bg

log = logging.getLogger(__name__)

CASES_CSV = Path(__file__).parent / "data" / "cases.csv"
IMAGES_DIR = Path(__file__).parent / "images"
RESULTS_DIR = Path(__file__).parent / "results"


def load_cases(filter_str: str | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(CASES_CSV) as f:
        for row in csv.DictReader(f):
            if filter_str and filter_str.lower() not in row["image"].lower():
                continue
            rows.append(row)
    return rows


def run_one(case: dict, mode: str) -> dict:
    """Run a single case/mode combination."""
    image_path = IMAGES_DIR / case["image"]
    if not image_path.exists():
        return {
            "image": case["image"], "foreground": case["foreground"],
            "mode": mode,
            "error": f"Image not found: {image_path}", "elapsed_s": 0,
        }

    prompts = [p.strip() for p in case["prompts"].split("|")]
    out_dir = RESULTS_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{image_path.stem}__{case['foreground']}.png"

    log.info("[%s] %s -> %s", mode, case["image"], case["foreground"])
    try:
        result = remove_bg(
            image=image_path, prompts=prompts, output=output_path, mode=mode,
            save_steps=(mode == "guided"),
        )
        vlm_data = None
        if result.vlm_decompose:
            vlm_data = {
                "mode": result.vlm_decompose.mode,
                "targets": result.vlm_decompose.targets,
                "include": result.vlm_decompose.include,
                "exclude": result.vlm_decompose.exclude,
                "error": result.vlm_decompose.error,
            }
        step_images = {}
        for step_name, step_path in result.step_images.items():
            try:
                step_images[step_name] = str(Path(step_path).relative_to(RESULTS_DIR.parent))
            except ValueError:
                step_images[step_name] = step_path
        judge_data = []
        for v in result.judge_verdicts:
            judge_data.append({
                "passed": v.passed,
                "error": v.error,
                "issues": [{"type": i.type, "description": i.description, "fix": i.fix}
                           for i in v.issues],
            })
        return {
            "image": case["image"], "foreground": case["foreground"],
            "scenario": case.get("scenario", ""), "difficulty": case.get("difficulty", ""),
            "description": case.get("description", ""),
            "should_exclude": case.get("should_exclude", ""),
            "rmbg_would": case.get("rmbg_would", ""),
            "prompts": prompts,
            "mode": mode,
            "error": None, "elapsed_s": result.elapsed_s,
            "output_png": str(result.output_path.relative_to(RESULTS_DIR.parent)),
            "preview_jpg": str(result.preview_path.relative_to(RESULTS_DIR.parent)),
            "sam_scores": result.sam_scores,
            "vlm_decompose": vlm_data,
            "step_images": step_images,
            "judge_verdicts": judge_data,
        }
    except Exception as e:
        log.error("[%s] %s/%s FAILED: %s", mode, case["image"], case["foreground"], e)
        return {
            "image": case["image"], "foreground": case["foreground"],
            "mode": mode, "error": str(e), "elapsed_s": 0,
        }


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

    parser = argparse.ArgumentParser(description="Run guided RMBG benchmark")
    parser.add_argument("--mode", choices=MODES, help="Run only this mode (default: all)")
    parser.add_argument("--filter", default=None, help="Substring filter on image column")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    cases = load_cases(args.filter)
    modes = [args.mode] if args.mode else list(MODES)
    if not cases:
        sys.exit("No cases matched. Check data/cases.csv and --filter.")

    jobs: list[tuple[dict, str]] = []
    for c in cases:
        for m in modes:
            jobs.append((c, m))

    log.info("Running %d case(s) x %d mode(s) = %d total jobs",
             len(cases), len(modes), len(jobs))

    all_raw: list[dict] = []
    t0 = time.monotonic()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RESULTS_DIR / "run_meta.json"

    def _flush_meta(done: bool = False) -> None:
        elapsed = time.monotonic() - t0
        sorted_results = sorted(all_raw, key=lambda r: (r["mode"], r["image"], r["foreground"]))
        meta = {
            "total_cases": len(jobs),
            "completed": len(all_raw),
            "succeeded": sum(1 for r in all_raw if not r.get("error")),
            "failed": sum(1 for r in all_raw if r.get("error")),
            "done": done,
            "total_elapsed_s": elapsed,
            "results": sorted_results,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    _flush_meta()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, case, mode): (case, mode)
                   for case, mode in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            all_raw.append(result)
            _flush_meta()
            status = "OK" if not result.get("error") else f"ERR: {result['error'][:60]}"
            log.info("  [%d/%d] [%s] %s/%s -> %s (%.1fs)",
                     len(all_raw), len(jobs), result["mode"], result["image"],
                     result["foreground"], status, result["elapsed_s"])

    _flush_meta(done=True)
    elapsed_total = time.monotonic() - t0
    log.info("Done: %d/%d succeeded in %.1fs. Meta -> %s",
             sum(1 for r in all_raw if not r.get("error")),
             len(jobs), elapsed_total, meta_path)


if __name__ == "__main__":
    main()
