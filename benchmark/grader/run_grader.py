"""Grading orchestration — iterate results and score each one."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .providers import grade_with_anthropic, grade_with_openai

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


def grade_result(entry: dict, provider: str, api_key: str, base_url: str) -> dict:
    """Grade a single benchmark result entry."""
    preview_path = Path(entry.get("preview_jpg", ""))
    output_path = Path(entry.get("output_png", ""))
    original_path = IMAGES_DIR / entry["image"]

    result_path = preview_path if preview_path.exists() else output_path
    if not result_path.exists():
        return dict(**entry, grade={"error": "Result not found: " + str(result_path)})
    if not original_path.exists():
        return dict(**entry, grade={"error": "Original not found: " + str(original_path)})

    foreground = entry.get("foreground", "")
    description = entry.get("description", foreground)
    should_exclude = entry.get("should_exclude", "")

    try:
        if provider == "anthropic":
            grade = grade_with_anthropic(
                original_path, result_path, foreground, description, should_exclude, api_key,
            )
        else:
            grade = grade_with_openai(
                original_path, result_path, foreground, description, should_exclude,
                api_key, base_url,
            )
    except Exception as e:
        grade = {"error": str(e)}

    return {
        "image": entry["image"],
        "foreground": entry["foreground"],
        "scenario": entry.get("scenario", ""),
        "difficulty": entry.get("difficulty", ""),
        "description": entry.get("description", ""),
        "should_exclude": entry.get("should_exclude", ""),
        "rmbg_would": entry.get("rmbg_would", ""),
        "output_png": entry.get("output_png", ""),
        "preview_jpg": entry.get("preview_jpg", ""),
        "mode": entry["mode"],
        "elapsed_s": entry.get("elapsed_s", 0),
        "grade": grade,
    }


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

    parser = argparse.ArgumentParser(description="Auto-grade guided RMBG benchmark results")
    parser.add_argument("--mode", default=None, help="Grade only this mode")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = args.api_key
    if not api_key:
        env_name = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
    if not api_key:
        env_name = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        sys.exit("No API key. Pass --api-key or set " + env_name)

    meta_path = RESULTS_DIR / "run_meta.json"
    if not meta_path.exists():
        sys.exit("No run results at " + str(meta_path) + ". Run benchmark/runner.py first.")

    meta = json.loads(meta_path.read_text())
    results = meta["results"]
    if args.mode:
        results = [r for r in results if r["mode"] == args.mode]

    successful = [r for r in results if not r.get("error")]
    if not successful:
        sys.exit("No successful results to grade.")

    log.info("Grading %d result(s) with %s ...", len(successful), args.provider)

    graded: list[dict] = []
    for i, entry in enumerate(successful, 1):
        log.info("  [%d/%d] %s / %s / %s", i, len(successful), entry["mode"], entry["image"], entry["foreground"])
        result = grade_result(entry, args.provider, api_key, args.base_url)
        graded.append(result)
        grade_data = result.get("grade", {})
        score = grade_data.get("overall", "ERR")
        if isinstance(score, (int, float)):
            log.info("    -> overall=%.2f", score)
        else:
            log.info("    -> %s", score)
        time.sleep(0.5)

    modes_seen = sorted(set(r["mode"] for r in graded))
    for mode in modes_seen:
        mode_results = [r for r in graded if r["mode"] == mode]
        out_path = RESULTS_DIR / ("grades_" + mode + ".json")
        out_path.write_text(json.dumps(mode_results, indent=2, default=str))
        scores = [r["grade"].get("overall", 0) for r in mode_results if "error" not in r.get("grade", {})]
        avg = sum(scores) / len(scores) if scores else 0
        log.info("[%s] %d graded, avg=%.3f -> %s", mode, len(mode_results), avg, out_path)


if __name__ == "__main__":
    main()
