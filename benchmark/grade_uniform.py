"""AG-184 uniform grading: run the SAME judge_result() check, once, no retries, over
every candidate's output — so pass rate is comparable across candidates that don't all
have their own internal self-correction loop (only A1 does in production).

Writes benchmark/results/judge_uniform_<candidate>.json:
  {candidate, judge_model, total, judged, passed, results: [{image, foreground,
   prompts, elapsed_s, passed, error, issues}]}

Usage:
  uv run python benchmark/grade_uniform.py --candidate b_rmbg_sam31
  uv run python benchmark/grade_uniform.py --candidate a1_judge_on --judge-model opus
  uv run python benchmark/grade_uniform.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from guided_remove_background.clients.vlm_judge import judge_result

from candidate_registry import ALL_CANDIDATES

log = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
IMAGES_DIR = BENCHMARK_DIR / "images"


def _composite_white(image_path: Path, out_path: Path) -> None:
    """Composite (possibly-transparent) image onto white — same as the pipeline's own
    judge step, so grading is apples-to-apples regardless of whether the candidate's
    output actually has alpha (edit-model candidates don't; see fal_edit.py)."""
    img = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.save(out_path, format="PNG")


def grade_one(record: dict, judge_model: str) -> dict:
    image_path = IMAGES_DIR / record["image"]
    result_path = BENCHMARK_DIR / record["output_png"]
    prompt = " ".join(record.get("prompts", []))

    judge_tmp = Path(tempfile.mktemp(suffix="_judge.png"))
    t0 = time.monotonic()
    try:
        _composite_white(result_path, judge_tmp)
        verdict = judge_result(image_path, judge_tmp, prompt, model=judge_model)
    finally:
        judge_tmp.unlink(missing_ok=True)
    elapsed = time.monotonic() - t0

    return {
        "image": record["image"], "foreground": record["foreground"],
        "prompts": record.get("prompts", []),
        "elapsed_s": elapsed,
        "passed": verdict.passed, "error": verdict.error,
        "issues": [{"type": i.type, "description": i.description, "fix": i.fix}
                   for i in verdict.issues],
    }


def grade_candidate(candidate_id: str, judge_model: str, concurrency: int) -> None:
    cand = ALL_CANDIDATES[candidate_id]
    meta_path = RESULTS_DIR / cand["file"]
    if not meta_path.exists():
        log.warning("[%s] No results file yet: %s — run it first.", candidate_id, meta_path)
        return

    meta = json.loads(meta_path.read_text())
    records = [r for r in meta.get("results", []) if cand["match"](r) and not r.get("error")]
    if not records:
        log.warning("[%s] No successful records to grade in %s", candidate_id, meta_path)
        return

    log.info("[%s] Grading %d case(s) with judge_model=%s ...", candidate_id, len(records), judge_model)
    out_path = RESULTS_DIR / f"judge_uniform_{candidate_id}.json"
    graded: list[dict] = []

    def _flush(done: bool = False) -> None:
        passed = sum(1 for g in graded if g["passed"] and not g["error"])
        judged = sum(1 for g in graded if not g["error"])
        out = {
            "candidate": candidate_id, "judge_model": judge_model, "done": done,
            "total": len(records), "completed": len(graded), "judged": judged, "passed": passed,
            "results": sorted(graded, key=lambda g: (g["image"], g["foreground"])),
        }
        out_path.write_text(json.dumps(out, indent=2, default=str))

    _flush()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(grade_one, r, judge_model): r for r in records}
        for fut in as_completed(futs):
            g = fut.result()
            graded.append(g)
            _flush()
            log.info("  [%d/%d] %s/%s -> %s", len(graded), len(records),
                      g["image"], g["foreground"], "PASS" if g["passed"] else "FAIL")

    _flush(done=True)
    passed = sum(1 for g in graded if g["passed"] and not g["error"])
    judged = sum(1 for g in graded if not g["error"])
    log.info("[%s] Done: %d/%d passed. -> %s", candidate_id, passed, judged, out_path)


def main() -> None:
    load_dotenv(BENCHMARK_DIR.parent / ".env", override=True)

    parser = argparse.ArgumentParser(description="Uniform single-pass judge grading across AG-184 candidates")
    parser.add_argument("--candidate", choices=list(ALL_CANDIDATES), help="Grade one candidate")
    parser.add_argument("--all", action="store_true", help="Grade every candidate that has results")
    parser.add_argument("--judge-model", default="opus", choices=["sonnet", "opus"])
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    if args.all:
        for cid in ALL_CANDIDATES:
            grade_candidate(cid, args.judge_model, args.concurrency)
    elif args.candidate:
        grade_candidate(args.candidate, args.judge_model, args.concurrency)
    else:
        raise SystemExit("Pass --candidate <id> or --all.")


if __name__ == "__main__":
    main()
