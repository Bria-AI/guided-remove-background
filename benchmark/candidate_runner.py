"""Generic AG-184 candidate runner — one script, pick the candidate with --candidate.

Candidate A1 (existing chain, judge on) and A2 (existing chain, judge off) are already
covered by `runner.py --mode guided` / `runner.py --mode guided --no-verify`. This
script covers the structurally different candidates: B (RMBG + SAM directly, no VLM,
no judge), C1-C3 (a single instruction-based edit-model call), and E1 (an RMBG alpha
pass on top of C1's output).

Every candidate writes the same shape of run_meta_<candidate>.json that
`runner.py`/`candidates.html` already expect (total_cases, completed, succeeded,
failed, done, total_elapsed_s, results[]), so the dashboard needs no changes when a
new candidate's results show up.

Usage:
  uv run python benchmark/candidate_runner.py --candidate b_rmbg_sam31
  uv run python benchmark/candidate_runner.py --candidate c1_fibo
  uv run python benchmark/candidate_runner.py --candidate e1_fibo_alpha
  uv run python benchmark/candidate_runner.py --list
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

from PIL import Image

from guided_remove_background.clients.bria_rmbg import call_rmbg
from guided_remove_background.clients.fal_edit import call_edit_model
from guided_remove_background.pipelines.rmbg_sam_direct import rmbg_sam_direct
from guided_remove_background.processing.output import save_preview, save_result

from candidate_registry import RUNNABLE_CANDIDATES as CANDIDATES

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


def _edit_instruction(prompts: list[str]) -> str:
    """Turn the case's prompt phrasing(s) into a single edit instruction, per the
    AG-184 ticket's own example phrasing ("remove the background and keep only ...")."""
    return f"Remove the background and keep only {' / '.join(prompts)}."


def _save_rgb(rgb, output_path: Path) -> tuple[Path, Path]:
    """Edit-model outputs are plain opaque RGB (no alpha — see fal_edit.py), so they
    can't go through save_result/save_preview which assume a 4-channel RGBA array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(output_path)
    preview_path = output_path.with_name(output_path.stem + "_preview.jpg")
    Image.fromarray(rgb, "RGB").save(preview_path, quality=92)
    return output_path, preview_path


def run_one(case: dict, candidate_id: str, cand: dict) -> dict:
    """Run a single case for one candidate. Returns the same dict shape runner.py writes."""
    image_path = IMAGES_DIR / case["image"]
    if not image_path.exists():
        return {
            "image": case["image"], "foreground": case["foreground"],
            "mode": candidate_id, "error": f"Image not found: {image_path}", "elapsed_s": 0,
        }

    prompts = [p.strip() for p in case["prompts"].split("|")]
    out_dir = RESULTS_DIR / cand["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{image_path.stem}__{case['foreground']}.png"

    log.info("[%s] %s -> %s", candidate_id, case["image"], case["foreground"])
    try:
        if cand["kind"] == "rmbg_sam_direct":
            result = rmbg_sam_direct(image_path, prompts, output_path)
            output_png, preview_jpg = result.output_path, result.preview_path
            elapsed_s, sam_scores = result.elapsed_s, result.sam_scores

        elif cand["kind"] == "edit_model":
            t0 = time.monotonic()
            instruction = _edit_instruction(prompts)
            rgb = call_edit_model(image_path, instruction, model=cand["model_slug"])
            if rgb is None:
                raise RuntimeError(f"{cand['model_slug']} returned no image")
            output_png, preview_jpg = _save_rgb(rgb, output_path)
            elapsed_s = time.monotonic() - t0
            sam_scores = {}

        else:
            raise ValueError(f"Unknown candidate kind: {cand['kind']}")

        return {
            "image": case["image"], "foreground": case["foreground"],
            "scenario": case.get("scenario", ""), "difficulty": case.get("difficulty", ""),
            "description": case.get("description", ""),
            "should_exclude": case.get("should_exclude", ""),
            "rmbg_would": case.get("rmbg_would", ""),
            "prompts": prompts,
            "mode": candidate_id,
            "error": None, "elapsed_s": elapsed_s,
            "output_png": str(Path(output_png).relative_to(RESULTS_DIR.parent)),
            "preview_jpg": str(Path(preview_jpg).relative_to(RESULTS_DIR.parent)),
            "sam_scores": sam_scores,
            "vlm_decompose": None,
            "step_images": {},
            "judge_verdicts": [],
        }
    except Exception as e:
        log.error("[%s] %s/%s FAILED: %s", candidate_id, case["image"], case["foreground"], e)
        return {
            "image": case["image"], "foreground": case["foreground"],
            "mode": candidate_id, "error": str(e), "elapsed_s": 0,
        }


def run_candidate(candidate_id: str, cases: list[dict], concurrency: int) -> None:
    cand = CANDIDATES[candidate_id]
    meta_path = RESULTS_DIR / cand["file"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_raw: list[dict] = []
    t0 = time.monotonic()

    def _flush_meta(done: bool = False) -> None:
        elapsed = time.monotonic() - t0
        sorted_results = sorted(all_raw, key=lambda r: (r["image"], r["foreground"]))
        meta = {
            "total_cases": len(cases),
            "completed": len(all_raw),
            "succeeded": sum(1 for r in all_raw if not r.get("error")),
            "failed": sum(1 for r in all_raw if r.get("error")),
            "done": done,
            "total_elapsed_s": elapsed,
            "candidate": candidate_id,
            "results": sorted_results,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    _flush_meta()
    log.info("[%s] Running %d case(s) ...", candidate_id, len(cases))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_one, case, candidate_id, cand): case for case in cases}
        for fut in as_completed(futures):
            result = fut.result()
            all_raw.append(result)
            _flush_meta()
            status = "OK" if not result.get("error") else f"ERR: {result['error'][:60]}"
            log.info("  [%d/%d] %s/%s -> %s (%.1fs)",
                     len(all_raw), len(cases), result["image"], result["foreground"],
                     status, result["elapsed_s"])

    _flush_meta(done=True)
    elapsed_total = time.monotonic() - t0
    log.info("[%s] Done: %d/%d succeeded in %.1fs. Meta -> %s",
             candidate_id, sum(1 for r in all_raw if not r.get("error")),
             len(cases), elapsed_total, meta_path)


def run_alpha_pass_one(record: dict, candidate_id: str, cand: dict) -> dict:
    """Re-matte a SOURCE candidate's already-generated output image (not the original
    photo) to extract a real alpha matte from its flat/near-solid background via an
    RMBG-2.0 saliency re-pass (kind="rmbg_realpha" — can silently drop objects the
    source already got right, see e1_fibo_alpha's registry entry)."""
    source_image = Path(__file__).parent / record["output_png"]
    out_dir = RESULTS_DIR / cand["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{Path(record['image']).stem}__{record['foreground']}.png"

    log.info("[%s] %s/%s -> re-matting %s", candidate_id, record["image"], record["foreground"], source_image.name)
    try:
        t0 = time.monotonic()
        sam_scores: dict[str, float] = {}
        rgba = call_rmbg(source_image)
        output_png = save_result(rgba, output_path)
        preview_jpg = save_preview(rgba, output_path)
        elapsed_s = time.monotonic() - t0

        return {
            "image": record["image"], "foreground": record["foreground"],
            "scenario": record.get("scenario", ""), "difficulty": record.get("difficulty", ""),
            "description": record.get("description", ""),
            "should_exclude": record.get("should_exclude", ""),
            "rmbg_would": record.get("rmbg_would", ""),
            "prompts": record.get("prompts", []),
            "mode": candidate_id,
            "error": None, "elapsed_s": elapsed_s,
            "output_png": str(Path(output_png).relative_to(RESULTS_DIR.parent)),
            "preview_jpg": str(Path(preview_jpg).relative_to(RESULTS_DIR.parent)),
            "sam_scores": sam_scores,
            "vlm_decompose": None,
            "step_images": {},
            "judge_verdicts": [],
        }
    except Exception as e:
        log.error("[%s] %s/%s FAILED: %s", candidate_id, record["image"], record["foreground"], e)
        return {
            "image": record["image"], "foreground": record["foreground"],
            "mode": candidate_id, "error": str(e), "elapsed_s": 0,
        }


def run_alpha_pass_candidate(candidate_id: str, concurrency: int) -> None:
    from candidate_registry import ALL_CANDIDATES  # source may be a runner.py candidate (a1/a2/rmbg_only), not just a RUNNABLE_CANDIDATES one

    cand = CANDIDATES[candidate_id]
    source_id = cand["source_candidate"]
    source_meta_path = RESULTS_DIR / ALL_CANDIDATES[source_id]["file"]
    if not source_meta_path.exists():
        sys.exit(f"Source candidate '{source_id}' has no results yet: {source_meta_path}")

    source_meta = json.loads(source_meta_path.read_text())
    match = ALL_CANDIDATES[source_id]["match"]
    records = [r for r in source_meta.get("results", []) if match(r) and not r.get("error")]
    if not records:
        sys.exit(f"No successful records in {source_meta_path} to re-matte.")

    meta_path = RESULTS_DIR / cand["file"]
    all_raw: list[dict] = []
    t0 = time.monotonic()

    def _flush_meta(done: bool = False) -> None:
        elapsed = time.monotonic() - t0
        sorted_results = sorted(all_raw, key=lambda r: (r["image"], r["foreground"]))
        meta = {
            "total_cases": len(records), "completed": len(all_raw),
            "succeeded": sum(1 for r in all_raw if not r.get("error")),
            "failed": sum(1 for r in all_raw if r.get("error")),
            "done": done, "total_elapsed_s": elapsed,
            "candidate": candidate_id, "source_candidate": source_id,
            "results": sorted_results,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    _flush_meta()
    log.info("[%s] Re-matting %d record(s) from '%s' ...", candidate_id, len(records), source_id)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_alpha_pass_one, r, candidate_id, cand): r for r in records}
        for fut in as_completed(futures):
            result = fut.result()
            all_raw.append(result)
            _flush_meta()
            status = "OK" if not result.get("error") else f"ERR: {result['error'][:60]}"
            log.info("  [%d/%d] %s/%s -> %s (%.1fs)",
                     len(all_raw), len(records), result["image"], result["foreground"],
                     status, result["elapsed_s"])

    _flush_meta(done=True)
    log.info("[%s] Done: %d/%d succeeded. Meta -> %s",
             candidate_id, sum(1 for r in all_raw if not r.get("error")), len(records), meta_path)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

    parser = argparse.ArgumentParser(description="Run one AG-184 candidate pipeline over the benchmark cases")
    parser.add_argument("--candidate", choices=list(CANDIDATES), help="Which candidate to run")
    parser.add_argument("--filter", default=None, help="Substring filter on image column")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--list", action="store_true", help="List available candidates and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        for cid, cand in CANDIDATES.items():
            extra = f" (source: {cand['source_candidate']})" if "source_candidate" in cand else ""
            print(f"  {cid:16s} kind={cand['kind']:16s} -> results/{cand['file']}{extra}")
        return

    if not args.candidate:
        sys.exit("Pass --candidate <id> or --list to see options.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    cand = CANDIDATES[args.candidate]
    if "source_candidate" in cand:
        run_alpha_pass_candidate(args.candidate, args.concurrency)
        return

    cases = load_cases(args.filter)
    if not cases:
        sys.exit("No cases matched. Check data/cases.csv and --filter.")

    run_candidate(args.candidate, cases, args.concurrency)


if __name__ == "__main__":
    main()
