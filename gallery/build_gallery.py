"""CLI entry point — build gallery.html from benchmark grades."""

from __future__ import annotations

import sys
from pathlib import Path

from .builder import build_html
from .stats import MODES, MODE_LABELS, build_case_index, compute_stats, load_grades

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results"
OUTPUT = Path(__file__).resolve().parent.parent / "gallery.html"


def main() -> None:
    all_grades = load_grades(RESULTS_DIR)
    if not all_grades:
        sys.exit("No grade files found. Run benchmark/grader/run_grader.py first.")

    cases = build_case_index(all_grades)
    stats = compute_stats(all_grades)

    html = build_html(all_grades, cases, stats)
    OUTPUT.write_text(html)
    print("Gallery written to", OUTPUT)
    print("Cases:", len(cases))
    for mode in MODES:
        if mode in stats:
            s = stats[mode]
            print("  {}: avg={:.3f} correct={:.0%} cases={}".format(
                MODE_LABELS.get(mode, mode), s["avg_overall"], s["correct_rate"], s["count"],
            ))


if __name__ == "__main__":
    main()
