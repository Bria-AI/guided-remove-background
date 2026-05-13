"""Grade data loading and statistics for guided RMBG benchmark."""

from __future__ import annotations

import json
from pathlib import Path

MODES = ["guided", "rmbg-only"]
MODE_LABELS = {"guided": "Guided RMBG", "rmbg-only": "Plain RMBG (baseline)"}
MODE_COLORS = {"guided": "#2563eb", "rmbg-only": "#d97706"}


def load_grades(results_dir: Path) -> dict[str, list[dict]]:
    all_grades = {}
    for mode in MODES:
        path = results_dir / ("grades_" + mode + ".json")
        if path.exists():
            all_grades[mode] = json.loads(path.read_text())
    return all_grades


def build_case_index(all_grades: dict) -> dict:
    """Build a dict keyed by (image, foreground) with grades from each mode."""
    cases: dict = {}
    for mode, entries in all_grades.items():
        for entry in entries:
            key = (entry["image"], entry["foreground"])
            if key not in cases:
                cases[key] = {
                    "image": entry["image"],
                    "foreground": entry["foreground"],
                    "scenario": entry.get("scenario", ""),
                    "difficulty": entry.get("difficulty", ""),
                    "description": entry.get("description", ""),
                    "should_exclude": entry.get("should_exclude", ""),
                    "rmbg_would": entry.get("rmbg_would", ""),
                    "modes": {},
                }
            cases[key]["modes"][mode] = entry
    return cases


def compute_stats(all_grades: dict) -> dict:
    stats = {}
    for mode, entries in all_grades.items():
        scores, times, total = [], [], 0
        inc_scores, exc_scores, edge_scores = [], [], []
        correct = 0

        for e in entries:
            g = e.get("grade", {})
            if "error" in g:
                continue
            total += 1
            scores.append(g.get("overall", 0))
            times.append(e.get("elapsed_s", 0))
            inc_scores.append(g.get("inclusion_accuracy", 0))
            exc_scores.append(g.get("exclusion_accuracy", 0))
            edge_scores.append(g.get("edge_quality", 0))
            if g.get("foreground_correct", False):
                correct += 1

        stats[mode] = {
            "count": total,
            "avg_overall": _avg(scores),
            "avg_inclusion": _avg(inc_scores),
            "avg_exclusion": _avg(exc_scores),
            "avg_edge_quality": _avg(edge_scores),
            "correct_rate": correct / total if total else 0,
            "median_latency": sorted(times)[len(times) // 2] if times else 0,
            "avg_latency": _avg(times),
        }
    return stats


def scenario_stats(all_grades: dict, scenarios: list[str]) -> dict:
    result = {}
    for scen in scenarios:
        result[scen] = {}
        for mode, entries in all_grades.items():
            scen_entries = [e for e in entries if e.get("scenario") == scen]
            scores = [e["grade"].get("overall", 0) for e in scen_entries if "error" not in e.get("grade", {})]
            result[scen][mode] = _avg(scores)
    return result


def score_color(score: float) -> str:
    if score >= 0.8:
        return "#16a34a"
    if score >= 0.5:
        return "#d97706"
    return "#dc2626"


def _avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0
