"""Build the gallery HTML — side-by-side RMBG baseline vs Guided RMBG."""

from __future__ import annotations

import json
from pathlib import Path

from .stats import MODE_COLORS, MODE_LABELS, MODES, scenario_stats, score_color

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def build_html(all_grades: dict, cases: dict, stats: dict) -> str:
    """Read template.html and replace marker comments with generated content."""
    template = TEMPLATE_PATH.read_text()
    modes_found = [m for m in MODES if m in all_grades]
    scenarios = sorted(set(c["scenario"] for c in cases.values() if c["scenario"]))

    scoreboard = _build_scoreboard(modes_found, stats)
    scen_options = "".join("<option value='{}'>{}</option>".format(s, s.title()) for s in scenarios)
    diff_options = "".join("<option value='{}'>{}</option>".format(d, d.title()) for d in ["easy", "medium", "hard"])
    rows = _build_rows(cases, modes_found)

    chart_data_json = json.dumps({
        "modes": modes_found,
        "labels": [MODE_LABELS.get(m, m) for m in modes_found],
        "colors": [MODE_COLORS.get(m, "#666") for m in modes_found],
        "stats": stats,
        "scenarios": scenarios,
        "scenario_stats": scenario_stats(all_grades, scenarios),
    }, default=str)

    html = template
    html = html.replace("<!-- SCOREBOARD -->", scoreboard)
    html = html.replace("<!-- SCEN_OPTIONS -->", scen_options)
    html = html.replace("<!-- DIFF_OPTIONS -->", diff_options)
    html = html.replace("<!-- ROWS -->", rows)
    html = html.replace("/* CHART_DATA_JSON */null", chart_data_json)
    return html


def _build_scoreboard(modes_found: list[str], stats: dict) -> str:
    cards = []
    for mode in modes_found:
        s = stats.get(mode, {})
        sc = score_color(s.get("avg_overall", 0))
        cards.append(
            "<div class='score-card' style='border-top: 4px solid {}'>"
            "<h3>{}</h3>"
            "<div class='big-score' style='color:{}'>{:.3f}</div>"
            "<div class='score-details'>"
            "<div>Foreground correct: {:.0%}</div>"
            "<div>Inclusion accuracy: {:.3f}</div>"
            "<div>Exclusion accuracy: {:.3f}</div>"
            "<div>Edge quality: {:.3f}</div>"
            "<div>Median latency: {:.1f}s</div>"
            "<div>Cases: {}</div>"
            "</div></div>".format(
                MODE_COLORS.get(mode, "#666"),
                MODE_LABELS.get(mode, mode),
                sc, s.get("avg_overall", 0),
                s.get("correct_rate", 0),
                s.get("avg_inclusion", 0),
                s.get("avg_exclusion", 0),
                s.get("avg_edge_quality", 0),
                s.get("median_latency", 0),
                s.get("count", 0),
            )
        )
    return "".join(cards)


def _build_rows(cases: dict, modes_found: list[str]) -> str:
    rows_html = []
    for key in sorted(cases.keys()):
        case = cases[key]
        img_file = case["image"]
        foreground = case["foreground"]
        scenario = case["scenario"]
        difficulty = case["difficulty"]
        description = case["description"]
        rmbg_would = case.get("rmbg_would", "")
        should_exclude = case.get("should_exclude", "")

        rmbg_data = case["modes"].get("rmbg-only")
        guided_data = case["modes"].get("guided")

        rmbg_cell = _img_cell(rmbg_data)
        guided_cell = _img_cell(guided_data)
        rmbg_score_cell = _score_cell(rmbg_data)
        guided_score_cell = _score_cell(guided_data)

        guided_overall = 0
        if guided_data:
            g = guided_data.get("grade", {})
            if "error" not in g:
                guided_overall = g.get("overall", 0)

        detail_sections = []
        for mode in modes_found:
            mdata = case["modes"].get(mode)
            if not mdata:
                continue
            grade = mdata.get("grade", {})
            has_error = "error" in grade
            overall = grade.get("overall", 0) if not has_error else 0
            sc = score_color(overall)
            detail_sections.append(
                "<div class='detail-mode'>"
                "<h4>{}</h4>"
                "<div class='detail-scores'>"
                "<span>Foreground correct: {}</span>"
                "<span>Inclusion: {:.2f}</span>"
                "<span>Exclusion: {:.2f}</span>"
                "<span>Edge: {:.2f}</span>"
                "<span>Overall: <b style='color:{}'>{:.2f}</b></span>"
                "<span>Time: {:.1f}s</span>"
                "</div>"
                "<p class='reasoning'>{}</p>"
                "{}</div>".format(
                    MODE_LABELS.get(mode, mode),
                    "Yes" if grade.get("foreground_correct") else "No",
                    grade.get("inclusion_accuracy", 0) if not has_error else 0,
                    grade.get("exclusion_accuracy", 0) if not has_error else 0,
                    grade.get("edge_quality", 0) if not has_error else 0,
                    sc, overall,
                    mdata.get("elapsed_s", 0),
                    grade.get("reasoning", ""),
                    "<p class='issues'>Issues: " + ", ".join(grade.get("issues", [])) + "</p>"
                    if grade.get("issues") else "",
                )
            )

        orig_path = "benchmark/images/" + img_file
        row_id = (img_file + "__" + foreground).replace(".", "_").replace(" ", "_")

        exclude_note = ""
        if should_exclude:
            exclude_note = "<br/><small style='color:#dc2626'>Exclude: " + should_exclude.replace("|", ", ") + "</small>"

        prompt_display = description if description else foreground
        rmbg_note = "<br/><span class='rmbg-would'>RMBG default: " + rmbg_would + "</span>" if rmbg_would else ""

        row = (
            "<tr class='case-row' data-scenario='{}' data-difficulty='{}' data-id='{}' "
            "data-guided-score='{}' onclick='toggleDetail(this)'>"
            "<td class='img-cell'><img src='{}' loading='lazy' /></td>"
            "<td class='guidance-cell'>"
            "<div class='prompt'>{}</div>"
            "<span class='tag scen-{}'>{}</span> "
            "<span class='tag diff-{}'>{}</span>"
            "{}{}</td>"
            "{}{}{}{}</tr>"
            "<tr class='detail-row' id='detail_{}'>"
            "<td colspan='6'><div class='detail-content'>{}</div></td></tr>"
        ).format(
            scenario, difficulty, row_id, guided_overall,
            orig_path, prompt_display,
            scenario, scenario, difficulty, difficulty,
            rmbg_note, exclude_note,
            rmbg_cell, guided_cell, rmbg_score_cell, guided_score_cell,
            row_id, "".join(detail_sections),
        )
        rows_html.append(row)

    return "".join(rows_html)


def _img_cell(mdata: dict | None) -> str:
    if not mdata:
        return "<td class='img-cell'><span class='na'>N/A</span></td>"
    preview = mdata.get("preview_jpg", "")
    output = mdata.get("output_png", "")
    img_src = preview if preview else output
    if img_src:
        return "<td class='img-cell'><img src='{}' loading='lazy' /></td>".format(img_src)
    return "<td class='img-cell'><span class='na'>No image</span></td>"


def _score_cell(mdata: dict | None) -> str:
    if not mdata:
        return "<td>-</td>"
    grade = mdata.get("grade", {})
    if "error" in grade:
        return "<td style='color:#dc2626'>ERR</td>"
    overall = grade.get("overall", 0)
    sc = score_color(overall)
    return "<td style='color:{}'>{:.2f}</td>".format(sc, overall)
