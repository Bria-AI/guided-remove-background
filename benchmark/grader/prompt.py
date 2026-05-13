"""Grading prompt template for guided background removal evaluation."""

from __future__ import annotations


def grading_prompt(foreground: str, description: str, should_exclude: str) -> str:
    """Build the VLM prompt for evaluating a guided RMBG result.

    The VLM sees: (1) original image, (2) background-removed result.
    It grades whether the user's foreground guidance was followed correctly.
    """
    exclude_note = ""
    if should_exclude:
        items = should_exclude.replace("|", ", ")
        exclude_note = (
            "\n\nIMPORTANT — the user explicitly wants these EXCLUDED from the result: "
            + items
            + "\nPenalize exclusion_accuracy if any of these appear."
        )

    lines = [
        "You are grading a GUIDED BACKGROUND REMOVAL result.",
        "The user wanted to remove the background but keep specific foreground content.",
        "",
        "You see two images:",
        "1. ORIGINAL: the full source image before any processing.",
        "2. RESULT: the image after background removal (transparent or checkerboard background).",
        "",
        "The user's guidance was: '{}'".format(foreground),
        "Context: {}".format(description),
        exclude_note,
        "",
        "Grade the RESULT on four criteria:",
        "",
        "1. foreground_correct (true/false): Does the result contain the content the user asked to keep?",
        "",
        "2. inclusion_accuracy (0.0-1.0): How completely are the REQUESTED items present?",
        "   1.0 = everything the user asked to keep is fully visible and intact.",
        "   0.0 = the requested foreground is missing entirely.",
        "",
        "3. exclusion_accuracy (0.0-1.0): How free is the result of UNREQUESTED content?",
        "   1.0 = only what the user asked for is present, background fully removed.",
        "   0.0 = the entire original image is still there, nothing was removed.",
        "   Note: some context (e.g. surface the object sits on) may be acceptable",
        "   if removing it would look unnatural. Use judgment.",
        "",
        "4. edge_quality (0.0-1.0): How clean are the alpha edges?",
        "   1.0 = smooth, natural edges with no artifacts.",
        "   0.0 = severe halos, jagged edges, or missing edge detail.",
        "",
        "Respond with ONLY valid JSON, no markdown fences:",
        '{"foreground_correct": true, "inclusion_accuracy": 1.0,',
        ' "exclusion_accuracy": 1.0, "edge_quality": 1.0,',
        ' "overall": 1.0, "reasoning": "Brief explanation", "issues": []}',
        "",
        "overall formula:",
        "  if foreground_correct is false: overall = max 0.1",
        "  else: inclusion_accuracy * 0.35 + exclusion_accuracy * 0.35 + edge_quality * 0.30",
    ]
    return chr(10).join(lines)
