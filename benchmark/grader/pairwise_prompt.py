"""Judge prompt for the AG-195 pairwise comparison: one case, two anonymous results.

Versioned with the repo so a run can be reproduced exactly; pairwise.py records
PROMPT_VERSION in every output file. Change the version whenever the wording changes.
"""

from __future__ import annotations

PROMPT_VERSION = "pairwise-v2"

SYSTEM = (
    "You are an exacting reviewer of image-editing results for a commercial background-removal "
    "product. You compare two results for the same request and pick the one a customer would "
    "rather ship. You judge only what is visible, and you are not told which system made which "
    "result."
)

# One wording per scenario, matching cases.csv's `scenario` column. An `include` ask ("the woman
# with the laptop") means the obvious foreground PLUS the named items -- a result that also keeps
# the desk and the chair is not "over-inclusion," it's the plain cut, correctly tweaked. An
# `exclude` ask ("without the lamp") means the obvious foreground MINUS the named items. Only
# `narrow` ("only the laptop") actually means just the named items and nothing else.
_FRAMING = {
    "include": "keeping the obvious foreground subject(s) PLUS: \"{instruction}\"",
    "exclude": "keeping the obvious foreground subject(s) EXCEPT: \"{instruction}\"",
    "narrow": "keeping ONLY: \"{instruction}\"",
}


def pairwise_prompt(instruction: str, should_exclude: str, scenario: str,
                     a_real_alpha: bool, b_real_alpha: bool) -> str:
    exclude_line = ""
    if should_exclude:
        exclude_line = (
            "The request explicitly says these must NOT be in the result: "
            + should_exclude.replace("|", ", ") + ".\n"
        )
    framing = _FRAMING[scenario].format(instruction=instruction)
    alpha_fact = (
        "Technical fact, not inferred from the image: Result A has a real alpha channel: "
        f"{'yes' if a_real_alpha else 'no'}. Result B has a real alpha channel: "
        f"{'yes' if b_real_alpha else 'no'}.\n"
    )
    return (
        f"The customer uploaded the ORIGINAL photo and asked to remove the background, {framing}.\n"
        f"{exclude_line}\n"
        f"{alpha_fact}\n"
        "Below are RESULT A and RESULT B. A grey-and-white checkerboard means that area is "
        "transparent. Any other background, including plain white or a solid colour, means the "
        "background was painted over, not removed. Trust the technical fact above over what the "
        "pixels appear to show -- a result can paint its own fake checkerboard.\n\n"
        "Compare them on these criteria, most important first:\n"
        "1. Selection: is exactly the requested content kept, complete, with everything else gone?\n"
        "2. Fidelity: are the kept people and objects the same as in the ORIGINAL, not redrawn, "
        "restyled, moved, resized or with details changed (faces, logos, text, colours, shapes)?\n"
        "3. Background removal: is the background actually transparent?\n"
        "4. Edges: clean cut-out edges, no halos, no chunks missing, fine detail kept.\n\n"
        "Call it a tie only when neither result is meaningfully better overall.\n\n"
        "Give 2-4 sentences of reasoning comparing A and B on the criteria above, then the "
        "verdicts. selection_winner and fidelity_winner judge only their own criterion; "
        "overall_winner weighs all four."
    )


_WINNER = {"type": "string", "enum": ["A", "B", "tie"]}

# Enforced through structured outputs, so every verdict parses.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "selection_winner": _WINNER,
        "fidelity_winner": _WINNER,
        "overall_winner": _WINNER,
    },
    "required": ["reasoning", "selection_winner", "fidelity_winner", "overall_winner"],
    "additionalProperties": False,
}
