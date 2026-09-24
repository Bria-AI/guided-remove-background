"""Single source of truth for AG-184's candidates: which run_meta file holds each
one's results, and how to pick that candidate's rows out of it (a file can hold more
than one mode/variant — e.g. run_meta.json has both "guided" and "rmbg-only").

Used by:
  - candidate_runner.py  (only the candidates with a "kind" — the ones it can run)
  - grade_uniform.py     (all candidates — grades whatever output each already has)
  - candidates.html reads run_meta files directly in JS; keep its CANDIDATES array
    in sync with the ids/files below if this registry changes.
"""

from __future__ import annotations

from guided_remove_background.clients.fal_edit import (
    FIBO_EDIT_1_5,
    GPT_IMAGE_2_EDIT,
    NANO_BANANA_2_EDIT,
)

# Candidates already produced by the existing runner.py (A1/A2) and the baseline
# ("rmbg_only") — no "kind"/"out_subdir" because candidate_runner.py doesn't run these.
ALL_CANDIDATES: dict[str, dict] = {
    "a1_judge_on": {
        "label": "A1 · Existing chain, judge ON",
        "file": "run_meta.json",
        "match": lambda r: r.get("mode") == "guided" and r.get("verify") is not False,
    },
    "a2_judge_off": {
        "label": "A2 · Existing chain, judge OFF",
        "file": "run_meta_noverify.json",
        "match": lambda r: r.get("mode") == "guided",
    },
    "rmbg_only": {
        "label": "RMBG baseline (no guidance)",
        "file": "run_meta.json",
        "match": lambda r: r.get("mode") == "rmbg-only",
    },
    "b_rmbg_sam31": {
        "label": "B · RMBG + SAM 3.1 (no VLM/judge)",
        "file": "run_meta_rmbg_sam_direct.json",
        "match": lambda r: True,
        "kind": "rmbg_sam_direct", "out_subdir": "rmbg_sam_direct",
    },
    "c1_fibo": {
        "label": "C1 · fibo-edit-1.5 (single call)",
        "file": "run_meta_edit_fibo.json",
        "match": lambda r: True,
        "kind": "edit_model", "model_slug": FIBO_EDIT_1_5, "out_subdir": "edit_fibo",
    },
    "c2_nanobanana2": {
        "label": "C2 · Nano Banana 2 edit",
        "file": "run_meta_edit_nanobanana2.json",
        "match": lambda r: True,
        "kind": "edit_model", "model_slug": NANO_BANANA_2_EDIT, "out_subdir": "edit_nanobanana2",
    },
    "c3_gptimage2": {
        "label": "C3 · GPT Image 2 edit",
        "file": "run_meta_edit_gptimage2.json",
        "match": lambda r: True,
        "kind": "edit_model", "model_slug": GPT_IMAGE_2_EDIT, "out_subdir": "edit_gptimage2",
    },
    # Closes C1's alpha gap by re-running RMBG on its already-generated flat-background
    # output, extracting a real alpha matte instead of re-photographing anything. Net
    # negative vs. C1 alone (81% vs. 90%) — RMBG's saliency re-decision on the flat
    # output can silently drop something Fibo had already correctly kept — but it's
    # the cheapest way to test whether "add a matting pass on top" helps at all, and
    # the answer is documented here as no.
    "e1_fibo_alpha": {
        "label": "E1 · fibo-edit-1.5 + RMBG alpha pass",
        "file": "run_meta_fibo_plus_alpha.json",
        "match": lambda r: True,
        "kind": "rmbg_realpha", "source_candidate": "c1_fibo", "out_subdir": "fibo_plus_alpha",
    },
}

# Candidates candidate_runner.py can actually execute (the rest are produced by runner.py).
RUNNABLE_CANDIDATES = {k: v for k, v in ALL_CANDIDATES.items() if "kind" in v}
