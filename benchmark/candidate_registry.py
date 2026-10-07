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
    # Same RMBG->VLM->SAM->merge chain as A1/A2, but the judge/Opus retry loop is
    # replaced with a single deterministic retry right after SAM: re-run a target with
    # a simplified prompt if SAM found nothing for it or scored it below 0.5. Tests
    # whether most of the judge loop's ~20pt pass-rate gain can be had without any
    # Opus call in the request path.
    "f1_det_retry": {
        "label": "F1 · Deterministic retry (no judge)",
        "file": "run_meta_det_retry.json",
        "match": lambda r: True,
        "kind": "deterministic_retry", "out_subdir": "det_retry",
    },
    # Bria's own object-extraction endpoint (single prompt -> single cutout), no VLM
    # decompose, no judge. Two variants of its own "remove_background" toggle: G1 uses
    # the raw SAM-segmentation alpha, G2 refines it with an RMBG pass. Tests whether
    # this off-the-shelf endpoint alone matches the existing chain on ADD-style cases.
    "g1_extract_object_rmbg_off": {
        "label": "G1 · extract-object, remove_background OFF",
        "file": "run_meta_extract_object_rmbg_off.json",
        "match": lambda r: True,
        "kind": "extract_object", "remove_background": False, "out_subdir": "extract_object_rmbg_off",
    },
    "g2_extract_object_rmbg_on": {
        "label": "G2 · extract-object, remove_background ON",
        "file": "run_meta_extract_object_rmbg_on.json",
        "match": lambda r: True,
        "kind": "extract_object", "remove_background": True, "out_subdir": "extract_object_rmbg_on",
    },
    # The real shipped v1.1 pipeline (AG-206): Gemini intent decompose + in-house SAM 3 +
    # RMBG, all in-process on Bria's own GPUs via the object_extraction fal app's
    # /guided-remove-background endpoint -- NOT this repo's own local remove_bg() (that's
    # A1/A2, a different, Claude-based research implementation never shipped to prod).
    "h1_guided_v1_1": {
        "label": "H1 · Guided v1.1 (Gemini, fal)",
        "file": "run_meta_guided_v1_1.json",
        "match": lambda r: True,
        "kind": "guided_v1_1", "out_subdir": "guided_v1_1",
    },
}

# Candidates candidate_runner.py can actually execute (the rest are produced by runner.py).
RUNNABLE_CANDIDATES = {k: v for k, v in ALL_CANDIDATES.items() if "kind" in v}
