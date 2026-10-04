# AG-194 / AG-195 benchmark run — October 2026

59 cases across 15 images (product, people, interior, multi-object), one output per
case per system, 6 systems × 59 = 354 outputs. Zero hard fails. AG-195's pairwise
grader then ran Bria against each competitor — see `pairwise_results.json` /
`pairwise_summary.json`.

This folder holds metadata only (JSON + README + logs), same as `may-2026/` and
`sep-2026/` — no image files are committed. The actual output images aren't kept
here; `run_meta*.json` records everything textual (scores, instructions, pass/fail,
latency, which output file each case produced), but re-running `runner.py` /
`candidate_runner.py` would call each API again and isn't guaranteed to reproduce
pixel-identical outputs (these models aren't deterministic). What's kept instead is
`review_shareable.html` — a self-contained export with all 413 images already
embedded (not committed, same pattern as the gitignored `benchmark_report.html` /
`candidates_report.html`) — share that file directly with whoever wants to look.

- Started: 2026-10-01T10:28:59Z
- Finished: 2026-10-01 (same session)
- Cases: `benchmark/data/cases.csv` (59 rows, `benchmark_class` column groups by
  product / people / interior / multi_object for AG-195's per-class win rates)
- Images: `benchmark/data/catalog.py` (15 active images; 12 earlier images retired as
  too busy, kept in the catalog for the Sep 2026 snapshot's reproducibility)

## Systems and versions

| System | run_meta file | What it is | Endpoint / model |
|---|---|---|---|
| `bria_a2` | `run_meta_noverify.json` | Bria, verify OFF — what the shipped route runs by default | in-process chain: Bria RMBG 2.0 → Claude decompose → SAM 3.1 → merge |
| `rmbg_only` | `run_meta.json` | No-guidance baseline — plain Bria RMBG, empty instruction | `POST /v2/image/edit/remove_background` (`engine.int.bria-api.com`) |
| `c1_fibo` | `run_meta_edit_fibo.json` | FIBO-Edit-1.5 with an instruction, single call | fal `bria/fibo-edit-1.5/edit` |
| `c2_nanobanana2` | `run_meta_edit_nanobanana2.json` | Nano Banana 2 edit | fal `fal-ai/nano-banana-2/edit` |
| `c3_gptimage2` | `run_meta_edit_gptimage2.json` | GPT Image 2 edit | fal `openai/gpt-image-2/edit` |
| `g2_extract_object_rmbg_on` | `run_meta_extract_object_rmbg_on.json` | Bria extract-object, `remove_background` ON | fal `bria/object-extraction/extract-object` |

No explicit version pins are exposed by any of these endpoints (fal model slugs and
the Bria RMBG route are unversioned aliases) — "version" here means the model slug /
route called, exactly as above, on the date this run started.

Intent model for the decompose step inside `bria_a2`: `claude-sonnet-4-5`. (AG-195's
pairwise grader is separate — see `benchmark/grader/pairwise_prompt.py`.)

## What's in this folder

- `run_meta*.json` — one per system, every case's result (scores, instructions,
  elapsed time, which output file it wrote at the time).
- `pairwise_results.json` / `pairwise_summary.json` — AG-195's per-comparison
  grader verdicts (with reasoning) and the aggregated win-rate table.
- `review_shareable.html` — self-contained, all images embedded. This is the file
  to share with the team (not committed — see note above).
- `logs/` — stdout/stderr per system for this run.

Alpha channel, confirmed directly on the files before they were removed: `bria_a2`,
`rmbg_only` and `g2_extract_object_rmbg_on` produced real RGBA; `c1_fibo`,
`c2_nanobanana2` and `c3_gptimage2` returned flat RGB with no alpha channel at all.
