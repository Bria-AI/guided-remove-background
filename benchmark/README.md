# Benchmark scripts

The original 58-case benchmark (`runner.py`, `data/cases.csv`, `live.html`,
`feedback_server.py`, `export_html.py`, `grader/`) is documented in the top-level
[README](../README.md). This file covers the **AG-184 candidate-comparison
scripts**, added to test alternatives to the existing 4-stage chain.

## Candidates

| ID | What it is | How it's run |
|---|---|---|
| `a1_judge_on` | Existing chain, judge ON (the baseline pipeline) | `runner.py --mode guided` |
| `a2_judge_off` | Existing chain, judge OFF | `runner.py --mode guided --no-verify` |
| `b_rmbg_sam31` | RMBG-2.0 ∪ SAM 3.1 on the raw prompt, no VLM, no judge | `candidate_runner.py --candidate b_rmbg_sam31` |
| `c1_fibo` | Single call to `bria/fibo-edit-1.5/edit` | `candidate_runner.py --candidate c1_fibo` |
| `c2_nanobanana2` | Single call to `fal-ai/nano-banana-2/edit` | `candidate_runner.py --candidate c2_nanobanana2` |
| `c3_gptimage2` | Single call to `openai/gpt-image-2/edit` | `candidate_runner.py --candidate c3_gptimage2` |
| `e1_fibo_alpha` | C1's output re-matted with an RMBG-2.0 alpha pass (adds real transparency) | `candidate_runner.py --candidate e1_fibo_alpha` (needs `c1_fibo` run first) |

`candidate_registry.py` is the single source of truth for this table — add a new
candidate there and both `candidate_runner.py` and `candidates.html` pick it up
automatically.

## Files

- **`candidate_registry.py`** — maps each candidate ID to its `run_meta_*.json` file,
  a `match` filter (a file can hold more than one mode), and — for candidates
  `candidate_runner.py` can run — a `kind` plus whatever that kind needs.
- **`candidate_runner.py`** — generic runner; `--candidate <id>` picks the pipeline via
  the registry, so one script covers B, C1-C3, and E1. `--list` shows all runnable
  candidates. `--filter <substr>` runs a subset of cases (by image filename) for quick
  sanity checks before committing to a full 58-case run.
- **`grade_uniform.py`** — grades any candidate's output images with a single-pass
  judge call (no retries), so quality is comparable across candidates that don't all
  have their own internal self-correction loop. Writes `judge_uniform_<id>.json`.
- **`candidates.html`** — the comparison dashboard: a candidate-switcher chip bar,
  pass/fail pills sourced from `judge_uniform_*.json`, click-to-expand per-case detail
  (reuses `live.html`'s pipeline-step visualization where a candidate has that data).
- **`export_candidates_html.py`** — bundles `candidates.html` plus every candidate's
  `run_meta_*.json`/`judge_uniform_*.json` and every referenced image into one
  self-contained HTML file (`candidates_report.html`), so it can be shared or opened
  without the feedback server running.

## Typical flow for a new candidate

```powershell
# 1. Add an entry to candidate_registry.py (or reuse an existing "kind" if it fits)
# 2. Sanity-check on a few cases first
.\.venv\Scripts\python.exe benchmark\candidate_runner.py --candidate <id> --filter cooking_scene -v

# 3. Full 58-case run
.\.venv\Scripts\python.exe benchmark\candidate_runner.py --candidate <id>

# 4. Grade it
.\.venv\Scripts\python.exe benchmark\grade_uniform.py --candidate <id>

# 5. View it (feedback_server.py must be running — see top-level README)
#    http://localhost:8899/candidates.html

# 6. Re-export the shareable report once results are final
.\.venv\Scripts\python.exe benchmark\export_candidates_html.py
```
