# Benchmark run, v1.1 (October 2026)

Re-run of the same 59-case / 15-image set against **v1.1**: the shipped pipeline
(RMBG baseline → Gemini intent decompose → per-target SAM → mode merge →
per-added-object crop+RMBG-vs-SAM refinement) (`candidate: h1_guided_v1_1`,
label "Bria v1.1 (Gemini intent, in-house SAM 3 + RMBG, fal)"). This replaces
the earlier `bria_a2` run in this folder as Bria's entry in the pairwise
comparison; `bria_a2`'s own numbers are kept byte-identical as `*.bria_a2.json`
for reference, not deleted.

- Date: 2026-10-06 (two batches — see "Two batches, not one" below)
- Cases: `benchmark/data/cases.csv` (59 rows)
- Images: `benchmark/data/catalog.py` (15 active images)
- Grader: Claude Opus, pairwise, tie allowed — `benchmark/grader/pairwise_prompt.py`
  (unchanged from the `bria_a2` run)

## What ran, and against what

295 pairwise comparisons (59 cases × 5 competitors), 0 errors, reproduced
exactly from `pairwise_results.json` (`"total": 295, "done": true`, zero
non-null `error` fields).

| System | run_meta file | What it is | Endpoint / model |
|---|---|---|---|
| `h1_guided_v1_1` (Bria) | `run_meta_guided_v1_1.json` | v1.1, as above | fal `object_extraction` app, `/guided-remove-background` |
| `rmbg_only` | `run_meta.json` | No-guidance baseline — plain Bria RMBG, empty instruction | `POST /v2/image/edit/remove_background` (`engine.int.bria-api.com`) |
| `c1_fibo` | `run_meta_edit_fibo.json` | FIBO-Edit-1.5 with an instruction, single call | fal `bria/fibo-edit-1.5/edit` |
| `c2_nanobanana2` | `run_meta_edit_nanobanana2.json` | Nano Banana 2 edit | fal `fal-ai/nano-banana-2/edit` |
| `c3_gptimage2` | `run_meta_edit_gptimage2.json` | GPT Image 2 edit | fal `openai/gpt-image-2/edit` |
| `g2_extract_object_rmbg_on` | `run_meta_extract_object_rmbg_on.json` | Bria extract-object, `remove_background` ON | fal `bria/object-extraction/extract-object` |

Only `h1_guided_v1_1` changed from the prior `bria_a2` run in this folder; the
five competitor systems are the same runs, not re-executed.

## Pipeline identity (what "v1.1" actually was)

- **Gemini model**: `vertex_ai/gemini-3-flash-preview` for reading the ask
  (`ObjectExtractionConfig.vlm_model` at the commits below).
- **Verify step**: off (`guided_judge_enabled=False`, the default).
- **fal app id**: not recorded, and not recoverable. This ran against an
  ephemeral `fal run` dev session (env var `GUIDED_V1_1_FAL_MODEL`), which
  serves at a throwaway app id (e.g. `bria/<uuid>/guided-remove-background`),
  not a stable published path — that id was never logged and isn't persisted
  anywhere. **This run is not byte-for-byte reproducible** for that reason;
  what below is both recorded and genuinely reproducible is which
  bria-everywhere commit was running inside that session.

### Two batches, not one

A narrow-mode bug was found and fixed partway through grading, and only the
affected cases were re-run against the fixed pipeline — the final
`run_meta_guided_v1_1.json` / `pairwise_results.json` mix outputs from two
different bria-everywhere commits:

| Batch | bria-everywhere commit | This repo's commit | Cases | When |
|---|---|---|---|---|
| Initial | `94878a0b` ("AG-206:log for test") | `52b0385` | 47 | 2026-10-06 09:26 |
| Narrow-mode fix re-run | `433d7ccd` ("AG-206:fixes") | `b7a5cbe` ("fix issue in narrow cases") | 12 | 2026-10-06 10:59 |

The bria-everywhere commit per batch is **inferred**, not logged at run time:
the latest commit touching `guided_remove_background.py` / `guided_merge.py`
as of each batch's own commit timestamp in *this* repo. The 12 re-run cases
(all `*__without_*`, i.e. remove-mode) are listed in
`run_meta_guided_v1_1.json`'s `bria_everywhere_pipeline.batches[1].cases`.

## Gate

13 cases: `yoga_studio / with_mat_and_plant`, `man_bench_mountains / man_with_bench`
and the 11 narrow cases plain remove background won against v1. A case passes when
the grader prefers guided over plain remove background (`overall_winner == "bria"`
against `rmbg_only` in `pairwise_results.json`).

**Result: 10 of 13 pass.** The three that lose to plain:

| Case | Ask type | Cause |
|---|---|---|
| courier_bench / man_and_bag | include | Keeps the man and the bag but drops the bench he sits on, so he floats, with bench-coloured fringe along his legs and the bag. A guided defect. |
| breakfast_plate / plate_and_coffee | include | Top-down shot with no obvious foreground; the plate is cut into fragments. Outside guided's home, which is photos with an obvious foreground. |
| breakfast_plate / just_the_egg_cup | narrow | Keeps the cup and drops the egg, with a rough halo on the rim. A one-object ask, which is extract-object's job. |

The other four losses to plain in the full set are `sneakers_on_boxes /
sneakers_with_boxes`, `laptop_pears / laptop_and_pears`, `yoga_studio /
without_candles` and `bedroom_desk / without_shelves`; the grader's reasoning for
each is in `pairwise_results.json`.

## Edge check (AI, 15 cases)

A vision-model pass over the pixels: the 11 narrow cases above plus four add and
remove cases on people and interiors, to look at the seam where an added object
meets the plain cut. Inputs and prompts are in `benchmark/data/cases.csv`.

| Case | Verdict | What's wrong (if anything) |
|---|---|---|
| sneakers_on_boxes / just_the_sneakers | Pass | Clean edges, both shoes intact, laces fully preserved |
| sneakers_on_boxes / sneakers_and_top_box | Pass | Clean, shoes+box contact edge intact |
| sneakers_on_boxes / left_sneaker | **Fail** | Shoelace tip hard-truncated with a straight cut, not a natural taper |
| sneakers_plant / front_sneaker | Pass | Clean, no halo |
| dropper_bottles_rocks / just_the_bottles | Pass | Glass bottles handled cleanly, rocks correctly excluded |
| watch_flatlay / just_the_coffee | **Fail** | Hollow object: the coffee's liquid surface is missing/transparent, only scattered foam flecks remain |
| courier_bench / man_and_bag | **Fail** | Visible bluish halo/fringe along the whole body silhouette, worst on the legs |
| breakfast_plate / plate_and_coffee | **Fail (severe)** | Plate shattered into disconnected fragments with jagged holes punched through it |
| breakfast_plate / just_the_egg_cup | **Fail** | The egg is missing from the cup; rough halo on the rim |
| laptop_pears / just_the_pears | Pass | Clean, stems preserved |
| laptop_pears / lying_pear | Pass (minor) | Faint yellow halo sliver on one edge, small |
| yoga_studio / with_mat_and_plant | Pass | Thin plant leaves held up well, no haloing |
| home_office / without_lamp | Pass | Clean |
| man_bench_mountains / man_with_bench | Pass | Clean, thin cigarette detail preserved |
| bedroom_desk / chair_and_desk | **Fail** | Disconnected stray background fragments left dangling below the chair/desk — leftover background, not part of either kept object |

**Result: 9 of 15 pass.** Besides the three gate losses, the edge defects to fix
in v1.2 are the truncated shoelace on `left_sneaker`, the hollow coffee on
`just_the_coffee` and the stray background fragments on `chair_and_desk`.

## Latency

Per-case wall time, `elapsed_s` in `run_meta_guided_v1_1.json` (n=59):

| p50 | p90 | mean | max |
|---|---|---|---|
| 18.6 s | 23.5 s | 19.7 s | 52.3 s |

This is client-side time in the benchmark runner — upload to fal,
`fal_client.subscribe()` (queue + inference), and the result download — not a
server-side-only pipeline-compute figure, and measured through the ephemeral
dev session above, not a stable deployed app. Expect it to differ (likely
lower, warm/reserved capacity) once this runs behind a real `fal deploy`.

## Images: not committed

Same policy as `may-2026/` and `sep-2026/` and the rest of this folder: this
is metadata only (JSON + README + logs). The 118 output files (59 PNG cutouts
+ 59 preview JPGs, 97.5 MB) this run produced are in
`.gitignore` (`benchmark/results/*/guided_v1_1/`) and are not tracked.
`run_meta_guided_v1_1.json` records everything textual (scores, instructions,
pass/fail, latency, which output file each case wrote); re-running
`candidate_runner.py` would call the API again and isn't guaranteed to
reproduce pixel-identical outputs. `guided_v1_1_review.html` / `review.html`
(self-contained exports with the images embedded) are the files to share with
whoever wants to look; they are not committed.

## What's in this folder

- `run_meta*.json` — one per system, every case's result (scores, instructions,
  elapsed time, which output file it wrote at the time). `run_meta_guided_v1_1.json`
  additionally carries the pipeline-identity and two-batch provenance above.
- `pairwise_results.json` / `pairwise_summary.json` — the grader's per-comparison
  verdicts (with reasoning) and the aggregated win-rate table, for `h1_guided_v1_1`.
- `pairwise_results.bria_a2.json` / `pairwise_summary.bria_a2.json` — the prior
  `bria_a2` run's own grading, kept byte-identical for reference.
- `guided_v1_1_review.html` / `review.html` — self-contained, all images
  embedded. Share these with the team (not committed — see note above).
