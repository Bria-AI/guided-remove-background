# Benchmark run, v0.3.2 (8 October 2026)

Re-run of the same 59-case / 15-image set against the guided pipeline in
`object-extraction-v0.3.2` (Bria-AI/bria-everywhere#512: no transparent seam
between two requested objects that touch). This is the run of record for the
launch page numbers.

- Date: 2026-10-08
- Cases: `benchmark/data/cases.csv` (59 rows); images: `benchmark/data/catalog.py`
- Guided: `POST /v2/image/edit/remove_background/guided` on `engine.int.bria-api.com`,
  served by the guided fal app pinned to `object-extraction-v0.3.2` (Bria-AI/fal#473),
  59/59 succeeded. Median 17.3 s end to end, p90 33.5 s.
- Competitors: the same outputs as the 7 October run of `object-extraction-v0.3.1`,
  not re-executed (plain remove background, FIBO-Edit-1.5, Nano Banana 2, GPT Image 2,
  extract-object).
- Grader: Claude Opus, pairwise, tie allowed (`benchmark/grader/run_pairwise.py`),
  295 comparisons, 1 error: extract-object has no output for
  `dropper_bottles_rocks/rocks_without_bottles`, so its row covers 58 cases.

## Results (ties count as half a win)

| vs | Preference | Wins / ties / losses |
|---|---|---|
| Plain remove background | 82.2% | 45 / 7 / 7 |
| GPT Image 2 | 92.4% | 54 / 1 / 4 |
| Nano Banana 2 | 93.2% | 55 / 0 / 4 |
| FIBO-Edit-1.5 | 98.3% | 58 / 0 / 1 |
| Extract-object | 70.7% | 30 / 22 / 6 (58 cases) |

Against plain remove background by scenario: narrow 94.0% (23/1/1), include 75.0%
(14/5/3), exclude 70.8% (8/1/3).

## What changed from v0.3.1

- Seam pixels (thin transparent gaps inside the plain cut) over all 59 outputs:
  27,083 → 3,797. The five seam cases (courier man_bench_bag_bike, home_office
  workspace_setup, man_pug_and_mug, sneakers_with_boxes, sneakers_and_top_box) are
  clean in this run and in three repeat runs of each.
- Repeat runs (3 extra per case) of eight cases show the intent model reads some
  ambiguous prompts differently from run to run: `just_the_egg_cup` lost the egg in
  1 run of 4; `chair_and_desk` kept the bed in 2 of 4; `workspace_setup` dropped the
  woman in 3 of 4. These are prompt-reading variance, not the merge.
- Known limits: `plate_and_coffee` drops dishes and cutlery that stick out past the
  plate (narrow keeps only what was named plus what the named objects enclose).

Output images are not committed (see `.gitignore`).
