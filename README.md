# Guided Remove Background

**Guided background removal** — tell `remove_background` what to keep.

Standard background removal (RMBG) guesses what the "foreground" is. Sometimes it's ambiguous (a living room — is the sofa the foreground? the coffee table? the rug?). Sometimes it keeps too much or too little. **Guided Remove Background lets the user guide the process** with natural language prompts — add items RMBG missed, remove items it kept, or narrow to specific objects. A VLM + SAM pipeline interprets user intent and precisely adjusts the RMBG baseline, then an agentic judge verifies the result and self-corrects if needed.

## Architecture

The pipeline has four layers, each with a single responsibility:

1. **RMBG** (Bria RMBG-2.0) — produces a baseline foreground mask with sub-pixel alpha edges. This is the starting point: the "obvious foreground."
2. **VLM** (Vision-Language Model) — reads the user's prompt and classifies their *intent* into a mode (`add`, `remove`, `narrow`, or `add_remove`) plus a list of target descriptions for SAM to find.
3. **SAM** (Segment Anything Model) — a dumb segmentation tool. Given text descriptions, it returns object masks. It has no concept of modes — it doesn't know or care what we do with its masks afterward.
4. **Judge** (VLM-based) — an agentic verification loop that evaluates the final result against the user's prompt. If it finds issues (missing items, unwanted items, mask artifacts), it triggers automated corrections (re-run SAM with better prompts, fill holes) for up to 2 retries. If corrections make things worse, it reverts to the pre-judge result.

The mode logic lives entirely in our pipeline, which combines the RMBG baseline with SAM masks differently depending on the VLM's classification:

```
User prompt ──► VLM classifies intent ──► SAM segments targets ──► Pipeline combines masks ──► Alpha blend ──► Judge verifies
                     │                          │                          │                                         │
                  mode + targets           per-object masks          RMBG ∪/−/∩ SAM                          pass / retry / revert
                                           (mode-agnostic)          based on mode
```

| Step | What Happens |
|------|-------------|
| 1. RMBG baseline | Bria RMBG-2.0 produces the default foreground mask with sub-pixel alpha |
| 2. VLM decomposition | A vision-language model sees the image + RMBG overlay, classifies user intent into a mode + target list |
| 3. SAM segmentation | SAM 3.1 segments each target object by text description (mode-agnostic) |
| 4. Mode-specific mask | Pipeline combines RMBG and SAM masks using the mode's formula |
| 5. Alpha blend | RMBG's precise alpha where available; feathered edges for SAM-only zones |
| 6. Agentic judge | VLM evaluates the final image, triggers corrections if needed, reverts if corrections fail |

### Four Modes

The core principle: **the obvious foreground (RMBG baseline) is preserved by default**. The user's prompt guides modifications to that baseline.

- **ADD** — Keep everything RMBG found, *plus* extra items: `final = RMBG ∪ SAM_targets`. Example: *"with the surfboard on the wall"*
- **REMOVE** — Keep everything RMBG found, *minus* specific items: `final = RMBG − SAM_targets`. Example: *"without the dog"*
- **NARROW** — Keep only the named items from RMBG's foreground: `final = RMBG ∩ SAM_targets`. Reserved for explicit "only/just" statements or when the user names the main subject directly. Example: *"only the chef and the pot"*
- **ADD_REMOVE** — Mixed intent, applies both: `final = (RMBG ∪ add_targets) − remove_targets`. Example: *"add the surfboard, remove the far chairs"*

### Pipeline Deep Dive

Each step in detail, including the post-processing we apply:

**Step 1 — RMBG baseline** (1 API call)
Bria RMBG-2.0 returns an RGBA image with sub-pixel alpha edges. We threshold at `alpha > 128` to get a binary `rmbg_mask`. This mask is the starting point for everything.

**Step 2 — VLM decomposition** (1 vision LLM call + 1 text-only LLM call for NARROW mode)
We send the VLM **two images**: the original photo and a green overlay showing what RMBG considers foreground. This lets the VLM see the baseline before deciding how to modify it. The VLM returns a mode + target list. For NARROW mode, a second text-only LLM call validates each target against the user's prompt and drops any the VLM hallucinated.

**Step 3 — SAM segmentation** (N API calls, parallelized)
Each target description is sent to SAM 3.1 as a separate text prompt, in parallel (up to 4 workers). SAM returns a binary mask + confidence score per target. For ADD mode, if SAM finds nothing, we retry with simplified 2-3 word descriptions.

**Step 4 — Mode-specific mask combination** (local, no APIs)
This is where the mode formulas are applied, with morphological post-processing:

- **ADD**: Dilate SAM masks (3 iterations), union with RMBG, then `binary_fill_holes` to close gaps at the seam between RMBG and SAM regions.
- **REMOVE**: Erode RMBG mask (8+ iterations) to find the "core." Per-target: dilate SAM mask, then check core protection — if a target is < 5% of foreground and > 90% inside the core, it's skipped (it's part of the main subject, not a removable item). Subtract surviving targets from RMBG.
- **NARROW**: Per-target: dilate SAM mask, compute RMBG coverage. If > 15% overlap (target is an RMBG foreground object): dilate the intersection, fill holes, clamp back to RMBG boundary. If < 15% (target is in RMBG's background): use dilated SAM mask directly.
- **ADD_REMOVE**: Apply ADD logic for add targets, REMOVE logic (with core protection) for remove targets, combine, then `binary_fill_holes`.

**Step 5 — Alpha channel construction** (local)
Where the final mask overlaps RMBG, we use RMBG's original sub-pixel alpha (best edge quality). Where the mask extends beyond RMBG (SAM-only zones), we build feathered alpha using distance transforms + Gaussian blur. For ADD/ADD_REMOVE modes, we bridge pixel-thin seam gaps between RMBG and SAM zones.

**Step 6 — Agentic judge** (1-3 vision LLM calls)
A VLM receives the original image, the final result (on checkerboard), and the user prompt. It evaluates whether the result matches user intent, looking for three specific problems:
- **Missing items** — a discrete object the user explicitly asked for is completely absent.
- **Unwanted items** — an object the user asked to remove is still fully present.
- **Mask artifacts** — large holes inside foreground objects.

If issues are found, the judge suggests fixes (`re-sam` with a better prompt, `fill-holes`). The pipeline applies the fix and re-evaluates, up to 2 retries. Safety guards prevent over-correction: a mask floor (20% of original size) stops excessive removal, and `shrink-mask`/`expand-mask` fixes are disabled. If all retries fail, the pipeline **reverts to the pre-judge result** rather than returning a broken correction.

**Cost per image**: 1 RMBG call + 1 vision LLM call + 0-1 validation LLM call + N SAM calls + 1-3 judge LLM calls. Total: 4-9 API calls.

## Examples

```bash
# Narrow: keep only the chef, stove, and pots (drop plates, countertop, etc.)
uv run guided-remove-background \
  --image cooking_scene.jpg \
  --prompts "the chef with the stove and pots" \
  --output result.png

# Remove: keep the person, drop the dog
uv run guided-remove-background \
  --image person_dog.jpg \
  --prompts "the person walking without the dog" \
  --output person_only.png

# Add: keep everything RMBG finds + add the staircase
uv run guided-remove-background \
  --image living_room.jpg \
  --prompts "all the furniture including the staircase" \
  --output full_room.png

# Add+Remove: add the surfboard, remove the far chairs
uv run guided-remove-background \
  --image cafe_interior.jpg \
  --prompts "add the surfboard, remove the far chairs" \
  --output adjusted.png

# Plain RMBG baseline (no guidance)
uv run guided-remove-background \
  --image living_room.jpg \
  --prompts "anything" \
  --output baseline.png \
  --mode rmbg-only
```

## Quickstart

```bash
git clone https://github.com/Bria-AI/guided-remove-background.git
cd guided-remove-background
cp .env.example .env   # add your API keys
make benchmark         # install → fetch images → run 58 cases → open dashboard
```

The dashboard opens at **http://localhost:8899/live.html** — browse every case with full pipeline step visualization, VLM reasoning, SAM scores, judge verdicts, and interactive feedback.

The playground at **http://localhost:8899/playground.html** lets you iterate on prompts interactively — pick an image, type a prompt, and see the full pipeline result with step-by-step visualization in a chat interface.

### Step by step

```bash
make setup             # install dependencies
make images            # download 15 test images from Pexels
make run               # run all 58 benchmark cases
make serve             # start dashboard at localhost:8899
make grade             # auto-grade results with VLM (optional)
make export            # export results as a single shareable HTML file
make help              # show all available commands
```

## Setup

```bash
cd guided-remove-background
uv sync
```

Create a `.env` file (or copy `.env.example`) with your API keys:

```
BRIA_API_KEY=...        # Bria.ai — background removal
FAL_KEY=...             # Fal.ai — SAM 3.1 segmentation
ANTHROPIC_API_KEY=...   # VLM decomposition + judge + grading
```

## Benchmark

58 test cases across 15 images in two scenario types:

1. **Ambiguous foreground** — scenes with no clear single subject (interiors, table settings, workspaces). The user's guidance defines the foreground.
2. **Adjustable foreground** — scenes with a clear default subject (a person, a group), but the user wants to adjust scope (add the yoga mat, remove the dog, keep only the laptop).

Each case has a scenario type (`include`, `exclude`, `narrow`) and a user prompt.

### Live Dashboard

The dashboard shows the full pipeline for each case: original image, RMBG baseline, VLM mode + targets, SAM masks with confidence scores, combined mask, alpha refinement, judge verdict (pass/fail/reverted), and the final result. Click any step card to expand it for a detailed view. Each case has like/dislike buttons with comment support for collecting feedback.

## Project Structure

```
guided-remove-background/
  src/guided_remove_background/     # Core package
    __init__.py                     # Version, MODES
    cli.py                          # CLI entry point
    remove_bg.py                    # Four-mode orchestrator + judge integration
    clients/
      http_utils.py                 # Shared HTTP retry, env helpers
      bria_rmbg.py                  # Bria RMBG-2.0 API client
      fal_sam.py                    # SAM 3.1 via Fal.ai client
      vlm_decompose.py             # VLM prompt decomposition (mode + targets)
      vlm_judge.py                  # Agentic judge — visual verification + self-correction
    processing/
      debug.py                     # Step recorder (saves intermediate visuals)
      output.py                    # Save result PNG + preview JPG
      mask_cleanup.py              # Morphological mask cleanup
      edge_band.py                 # Edge-band refinement
      sanity.py                    # Sanity guards (RMBG/SAM agreement, bloat)

  benchmark/                        # Benchmark suite
    data/
      cases.csv                    # 58 test cases (image, scenario, prompt)
      catalog.py                   # Image URL catalog (15 curated images)
    fetch_images.py                # Download benchmark images
    runner.py                      # Batch runner with step recording
    feedback_server.py             # HTTP server with feedback API
    live.html                      # Live dashboard with pipeline step visualization
    playground.html                # Interactive prompt playground
    export_html.py                 # Export results as shareable single-file HTML
    grader/
      prompt.py                    # VLM grading prompt
      providers.py                 # Anthropic + OpenAI grading
      run_grader.py                # Grading orchestration
```

## API Keys

| Key | Service | Purpose |
|-----|---------|---------|
| `BRIA_API_KEY` | [Bria.ai](https://bria.ai) | Background removal (RMBG-2.0) |
| `FAL_KEY` | [Fal.ai](https://fal.ai) | SAM 3.1 segmentation |
| `ANTHROPIC_API_KEY` | Anthropic | VLM decomposition + judge + grading |
| `OPENAI_API_KEY` | OpenAI | VLM decomposition + grading (alternative) |
