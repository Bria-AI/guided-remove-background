# Guided Remove Background

**Guided background removal** — tell `remove_background` what to keep.

Standard background removal (RMBG) guesses what the "foreground" is. Sometimes it's ambiguous (a living room — is the sofa the foreground? the coffee table? the rug?). Sometimes it keeps too much (you wanted the chef and stove, but it also kept the plates and countertop). **Guided Remove Background lets the user guide the process** with natural language prompts, and a VLM + SAM pipeline ensures only the requested items remain.

## Architecture

```
User prompt ──► VLM classifies intent ──► SAM segments targets ──► Mask logic ──► Alpha blend
                     │                          │                       │
                  mode + targets           per-object masks       RMBG edges where
                  (narrow/add/remove)                             available, feathered
                                                                  edges for SAM-only
```

| Step | What Happens |
|------|-------------|
| 1. RMBG baseline | Bria RMBG-2.0 removes the background with sub-pixel alpha edges |
| 2. VLM decomposition | A vision-language model classifies user intent into a mode + target list |
| 3. SAM segmentation | SAM 3.1 segments each target object in the image |
| 4. Mode-specific mask | Combine RMBG and SAM masks based on the classified mode |
| 5. Alpha blend | Use RMBG's precise alpha where it has data; feathered edges for SAM-only zones |

### Three Modes

The VLM classifies every user prompt into one of three modes based on **specificity**:

- **NARROW** (default) — User names specific items: "the chef with the stove and pots". SAM finds exactly those items; everything else is dropped. Targets = all items the user mentioned.
- **ADD** — User wants everything RMBG keeps plus extras: "all the furniture including the staircase". Targets = only the extras RMBG would miss.
- **REMOVE** — User wants RMBG's result minus specific items: "the person without the dog". Targets = items to subtract.

The key principle: when a user names specific items, they're giving a **complete list**. Anything not mentioned — even if RMBG kept it — should not be in the result.

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

The dashboard opens at **http://localhost:8899/live.html** — browse every case with full pipeline step visualization, VLM reasoning, and interactive feedback.

### Step by step

```bash
make setup             # install dependencies
make images            # download 15 test images from Pexels
make run               # run all 58 benchmark cases (~8 min)
make serve             # start dashboard at localhost:8899
make grade             # auto-grade results with VLM (optional)
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
ANTHROPIC_API_KEY=...   # VLM decomposition + grading
```

## Benchmark

58 test cases across 15 images in two scenario types:

1. **Ambiguous foreground** — scenes with no clear single subject (interiors, table settings, workspaces). The user's guidance defines the foreground.
2. **Adjustable foreground** — scenes with a clear default subject (a person, a group), but the user wants to adjust scope (add the yoga mat, remove the dog, keep only the laptop).

Each case has a scenario type (`include`, `exclude`, `narrow`) and a user prompt.

### Live Dashboard

The dashboard shows for each case: the original image, RMBG baseline, VLM mode + targets, SAM masks, combined mask, alpha refinement, and final result. Each case has like/dislike buttons with comment support for iterative improvement.

## Project Structure

```
guided-remove-background/
  src/guided_remove_background/     # Core package
    __init__.py                     # Version, MODES
    cli.py                          # CLI entry point
    remove_bg.py                    # Three-mode orchestrator (narrow/add/remove)
    clients/
      http_utils.py                 # Shared HTTP retry, env helpers
      bria_rmbg.py                  # Bria RMBG-2.0 API client
      fal_sam.py                    # SAM 3.1 via Fal.ai client
      vlm_decompose.py             # VLM prompt decomposition (mode + targets)
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
    live.html                      # Live dashboard with step visualization
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
| `ANTHROPIC_API_KEY` | Anthropic | VLM decomposition + grading |
| `OPENAI_API_KEY` | OpenAI | VLM decomposition + grading (alternative) |
