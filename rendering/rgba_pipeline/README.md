# RGBA Pipeline

Reproducible Blender/Cycles scene construction for joint RGBA targets. A sample is defined by `TARGET`, `ENVIRONMENT`, `BACKPLATE`, a camera, and a versioned `scene_recipe.json`.

## Quick start

Run these commands from the RABench repository root. The examples assume `uv`
is installed and Blender is available on this machine.

```bash
cd rendering/rgba_pipeline
uv sync --frozen --extra dev
export RGBA_BLENDER_BIN=/path/to/blender
uv run python -m rgba_pipeline doctor
uv run python -m rgba_pipeline assets fetch
uv run python -m rgba_pipeline assets prepare
uv run python -m rgba_pipeline generate --suite acceptance_v0 --run-name my_acceptance_run
uv run python -m rgba_pipeline batch --manifest runs/my_acceptance_run/manifest.jsonl --resume
uv run python -m rgba_pipeline validate --run runs/my_acceptance_run
uv run python -m rgba_pipeline calibrate --run runs/my_acceptance_run
uv run python -m rgba_pipeline report --run runs/my_acceptance_run
```

If Blender is not on `PATH`, set `RGBA_BLENDER_BIN` to its executable path
before running the commands. For example, on the project server:

```bash
export RGBA_BLENDER_BIN=/data/gaoshaohan/blender/blender
```

The pipeline downloads source assets into `assets/` and writes recipes,
scenes, renders, and reports under `runs/`. These local inputs and generated
outputs are excluded from Git; only `assets/candidates.json`, which records
the selected sources and licenses, is included in this repository.

`assets fetch` is deliberately serialized and stores all bytes, failures, hashes, and source URLs in `assets/download_ledger.jsonl`. It stops at the configured 1 GiB download limit and rejects any request that would make the asset cache exceed 2 GiB or the hard 10 GiB ceiling.

`batch` runs one clean Blender process per recipe. It supports `joint`, `independent`, `intervention`, `core_probe`, `physical_full`, `without_target`, and `geometry_debug` variants. Core samples are checked by eight independently rendered backplates. Physical samples retain actual full-scene / no-target renders and never claim their renderer alpha is universal compositing truth.

This checkout currently defaults to the project server Blender at `/data/gaoshaohan/blender/blender`. Set `RGBA_BLENDER_BIN` to your local executable on other machines. The build scripts require a Cycles GPU and try OptiX, then CUDA; they do not silently use the CPU.

The acceptance suite has four contact/color Core samples, four neutral-veil Core samples, and four glass/liquid Physical samples. It is development data, not a train/test split.

Each generated sample contains its `scene_recipe.json`, a Blender scene file, float32 linear-premultiplied EXR, alpha EXR, straight RGBA PNG, black/white/checker previews, render log, and `qa.json`.  The `diagnostics/` directory contains the direct-geometry debug render plus diagnostic visibility and interaction masks; these masks are not included in the data RGB or alpha.  Open `runs/<name>/report.html` locally to inspect the acceptance cards and budget summary.

`calibrate` performs the export convention checks on the run: empty transparent film, an opaque target, and alpha-over agreement for the single- and double-veil cases. It stores `calibration.json` beside the report.

To reconstruct a single recipe in a new Blender process, use `build --recipe ...`; to render one variant into an arbitrary directory, use `render --recipe ... --variant joint --output ...`.  Add a new task by generating recipes that satisfy the documented JSON contract and reuse the generic `build`, `render`, `batch`, and `validate` commands; do not put template-specific placement logic in the renderer.

## Expansion run: effects QA and final snapshot

The `expansion60_v1` workflow now records a category-specific intervention in each recipe: object color for simple controls, vertical separation for contact shadows, neutralization of the reflecting/bleeding neighbor, removal of camera-invisible environment objects, coverage changes for translucent layers, and depth separation for interleaved objects. TR and OC recipes render at 512 spp; other categories use 256 spp.

After the formal batch and second-seed noise suite, run `python -m rgba_pipeline effect-qa --run runs/exp60_20260930_v1 --max-workers 8`. It reuses a complete intervention render when its recipe directive matches, computes noise-aware regional effects, exports per-object isolated depth EXRs for OC, and writes per-sample `diagnostics/category_effect_qa.json` plus run-level `effect_summary.json`. `python -m rgba_pipeline report --run runs/exp60_20260930_v1` includes those results and quality-paused scenes.

Refresh the reproducibility snapshot at the end with `python scripts/refresh_exp60_snapshot.py runs/exp60_20260930_v1`; it records source-file SHA-256 hashes and updates the run's `experiment.json` snapshot inventory.
