# RGBA Pipeline

Reproducible Blender/Cycles scene construction for joint RGBA targets. A sample is defined by `TARGET`, `ENVIRONMENT`, `BACKPLATE`, a camera, and a versioned `scene_recipe.json`.

## P1: fixed paired dataset and Luna pilot

`p1_pairs_v1` produces **24 RGB/RGBA pairs**, using 12 semantic units, three
editable scenes, two camera presets, and explicit scene/material variants.
Each pair shares the camera, object geometry, canvas position, scale, and
1024×768 dimensions. The context RGB retains authored scene illumination;
the target RGBA is rendered alone under a constant white world (strength 1),
with no directional lamps. It uses 512 samples and seed 42 by default.

The earlier black world plus three lamps above the object left lower surfaces
underlit and produced dark environment reflections. The new full-direction white
illumination removes that directional bias. Source PBR maps, intrinsically dark
materials, cavities, and self shading remain; the output is not a flat albedo map.
The glass, liquid, and straw are one semantic unit with three registered
components. A source GLB containing a cup and saucer also stays together.

After `mlx worker login`, run this in the interactive GPU shell with the required
`HF_TOKEN` already exported:

```bash
bash --noprofile --norc -i rendering/rgba_pipeline/scripts/run_p1.sh
```

The script activates `anime`, generates recipes, renders on two GPUs, runs the
three-case Luna pilot, and creates the report. Adjustable options appear directly
after the Python commands. Existing unchanged image and annotation results are
reused. Rendering again preserves Luna results whose image file metadata still
matches; changed recipes are archived before replacement.
The standalone `python -m rgba_pipeline validate --run runs/p1_small_batch`
command repeats the pair checks without a GPU or any Luna call.

The preset registry and implementation are separated by responsibility:

| Presets in `configs/p1/` | Implementation in `blender/` | Responsibility |
| --- | --- | --- |
| `scenes.json` | `scene_templates.py` | Native scene, removals, anchor, support height |
| `materials.json` | `pbr_materials.py` | Source PBR, surface replacements, optical materials |
| `objects.json` | `object_builder.py`, `geometry_support.py` | Asset or assembly, component transforms, bounds |
| `cameras.json` | `camera_support.py` | View presets and shared camera construction |
| `lighting.json` | `lighting_support.py` | Authored context fill multiplier; isolated neutral world |
| `export.json` | `render_export.py` | Resolution, sampling, view transform, float EXR and PNG conventions |
| `suite.json` | `rgba_pipeline/presets.py`, `p1.py` | Fixed 24 cases and the three pilot IDs |

The export convention fields describe the fixed EXR/PNG contract; they do not
select alternative encodings. Context illumination keeps the authored world and
lights; its fill multiplier is adjustable. Target illumination is separately
controlled by `uniform_white`.

Luna uses the same `VLMClient` and YAML configuration interface as AlphaLift's
`scripts/data_pipeline/filter/curate_prismlayers_vlm.py`. The defaults point to
the AlphaLift checkout beside the workspace; `RGBA_LUNA_CLIENT` / `RGBA_LUNA_CONFIG`
or `--client-module` / `--teacher-config` can override those paths. Credentials stay in the external configuration.

The pilot selects **only** `P09_orange_a`, `P16_moka_b`, and `P24_drink_b`.
For each, Luna receives the scene RGB, target checker preview, black/white
close-up, and alpha image. It first evaluates **image quality only**: geometry,
alpha, isolation, neutral lighting, pair correspondence, and material integrity.
A failed or uncertain check discards the sample. Kept samples receive three
direct English fields: `short_prompt`, `detailed_prompt`, `extraction_prompt`.
There is no rewrite stage or annotation-quality scoring. Annotation validation
only checks the JSON fields and nonempty string types. The two model prompts are
editable in `configs/p1/prompts/`.

| Output under `runs/p1_small_batch/` | Content |
| --- | --- |
| `dataset.jsonl` | All 24 pairs; the other 21 explicitly have `luna_status: not_run` |
| `annotated.jsonl` | Only the kept, annotated pilot samples |
| `samples/<id>/condition/rgb.png` | Final scene RGB |
| `samples/<id>/target/straight_rgba.png` | Same-camera neutral RGBA |
| `samples/<id>/target/linear_premult.exr`, `alpha.exr` | Float32 linear target and alpha |
| `samples/<id>/qa.json` | Camera registration, framing, assembly and buffer checks |
| `luna/<id>/image_quality.json`, `annotation.json` | Model decisions/evidence and generated English fields |
| `luna/<id>/input_views/`, `result.json` | Actual submitted images and reusable result |
| `report.html`, `pairs_overview.jpg`, `pairs_preview.jpg` | All pairs and a six-pair preview |
| `preset_snapshot/`, `source_snapshot/` | Presets and source modules captured during generation |

P1 checks its extraction/relighting contract; it does not claim to pass the
legacy joint-target Core/Physical compositing probes. Glass alpha remains
`renderer_alpha_only`, specific to this rendering and illumination.

## Single-object RGBA

The single-object workflow renders every selected semantic object separately in
an empty scene under the same uniform neutral white illumination. It never loads
the source scene `.blend`, its geometry, HDRI, or lights. Other target objects are
not instantiated. Multiple meshes in one source GLB stay together as one object.
Its source PBR materials and self shading are retained.

```bash
python -m rgba_pipeline isolate --run runs/template_pbr_demo --output runs/single_object_demo --resolution 1024 --samples 512 --seed 42 --framing object --max-workers 2 --resume
```

Or run `bash -i scripts/run_isolated_objects.sh` in the interactive GPU worker.
Use `--sample-ids T02_coffee` and `--object-ids subject` to select one object.
`--framing object` centers each object and fits it to the image while preserving
the original viewing direction; `--framing scene` keeps the original camera and
aspect ratio. The default world emits constant white light from every direction
at strength 1, with no area lamps. Saved version-1 recipes retain their earlier
three-lamp/black-world settings for reproducibility. Each output contains exactly
one semantic unit, including all of its declared components.

Outputs are under `runs/single_object_demo/samples/<sample_id>__<object_id>/`:
`straight_rgba.png`, `linear_premult.exr`, `alpha.exr`, display previews,
`scene_recipe.json`, `scene.blend`, and `isolation.json` recording the actual
geometry, camera and lights. `qa.json` checks scene isolation and buffer integrity.
`report.html` links to individual files; `single_objects.jpg` is only a gallery
of those separate renders.

Glass transmits to transparent film and keeps its material's refraction and
reflection. This alpha is specific to the isolated render and its lighting; it
does not reproduce refraction of arbitrary replacement backgrounds. Original
texture content is preserved, including any appearance baked into source maps.
Schema 1.3 records this single-object protocol separately from legacy Core and
Physical evaluation outputs.

## PBR scene templates

`template_pbr_v1` uses a fixed selection of **16 TexVerse PBR objects** and three
editable Poly Haven scenes: Blue Wall, Pawn Shop, and The Shed. The five example
compositions cover a camera and clock, a metal coffee pot and cup, a boot and
instant camera, a glass of water with a straw, and a paired table-material change.
Source geometry, UVs, material partitions, and embedded 2K/4K texture maps are kept.
Source IDs, authors, licenses, and download URLs are in
`assets/texverse_pbr_16.json` and `assets/scene_sources.json`.

On this workspace, enter an interactive GPU shell with `mlx worker login`, activate
`anime`, and export the required `HF_TOKEN`. Then run:

```bash
bash -i rendering/rgba_pipeline/scripts/run_template_pbr.sh
```

The script contains editable parameters directly after each Python command.
It uses the workspace's Python 3.11 + bpy 4.2 runtime, with image I/O dependencies
in `.runtime/bpy` and host OpenEXR in `.runtime/host`. For a fresh environment,
install the project's Python dependencies into the orchestration environment and
`OpenEXR`, `Pillow`, and `opencolorio` into the Blender Python environment.
`RGBA_BPY_PYTHON`, `RGBA_BPY_RUNTIME`, and `RGBA_BPY_LIBRARIES` override runtime
locations; an existing `RGBA_BLENDER_BIN` takes precedence.

To render only the three main outputs before running all QA probes:

```bash
python -m rgba_pipeline batch --manifest runs/template_pbr_demo/acquisition_manifest.jsonl --max-workers 5 --resume --variants joint,scene_full,scene_without_target
```

Partial rendering is recorded as `rendered_subset`, with QA `not_run`. Running
`batch --resume` without `--variants` completes the original probes and QA.
The existing compositing, physical C/B, interaction, noise, and effect-strength
formulas and thresholds are unchanged. The template calibration uses the same
empty-film and opaque-target gates; veil/film-specific gates are not applicable
to this small suite.

Edit `configs/template_pbr_v1.json` to change the template, target asset bindings,
real-world sizes, rotations, support heights, camera, fill light, or material
preset. Scene template IDs are separate from evaluation categories. A source
object is replaced by removing the named scene object and binding an asset to a
slot. Assets are uniformly scaled and aligned to the support height. Before a
joint render, conservative camera bounds, support contact, and intersections
between target meshes are checked. Declared containment is allowed. These checks
do not replace visual inspection for target/environment intersections or occlusion.
Source
materials are preserved unless an explicit override is supplied. Material presets
copy selected material instances; PBR replacement maps use sRGB for base color
and Non-Color for roughness, metallic and tangent-space normals.

Each sample includes:

| Path | Content |
| --- | --- |
| `joint/straight_rgba.png` | Display-space straight RGBA of the target group |
| `joint/linear_premult.exr`, `joint/alpha.exr` | Float32 scene-linear premultiplied target and linear alpha |
| `variants/scene_full/rgb.png` | Complete scene, including target and environment |
| `variants/scene_without_target/rgb.png` | The same scene with the target group removed |
| `scene.blend`, `scene_recipe.json` | Editable scene and reconstruction recipe |
| `joint/placement.json` | Final world-space target bounds |
| `joint/placement_checks.json` | Camera framing, support contact, and target collision checks |
| `qa.json`, `diagnostics/` | Existing quantitative checks and diagnostics |

PNG previews use AgX; numerical QA reads untransformed EXR. No denoising is applied
to the quantitative renders. Environment geometry remains visible to reflection,
transmission, and shadow rays when its camera visibility is disabled for RGBA.
The glass uses a closed, thick-walled mesh, explicit IOR and absorption, and
Cycles transmission. Its alpha is labelled `renderer_alpha_only`; the scene RGB
contains the actual background refraction. The report displays RGB, RGBA, and
the background together at `runs/template_pbr_demo/report.html`.

Schema 1.2 uses explicit recipe IDs and direct recipe comparisons for resume.
This template workflow does not compute asset or recipe hashes. Legacy schema
1.0/1.1 workflows keep their original identity contract.

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
before running the commands. The existing workspace can also use its standalone
bpy runtime. See the [environment guide](../../docs/environment.md) for the
Blender, Conda, optional Luna and temporary-directory settings.

The pipeline downloads source assets into `assets/` and writes recipes,
scenes, renders, and reports under `runs/`. These local inputs and generated
outputs are excluded from Git; the small asset and scene source manifests are
included in this repository. Download staging under the workspace's `codex_tmp/`
is disposable: official scene files and textures live in `assets/scenes/`,
and successful default scene downloads remove their temporary ZIP files.

`assets fetch` is deliberately serialized and stores byte counts, failures, and source URLs in `assets/download_ledger.jsonl`. Its GLB download limit is 1 GiB and GLB cache limit is 2 GiB. Native scene archives and scene textures are fetched separately; they are outside that GLB budget.

`batch` runs one clean Blender process per recipe. It supports `joint`, `independent`, `intervention`, `core_probe`, `physical_full`, `without_target`, and `geometry_debug` variants. Core samples are checked by eight independently rendered backplates. Physical samples retain actual full-scene / no-target renders and never claim their renderer alpha is universal compositing truth.

The pipeline uses `RGBA_BLENDER_BIN` or Blender on `PATH`, then falls back to the configured standalone bpy runtime. The build scripts require a Cycles GPU and try OptiX, then CUDA; they do not silently use the CPU.

The acceptance suite has four contact/color Core samples, four neutral-veil Core samples, and four glass/liquid Physical samples. It is development data, not a train/test split.

Each generated sample contains its `scene_recipe.json`, a Blender scene file, float32 linear-premultiplied EXR, alpha EXR, straight RGBA PNG, black/white/checker previews, render log, and `qa.json`.  The `diagnostics/` directory contains the direct-geometry debug render plus diagnostic visibility and interaction masks; these masks are not included in the data RGB or alpha.  Open `runs/<name>/report.html` locally to inspect the acceptance cards and budget summary.

`calibrate` performs the export convention checks on the run: empty transparent film, an opaque target, and alpha-over agreement for the single- and double-veil cases. It stores `calibration.json` beside the report.

To reconstruct a single recipe in a new Blender process, use `build --recipe ...`; to render one variant into an arbitrary directory, use `render --recipe ... --variant joint --output ...`.  Add a new task by generating recipes that satisfy the documented JSON contract and reuse the generic `build`, `render`, `batch`, and `validate` commands; do not put template-specific placement logic in the renderer.

## Expansion run: effects QA and final snapshot

The `expansion60_v1` workflow now records a category-specific intervention in each recipe: object color for simple controls, vertical separation for contact shadows, neutralization of the reflecting/bleeding neighbor, removal of camera-invisible environment objects, coverage changes for translucent layers, and depth separation for interleaved objects. TR and OC recipes render at 512 spp; other categories use 256 spp.

After the formal batch and second-seed noise suite, run `python -m rgba_pipeline effect-qa --run runs/exp60_20260930_v1 --max-workers 8`. It reuses a complete intervention render when its recipe directive matches, computes noise-aware regional effects, exports per-object isolated depth EXRs for OC, and writes per-sample `diagnostics/category_effect_qa.json` plus run-level `effect_summary.json`. `python -m rgba_pipeline report --run runs/exp60_20260930_v1` includes those results and quality-paused scenes.

The existing expansion run and its recorded source snapshot remain local historical
results. Current template and P1 workflows capture their presets and source files
without a checksum verification step.
