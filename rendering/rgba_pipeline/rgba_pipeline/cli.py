from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .assets import MANIFEST, PREPARED_ASSETS, fetch_selected, load_candidates, prepared_manifest, raw_assets, write_prepared_manifest
from .io import append_jsonl, read_json, write_json
from .expansion import generate_expansion_suite
from .runner import eligible_gpus, run_expansion_batch
from .paths import ASSET_ROOT, BLENDER_BIN, PROJECT_ROOT, RUNS_ROOT, ensure_layout
from .recipes import generate_acceptance_suite, validate_recipe
from .render import render_recipe, run_variant
from .report import make_report
from .noise import render_noise_suite
from .effects import run_effect_suite
from .validate import calibration_checks, validate_sample


def doctor(_: argparse.Namespace) -> int:
    ensure_layout(); data = {"project": str(PROJECT_ROOT), "version": __version__, "python": sys.version, "blender": str(BLENDER_BIN), "blender_exists": BLENDER_BIN.exists(), "disk": shutil.disk_usage(PROJECT_ROOT)._asdict()}
    if BLENDER_BIN.exists():
        probe = subprocess.run([str(BLENDER_BIN), "--version"], text=True, capture_output=True); data["blender_version"] = probe.stdout.splitlines()[:3]
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        probe = subprocess.run([nvidia, "--query-gpu=index,uuid,name,memory.free,utilization.gpu,driver_version", "--format=csv,noheader"], text=True, capture_output=True)
        data["gpus"] = probe.stdout.splitlines() if probe.returncode == 0 else {"error": probe.stderr.strip()}
    try:
        import OpenEXR, PIL, numpy  # noqa: F401
        data["host_image_io"] = "available"
    except Exception as exc: data["host_image_io"] = f"unavailable: {exc}"
    write_json(PROJECT_ROOT / "environment.json", data); print(json.dumps(data, indent=2)); return 0


def asset_fetch(args: argparse.Namespace) -> int:
    paths = fetch_selected(Path(args.candidate_file) if args.candidate_file else None); print("\n".join(map(str, paths))); return 0


def asset_prepare(_: argparse.Namespace) -> int:
    raw = list(raw_assets())
    if len(raw) < 3: raise RuntimeError("Need at least three downloaded GLB files")
    candidates = {candidate.uid: candidate for candidate in load_candidates(None)}
    records=[]
    for index, source in enumerate(raw[:3]):
        asset_id, uid = source.stem.rsplit("_", 1)
        prepared = PREPARED_ASSETS / f"{asset_id}.glb"; meta = PREPARED_ASSETS / f"{asset_id}.json"
        command=[str(BLENDER_BIN),"--background","--factory-startup","--python-exit-code","1","--python",str(PROJECT_ROOT/"blender"/"prepare_asset.py"),"--","--source",str(source),"--output",str(prepared),"--metadata",str(meta)]
        result=subprocess.run(command,text=True,capture_output=True)
        if result.returncode: raise RuntimeError(f"Asset preparation failed for {source}: {result.stderr[-2000:]}")
        candidate = candidates[uid]
        item=read_json(meta); item.update({"asset_id":asset_id,"uid":uid,"lineage_id":f"objaverse:{uid}","source_url":candidate.source_page,"download_url":candidate.url,"license":candidate.license})
        records.append(item)
    write_prepared_manifest(records); print(MANIFEST); return 0


def generate(args: argparse.Namespace) -> int:
    if args.suite == "acceptance_v0":
        print(generate_acceptance_suite(args.run_name)); return 0
    if args.suite == "expansion60_v1":
        root = RUNS_ROOT / args.run_name
        print(generate_expansion_suite(root)); return 0
    raise ValueError(f"unknown suite {args.suite!r}")


def build(args: argparse.Namespace) -> int:
    recipe=Path(args.recipe).resolve(); data=read_json(recipe); errors=validate_recipe(data)
    if errors: raise ValueError(errors)
    output=Path(args.output).resolve() if args.output else recipe.parent.parent/"samples"/data["identity"]["sample_id"]
    output.mkdir(parents=True,exist_ok=True); shutil.copy2(recipe,output/"scene_recipe.json")
    # Geometry-debug creates the reproducible .blend without requiring all render variants.
    run_variant(recipe,output/"diagnostics"/"geometry_debug","geometry_debug"); print(output); return 0


def render(args: argparse.Namespace) -> int:
    recipe=Path(args.recipe).resolve(); output=Path(args.output).resolve() if args.output else recipe.parent.parent/"samples"/read_json(recipe)["identity"]["sample_id"] / "joint"
    run_variant(recipe,output,args.variant,args.probe); return 0


def batch(args: argparse.Namespace) -> int:
    manifest=Path(args.manifest).resolve(); run=manifest.parent
    if manifest.name == "acquisition_manifest.jsonl":
        selected = set(args.pilot_scenes.split(",")) if args.pilot_scenes else None
        result = run_expansion_batch(run, max_workers=args.max_workers, resume=args.resume, scene_ids=selected)
        print(json.dumps(result, indent=2))
        return 1 if result["failures"] else 0
    status=run/"batch_status.jsonl"; failures=[]
    for row in (json.loads(line) for line in manifest.read_text().splitlines() if line.strip()):
        recipe=Path(row["recipe"]); sample=run/"samples"/row["sample_id"]
        if args.resume and (sample / "scene_recipe.json").exists():
            existing = read_json(sample / "scene_recipe.json")
            if existing.get("identity", {}).get("recipe_hash") != row["recipe_hash"]:
                archive = run / "archive" / f"{row['sample_id']}_{existing.get('identity', {}).get('recipe_hash', 'unknown')[:12]}"
                if archive.exists():
                    raise RuntimeError(f"Refusing to overwrite archived result: {archive}")
                sample.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(sample), str(archive))
                sample.mkdir(parents=True, exist_ok=True)
        sample.mkdir(parents=True, exist_ok=True)
        shutil.copy2(recipe,sample/"scene_recipe.json")
        data = read_json(recipe)
        try:
            result=render_recipe(recipe,sample,resume=args.resume); append_jsonl(status,{**result,"status":"succeeded"})
            # Promote the aggregate sample metadata and the generic geometry
            # diagnostics to the stable locations promised by the data
            # contract.  Variant-specific logs remain beside their renders.
            write_json(sample / "metadata.json", {
                "schema_version": data.get("schema_version"),
                "sample_id": data["identity"]["sample_id"],
                "scene_root_id": data["identity"]["scene_root_id"],
                "template_id": data["identity"]["template_id"],
                "generator_version": data.get("generator_version"),
                "target_ids": data["sets"].get("TARGET", []),
                "environment_ids": data["sets"].get("ENVIRONMENT", []),
                "backplate_ids": data["sets"].get("BACKPLATE", []),
                "asset_lineage_ids": data["identity"].get("asset_lineage_ids", []),
                "representation_class": "physical" if data["background_protocol"]["mode"] == "physical" else "core_candidate",
            })
            write_json(sample / "render_log.json", result)
            debug = sample / "variants" / "geometry_debug"
            diagnostics = sample / "diagnostics"; diagnostics.mkdir(exist_ok=True)
            if (debug / "linear_premult.exr").exists(): shutil.copy2(debug / "linear_premult.exr", diagnostics / "geometry.exr")
            if (debug / "object_id_map.json").exists(): shutil.copy2(debug / "object_id_map.json", diagnostics / "object_id_map.json")
        except Exception as exc:
            failures.append({"sample_id":row["sample_id"],"error":str(exc)}); append_jsonl(status,{"sample_id":row["sample_id"],"status":"failed","error":str(exc)})
    if failures: write_json(run/"batch_failures.json",failures); raise RuntimeError(f"{len(failures)} batch task(s) failed")
    return 0


def validate(args: argparse.Namespace) -> int:
    run=Path(args.run).resolve(); reports=[]
    for sample in sorted((run/"samples").glob("*")): reports.append(validate_sample(sample))
    write_json(run/"validation.json",reports); print(json.dumps(reports,indent=2)); return 0 if all(x["qa_status"]=="pass" for x in reports) else 1


def report(args: argparse.Namespace) -> int: print(make_report(Path(args.run).resolve())); return 0


def noise(args: argparse.Namespace) -> int:
    result = render_noise_suite(Path(args.run).resolve(), max_workers=args.max_workers, resume=True)
    print(json.dumps(result, indent=2))
    return 1 if result["failures"] else 0


def effect_qa(args: argparse.Namespace) -> int:
    result = run_effect_suite(Path(args.run).resolve(), max_workers=args.max_workers, resume=True)
    print(json.dumps(result, indent=2))
    return 1 if (result["intervention_failures"] or result["occlusion_depth_failures"]
                 or result["category_qa_counts"].get("fail", 0)) else 0


def calibrate(args: argparse.Namespace) -> int:
    run = Path(args.run).resolve()
    empty = run / "calibration" / "empty_core"
    expansion_recipe = run / "recipes" / "SC01_L1_C1.json"
    if expansion_recipe.exists():
        recipe = expansion_recipe
        if not (empty / "alpha.exr").exists():
            devices = eligible_gpus(1)
            if not devices: raise RuntimeError("calibration waits for one idle GPU")
            empty.mkdir(parents=True, exist_ok=True)
            shutil.copy2(recipe, empty / "scene_recipe.json")
            run_variant(recipe, empty, "empty_core", gpu_uuid=devices[0]["uuid"])
    else:
        recipe = run / "recipes" / "C02.json"
        if not (empty / "alpha.exr").exists(): run_variant(recipe, empty, "empty_core")
    result = calibration_checks(run)
    write_json(run / "calibration.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


def parser() -> argparse.ArgumentParser:
    root=argparse.ArgumentParser(prog="python -m rgba_pipeline"); subs=root.add_subparsers(dest="command",required=True)
    p=subs.add_parser("doctor");p.set_defaults(func=doctor)
    p=subs.add_parser("assets"); child=p.add_subparsers(dest="asset_command",required=True);f=child.add_parser("fetch");f.add_argument("--candidate-file");f.set_defaults(func=asset_fetch);q=child.add_parser("prepare");q.set_defaults(func=asset_prepare)
    p=subs.add_parser("generate");p.add_argument("--suite",required=True);p.add_argument("--run-name",default="acceptance_v0");p.set_defaults(func=generate)
    p=subs.add_parser("build");p.add_argument("--recipe",required=True);p.add_argument("--output");p.set_defaults(func=build)
    p=subs.add_parser("render");p.add_argument("--recipe",required=True);p.add_argument("--variant",required=True);p.add_argument("--output");p.add_argument("--probe");p.set_defaults(func=render)
    p=subs.add_parser("batch");p.add_argument("--manifest",required=True);p.add_argument("--resume",action="store_true");p.add_argument("--max-workers",type=int,default=8);p.add_argument("--pilot-scenes",help="comma-separated scene IDs for the formal pilot");p.set_defaults(func=batch)
    p=subs.add_parser("validate");p.add_argument("--run",required=True);p.set_defaults(func=validate)
    p=subs.add_parser("calibrate");p.add_argument("--run",required=True);p.set_defaults(func=calibrate)
    p=subs.add_parser("noise");p.add_argument("--run",required=True);p.add_argument("--max-workers",type=int,default=8);p.set_defaults(func=noise)
    p=subs.add_parser("effect-qa");p.add_argument("--run",required=True);p.add_argument("--max-workers",type=int,default=8);p.set_defaults(func=effect_qa)
    p=subs.add_parser("report");p.add_argument("--run",required=True);p.set_defaults(func=report);return root
def main(argv=None):
    try:
        args = parser().parse_args(argv)
        return args.func(args)
    except Exception as exc: print(f"ERROR: {exc}",file=sys.stderr);return 2
