from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .assets import MANIFEST, PREPARED_ASSETS, fetch_selected, load_candidates, prepared_manifest, raw_assets, write_prepared_manifest
from .io import append_jsonl, read_json, write_json
from .expansion import generate_expansion_suite
from .runner import eligible_gpus, run_expansion_batch
from .paths import ASSET_ROOT, BLENDER_BIN, BPY_PYTHON, BPY_RUNTIME, PROJECT_ROOT, RUNS_ROOT, WORKSPACE_ROOT, ensure_layout
from .recipes import generate_acceptance_suite, validate_recipe
from .render import render_recipe, run_variant, blender_script_command, blender_environment
from .report import make_report
from .noise import render_noise_suite
from .effects import run_effect_suite
from .validate import calibration_checks, validate_sample


def doctor(_: argparse.Namespace) -> int:
    ensure_layout(); data = {"project": str(PROJECT_ROOT), "version": __version__, "python": sys.version, "blender": str(BLENDER_BIN), "blender_exists": BLENDER_BIN.exists(), "disk": shutil.disk_usage(PROJECT_ROOT)._asdict()}
    if BLENDER_BIN.exists():
        probe = subprocess.run([str(BLENDER_BIN), "--version"], text=True, capture_output=True); data["blender_version"] = probe.stdout.splitlines()[:3]
    elif BPY_RUNTIME.is_dir():
        probe = subprocess.run([str(BPY_PYTHON), "-c", "import bpy; print(bpy.app.version_string)"],
                               text=True, capture_output=True, env=blender_environment())
        data.update({"backend": "bpy", "bpy_runtime": str(BPY_RUNTIME),
                     "blender_version": probe.stdout.strip() if probe.returncode == 0 else probe.stderr[-1000:]})
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
    paths = fetch_selected(Path(args.candidate_file) if args.candidate_file else None, count=args.count, max_file_mib=args.max_file_mib); print("\n".join(map(str, paths))); return 0


def asset_prepare(args: argparse.Namespace) -> int:
    if args.preserve_source:
        candidates = load_candidates(Path(args.candidate_file) if args.candidate_file else None)
        records = []
        for candidate in candidates[:args.count]:
            source = ASSET_ROOT / 'raw' / f'{candidate.asset_id}_{candidate.uid}.glb'
            if not source.is_file(): raise FileNotFoundError(source)
            meta = PREPARED_ASSETS / f'{candidate.asset_id}.json'
            meta.parent.mkdir(parents=True, exist_ok=True)
            command = blender_script_command(PROJECT_ROOT/'blender/inspect_asset.py', ['--source', str(source), '--output', str(meta), '--no-fingerprint'])
            result = subprocess.run(command, text=True, capture_output=True, env=blender_environment())
            if result.returncode: raise RuntimeError(f'Asset inspection failed for {candidate.asset_id}: {result.stderr[-2000:]}')
            inspection = read_json(meta)
            record = {**asdict(candidate), 'local_path': str(source), 'prepared_path': str(source),
                      'lineage_id': f'sketchfab:{candidate.uid}', 'inspection_path': str(meta),
                      'source_bounds': inspection['world_bounds'], 'material_policy': 'preserve_source',
                      'material_count': inspection['material_count'], 'image_count': inspection['image_count']}
            records.append(record)
            print('ASSET_READY', candidate.asset_id, candidate.name, flush=True)
        if len(records) != args.count: raise ValueError(f'expected {args.count} assets, found {len(records)}')
        write_prepared_manifest(records); print(MANIFEST); return 0
    raw = list(raw_assets())
    if len(raw) < args.count: raise RuntimeError(f"Need at least {args.count} downloaded GLB files")
    candidates = {candidate.uid: candidate for candidate in load_candidates(Path(args.candidate_file) if args.candidate_file else None)}
    records=[]
    for index, source in enumerate(raw[:args.count]):
        asset_id, uid = source.stem.rsplit("_", 1)
        prepared = PREPARED_ASSETS / f"{asset_id}.glb"; meta = PREPARED_ASSETS / f"{asset_id}.json"
        command=blender_script_command(PROJECT_ROOT/'blender/prepare_asset.py',['--source',str(source),'--output',str(prepared),'--metadata',str(meta)])
        result=subprocess.run(command,text=True,capture_output=True,env=blender_environment())
        if result.returncode: raise RuntimeError(f"Asset preparation failed for {source}: {result.stderr[-2000:]}")
        candidate = candidates[uid]
        item=read_json(meta); item.update({"asset_id":asset_id,"uid":uid,"lineage_id":f"objaverse:{uid}","source_url":candidate.source_page,"download_url":candidate.url,"license":candidate.license})
        records.append(item)
    write_prepared_manifest(records); print(MANIFEST); return 0


def generate(args: argparse.Namespace) -> int:
    if args.suite == 'p1_pairs_v1':
        from .p1 import generate_p1
        print(generate_p1(args.run_name,samples=args.samples,resolution=args.resolution,seed=args.seed)); return 0
    if args.suite == 'template_pbr_v1':
        from .template_recipes import generate_template_suite
        print(generate_template_suite(args.run_name, resolution=args.resolution, samples=args.samples, seed=args.seed)); return 0
    if args.suite == "acceptance_v0":
        print(generate_acceptance_suite(args.run_name)); return 0
    if args.suite == "expansion60_v1":
        root = RUNS_ROOT / args.run_name
        print(generate_expansion_suite(root)); return 0
    raise ValueError(f"unknown suite {args.suite!r}")


def build(args: argparse.Namespace) -> int:
    recipe=Path(args.recipe).resolve(); data=read_json(recipe); errors=validate_recipe(data)
    if errors: raise ValueError(errors)
    output=Path(args.output).resolve() if args.output else recipe.parent if data.get('isolation') else recipe.parent.parent/"samples"/data["identity"]["sample_id"]
    output.mkdir(parents=True,exist_ok=True)
    if recipe != output/"scene_recipe.json": shutil.copy2(recipe,output/"scene_recipe.json")
    if data.get('isolation'):
        run_variant(recipe,output,'isolated_object'); print(output); return 0
    # The joint entrypoint saves scene.blend before rendering the target preview.
    run_variant(recipe,output/"joint","joint"); print(output); return 0


def render(args: argparse.Namespace) -> int:
    recipe=Path(args.recipe).resolve(); output=Path(args.output).resolve() if args.output else recipe.parent if args.variant == 'isolated_object' else recipe.parent.parent/"samples"/read_json(recipe)["identity"]["sample_id"] / "joint"
    run_variant(recipe,output,args.variant,args.probe); return 0


def batch(args: argparse.Namespace) -> int:
    manifest=Path(args.manifest).resolve(); run=manifest.parent
    variant_filter = set(args.variants.split(',')) if args.variants else None
    if variant_filter:
        valid={'joint','independent','intervention','geometry_debug','core_probe','core_backplate','physical_full','without_target','scene_full','scene_without_target'}
        if variant_filter-valid: raise ValueError(f'unknown variants: {sorted(variant_filter-valid)}')
    if manifest.name == "acquisition_manifest.jsonl":
        selected = set(args.pilot_scenes.split(",")) if args.pilot_scenes else None
        result = run_expansion_batch(run, max_workers=args.max_workers, resume=args.resume, scene_ids=selected, variant_filter=variant_filter)
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
            result=render_recipe(recipe,sample,resume=args.resume,variant_filter=variant_filter); append_jsonl(status,{**result,"status":"succeeded"})
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
    experiment = read_json(run/'experiment.json') if (run/'experiment.json').exists() else {}
    if experiment.get('suite') == 'p1_pairs_v1':
        from .p1 import validate_pair
        for line in (run/'render_manifest.jsonl').read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line); sample = run/'samples'/row['id']
            try:
                recipes = {'source':read_json(Path(row['recipe'])), 'target':read_json(Path(row['target_recipe']))}
                if read_json(sample/'recipes.json') != recipes:
                    raise ValueError('rendered sample does not match the current recipe')
                reports.append(validate_pair(sample,recipes['source'],recipes['target']))
            except Exception as exc:
                reports.append({'sample_id':row['id'],'qa_status':'fail','errors':[str(exc)]})
        write_json(run/'validation.json',reports)
        failures = [item for item in reports if item['qa_status'] != 'pass']
        print(json.dumps({'samples':len(reports),'passed':len(reports)-len(failures),'failures':failures},indent=2))
        return 1 if failures else 0
    for sample in sorted((run/"samples").glob("*")): reports.append(validate_sample(sample))
    write_json(run/"validation.json",reports); print(json.dumps(reports,indent=2)); return 0 if all(x["qa_status"]=="pass" for x in reports) else 1


def report(args: argparse.Namespace) -> int: print(make_report(Path(args.run).resolve())); return 0


def isolate(args: argparse.Namespace) -> int:
    from .isolation import render_isolated_run
    source = Path(args.run).resolve()
    output = Path(args.output).resolve() if args.output else source.with_name(source.name + '_isolated')
    result = render_isolated_run(source, output, resolution=args.resolution, samples=args.samples,
                                 seed=args.seed, framing=args.framing, max_workers=args.max_workers,
                                 resume=args.resume, sample_ids=set(args.sample_ids.split(',')) if args.sample_ids else None,
                                 object_ids=set(args.object_ids.split(',')) if args.object_ids else None)
    print(json.dumps(result, indent=2))
    return 1 if result['failures'] else 0


def p1_render(args: argparse.Namespace) -> int:
    from .p1 import render_p1
    result=render_p1(Path(args.run),max_workers=args.max_workers,
                     sample_ids=set(args.sample_ids.split(',')) if args.sample_ids else None,resume=args.resume)
    print(json.dumps(result,indent=2));return 1 if result['failures'] else 0


def luna_annotate(args: argparse.Namespace) -> int:
    from .luna import annotate_pilot
    result=annotate_pilot(Path(args.run),client_module=Path(args.client_module),teacher_config=Path(args.teacher_config),
                          model_id=args.model_id,reasoning_effort=args.reasoning_effort,max_tokens=args.max_tokens,
                          workers=args.workers,limit=args.limit,
                          sample_ids=set(args.sample_ids.split(',')) if args.sample_ids else None,resume=args.resume)
    print(json.dumps(result,indent=2));return 1 if result['errors'] else 0


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
    experiment = read_json(run / "experiment.json") if (run / "experiment.json").exists() else {}
    if experiment.get("suite") == "template_pbr_v1" or expansion_recipe.exists():
        recipe = (next(path for path in sorted((run / "recipes").glob("*.json"))
                       if read_json(path)["background_protocol"]["mode"] == "core")
                  if experiment.get("suite") == "template_pbr_v1" else expansion_recipe)
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
    p=subs.add_parser("assets"); child=p.add_subparsers(dest="asset_command",required=True);f=child.add_parser("fetch");f.add_argument("--candidate-file");f.add_argument('--count',type=int,default=3);f.add_argument('--max-file-mib',type=int,default=64);f.set_defaults(func=asset_fetch);q=child.add_parser("prepare");q.add_argument('--candidate-file');q.add_argument('--count',type=int,default=3);q.add_argument('--preserve-source',action='store_true');q.set_defaults(func=asset_prepare)
    p=subs.add_parser("generate");p.add_argument("--suite",required=True);p.add_argument("--run-name",default="acceptance_v0");p.add_argument('--resolution',type=int,default=1024);p.add_argument('--samples',type=int,default=256);p.add_argument('--seed',type=int,default=42);p.set_defaults(func=generate)
    p=subs.add_parser("build");p.add_argument("--recipe",required=True);p.add_argument("--output");p.set_defaults(func=build)
    p=subs.add_parser("render");p.add_argument("--recipe",required=True);p.add_argument("--variant",required=True);p.add_argument("--output");p.add_argument("--probe");p.set_defaults(func=render)
    p=subs.add_parser('isolate');p.add_argument('--run',required=True);p.add_argument('--output');p.add_argument('--sample-ids');p.add_argument('--object-ids');p.add_argument('--resolution',type=int,default=1024);p.add_argument('--samples',type=int,default=256);p.add_argument('--seed',type=int,default=42);p.add_argument('--framing',choices=['object','scene'],default='object');p.add_argument('--max-workers',type=int,default=2);p.add_argument('--resume',action='store_true');p.set_defaults(func=isolate)
    p=subs.add_parser('p1-render');p.add_argument('--run',required=True);p.add_argument('--sample-ids');p.add_argument('--max-workers',type=int,default=2);p.add_argument('--resume',action='store_true');p.set_defaults(func=p1_render)
    p=subs.add_parser('luna-annotate');p.add_argument('--run',required=True);p.add_argument('--client-module',default=os.environ.get('RGBA_LUNA_CLIENT', str(WORKSPACE_ROOT.parent/'AlphaLift/alphalift/teachers/client.py')));p.add_argument('--teacher-config',default=os.environ.get('RGBA_LUNA_CONFIG', str(WORKSPACE_ROOT.parent/'AlphaLift/configs/data/gpt_config.yaml')));p.add_argument('--model-id',default='gpt-5.6-luna');p.add_argument('--reasoning-effort',default='high');p.add_argument('--max-tokens',type=int,default=4096);p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int,default=3);p.add_argument('--sample-ids');p.add_argument('--resume',action='store_true');p.set_defaults(func=luna_annotate)
    p=subs.add_parser("batch");p.add_argument("--manifest",required=True);p.add_argument("--resume",action="store_true");p.add_argument("--max-workers",type=int,default=8);p.add_argument("--pilot-scenes",help="comma-separated scene IDs for the formal pilot");p.add_argument('--variants',help='render only these comma-separated variants; QA is not marked passed for partial renders');p.set_defaults(func=batch)
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
