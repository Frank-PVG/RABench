from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .paths import BLENDER_BIN, BLENDER_SCRIPT, BPY_PYTHON, BPY_RUNTIME, BPY_LIBRARIES, TEMP_ROOT


def blender_script_command(script: Path, arguments: list[str]) -> list[str]:
    if not BLENDER_BIN.exists() and BPY_RUNTIME.is_dir():
        return [str(BPY_PYTHON), str(script), "--", *arguments]
    return [str(BLENDER_BIN), "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(script), "--", *arguments]


def blender_command(recipe: Path, output: Path, variant: str) -> list[str]:
    return blender_script_command(BLENDER_SCRIPT, ["--recipe", str(recipe), "--output", str(output), "--variant", variant])


def blender_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if not BLENDER_BIN.exists() and BPY_RUNTIME.is_dir():
        environment['PYTHONPATH'] = os.pathsep.join([str(BPY_RUNTIME), str(BLENDER_SCRIPT.parent.parent / '.runtime/bpy')])
        environment['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [str(BPY_LIBRARIES), environment.get('LD_LIBRARY_PATH', '')]))
        environment['PYTHONNOUSERSITE'] = '1'
    for variable, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'), ('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp')]:
        directory = TEMP_ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(directory)
    return environment


def run_variant(recipe_path: Path, output: Path, variant: str, probe: str | None = None,
                gpu_uuid: str | None = None) -> dict[str, Any]:
    if not BLENDER_BIN.exists() and not BPY_RUNTIME.is_dir():
        raise FileNotFoundError(f"Blender not found: {BLENDER_BIN}")
    output.mkdir(parents=True, exist_ok=True)
    command = blender_command(recipe_path, output, variant)
    environment = blender_environment()
    if probe:
        environment["RGBA_PROBE"] = probe
    if gpu_uuid:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        environment["RGBA_CYCLES_DEVICE_UUID"] = gpu_uuid
    start = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    log = {"command": command, "variant": variant, "probe": probe, "exit_code": completed.returncode,
           "elapsed_seconds": round(time.monotonic() - start, 3), "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:]}
    write_json(output / "host_process.json", log)
    if completed.returncode:
        raise RuntimeError(f"Blender failed: {output / 'host_process.json'}")
    return log


def variants(recipe: dict[str, Any]) -> list[tuple[str, str | None]]:
    if recipe.get('isolation'):
        return [('isolated_object', None)]
    if recipe["background_protocol"]["mode"] == "core":
        pairs = [("joint", None), ("independent", None), ("intervention", None), ("geometry_debug", None)]
        for probe in recipe["background_protocol"]["probes"]:
            pairs.extend([("core_probe", probe), ("core_backplate", probe)])
    else:
        pairs = [("joint", None), ("geometry_debug", None)]
        for probe in recipe["background_protocol"]["probes"]:
            pairs.extend([("physical_full", probe), ("without_target", probe)])
    if recipe.get('scene_template'):
        pairs.extend([('scene_full', None), ('scene_without_target', None)])
    return pairs


def variant_output_complete(sample_root: Path, recipe: dict[str, Any], variant: str, probe: str | None) -> bool:
    """Return whether a variant has all of its required primary outputs.

    `independent` is a fan-out variant: its EXRs live below one directory per
    semantic object.  Checking only the fan-out parent was the source of a
    subtle resume bug that caused already completed layers to be rendered
    again.
    """
    modern = recipe.get("schema_version") in {"1.1", "1.2"}
    if variant == "joint":
        folder = sample_root / "joint"
        names = ["linear_premult.exr", "alpha.exr", "straight_rgba.png", "render_log.json"]
        if modern: names += ["alpha_preview.png"]
        return all((folder / name).exists() for name in names)
    if variant == "independent":
        order = recipe.get("diagnostics", {}).get("independent_order", [])
        return bool(order) and all(
            all((sample_root / "variants" / "independent" / "independent" / object_id / name).exists()
                for name in (("linear_premult.exr", "alpha.exr", "straight_rgba.png", "alpha_preview.png")
                             if modern else ("linear_premult.exr", "alpha.exr", "straight_rgba.png")))
            for object_id in order
        )
    root = "probes" if variant in {"core_probe", "core_backplate"} else "physical" if variant in {"physical_full", "without_target"} else "variants"
    folder = sample_root / root / probe / variant if probe else sample_root / root / variant
    names = ["linear_premult.exr", "alpha.exr", "straight_rgba.png", "render_log.json"]
    if modern: names += ["alpha_preview.png"]
    if variant in {'scene_full','scene_without_target'}: names += ['rgb.png']
    return all((folder / name).exists() for name in names)


def render_recipe(recipe_path: Path, sample_root: Path, *, resume: bool = False,
                  gpu_uuid: str | None = None, variant_filter: set[str] | None = None) -> dict[str, Any]:
    recipe = read_json(recipe_path); completed: list[dict[str, Any]] = []
    for variant, probe in variants(recipe):
        if variant_filter is not None and variant not in variant_filter:
            continue
        folder = sample_root / ("probes" if variant in {"core_probe", "core_backplate"} else "physical" if variant in {"physical_full", "without_target"} else "variants")
        if variant == "joint": folder = sample_root / "joint"
        if probe: folder = folder / probe / variant
        else: folder = folder / variant if variant != "joint" else folder
        if resume and variant_output_complete(sample_root, recipe, variant, probe):
            completed.append({"variant": variant, "probe": probe, "status": "resumed"}); continue
        run_variant(recipe_path, folder, variant, probe, gpu_uuid=gpu_uuid)
        completed.append({"variant": variant, "probe": probe, "status": "rendered"})
    return {"sample_id": recipe["identity"]["sample_id"], "variants": completed}
