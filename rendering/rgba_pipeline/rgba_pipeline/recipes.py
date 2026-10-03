from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .assets import prepared_manifest
from .io import append_jsonl, canonical_hash, write_json
from .model import GENERATOR_VERSION, SCHEMA_VERSION
from .paths import RUNS_ROOT, ensure_layout


PROBES = ["black", "white", "red", "blue", "checker_fine", "checker_coarse", "noise", "stripes"]


def _render(kind: str, highres: bool = False) -> dict[str, Any]:
    physical = kind == "physical"
    return {
        "engine": "CYCLES", "resolution": 1024 if highres else 512,
        "samples": 512 if physical else 256, "preview_samples": 32,
        "max_bounces": 16 if physical else 12, "transparent_bounces": 16, "clamp_direct": 3.0, "clamp_indirect": 3.0,
        "transmission_bounces": 12 if physical else 8, "pixel_filter": "BOX",
        "denoise": False, "adaptive_sampling": False, "transparent_film": True,
        "color_space": "scene_linear", "png_rgb_encoding": "sRGB", "alpha_encoding": "linear",
    }


def _camera(index: int) -> dict[str, Any]:
    return ({"location": [0.38, -0.62, 0.32], "look_at": [0.0, 0.0, 0.09], "lens_mm": 50,
             "sensor_width_mm": 36, "clip_start": 0.01, "clip_end": 100.0}
            if index == 1 else
            {"location": [-0.46, -0.58, 0.40], "look_at": [0.0, 0.0, 0.09], "lens_mm": 50,
             "sensor_width_mm": 36, "clip_start": 0.01, "clip_end": 100.0})


def _environment(index: int) -> dict[str, Any]:
    key = ([-0.35, -0.45, 0.75] if index == 1 else [0.42, -0.40, 0.68])
    return {"world_color": [0.035, 0.035, 0.035], "lights": [
        {"id": "key", "type": "AREA", "location": key, "energy": 360.0, "size": 0.35},
        {"id": "fill", "type": "AREA", "location": [-key[0], 0.2, 0.45], "energy": 45.0, "size": 0.4},
    ], "environment_objects": []}


def _base(sample_id: str, template: str, representation: str, camera: int, light: int, highres: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "identity": {"sample_id": sample_id, "scene_root_id": sample_id, "template_id": template,
                     "split": "development_acceptance", "asset_lineage_ids": []},
        "sets": {"TARGET": [], "ENVIRONMENT": [], "BACKPLATE": []}, "objects": [], "relations": [],
        "camera": _camera(camera), "environment": _environment(light),
        "background_protocol": {"mode": representation, "probes": PROBES if representation == "core" else PROBES,
                                "core_backplate_ray_visibility": {"camera": True, "diffuse": False, "glossy": False,
                                                                    "transmission": False, "volume": False, "shadow": False}},
        "render": _render("physical" if representation == "physical" else "core", highres),
        "seeds": {"master": int(sample_id[1:]) + 20260921, "geometry": 11, "material": 23, "camera": camera, "lighting": light, "render": 47},
        "diagnostics": {"variants": ["joint", "independent", "intervention", "geometry_debug"] if representation == "core"
                                      else ["joint", "physical_full", "without_target", "geometry_debug"],
                        "independent_order": [], "intervention": {}},
        "semantics": {"objects": [], "relations": [], "materials": [], "lighting": "fixed environment E", "expected_effects": [], "target_group": []},
    }


def _asset_object(asset: dict[str, Any], object_id: str, color: list[float]) -> dict[str, Any]:
    return {"id": object_id, "role": "asset", "asset_id": asset["asset_id"], "asset_path": asset["prepared_path"],
            "source_uid": asset["uid"], "lineage_id": asset["lineage_id"], "transform": {"location": [0, 0, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
            "material": {"preset": "opaque_principled", "base_color": color, "roughness": 0.42, "metallic": 0.0},
            "semantic_object": asset["asset_id"]}


def contact(sample_id: str, asset: dict[str, Any], camera: int, light: int) -> dict[str, Any]:
    recipe = _base(sample_id, "contact_color", "core", camera, light, sample_id == "C01")
    subject = _asset_object(asset, "subject", [0.78, 0.025, 0.025, 1.0])
    pedestal = {"id": "pedestal", "role": "procedural", "geometry": {"type": "rounded_box", "size": [0.36, 0.30, 0.035], "bevel": 0.008},
                "transform": {"location": [0, 0, -0.0175], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                "material": {"preset": "opaque_principled", "base_color": [0.82, 0.82, 0.82, 1], "roughness": 0.58}, "semantic_object": "white pedestal"}
    recipe["objects"] = [subject, pedestal]; recipe["sets"]["TARGET"] = ["subject", "pedestal"]
    recipe["relations"] = [{"type": "contact", "subject": "subject", "support": "pedestal", "tolerance_m": 0.002}]
    recipe["identity"]["asset_lineage_ids"] = [asset["lineage_id"]]
    recipe["diagnostics"].update({"independent_order": ["pedestal", "subject"], "intervention": {"object": "subject", "parameter": "material.base_color", "value": [0.31, 0.31, 0.31, 1.0]}})
    recipe["semantics"].update({"objects": ["red asset", "white pedestal"], "relations": ["red asset contacts white pedestal"], "materials": ["opaque red", "opaque white"], "expected_effects": ["contact shadow", "red indirect color spill"], "target_group": ["subject", "pedestal"]})
    return recipe


def veil(sample_id: str, asset: dict[str, Any], camera: int, light: int, coverage: float, layers: int) -> dict[str, Any]:
    recipe = _base(sample_id, "neutral_veil", "core", camera, light, sample_id == "V02")
    subject = _asset_object(asset, "subject", [0.025, 0.12, 0.72, 1.0])
    objects = [subject]
    target = ["subject"]
    for index in range(layers):
        veil_id = f"veil_{index + 1}"
        objects.append({"id": veil_id, "role": "procedural", "geometry": {"type": "veil", "width": 0.42, "height": 0.34, "grid": 160, "fold_amplitude": 0.018, "fold_phase": index * 1.7, "coverage": coverage, "construction": "opaque_woven_microgeometry"},
                        # `veil()` is authored in the X/Z plane, with Y as its
                        # small folded depth.  Keep it upright in front of the
                        # subject rather than rotating it flat onto the ground.
                        "transform": {"location": [0.0 + index * 0.025, -0.12 - index * 0.018, 0.14 + index * 0.018], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                        "material": {"preset": "opaque_principled", "base_color": [0.78, 0.78, 0.78, 1], "roughness": 0.66, "alpha_semantics": "subpixel_coverage_from_opaque_woven_microgeometry"}, "semantic_object": "neutral woven veil"})
        target.append(veil_id)
    recipe["objects"] = objects; recipe["sets"]["TARGET"] = target
    recipe["relations"] = [{"type": "covers", "cover": vid, "subject": "subject", "requires_hanging_region": True} for vid in target[1:]]
    recipe["identity"]["asset_lineage_ids"] = [asset["lineage_id"]]
    recipe["diagnostics"].update({"independent_order": ["subject", *reversed(target[1:])], "intervention": {"object": "veil_1", "parameter": "material.coverage", "value": min(0.9, coverage + 0.2)}})
    recipe["semantics"].update({"objects": ["blue asset"] + ["neutral woven veil"] * layers, "relations": ["veil covers object and hangs beyond it"], "materials": ["opaque blue", "neutral woven microgeometry with subpixel alpha"], "expected_effects": ["internal continuous alpha", "overlap attenuation"], "target_group": target})
    return recipe


def glass(sample_id: str, camera: int, light: int, liquid: str, roughness: float) -> dict[str, Any]:
    recipe = _base(sample_id, "glass_liquid", "physical", camera, light, sample_id == "P01")
    cup = {"id": "cup", "role": "procedural", "geometry": {"type": "cup", "outer_radius": 0.04, "height": 0.10, "wall": 0.002, "bottom": 0.003}, "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "material": {"preset": "glass", "ior": 1.45, "roughness": roughness}, "semantic_object": "glass cup"}
    fluid = {"id": "liquid", "role": "procedural", "geometry": {"type": "liquid", "radius": 0.037, "height": 0.061, "bottom_gap": 0.001, "wall_gap": 0.001}, "transform": {"location": [0, 0, 0.003], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "material": {"preset": "liquid", "ior": 1.333, "absorption_color": [0.72, 0.90, 1.0] if liquid == "clear" else [0.45, 0.78, 0.92], "absorption_density": 0.05 if liquid == "clear" else 2.0}, "semantic_object": f"{liquid} liquid"}
    straw = {"id": "straw", "role": "procedural", "geometry": {"type": "straw", "radius": 0.004, "points": [[-0.015, 0, 0.025], [-0.007, 0, 0.12], [0.025, 0, 0.15]]}, "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "material": {"preset": "opaque_principled", "base_color": [0.85, 0.20, 0.04, 1], "roughness": 0.3}, "semantic_object": "bent straw"}
    recipe["objects"] = [cup, fluid, straw]; recipe["sets"]["TARGET"] = ["cup", "liquid", "straw"]
    recipe["relations"] = [{"type": "contains", "container": "cup", "content": "liquid", "wall_gap_m": 0.001}, {"type": "contains", "container": "cup", "content": "straw", "wall_gap_m": 0.001}]
    recipe["diagnostics"].update({"physical_backplates": PROBES, "renderer_alpha_semantics": "renderer_alpha_only"})
    recipe["semantics"].update({"objects": ["glass cup", f"{liquid} liquid", "bent straw"], "relations": ["cup contains liquid and straw"], "materials": ["refractive glass", "absorbing liquid"], "expected_effects": ["refraction", "transmission", "background dependence"], "target_group": ["cup", "liquid", "straw"]})
    return recipe


def validate_recipe(recipe: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    needed = {"schema_version", "generator_version", "identity", "sets", "objects", "relations", "camera", "environment", "background_protocol", "render", "seeds", "diagnostics", "semantics"}
    errors.extend(sorted(needed - recipe.keys()))
    version = recipe.get("schema_version")
    if version not in {"1.0", "1.1", "1.2", "1.3"}: errors.append(f"unsupported schema version {version!r}")
    ids = [obj.get("id") for obj in recipe.get("objects", [])]
    if len(ids) != len(set(ids)): errors.append("duplicate object ids")
    target = recipe.get("sets", {}).get("TARGET", [])
    if not target: errors.append("TARGET is empty")
    if set(target) - set(ids): errors.append("TARGET has invalid references")
    environment = recipe.get("sets", {}).get("ENVIRONMENT", [])
    environment_ids=set(ids) | ({'__scene_environment__'} if recipe.get('scene_template') else set())
    if set(environment) - environment_ids: errors.append("ENVIRONMENT has invalid references")
    if recipe.get("background_protocol", {}).get("mode") not in ({"isolated"} if version == '1.3' else {"core", "physical"}): errors.append("invalid background mode")
    if version == "1.1":
        identity = recipe.get("identity", {})
        for field in ("scene_root_id", "category", "camera_id", "light_id", "scene_content_hash", "recipe_hash"):
            if not identity.get(field): errors.append(f"schema 1.1 identity missing {field}")
        for obj in recipe.get("objects", []):
            if obj.get("role") == "asset" and obj.get("material_policy") not in {"preserve_source", "explicit_override"}:
                errors.append(f"schema 1.1 asset {obj.get('id')} has no explicit material policy")
        if identity.get("recipe_hash"):
            from copy import deepcopy
            from .io import canonical_hash
            normalized = deepcopy(recipe); expected = normalized["identity"].pop("recipe_hash")
            if canonical_hash(normalized) != expected: errors.append("recipe hash mismatch")
    if version == '1.2':
        for field in ('sample_id', 'scene_root_id', 'recipe_id', 'category', 'camera_id', 'light_id'):
            if not recipe.get('identity', {}).get(field): errors.append(f'schema 1.2 identity missing {field}')
        template = recipe.get('scene_template', {})
        if not template.get('id') or not template.get('blend_file'): errors.append('template id and blend_file are required')
        for obj in recipe.get('objects', []):
            if obj.get('role') == 'asset' and obj.get('material_policy') != 'preserve_source':
                errors.append('template asset must preserve source PBR materials')
    if version == '1.3':
        oid = recipe.get('isolation', {}).get('object_id')
        if ids != [oid] or target != [oid]: errors.append('isolation requires exactly one semantic object')
        if environment or recipe.get('sets', {}).get('BACKPLATE'): errors.append('isolated recipe contains environment/backplate objects')
        if recipe.get('scene_template'): errors.append('isolated recipe must not load a scene template')
        if recipe.get('environment', {}).get('lights'): errors.append('isolated recipe must not reuse scene lights')
        if recipe.get('relations'): errors.append('isolated recipe must not contain inter-object relations')
        if recipe.get('isolation', {}).get('protocol') not in {'neutral-white-object-v1','neutral-white-object-v2'}: errors.append('unknown isolation protocol')
    return errors


def generate_acceptance_suite(run_name: str = "acceptance_v0") -> Path:
    ensure_layout()
    assets = prepared_manifest()
    if len(assets) < 3:
        raise RuntimeError("Need three prepared Objaverse assets; run `assets fetch` then `assets prepare`.")
    root = RUNS_ROOT / run_name; recipe_root = root / "recipes"; recipe_root.mkdir(parents=True, exist_ok=True)
    mappings = [
        contact("C01", assets[0], 1, 1), contact("C02", assets[0], 2, 2), contact("C03", assets[2], 1, 2), contact("C04", assets[2], 2, 1),
        veil("V01", assets[0], 1, 1, .35, 1), veil("V02", assets[0], 2, 2, .35, 2), veil("V03", assets[1], 1, 2, .60, 1), veil("V04", assets[1], 2, 1, .35, 2),
        glass("P01", 1, 1, "clear", .0), glass("P02", 2, 2, "clear", .0), glass("P03", 1, 2, "tinted", .0), glass("P04", 2, 1, "clear", .22),
    ]
    manifest = root / "manifest.jsonl"; manifest.unlink(missing_ok=True)
    for recipe in mappings:
        errors = validate_recipe(recipe)
        if errors: raise ValueError(f"{recipe['identity']['sample_id']}: {errors}")
        recipe["identity"]["recipe_hash"] = canonical_hash(recipe)
        path = recipe_root / f"{recipe['identity']['sample_id']}.json"; write_json(path, recipe)
        append_jsonl(manifest, {"sample_id": recipe["identity"]["sample_id"], "recipe": str(path), "recipe_hash": recipe["identity"]["recipe_hash"], "status": "pending"})
    return manifest
