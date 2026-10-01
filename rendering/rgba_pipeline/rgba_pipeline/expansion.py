from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .experiment import EXPECTED_VARIANTS, SCENES, stable_seed
from .io import canonical_hash, read_json, write_json
from .model import EXPANSION_SCHEMA_VERSION

PROBES = ["black", "white", "red", "blue", "checker_fine", "checker_coarse", "noise", "stripes"]
COLORS = {
    "red": [0.8, 0.025, 0.02, 1], "blue": [0.02, 0.12, 0.8, 1],
    "green": [0.02, 0.55, 0.06, 1], "orange": [1, 0.22, 0.015, 1],
    "magenta": [0.8, 0.02, 0.48, 1], "yellow": [0.9, 0.65, 0.01, 1],
    "white": [0.82, 0.82, 0.82, 1], "warm": [0.55, 0.3, 0.12, 1],
    "neutral": [0.48, 0.48, 0.48, 1],
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _asset(record: dict[str, Any], object_id: str, location: list[float], *,
           rotation: list[float] | None = None, overrides: dict[str, Any] | None = None,
           include_bounds: bool = False) -> dict[str, Any]:
    dataset = record.get("dataset", "Objaverse 1.0")
    lineage = record.get("lineage_id") or f"{dataset.lower().replace(' ', '_')}:{record['uid']}"
    normalization = dict(record.get("normalization") or {})
    inspection_path = record.get("inspection_path") if include_bounds else None
    if inspection_path and Path(inspection_path).is_file() and "normalized_extent_m" not in normalization:
        inspection = read_json(Path(inspection_path))
        extent = inspection.get("world_bounds", {}).get("extent")
        scale = float(normalization.get("scale", 0.0))
        if extent and scale > 0:
            normalization["normalized_extent_m"] = [float(v) * scale for v in extent]
    return {
        "id": object_id, "role": "asset", "asset_id": record.get("asset_id", record["uid"]),
        "asset_path": record["local_path"], "source_uid": record["uid"],
        "source_sha256": record.get("sha256") or record.get("source_sha256"), "lineage_id": lineage,
        "source_dataset": dataset, "source_license": record.get("license"),
        "source_url": record.get("source_url") or record.get("model_url"),
        # Reuse the Blender inspection result so every camera/light variant
        # shares identical source-space normalization.
        "normalization": normalization,
        "component_count": record.get("component_count"),
        "geometry_fingerprint": record.get("geometry_fingerprint"),
        "transform": {"location": location, "rotation": rotation or [0, 0, 0], "scale": [1, 1, 1]},
        "material_policy": "preserve_source", "surface_overrides": overrides or {},
        "material": {"preset": "opaque_principled", "base_color": [0.8, 0.8, 0.8, 1], "roughness": 0.5, "metallic": 0.0},
        "semantic_object": record.get("name", object_id),
    }


def _proc(object_id: str, kind: str, size: list[float], location: list[float], color: list[float], *,
          rotation: list[float] | None = None, roughness: float = 0.55, metallic: float = 0.0,
          bevel: float = 0.004, semantic: str | None = None) -> dict[str, Any]:
    geometry = {"type": kind}
    if kind == "rounded_box": geometry.update({"size": size, "bevel": bevel})
    elif kind == "sphere": geometry.update({"radius": size[0] / 2, "segments": 48, "rings": 24})
    elif kind == "thin_film": geometry.update({"width": size[0], "height": size[2], "thickness": size[1]})
    elif kind == "veil": geometry.update({"width": size[0], "height": size[2], "grid": 96,
                                           "fold_amplitude": 0.012, "coverage": 0.3})
    elif kind == "cup": geometry.update({"outer_radius": size[0] / 2, "height": size[2], "wall": 0.002, "bottom": 0.003})
    elif kind == "straw": geometry.update({"radius": size[0] / 2,
                                            "points": [[-size[0], 0, 0], [0, 0, size[2] * .5], [size[0], 0, size[2]]]})
    return {
        "id": object_id, "role": "procedural", "geometry": geometry,
        "transform": {"location": location, "rotation": rotation or [0, 0, 0], "scale": [1, 1, 1]},
        "material": {"preset": "opaque_principled", "base_color": color,
                     "roughness": roughness, "metallic": metallic},
        "semantic_object": semantic or object_id,
    }


def _find_helper(selected: dict[str, dict[str, Any]], task_id: str, queries: tuple[str, ...], exclude_uid: str) -> dict[str, Any] | None:
    options = []
    for scene_id, record in selected.items():
        if scene_id == task_id or record.get("uid") == exclude_uid or not record.get("local_path"):
            continue
        text = _norm(" ".join([record.get("name", ""), *record.get("tags", [])]))
        for priority, query in enumerate(queries):
            tokens = [x for x in _norm(query).split("_") if len(x) > 2]
            score = sum(token in text for token in tokens)
            if score:
                options.append((-score, priority, int(record.get("glb_size") or 0), scene_id, record))
    return min(options, default=(0, 0, 0, "", None))[-1]


def _camera(camera_id: int, category: str) -> dict[str, Any]:
    if camera_id == 1:
        location = [0.38, -0.62, 0.32]
        if category == "simple_control":
            location = [0.57, -0.93, 0.41]
        return {"location": location, "look_at": [0, 0, .09], "lens_mm": 50,
                "sensor_width_mm": 36, "clip_start": .01, "clip_end": 100}
    location = [-.46, -.58, .40]
    if category == "simple_control":
        location = [-.69, -.87, .52]
    return {"location": location, "look_at": [0, 0, .09], "lens_mm": 50,
            "sensor_width_mm": 36, "clip_start": .01, "clip_end": 100}


def _environment(light_id: int) -> dict[str, Any]:
    key = [-.35, -.45, .75] if light_id == 1 else [.42, -.40, .68]
    return {"world_color": [.035, .035, .035], "lights": [
        {"id": "key", "type": "AREA", "location": key, "energy": 360, "size": .35},
        {"id": "fill", "type": "AREA", "location": [-key[0], .2, .45], "energy": 45, "size": .4},
    ], "environment_objects": []}


def _pause_reason(task_id: str, asset: dict[str, Any], selected: dict[str, dict[str, Any]], task: Any) -> str | None:
    if task_id in {"OC01", "OC02", "OC03", "OC07", "OC08", "OC09"}:
        aux = _find_helper(selected, task_id, task.auxiliary_search, asset["uid"])
        if aux is None:
            return "no second dataset asset matches the required chain, branch, mesh handle, or cable interaction"
    return None


def _objects_for(task: Any, primary: dict[str, Any], selected: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    sid = task.scene_id
    objects: list[dict[str, Any]] = []
    target: list[str] = []
    environment_ids: list[str] = []
    relations: list[dict[str, Any]] = []
    lineage: set[str] = {primary.get("lineage_id") or f"objaverse:{primary['uid']}"}

    def add_primary(object_id: str = "subject", loc: list[float] | None = None, overrides: dict[str, Any] | None = None) -> None:
        objects.append(_asset(primary, object_id, loc or [0, 0, 0], overrides=overrides, include_bounds=sid == "CT12"))
        target.append(object_id)

    def add_aux(query: tuple[str, ...], object_id: str, loc: list[float], *, fallback_kind: str = "rounded_box",
                size: list[float] | None = None, color: list[float] | None = None, as_target: bool = True,
                rotation: list[float] | None = None, overrides: dict[str, Any] | None = None) -> None:
        match = _find_helper(selected, sid, query, primary["uid"])
        if match:
            obj = _asset(match, object_id, loc, rotation=rotation, overrides=overrides, include_bounds=sid == "CT12")
            lineage.add(match.get("lineage_id") or f"objaverse:{match['uid']}")
        else:
            fallback_size = size or [.08, .08, .08]
            center = loc.copy(); center[2] += fallback_size[2] / 2
            obj = _proc(object_id, fallback_kind, fallback_size, center, color or COLORS["neutral"],
                        rotation=rotation, semantic="procedural auxiliary")
        objects.append(obj)
        (target if as_target else environment_ids).append(object_id)

    def support(object_id: str = "support", color: list[float] | None = None, size: list[float] | None = None) -> None:
        objects.append(_proc(object_id, "rounded_box", size or [.29, .24, .02], [0, 0, -.01], color or COLORS["white"], semantic="finite contact support"))
        target.append(object_id)
        relations.append({"type": "contact", "subject": "subject", "support": object_id, "tolerance_m": .002})

    if sid.startswith("SC"):
        add_primary(loc=[-.18, 0, 0])
        query = task.auxiliary_search
        kind = "sphere" if any(x in " ".join(query) for x in ("ball", "sphere")) else "rounded_box"
        size = [.09, .09, .09] if kind == "sphere" else [.08, .07, .06]
        add_aux(query, "companion", [.18, 0, 0], fallback_kind=kind, size=size, color=COLORS["neutral"])
        relations.append({"type": "separated", "objects": ["subject", "companion"], "minimum_gap_m": .02})
    elif sid.startswith("CT"):
        if sid in {"CT02", "CT03", "CT09"}:
            add_primary(loc=[0, 0, .02])
            plate = _proc("shallow_dish", "rounded_box", [.22, .22, .02], [0, 0, .01], COLORS["white"], semantic="shallow contact dish")
            objects.append(plate); target.append("shallow_dish")
            relations.append({"type": "contact", "subject": "subject", "support": "shallow_dish", "tolerance_m": .002})
        else:
            add_primary()
            support(size=[.30, .26, .02])
        if sid == "CT01":
            cloth = _proc("finite_cloth", "veil", [.30, .30, .002], [0, 0, .001], [.55, .55, .55, 1], rotation=[1.5708, 0, 0], semantic="finite static woven cloth")
            cloth["geometry"].update({"grid": 112, "fold_amplitude": .002, "coverage": .38})
            objects.append(cloth); target.append("finite_cloth")
            relations.append({"type": "contact", "subject": "subject", "support": "finite_cloth", "tolerance_m": .002})
        elif sid == "CT12":
            add_aux(task.auxiliary_search, "second_part", [0, 0, .20], size=[.10, .08, .04], color=COLORS["warm"])
            relations.append({"type": "contact", "subject": "second_part", "support": "subject", "tolerance_m": .01})
    elif sid.startswith("RF"):
        metal = int(sid[2:]) >= 7
        palette = {
            "RF01": "red", "RF02": "blue", "RF03": "green", "RF04": "orange",
            "RF05": "magenta", "RF06": "yellow", "RF07": "red", "RF08": "blue",
            "RF09": "green", "RF10": "magenta", "RF11": "orange", "RF12": "yellow",
        }
        neighbor_color = COLORS[palette[sid]]
        primary_overrides = ({"metallic": 1.0, "roughness": .22} if metal else
                             {"base_color_tint": [.92, .92, .92], "metallic": 0.0, "roughness": .48})
        add_primary(loc=[-.09, 0, 0], overrides=primary_overrides)
        q = task.auxiliary_search
        fallback_kind = "sphere" if sid == "RF04" else "rounded_box"
        helper_overrides = {"base_color_tint": neighbor_color[:3]}
        add_aux(q, "neighbor", [.12, 0, 0], fallback_kind=fallback_kind,
                size=[.08, .08, .08], color=neighbor_color, overrides=helper_overrides)
        relations.append({"type": "near", "objects": ["subject", "neighbor"], "expected_effect": "reflection_or_indirect_color"})
    elif sid.startswith("EN"):
        add_primary(overrides={"metallic": 0.95, "roughness": 0.24} if sid in {"EN04", "EN05", "EN06"} else {})
        if sid in {"EN01", "EN02"}:
            # Camera-invisible frame bars intercept key light and cast a window-shaped pattern.
            bars = [([.28, .018, .36], [-.16, -.20, .18]), ([.28, .018, .36], [.16, -.20, .18]),
                    ([.018, .018, .36], [0, -.20, .18])]
            if sid == "EN02": bars.extend([([.30, .018, .018], [0, -.20, .18]), ([.30, .018, .018], [0, -.20, .06])])
            for i, (size, loc) in enumerate(bars):
                oid = f"window_frame_{i+1}"
                objects.append(_proc(oid, "rounded_box", size, loc, [.2, .2, .2, 1], semantic="camera-invisible window frame"))
                environment_ids.append(oid)
        else:
            color = COLORS["warm"] if sid in {"EN03", "EN04"} else COLORS["white"]
            panel = _proc("reflector", "rounded_box", [.36, .012, .30], [0, .18, .15], color, roughness=.3,
                          metallic=1 if sid in {"EN04", "EN05", "EN06"} else 0, semantic="camera-invisible environment panel")
            objects.append(panel); environment_ids.append("reflector")
            if sid == "EN06":
                side = _proc("side_reflector", "rounded_box", [.012, .22, .28], [.18, 0, .14], [.12, .12, .12, 1], roughness=.25, metallic=1, semantic="camera-invisible asymmetric reflector")
                objects.append(side); environment_ids.append("side_reflector")
    elif sid.startswith("TR"):
        add_primary()
        layers = 1 if sid in {"TR01", "TR02", "TR03", "TR11"} else 2 if sid in {"TR04", "TR05", "TR06", "TR07", "TR12", "TR13"} else 3
        film = sid in {"TR11", "TR12", "TR13", "TR14", "TR15"}
        veil_coverages = (.25, .35, .50)
        veil_coverage = veil_coverages[(int(sid[2:]) - 1) % len(veil_coverages)]
        film_effective_coverages = (.25, .40, .55)
        film_effective_coverage = film_effective_coverages[(int(sid[2:]) - 11) % len(film_effective_coverages)]
        # The film is a closed thin box. The per-face mix is calibrated so two interfaces
        # yield the requested effective opacity: 1 - (1 - interface_mix)^2.
        film_interface_coverage = 1 - (1 - film_effective_coverage) ** .5
        for index in range(layers):
            oid = f"cover_{index+1}"
            y = -.11 - index * .018
            z = .105 + index * .012
            if film:
                obj = _proc(oid, "thin_film", [.31, .0002, .28], [index * .012, y, z], [.78, .78, .78, 1], semantic="neutral no-refraction thin film")
                obj["material"] = {"preset": "neutral_thin_coverage", "coverage": film_interface_coverage,
                                   "base_color": [.78, .78, .78, 1]}
                obj["calibration_target_effective_opacity"] = film_effective_coverage
            else:
                obj = _proc(oid, "veil", [.32, .31, .30], [index * .018, y, z], [.78, .78, .78, 1], semantic="neutral woven veil")
                obj["geometry"].update({"grid": 160, "fold_amplitude": .018, "fold_phase": index * 1.7,
                                        "coverage": veil_coverage, "construction": "opaque_woven_microgeometry"})
            objects.append(obj); target.append(oid)
            relations.append({"type": "covers", "cover": oid, "subject": "subject", "requires_hanging_region": True})
    elif sid.startswith("OC"):
        add_primary()
        if sid in {"OC01", "OC02", "OC09"}:
            # Use two independently sourced meshes so interleaving is not inferred
            # from a single mesh's disconnected-component count.
            add_aux(task.auxiliary_search, "interleaving_helper", [.018, -.025, .004],
                    rotation=[.10, 0, 1.5708])
            relations.append({"type": "alternating_depth_occlusion",
                "objects": ["subject", "interleaving_helper"], "minimum_crossings": 2,
                "requires_distinct_dataset_assets": True})
        elif sid in {"OC04", "OC05", "OC06"}:
            grid = _proc("sparse_grid", "veil", [.32, .30, .32], [0, -.025, .14], [.34, .34, .34, 1], semantic="reused sparse woven grid")
            grid["geometry"].update({"grid": 72, "fold_amplitude": .004, "coverage": .13})
            objects.append(grid); target.append("sparse_grid")
            relations.append({"type": "alternating_depth_occlusion", "objects": ["subject", "sparse_grid"], "minimum_crossings": 2})
        else:
            add_aux(task.auxiliary_search, "interleaving_helper", [.025, -.025, 0], size=[.12, .06, .12], color=COLORS["warm"])
            relations.append({"type": "alternating_depth_occlusion", "objects": ["subject", "interleaving_helper"], "minimum_crossings": 2})
    else:
        raise ValueError(f"unhandled scene {sid}")

    return objects, target, environment_ids, relations, {"lineage": sorted(lineage)}


def _intervention_config(task: Any, objects: list[dict[str, Any]], target: list[str],
                         environment_ids: list[str], relations: list[dict[str, Any]]) -> dict[str, Any]:
    category = task.category
    if category == "simple_control":
        return {"kind": "color", "object": "companion", "value": [0.78, 0.04, 0.72, 1.0],
                "roi_object_ids": ["companion"]}
    if category == "contact_shadow":
        contacts = [r for r in relations if r.get("type") == "contact"]
        selected = next((r for r in contacts if r.get("support") == "subject"), contacts[0])
        supports = sorted({r["support"] for r in contacts if r["support"] != selected["subject"]})
        return {"kind": "translate", "object": selected["subject"], "offset": [0.0, 0.0, 0.02],
                "roi_object_ids": supports}
    if category == "reflection_color_spill":
        return {"kind": "neutral_color", "object": "neighbor", "value": [0.5, 0.5, 0.5, 1.0],
                "roi_object_ids": ["subject"]}
    if category == "environmental_influence":
        return {"kind": "remove_environment", "object": "environment", "roi_object_ids": ["subject"]}
    if category == "translucent_overlap":
        by_id = {obj["id"]: obj for obj in objects}
        cover = by_id["cover_1"]
        if cover.get("geometry", {}).get("type") == "veil":
            value = min(0.85, float(cover["geometry"]["coverage"]) + 0.2)
        else:
            value = min(0.85, float(cover["material"]["coverage"]) + 0.15)
        return {"kind": "coverage", "object": "cover_1", "value": value,
                "roi_object_ids": [obj_id for obj_id in target if obj_id.startswith("cover_")]}
    if category == "interleaved_occlusion":
        return {"kind": "translate", "object": "subject", "offset": [0.0, 0.25, 0.0],
                "roi_object_ids": target.copy()}
    raise ValueError(f"no intervention defined for {category}")


def _recipe(task: Any, primary: dict[str, Any], selected: dict[str, dict[str, Any]], camera_id: int, light_id: int) -> dict[str, Any]:
    objects, target, environment_ids, relations, lineage = _objects_for(task, primary, selected)
    sid = task.scene_id
    sample_id = f"{sid}_L{light_id}_C{camera_id}"
    asset_hashes = sorted(x.get("source_sha256") or x.get("uid", "") for x in objects if x["role"] == "asset")
    shared = {"scene_id": sid, "task": task.category, "objects": objects, "target": target,
              "environment_objects": environment_ids, "relations": relations, "asset_hashes": asset_hashes}
    scene_hash = canonical_hash(shared)
    camera = _camera(camera_id, task.category)
    environment = _environment(light_id)
    recipe = {
        "schema_version": EXPANSION_SCHEMA_VERSION, "generator_version": "expansion60-1.1",
        "identity": {"sample_id": sample_id, "scene_root_id": sid, "template_id": task.category,
                     "split": "expansion60_development", "asset_lineage_ids": lineage["lineage"],
                     "category": task.category, "camera_id": f"C{camera_id}", "light_id": f"L{light_id}",
                     "scene_content_hash": scene_hash},
        "sets": {"TARGET": target, "ENVIRONMENT": environment_ids, "BACKPLATE": []},
        "objects": objects, "relations": relations, "camera": camera, "environment": environment,
        "background_protocol": {"mode": "core", "probes": PROBES,
                                "core_backplate_ray_visibility": {"camera": True, "diffuse": False, "glossy": False,
                                                                     "transmission": False, "volume": False, "shadow": False}},
        "render": {"engine": "CYCLES", "resolution": 1024,
                   "samples": 512 if sid.startswith(("TR", "OC")) else 256, "preview_samples": 32,
                   "max_bounces": 12, "transparent_bounces": 16, "transmission_bounces": 8,
                   "clamp_direct": 3.0, "clamp_indirect": 3.0, "pixel_filter": "BOX", "denoise": False,
                   "adaptive_sampling": False, "transparent_film": True, "color_space": "scene_linear",
                   "png_rgb_encoding": "sRGB", "alpha_encoding": "linear"},
        "seeds": {"master": stable_seed(sid), "geometry": stable_seed(sid, "geometry"),
                  "material": stable_seed(sid, "material"), "camera": camera_id,
                  "lighting": light_id, "render": stable_seed(sid, f"L{light_id}C{camera_id}")},
        "diagnostics": {"variants": ["joint", "independent", "intervention", "geometry_debug"],
                        "independent_order": target.copy(),
                        "intervention": _intervention_config(task, objects, target, environment_ids, relations)},
        "semantics": {"objects": [x.get("semantic_object", x["id"]) for x in objects],
                      "relations": relations, "materials": [x.get("material_policy", "procedural") for x in objects],
                      "lighting": f"controlled light condition L{light_id}",
                      "expected_effects": [task.category], "target_group": target.copy()},
    }
    recipe["identity"]["recipe_hash"] = canonical_hash(recipe)
    return recipe


def generate_expansion_suite(run: Path) -> Path:
    selection_path = run / "selection" / "selected_assets.json"
    selected_doc = json.loads(selection_path.read_text())
    selected = selected_doc.get("selected", selected_doc)
    root = run / "recipes"; root.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "acquisition_manifest.jsonl"
    previous_rows = {x.get("sample_id"): x for x in
                     (json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip())
                     if x.get("sample_id")} if manifest_path.exists() else {}
    rows = []
    scene_plan_path = run / "scene_plan.jsonl"
    scene_rows = {x["scene_id"]: x for x in (json.loads(line) for line in scene_plan_path.read_text().splitlines() if line.strip())}
    for task in SCENES:
        primary = selected.get(task.scene_id)
        if not primary or not primary.get("local_path"):
            scene_rows[task.scene_id].update({"status": "paused_asset_missing", "pause_reason": "no selected, downloaded, validated dataset asset"})
            rows.append({"experiment_id": scene_rows[task.scene_id]["experiment_id"], "scene_id": task.scene_id,
                         "status": "paused_asset_missing", "reason": scene_rows[task.scene_id]["pause_reason"]})
            continue
        reason = _pause_reason(task.scene_id, primary, selected, task)
        if reason:
            scene_rows[task.scene_id].update({"status": "paused_asset_missing", "pause_reason": reason,
                                             "primary_uid": primary["uid"]})
            rows.append({"experiment_id": scene_rows[task.scene_id]["experiment_id"], "scene_id": task.scene_id,
                         "status": "paused_asset_missing", "reason": reason, "primary_uid": primary["uid"]})
            continue
        scene_rows[task.scene_id].update({"status": "asset_selected", "primary_uid": primary["uid"],
                                         "asset_path": primary["local_path"], "asset_sha256": primary.get("sha256")})
        for light_id in (1, 2):
            for camera_id in (1, 2):
                recipe = _recipe(task, primary, selected, camera_id, light_id)
                from .recipes import validate_recipe
                errors = validate_recipe(recipe)
                if errors: raise ValueError(f"{recipe['identity']['sample_id']}: {errors}")
                path = root / f"{recipe['identity']['sample_id']}.json"
                write_json(path, recipe)
                rows.append({"experiment_id": scene_rows[task.scene_id]["experiment_id"],
                             "scene_id": task.scene_id, "sample_id": recipe["identity"]["sample_id"],
                             "camera_id": recipe["identity"]["camera_id"], "light_id": recipe["identity"]["light_id"],
                             "recipe": str(path), "recipe_hash": recipe["identity"]["recipe_hash"],
                             "scene_content_hash": recipe["identity"]["scene_content_hash"],
                             "status": (previous_rows.get(recipe["identity"]["sample_id"], {}).get("status", "pending")
                                        if previous_rows.get(recipe["identity"]["sample_id"], {}).get("recipe_hash") == recipe["identity"]["recipe_hash"]
                                        else "pending")})
    scene_plan_path.write_text("".join(json.dumps(scene_rows[x.scene_id], ensure_ascii=False) + "\n" for x in SCENES))
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return manifest_path
