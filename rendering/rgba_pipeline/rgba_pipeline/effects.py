from __future__ import annotations

import copy
import json
import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

import OpenEXR
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .io import canonical_hash, read_json, write_json
from .render import run_variant
from .runner import MAX_WORKERS, eligible_gpus
from .validate import read_exr


_STATUS_LOCK = threading.Lock()


def _alpha(path: Path) -> np.ndarray:
    image = read_exr(path)
    return image[..., 0] if image.ndim == 3 else image


def _noise_sigma(sample: Path) -> tuple[float, float]:
    p = sample / "noise_estimate" / "noise_statistics.json"
    if not p.exists():
        return 0.0, 0.0
    metrics = read_json(p).get("statistics", {}).get("target_support_union", {})
    return float(metrics.get("premult_rgb_sigma_rms") or 0.0), float(metrics.get("alpha_sigma_rms") or 0.0)


def _intervention_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    derived = copy.deepcopy(recipe)
    ident = derived["identity"]
    parent_hash = ident["recipe_hash"]
    category = ident["category"]
    objects = {obj["id"]: obj for obj in derived["objects"]}
    if category == "simple_control":
        cfg = {"kind": "color", "object": "companion", "value": [0.78, 0.04, 0.72, 1.0],
               "roi_object_ids": ["companion"]}
    elif category == "contact_shadow":
        contacts = [r for r in derived.get("relations", []) if r.get("type") == "contact"]
        selected = next((r for r in contacts if r.get("support") == "subject"), contacts[0] if contacts else None)
        if not selected:
            raise ValueError("contact intervention has no declared support object")
        supports = sorted({r["support"] for r in contacts if r["support"] != selected["subject"]})
        cfg = {"kind": "translate", "object": selected["subject"], "offset": [0.0, 0.0, 0.02],
               "roi_object_ids": supports}
    elif category == "reflection_color_spill":
        cfg = {"kind": "neutral_color", "object": "neighbor", "value": [0.5, 0.5, 0.5, 1.0],
               "roi_object_ids": ["subject"]}
    elif category == "environmental_influence":
        if not derived.get("sets", {}).get("ENVIRONMENT"):
            raise ValueError("environment intervention has no environment objects")
        cfg = {"kind": "remove_environment", "object": "environment", "roi_object_ids": ["subject"]}
    elif category == "translucent_overlap":
        cover = objects["cover_1"]
        if cover.get("geometry", {}).get("type") == "veil":
            value = min(0.85, float(cover["geometry"]["coverage"]) + 0.2)
        elif cover.get("material", {}).get("preset") == "neutral_thin_coverage":
            value = min(0.85, float(cover["material"]["coverage"]) + 0.15)
        else:
            raise ValueError("translucency intervention cannot identify coverage implementation")
        cfg = {"kind": "coverage", "object": "cover_1", "value": value,
               "roi_object_ids": [obj for obj in objects if obj.startswith("cover_")]}
    elif category == "interleaved_occlusion":
        cfg = {"kind": "translate", "object": "subject", "offset": [0.0, 0.25, 0.0],
               "roi_object_ids": list(derived["sets"]["TARGET"])}
    else:
        raise ValueError(f"no categorical intervention for {category}")
    derived.setdefault("diagnostics", {})["intervention"] = cfg
    ident.pop("recipe_hash", None)
    ident["parent_recipe_hash"] = parent_hash
    ident["recipe_hash"] = canonical_hash(derived)
    return derived


def _roi_mask(sample: Path, object_ids: list[str]) -> np.ndarray:
    mask = None
    for object_id in object_ids:
        path = sample / "variants" / "independent" / "independent" / object_id / "alpha.exr"
        if not path.exists():
            raise FileNotFoundError(f"missing independent ROI layer: {object_id}")
        current = _alpha(path) > 0.01
        mask = current if mask is None else mask | current
    if mask is None:
        raise ValueError("empty intervention ROI")
    return mask


def _intervention_metrics(sample: Path, recipe: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    primary = read_exr(sample / "joint" / "linear_premult.exr")[..., :3].astype(np.float64)
    altered = read_exr(sample / "variants" / "intervention" / "linear_premult.exr")[..., :3].astype(np.float64)
    primary_a = _alpha(sample / "joint" / "alpha.exr").astype(np.float64)
    altered_a = _alpha(sample / "variants" / "intervention" / "alpha.exr").astype(np.float64)
    roi = _roi_mask(sample, derived["diagnostics"]["intervention"]["roi_object_ids"])
    sigma_rgb, _ = _noise_sigma(sample)
    threshold = max(0.003, 3.0 * sigma_rgb)
    delta = np.mean(np.abs(altered - primary), axis=2)
    effect = roi & (delta > threshold)
    count = int(roi.sum())
    effect_fraction = float(effect.sum() / count) if count else 0.0
    alpha_delta = np.abs(altered_a - primary_a)
    metrics = {
        "sample_id": recipe["identity"]["sample_id"],
        "category": recipe["identity"]["category"],
        "parent_recipe_hash": recipe["identity"]["recipe_hash"],
        "intervention_recipe_hash": derived["identity"]["recipe_hash"],
        "intervention": derived["diagnostics"]["intervention"],
        "roi_pixels": count,
        "effect_threshold_scene_linear": threshold,
        "noise_sigma_rgb": sigma_rgb,
        "effect_pixel_count": int(effect.sum()),
        "effect_fraction_in_roi": effect_fraction,
        "rgb_delta_mae_in_roi": float(delta[roi].mean()) if count else 0.0,
        "rgb_delta_p95_in_roi": float(np.percentile(delta[roi], 95)) if count else 0.0,
        "alpha_delta_mae_full_frame": float(alpha_delta.mean()),
        "alpha_delta_max": float(alpha_delta.max()),
    }
    passed = effect_fraction >= 0.01
    if recipe["identity"]["category"] == "environmental_influence":
        metrics["alpha_invariant"] = metrics["alpha_delta_max"] <= 0.001
        passed = passed and metrics["alpha_invariant"]
    metrics["status"] = "pass" if passed else "fail"
    metrics["errors"] = [] if passed else ["intervention effect is below 1% of its semantic ROI after the noise threshold"]
    if recipe["identity"]["category"] == "environmental_influence" and not metrics.get("alpha_invariant", False):
        metrics["errors"].append("removing the camera-invisible environment changed alpha by more than 0.001")
    return metrics


def _intervention_job(row: dict[str, Any], run: Path, gpu: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    sample = run / "samples" / row["sample_id"]
    recipe = read_json(sample / "scene_recipe.json")
    if recipe["identity"]["recipe_hash"] != row["recipe_hash"]:
        raise ValueError(f"sample recipe hash mismatch: {row['sample_id']}")
    derived = _intervention_recipe(recipe)
    diag = sample / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    recipe_path = diag / "intervention_recipe.json"
    output = sample / "variants" / "intervention"
    metrics_path = diag / "intervention_metrics.json"
    if resume and metrics_path.exists() and (output / "linear_premult.exr").exists():
        prior = read_json(metrics_path)
        if prior.get("parent_recipe_hash") == row["recipe_hash"] and prior.get("intervention_recipe_hash") == derived["identity"]["recipe_hash"]:
            return {"sample_id": row["sample_id"], "status": prior["status"], "resumed": True}
    write_json(recipe_path, derived)
    log_path = output / "render_log.json"
    host_log_path = output / "host_process.json"
    required = ("linear_premult.exr", "alpha.exr", "straight_rgba.png", "alpha_preview.png")
    formal_directive = recipe.get("diagnostics", {}).get("intervention", {})
    same_directive = formal_directive == derived.get("diagnostics", {}).get("intervention", {})
    reusable_formal = (same_directive and log_path.exists() and host_log_path.exists()
                       and all((output / name).is_file() for name in required)
                       and read_json(log_path).get("variant") == "intervention"
                       and read_json(log_path).get("sample_id") == row["sample_id"])
    if not reusable_formal:
        run_variant(recipe_path, output, "intervention", gpu_uuid=gpu["uuid"])
    metrics = _intervention_metrics(sample, recipe, derived)
    metrics["render_reused_from_formal_bundle"] = reusable_formal
    metrics["render_source_recipe_hash"] = row["recipe_hash"]
    metrics["gpu"] = gpu
    write_json(metrics_path, metrics)
    return {"sample_id": row["sample_id"], "status": metrics["status"], "gpu": gpu,
            "effect_fraction_in_roi": metrics["effect_fraction_in_roi"]}


def _occlusion_job(row: dict[str, Any], run: Path, gpu: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    sample = run / "samples" / row["sample_id"]
    recipe_path = Path(row["recipe"])
    recipe = read_json(recipe_path)
    if recipe["identity"]["recipe_hash"] != row["recipe_hash"]:
        raise ValueError(f"sample recipe hash mismatch: {row['sample_id']}")
    output = sample / "diagnostics" / "occlusion_depth"
    manifest = output / "depth_manifest.json"
    expected = [f"{object_id}.exr" for object_id in recipe["sets"]["TARGET"]]
    if resume and manifest.exists() and all((output / name).exists() for name in expected):
        return {"sample_id": row["sample_id"], "status": "resumed"}
    run_variant(recipe_path, output, "occlusion_depth", gpu_uuid=gpu["uuid"])
    return {"sample_id": row["sample_id"], "status": "succeeded", "gpu": gpu}


def _dispatch(jobs: list[dict[str, Any]], worker: Callable[..., dict[str, Any]], run: Path,
              max_workers: int, *, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending = list(jobs)
    active: dict[Any, tuple[dict[str, Any], dict[str, Any]]] = {}
    done_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    no_gpu_since: float | None = None
    limit = max(1, min(MAX_WORKERS, int(max_workers)))
    with ThreadPoolExecutor(max_workers=limit) as pool:
        while pending or active:
            available = eligible_gpus(limit)
            assigned = {gpu["uuid"] for _, gpu in active.values()}
            available = [gpu for gpu in available if gpu["uuid"] not in assigned]
            while pending and available and len(active) < limit:
                row, gpu = pending.pop(0), available.pop(0)
                active[pool.submit(worker, row, run, gpu, resume=resume)] = (row, gpu)
            if not active:
                if not pending:
                    break
                if no_gpu_since is None:
                    no_gpu_since = time.monotonic()
                if time.monotonic() - no_gpu_since >= 600:
                    blocked = [{"sample_id": row["sample_id"], "status": "waiting_gpu_idle",
                                "reason": "no GPU met the idle and memory gate for 10 minutes"} for row in pending]
                    done_rows.extend(blocked); failures.extend(blocked)
                    break
                time.sleep(2.0)
                continue
            no_gpu_since = None
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                row, gpu = active.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = {"sample_id": row["sample_id"], "status": "failed", "gpu": gpu, "error": str(exc)}
                    failures.append(status)
                done_rows.append(status)
                with _STATUS_LOCK, (run / "effect_status.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(status, ensure_ascii=False) + "\n")
    return done_rows, failures


def _scalar_channel(path: Path, channel_hint: str) -> np.ndarray:
    with OpenEXR.File(str(path)) as image:
        channels = image.channels()
        matches = [name for name in channels if channel_hint.lower() in name.lower()]
        if not matches:
            raise KeyError(f"no {channel_hint} channel in {path}: {list(channels)}")
        arr = np.asarray(channels[matches[0]].pixels, dtype=np.float32)
        return np.squeeze(arr)


def _depth_metrics(sample: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    out = sample / "diagnostics" / "occlusion_depth"
    ids = list(recipe["sets"]["TARGET"])
    tol = 0.001
    depth = {oid: _scalar_channel(out / f"{oid}.exr", "Depth") for oid in ids}
    alpha = {oid: _alpha(sample / "variants" / "independent" / "independent" / oid / "alpha.exr") for oid in ids}
    pairs = []
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            overlap = (alpha[a_id] > 0.5) & (alpha[b_id] > 0.5)
            da, db = depth[a_id], depth[b_id]
            valid = overlap & np.isfinite(da) & np.isfinite(db) & (da < 1e8) & (db < 1e8)
            delta = da[valid] - db[valid]
            a_front = int(np.count_nonzero(delta < -tol))
            b_front = int(np.count_nonzero(delta > tol))
            pairs.append({"object_a": a_id, "object_b": b_id, "projected_overlap_pixels": int(valid.sum()),
                          "a_front_pixels": a_front, "b_front_pixels": b_front,
                          "depth_tolerance_m": tol,
                          "reciprocal_order_pass": a_front >= 64 and b_front >= 64,
                          "fixed_global_layer_order_impossible": a_front >= 64 and b_front >= 64})
    passed = bool(pairs) and any(x["reciprocal_order_pass"] for x in pairs)
    return {"sample_id": recipe["identity"]["sample_id"], "category": "interleaved_occlusion",
            "recipe_hash": recipe["identity"]["recipe_hash"], "status": "pass" if passed else "fail",
            "pairs": pairs, "errors": [] if passed else ["no object pair has 64 valid overlap pixels in both depth orders"]}


def _rotation_matrix(euler: list[float]) -> np.ndarray:
    x, y, z = euler
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _project_cover_polygon(obj: dict[str, Any], camera: dict[str, Any], width: int, height: int) -> list[tuple[float, float]]:
    geometry = obj["geometry"]
    if geometry["type"] == "veil":
        half_x, half_z = geometry["width"] / 2, geometry["height"] / 2
    else:
        half_x, half_z = geometry["width"] / 2, geometry["height"] / 2
    transform = obj["transform"]
    rotation = _rotation_matrix(transform.get("rotation", [0, 0, 0]))
    scale = np.asarray(transform.get("scale", [1, 1, 1]), dtype=np.float64)
    origin = np.asarray(transform["location"], dtype=np.float64)
    cam = np.asarray(camera["location"], dtype=np.float64)
    target = np.asarray(camera["look_at"], dtype=np.float64)
    forward = target - cam; forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    sensor_w = float(camera["sensor_width_mm"]) / float(camera["lens_mm"])
    sensor_h = sensor_w * height / width
    points = []
    for x, z in ((-half_x, -half_z), (half_x, -half_z), (half_x, half_z), (-half_x, half_z)):
        world = origin + rotation @ (scale * np.asarray([x, 0.0, z]))
        rel = world - cam
        depth = float(np.dot(rel, forward))
        px = (0.5 + float(np.dot(rel, right)) / (depth * sensor_w)) * width
        py = (0.5 - float(np.dot(rel, up)) / (depth * sensor_h)) * height
        points.append((px, py))
    return points


def _polygon_mask(polygons: list[list[tuple[float, float]]], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return np.asarray(image) > 0


def _erode(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return mask
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(image.filter(ImageFilter.MinFilter(2 * radius + 1))) > 0


def _composite_alpha(layers: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(layers[0], dtype=np.float64)
    for layer in layers:
        result = layer + (1.0 - layer) * result
    return result


def _translucency_metrics(sample: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    root = recipe["identity"]["scene_root_id"]
    ids = [obj_id for obj_id in recipe["sets"]["TARGET"] if obj_id.startswith("cover_")]
    subject_id = "subject"
    joint_alpha = _alpha(sample / "joint" / "alpha.exr").astype(np.float64)
    h, w = joint_alpha.shape
    objects = {obj["id"]: obj for obj in recipe["objects"]}
    polygons = [_project_cover_polygon(objects[oid], recipe["camera"], w, h) for oid in ids]
    geom = _polygon_mask(polygons, (w, h))
    interior = _erode(geom, 2)
    subject = _alpha(sample / "variants" / "independent" / "independent" / subject_id / "alpha.exr") > .01
    hanging = interior & ~subject
    fractional = (joint_alpha >= .05) & (joint_alpha <= .95)
    cover_alpha = [_alpha(sample / "variants" / "independent" / "independent" / oid / "alpha.exr").astype(np.float64) for oid in ids]
    expected_alpha = _composite_alpha(cover_alpha + [
        _alpha(sample / "variants" / "independent" / "independent" / subject_id / "alpha.exr").astype(np.float64)])
    residual = np.abs(joint_alpha - expected_alpha)
    overlap_geom = np.ones((h, w), dtype=bool)
    for polygon in polygons:
        overlap_geom &= _polygon_mask([polygon], (w, h))
    alpha_noise = _noise_sigma(sample)[1]
    if len(cover_alpha) > 1:
        stack = np.zeros_like(cover_alpha[0])
        max_layer = np.zeros_like(stack)
        for layer in cover_alpha:
            stack = layer + (1.0 - layer) * stack
            max_layer = np.maximum(max_layer, layer)
        attenuation = overlap_geom & (stack > max_layer + max(.01, 3 * alpha_noise))
        attenuation_fraction = float(attenuation.sum() / max(int(overlap_geom.sum()), 1))
    else:
        attenuation_fraction = None
    target_effective = [objects[oid].get("calibration_target_effective_opacity") for oid in ids]
    opacity_measurements = []
    if any(value is not None for value in target_effective):
        for oid, layer, polygon, target_opacity in zip(ids, cover_alpha, polygons, target_effective):
            region = _erode(_polygon_mask([polygon], (w, h)), 2)
            values = layer[region]
            observed = float(np.median(values)) if values.size else 0.0
            opacity_measurements.append({"object_id": oid, "target_effective_opacity": target_opacity,
                                         "observed_median_alpha": observed,
                                         "absolute_error": abs(observed - float(target_opacity))})
    fractional_fraction = float((fractional & hanging).sum() / max(int(hanging.sum()), 1))
    passed = (float(interior.sum() / joint_alpha.size) >= .01 and fractional_fraction >= .20 and bool((geom & subject).any()))
    if len(ids) > 1:
        passed = passed and attenuation_fraction is not None and attenuation_fraction > .0
    if opacity_measurements:
        passed = passed and all(x["absolute_error"] <= .10 for x in opacity_measurements)
    return {"sample_id": recipe["identity"]["sample_id"], "category": "translucent_overlap",
            "recipe_hash": recipe["identity"]["recipe_hash"], "status": "pass" if passed else "fail",
            "cover_layer_count": len(ids), "hanging_envelope_pixels": int(hanging.sum()),
            "hanging_envelope_fraction_full_frame": float(hanging.sum() / joint_alpha.size),
            "fractional_alpha_fraction_in_hanging_envelope": fractional_fraction,
            "cover_over_subject_pixels": int((geom & subject).sum()),
            "layer_sequence_alpha_mae": float(residual.mean()), "layer_sequence_alpha_p95": float(np.percentile(residual, 95)),
            "multi_layer_attenuation_fraction": attenuation_fraction,
            "effective_film_opacity": opacity_measurements,
            "errors": [] if passed else ["coverage, hanging-region fractional alpha, layer attenuation, or effective film opacity failed"]}


def _contact_gap(recipe: dict[str, Any], relation: dict[str, Any]) -> float:
    objects = {obj["id"]: obj for obj in recipe["objects"]}
    subject = objects[relation["subject"]]
    support = objects[relation["support"]]
    subject_bottom = float(subject["transform"]["location"][2])
    subject_bottom += float(subject.get("normalization", {}).get("lowest_z_m", 0.0)) * float(subject["transform"].get("scale", [1, 1, 1])[2])
    sg = support.get("geometry", {})
    st = support["transform"]
    if support.get("role") == "asset":
        extent = support.get("normalization", {}).get("normalized_extent_m", [.2, .2, .2])
        support_top = float(st["location"][2]) + float(extent[2]) * float(st.get("scale", [1, 1, 1])[2])
    elif sg.get("type") == "veil":
        support_top = float(st["location"][2])
    else:
        support_top = float(st["location"][2]) + float(sg.get("size", [0, 0, 0])[2]) * float(st.get("scale", [1, 1, 1])[2]) / 2
    return subject_bottom - support_top


def _scene_interaction(sample: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    order = recipe["diagnostics"]["independent_order"]
    layers = []
    composed = None
    composed_alpha = None
    for oid in order:
        root = sample / "variants" / "independent" / "independent" / oid
        p = read_exr(root / "linear_premult.exr")[..., :3].astype(np.float64)
        a = _alpha(root / "alpha.exr").astype(np.float64)
        if composed is None:
            composed, composed_alpha = p, a
        else:
            composed = p + (1 - a[..., None]) * composed
            composed_alpha = a + (1 - a) * composed_alpha
        layers.append((p, a))
    joint = read_exr(sample / "joint" / "linear_premult.exr")[..., :3].astype(np.float64)
    joint_alpha = _alpha(sample / "joint" / "alpha.exr").astype(np.float64)
    mask = (joint_alpha > .01) | (composed_alpha > .01)
    sigma, _ = _noise_sigma(sample)
    threshold = max(.003, 3 * sigma)
    delta = np.mean(np.abs(joint - composed), axis=2)
    fraction = float(((delta > threshold) & mask).sum() / max(int(mask.sum()), 1))
    mae = float(delta[mask].mean()) if mask.any() else 0.0
    return {"control_interaction_fraction": fraction, "joint_independent_mae_visible": mae,
            "noise_threshold_scene_linear": threshold, "visible_pixels": int(mask.sum())}


def _control_separation(sample: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    ids = recipe["sets"]["TARGET"]
    masks = [_alpha(sample / "variants" / "independent" / "independent" / oid / "alpha.exr") > .5 for oid in ids]
    overlap = masks[0] & masks[1]
    minimum = min(int(masks[0].sum()), int(masks[1].sum()))
    return {"projected_overlap_pixels": int(overlap.sum()),
            "overlap_fraction_smaller_object": float(overlap.sum() / max(minimum, 1))}


def _category_metrics(sample: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    category = recipe["identity"]["category"]
    metrics_path = sample / "diagnostics" / "intervention_metrics.json"
    intervention = read_json(metrics_path) if metrics_path.exists() else {"status": "missing", "errors": ["missing intervention metrics"]}
    checks: dict[str, Any] = {"intervention": intervention}
    passed = intervention.get("status") == "pass"
    errors = list(intervention.get("errors", []))
    if category == "simple_control":
        separation = _control_separation(sample, recipe)
        interaction = _scene_interaction(sample, recipe)
        checks.update({"separation": separation, "independence": interaction})
        if separation["overlap_fraction_smaller_object"] >= .01:
            errors.append("simple-control objects overlap in projection")
        if interaction["control_interaction_fraction"] > .05:
            errors.append("simple-control interaction residual exceeds 5% of visible target pixels")
        passed = passed and separation["overlap_fraction_smaller_object"] < .01 and interaction["control_interaction_fraction"] <= .05
    elif category == "contact_shadow":
        relations = [r for r in recipe.get("relations", []) if r.get("type") == "contact"]
        gaps = [{"subject": r["subject"], "support": r["support"], "gap_m": _contact_gap(recipe, r),
                 "tolerance_m": r.get("tolerance_m", .002)} for r in relations]
        checks["contact_relations"] = gaps
        if not gaps or any(abs(x["gap_m"]) > x["tolerance_m"] for x in gaps):
            errors.append("contact relation gap exceeds its declared tolerance")
            passed = False
    elif category == "environmental_influence":
        if not recipe["sets"].get("ENVIRONMENT"):
            errors.append("environment objects are missing")
            passed = False
    elif category == "translucent_overlap":
        transparency = _translucency_metrics(sample, recipe)
        checks["translucency"] = transparency
        passed = passed and transparency["status"] == "pass"
        errors.extend(transparency["errors"])
    elif category == "interleaved_occlusion":
        depth_path = sample / "diagnostics" / "occlusion_depth" / "depth_manifest.json"
        if depth_path.exists():
            depth = _depth_metrics(sample, recipe)
        else:
            depth = {"status": "missing", "errors": ["missing isolated depth renders"]}
        checks["occlusion"] = depth
        passed = passed and depth.get("status") == "pass"
        errors.extend(depth.get("errors", []))
    checks["status"] = "pass" if passed else "fail"
    checks["errors"] = list(dict.fromkeys(errors))
    checks["sample_id"] = recipe["identity"]["sample_id"]
    checks["recipe_hash"] = recipe["identity"]["recipe_hash"]
    write_json(sample / "diagnostics" / "category_effect_qa.json", checks)
    return checks


def _collect_samples(run: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run / "acquisition_manifest.jsonl").read_text().splitlines() if line.strip()]


def run_effect_suite(run: Path, *, max_workers: int = MAX_WORKERS, resume: bool = True) -> dict[str, Any]:
    run = Path(run).resolve()
    rows = _collect_samples(run)
    intervention_done, intervention_failures = _dispatch(rows, _intervention_job, run, max_workers, resume=resume)
    occlusion_rows = [row for row in rows if read_json(Path(row["recipe"]))["identity"]["category"] == "interleaved_occlusion"]
    occlusion_done, occlusion_failures = _dispatch(occlusion_rows, _occlusion_job, run, max_workers, resume=resume)
    samples = []
    for row in rows:
        sample = run / "samples" / row["sample_id"]
        recipe = read_json(sample / "scene_recipe.json")
        samples.append(_category_metrics(sample, recipe))
    category_by_root: dict[str, list[dict[str, Any]]] = {}
    for item in samples:
        recipe = read_json(run / "samples" / item["sample_id"] / "scene_recipe.json")
        category_by_root.setdefault(recipe["identity"]["scene_root_id"], []).append(item)
    plan_path = run / "scene_plan.jsonl"
    if plan_path.exists():
        plan = [json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
        for row in plan:
            checks = category_by_root.get(row["scene_id"], [])
            if len(checks) == 4 and all(x["status"] == "pass" for x in checks):
                row["status"] = "completed"
                row.pop("pause_reason", None)
            elif any(x["status"] == "fail" for x in checks):
                row["status"] = "paused_quality_failed"
                row["pause_reason"] = "; ".join(sorted({e for x in checks for e in x.get("errors", [])}))
        plan_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in plan))
    summary = {
        "run": str(run), "submitted_interventions": len(rows),
        "completed_interventions": sum(x.get("status") in {"pass", "resumed"} for x in intervention_done),
        "intervention_failures": intervention_failures,
        "submitted_occlusion_depth": len(occlusion_rows),
        "completed_occlusion_depth": sum(x.get("status") in {"succeeded", "resumed"} for x in occlusion_done),
        "occlusion_depth_failures": occlusion_failures,
        "category_qa_counts": {key: sum(x["status"] == key for x in samples) for key in ("pass", "fail")},
        "scene_quality_status_counts": {"completed": sum(len(v) == 4 and all(x["status"] == "pass" for x in v) for v in category_by_root.values()),
                                         "paused_quality_failed": sum(any(x["status"] == "fail" for x in v) for v in category_by_root.values())},
        "category_qa": samples,
    }
    write_json(run / "effect_summary.json", summary)
    return summary
