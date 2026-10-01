from __future__ import annotations

import copy
import json
import math
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import stable_seed
from .io import canonical_hash, read_json, write_json
from .render import run_variant
from .runner import MIN_DISK_FREE_BYTES, MAX_WORKERS, eligible_gpus
from .validate import read_exr


JOINT_OUTPUTS = (
    "linear_premult.exr",
    "alpha.exr",
    "straight_rgba.png",
    "alpha_preview.png",
    "preview_black.png",
    "preview_white.png",
    "preview_checker.png",
    "render_log.json",
)


def _region_metrics(delta_p: np.ndarray, delta_a: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"pixel_count": 0, "premult_rgb_sigma_rms": None,
                "premult_rgb_abs_delta_p95": None, "alpha_sigma_rms": None,
                "alpha_abs_delta_p95": None}
    region_p = delta_p[mask]
    region_a = delta_a[mask]
    return {
        "pixel_count": count,
        "premult_rgb_sigma_rms": float(np.sqrt(np.mean(np.square(region_p), dtype=np.float64) / 2.0)),
        "premult_rgb_abs_delta_p95": float(np.percentile(np.mean(np.abs(region_p), axis=1), 95)),
        "alpha_sigma_rms": float(np.sqrt(np.mean(np.square(region_a), dtype=np.float64) / 2.0)),
        "alpha_abs_delta_p95": float(np.percentile(np.abs(region_a), 95)),
    }


def estimate_noise(primary_p: np.ndarray, second_p: np.ndarray,
                   primary_a: np.ndarray, second_a: np.ndarray) -> dict[str, Any]:
    """Estimate per-render Monte Carlo noise from two independently seeded renders.

    For independent renders with equal variance, sigma ~= RMS(P2-P1)/sqrt(2).
    P is measured in scene-linear premultiplied RGB; alpha is reported separately.
    """
    if primary_p.shape != second_p.shape or primary_a.shape != second_a.shape:
        raise ValueError("the two renders must have matching dimensions")
    if primary_p.ndim != 3 or primary_p.shape[-1] < 3 or primary_a.shape != primary_p.shape[:2] or second_a.shape != primary_p.shape[:2]:
        raise ValueError("expected HxWxRGB(A) premultiplied images and HxW alpha arrays")
    p1 = np.asarray(primary_p[..., :3], dtype=np.float64)
    p2 = np.asarray(second_p[..., :3], dtype=np.float64)
    a1 = np.asarray(primary_a, dtype=np.float64)
    a2 = np.asarray(second_a, dtype=np.float64)
    if not all(np.all(np.isfinite(x)) for x in (p1, p2, a1, a2)):
        raise ValueError("noise inputs contain NaN or Inf")
    delta_p = p2 - p1
    delta_a = a2 - a1
    support = (a1 > 0.01) | (a2 > 0.01)
    transparent = (a1 < 1e-4) & (a2 < 1e-4)
    all_pixels = np.ones(a1.shape, dtype=bool)
    return {
        "estimator": "sigma = RMS(render_seed2 - render_seed1) / sqrt(2)",
        "premultiplied_rgb_space": "scene-linear",
        "support_definition": "alpha > 0.01 in either render",
        "full_frame": _region_metrics(delta_p, delta_a, all_pixels),
        "target_support_union": _region_metrics(delta_p, delta_a, support),
        "transparent_background": _region_metrics(delta_p, delta_a, transparent),
    }


def _noise_job(recipe_path: Path, run: Path, gpu: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    recipe = read_json(recipe_path)
    identity = recipe["identity"]
    sample_id = identity["sample_id"]
    sample = run / "samples" / sample_id
    if not sample.is_dir() or not (sample / "scene_recipe.json").is_file():
        raise FileNotFoundError(f"base sample is missing: {sample}")
    parent_hash = identity["recipe_hash"]
    base_recipe = read_json(sample / "scene_recipe.json")
    if base_recipe.get("identity", {}).get("recipe_hash") != parent_hash:
        raise ValueError(f"base sample recipe hash mismatch: {sample_id}")
    if recipe.get("background_protocol", {}).get("mode") != "core":
        return {"sample_id": sample_id, "status": "skipped_non_core"}

    primary_seed = int(recipe["seeds"]["render"])
    second_seed = stable_seed(sample_id, "noise_estimate_render_seed_2") & 0x7FFFFFFF
    if second_seed == primary_seed:
        second_seed = (second_seed + 1) & 0x7FFFFFFF
    root = sample / "noise_estimate"
    output = root / "joint_seed2"
    stats_path = root / "noise_statistics.json"
    derived_path = root / "recipe_seed2.json"

    if resume and all((output / name).is_file() for name in JOINT_OUTPUTS) and stats_path.is_file():
        previous = read_json(stats_path)
        if previous.get("source_recipe_hash") == parent_hash and previous.get("second_seed") == second_seed:
            return {"sample_id": sample_id, "status": "resumed", "second_seed": second_seed,
                    "noise": previous.get("statistics")}

    derived = copy.deepcopy(recipe)
    derived["seeds"]["render"] = second_seed
    derived["identity"].pop("recipe_hash", None)
    derived["identity"]["recipe_hash"] = canonical_hash(derived)
    write_json(derived_path, derived)
    run_variant(derived_path, output, "joint", gpu_uuid=gpu["uuid"])

    p1 = read_exr(sample / "joint" / "linear_premult.exr")
    p2 = read_exr(output / "linear_premult.exr")
    a1 = read_exr(sample / "joint" / "alpha.exr")[..., 0]
    a2 = read_exr(output / "alpha.exr")[..., 0]
    statistics = estimate_noise(p1, p2, a1, a2)
    record = {
        "sample_id": sample_id,
        "source_recipe_hash": parent_hash,
        "seed2_recipe_hash": derived["identity"]["recipe_hash"],
        "primary_seed": primary_seed,
        "second_seed": second_seed,
        "gpu": gpu,
        "primary_render": "../joint/linear_premult.exr",
        "second_render": "joint_seed2/linear_premult.exr",
        "statistics": statistics,
        "finished_at": time.time(),
    }
    write_json(stats_path, record)
    return {"sample_id": sample_id, "status": "succeeded", "second_seed": second_seed,
            "gpu": gpu, "noise": statistics}


def render_noise_suite(run: Path, *, max_workers: int = MAX_WORKERS, resume: bool = True) -> dict[str, Any]:
    run = Path(run).resolve()
    manifest = run / "acquisition_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    recipes = [Path(row["recipe"]) for row in rows]
    disk = shutil.disk_usage(run)
    if disk.free < MIN_DISK_FREE_BYTES:
        raise RuntimeError(f"blocked_storage: free={disk.free} bytes, require={MIN_DISK_FREE_BYTES}")

    device_limit = max(1, min(MAX_WORKERS, int(max_workers)))
    pending = list(recipes)
    active: dict[Any, tuple[Path, dict[str, Any]]] = {}
    statuses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    status_path = run / "noise_status.jsonl"
    no_gpu_since: float | None = None
    gpu_idle_poll_seconds = 2.0
    gpu_idle_timeout_seconds = 600.0
    with ThreadPoolExecutor(max_workers=device_limit) as pool:
        while pending or active:
            available = eligible_gpus(device_limit)
            already = {gpu["uuid"] for _, gpu in active.values()}
            available = [gpu for gpu in available if gpu["uuid"] not in already]
            while pending and available and len(active) < device_limit:
                recipe_path = pending.pop(0)
                gpu = available.pop(0)
                future = pool.submit(_noise_job, recipe_path, run, gpu, resume=resume)
                active[future] = (recipe_path, gpu)
            if not active:
                if pending:
                    now = time.monotonic()
                    if no_gpu_since is None:
                        no_gpu_since = now
                    if now - no_gpu_since < gpu_idle_timeout_seconds:
                        time.sleep(gpu_idle_poll_seconds)
                        continue
                    waiting = [{"sample_id": read_json(p)["identity"]["sample_id"],
                                "status": "waiting_gpu_idle", "reason": "no GPU met the idle and memory gate for 10 minutes"}
                               for p in pending]
                    statuses.extend(waiting)
                    failures.extend(waiting)
                break
            no_gpu_since = None
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                recipe_path, gpu = active.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = {"sample_id": read_json(recipe_path)["identity"]["sample_id"],
                              "status": "failed", "gpu": gpu, "error": str(exc)}
                    failures.append(status)
                statuses.append(status)
                with status_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(status, ensure_ascii=False) + "\n")
            disk_now = shutil.disk_usage(run)
            if disk_now.free < MIN_DISK_FREE_BYTES:
                storage_failures = [{"sample_id": read_json(p)["identity"]["sample_id"],
                                     "status": "blocked_storage", "free_bytes": disk_now.free}
                                    for p in pending]
                statuses.extend(storage_failures)
                failures.extend(storage_failures)
                break

    summary = {
        "run": str(run),
        "submitted": len(recipes),
        "completed": sum(x.get("status") in {"succeeded", "resumed", "skipped_non_core"} for x in statuses),
        "failures": failures,
        "status_path": str(status_path),
        "statistics_root": str(run / "samples"),
    }
    write_json(run / "noise_summary.json", summary)
    return summary
