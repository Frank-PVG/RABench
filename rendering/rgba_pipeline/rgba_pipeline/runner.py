from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .io import write_json
from .render import render_recipe
from .validate import validate_sample

MIN_GPU_FREE_MIB = 16_384
MAX_WORKERS = 8
MIN_DISK_FREE_BYTES = 20 * 1024**3
INITIAL_OUTPUT_RESERVE_BYTES = 64 * 1024**3


def _nvidia_rows(query: str) -> list[list[str]]:
    result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True)
    return [[item.strip() for item in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def eligible_gpus(max_workers: int = MAX_WORKERS) -> list[dict[str, Any]]:
    rows = _nvidia_rows("--query-gpu=index,uuid,name,memory.free,utilization.gpu")
    try:
        apps = _nvidia_rows("--query-compute-apps=gpu_uuid,pid,process_name,used_memory")
    except subprocess.CalledProcessError:
        apps = []
    busy = {row[0] for row in apps if len(row) >= 2}
    devices = []
    for row in rows:
        if len(row) < 5: continue
        index, uuid, name, free_mib, utilization = row[:5]
        if uuid in busy or int(free_mib) < MIN_GPU_FREE_MIB or int(utilization) >= 10: continue
        devices.append({"index": int(index), "uuid": uuid, "name": name,
                        "memory_free_mib": int(free_mib), "utilization_percent": int(utilization)})
    return devices[:max(0, min(MAX_WORKERS, int(max_workers)))]


def _job(row: dict[str, Any], run: Path, gpu: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    recipe_path = Path(row["recipe"])
    sample = run / "samples" / row["sample_id"]
    sample.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(recipe_path.read_text())
    previous = sample / "scene_recipe.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get("identity", {}).get("recipe_hash") != row["recipe_hash"]:
            archive = run / "archive" / f"{row['sample_id']}_{old.get('identity', {}).get('recipe_hash', 'unknown')[:12]}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists(): raise RuntimeError(f"refusing to overwrite archived bundle {archive}")
            shutil.move(str(sample), str(archive))
            sample.mkdir(parents=True, exist_ok=True)
    shutil.copy2(recipe_path, previous)
    result = render_recipe(recipe_path, sample, resume=resume, gpu_uuid=gpu["uuid"])
    metadata = {
        "schema_version": recipe["schema_version"], "experiment_id": recipe.get("experiment_id"),
        "sample_id": recipe["identity"]["sample_id"], "scene_root_id": recipe["identity"]["scene_root_id"],
        "category": recipe["identity"]["category"], "camera_id": recipe["identity"]["camera_id"],
        "light_id": recipe["identity"]["light_id"], "scene_content_hash": recipe["identity"]["scene_content_hash"],
        "recipe_hash": recipe["identity"]["recipe_hash"], "target_ids": recipe["sets"]["TARGET"],
        "environment_ids": recipe["sets"]["ENVIRONMENT"], "asset_lineage_ids": recipe["identity"]["asset_lineage_ids"],
        "gpu": gpu, "renderer": "Blender Cycles", "variant_count": len(result["variants"]),
    }
    write_json(sample / "metadata.json", metadata)
    write_json(sample / "render_log.json", {**result, "gpu": gpu, "finished_at": time.time()})
    qa = validate_sample(sample)
    return {"sample_id": row["sample_id"], "scene_id": row["scene_id"], "status": "succeeded" if qa["qa_status"] == "pass" else "qa_failed",
            "qa_status": qa["qa_status"], "gpu": gpu, "errors": qa["errors"], "metrics": qa["metrics"]}


def _group_rows_by_scene(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(row)
    return [{"scene_id": scene_id, "rows": grouped[scene_id]} for scene_id in grouped]


def _scene_job(group: dict[str, Any], run: Path, gpu: dict[str, Any], *, resume: bool) -> list[dict[str, Any]]:
    """Keep one GPU worker on a scene and render its remaining camera/light combinations in order."""
    statuses = []
    for row in group["rows"]:
        try:
            statuses.append(_job(row, run, gpu, resume=resume))
        except Exception as exc:
            statuses.append({"sample_id": row["sample_id"], "scene_id": row["scene_id"],
                             "status": "failed", "gpu": gpu, "error": str(exc)})
    return statuses


def run_expansion_batch(run: Path, *, max_workers: int = MAX_WORKERS, resume: bool = True,
                        scene_ids: set[str] | None = None) -> dict[str, Any]:
    run = Path(run).resolve()
    manifest = run / "acquisition_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    retryable = {"pending", "failed", "qa_failed", "waiting_gpu_idle", "blocked_storage"}
    rows = [x for x in rows if (x.get("status") in retryable or not resume)
            and (scene_ids is None or x["scene_id"] in scene_ids)]
    if not rows: return {"submitted": 0, "completed": 0, "waiting": 0, "failures": []}
    disk = shutil.disk_usage(run)
    if disk.free < MIN_DISK_FREE_BYTES + INITIAL_OUTPUT_RESERVE_BYTES:
        raise RuntimeError(f"blocked_storage: free={disk.free} bytes, require={MIN_DISK_FREE_BYTES + INITIAL_OUTPUT_RESERVE_BYTES}")
    statuses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    device_limit = max(1, min(MAX_WORKERS, int(max_workers)))
    active: dict[Any, tuple[dict[str, Any], dict[str, Any]]] = {}
    pending = _group_rows_by_scene(rows)
    with ThreadPoolExecutor(max_workers=device_limit) as pool:
        while pending or active:
            available = eligible_gpus(device_limit)
            already = {item[1]["uuid"] for item in active.values()}
            available = [gpu for gpu in available if gpu["uuid"] not in already]
            while pending and available and len(active) < device_limit:
                group = pending.pop(0); gpu = available.pop(0)
                future = pool.submit(_scene_job, group, run, gpu, resume=resume)
                active[future] = (group, gpu)
            if not active:
                statuses.extend({"sample_id": row["sample_id"], "scene_id": group["scene_id"],
                                 "status": "waiting_gpu_idle", "reason": "no GPU met the idle and memory gate"}
                                for group in pending for row in group["rows"])
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                group, gpu = active.pop(future)
                try:
                    records = future.result()
                except Exception as exc:
                    records = [{"sample_id": row["sample_id"], "scene_id": row["scene_id"],
                                "status": "failed", "gpu": gpu, "error": str(exc)} for row in group["rows"]]
                statuses.extend(records)
                failures.extend(record for record in records if record["status"] != "succeeded")
                disk_now = shutil.disk_usage(run)
                if disk_now.free < MIN_DISK_FREE_BYTES:
                    statuses.extend({"sample_id": row["sample_id"], "scene_id": pending_group["scene_id"],
                                     "status": "blocked_storage", "free_bytes": disk_now.free}
                                    for pending_group in pending for row in pending_group["rows"])
                    pending.clear()
                    break
    status_path = run / "batch_status.jsonl"
    with status_path.open("a", encoding="utf-8") as out:
        for row in statuses: out.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_sample = {x["sample_id"]: x for x in statuses if x.get("sample_id")}
    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for row in manifest_rows:
        status = by_sample.get(row.get("sample_id"))
        if status: row["status"] = status["status"]
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows))
    return {"submitted": len(rows), "completed": sum(x.get("status") in {"succeeded", "qa_failed", "failed"} for x in statuses),
            "waiting": sum(x.get("status") == "waiting_gpu_idle" for x in statuses), "failures": failures,
            "status_path": str(status_path)}
