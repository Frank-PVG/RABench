from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import struct
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from rgba_pipeline.budget import BudgetedDownloader, ExperimentBudgetExceeded
from rgba_pipeline.experiment import EXPERIMENT_ID, SCENES, TOTAL_DOWNLOAD_BYTES
from rgba_pipeline.io import write_json
from rgba_pipeline.paths import BLENDER_BIN, PROJECT_ROOT

RUN = PROJECT_ROOT / "runs" / EXPERIMENT_ID
NASA_MODEL_PAGE = "https://science.nasa.gov/3d-resources/explorer-jupiter-c-rocket/"
NASA_MODEL_URL = "https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/model/explorer-jupiter-c-rocket/Explorer%20Jupiter-C%20Rocket.glb"
NASA_PREVIEW_URL = "https://assets.science.nasa.gov/dynamicimage/assets/science/cds/3d/resources/model/explorer-jupiter-c-rocket/Explorer%20Jupiter-C%20Rocket.png?crop=faces%2Cfocalpoint&fit=clip&h=1080&w=1920"
POLYHAVEN_INFO = "https://api.polyhaven.com/info/{}"
POLYHAVEN_FILES = "https://api.polyhaven.com/files/{}"
OBJAVERSE_BASE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"

# One manually reviewed, distinct primary object per scene root. Objaverse is
# preferred; the alternate datasets are only used for the recorded semantic gaps.
TARGETS: dict[str, tuple[str, str, str]] = {
    "SC01": ("objaverse", "e30380ec13d740719b7de6e3113add72", "Objaverse candidate preview: GTR Interceptor MK2"),
    "SC02": ("objaverse", "3f71baedb7424ac99bccda9bf4004fce", "Objaverse candidate preview: Polaroid camera"),
    "SC03": ("objaverse", "3b12b5a5f067430b9c7a2d4b7d1dc467", "Objaverse candidate preview: shoe"),
    "SC04": ("objaverse", "03e4ffa802654cea865fc8ebfc0d27ed", "Objaverse candidate preview: apple"),
    "SC05": ("objaverse", "db415d2f3d9d421c88caddad08cfe9f3", "Objaverse candidate preview: alarm clock"),
    "SC06": ("objaverse", "364f91c2d74744b7bc4f92f116e73352", "Objaverse candidate preview: bird model"),
    "CT01": ("objaverse", "3146c02a2b5e41b9ac8481c66c8e1ba1", "Objaverse candidate preview: cup"),
    "CT02": ("objaverse", "19bacf81f304491f8f472824279e5d66", "Objaverse candidate preview: coffee cup"),
    "CT03": ("objaverse", "0540daec5ee54e7bacb3e346776be39c", "Objaverse candidate preview: teapot"),
    "CT04": ("objaverse", "8eb56daa051f4a42a74900932180199c", "Objaverse candidate preview: bottle"),
    "CT05": ("objaverse", "0adb377236c84c55be35878cc4d2a18a", "Objaverse candidate preview: deity sculpture"),
    "CT06": ("objaverse", "0649fdcb431c41a0ac2529940f86c7a1", "Objaverse candidate preview: toy vehicle"),
    "CT07": ("gso", "50_BLOCKS", "Targeted Google Scanned Objects catalog + model preview"),
    "CT08": ("objaverse", "546af06b942a4267a67d2e6838dd7220", "Expanded Objaverse metadata + preview: Shires chesspiece"),
    "CT09": ("objaverse", "4eef3df05a234a36bc9c385be5284078", "Objaverse candidate preview: candle holder"),
    "CT10": ("objaverse", "83c04eea513c4a57a805a077df97d55d", "Objaverse candidate preview: seashell"),
    "CT11": ("objaverse", "bccaad9c6cef4e91a7bbf71666e45745", "Objaverse candidate preview: jar"),
    "CT12": ("gso", "FAIRY_TALE_BLOCKS", "Targeted Google Scanned Objects catalog + model preview"),
    "RF01": ("gso", "Threshold_Bead_Cereal_Bowl_White", "Objaverse targeted bowl search found no suitable white candidate; exact GSO white bowl listing and preview selected as fallback"),
    "RF02": ("objaverse", "ce5c3a6781574d0093ee3b7b67d61851", "Expanded Objaverse metadata + preview: pale stone bust"),
    "RF03": ("objaverse", "78ebe1fb7b354fd8a000b55775bf9570", "Objaverse candidate preview: jar vessel"),
    "RF04": ("objaverse", "e77092787ecd48389d0239ab983f5a6e", "Expanded Objaverse metadata + preview: jug"),
    "RF05": ("objaverse", "c59c34fda54e475d9ad129f709d723c3", "Expanded Objaverse metadata + preview: pale stone mouse sculpture"),
    "RF06": ("objaverse", "c40379cc02d24defa9ce324738153930", "Objaverse candidate preview: cup"),
    "RF07": ("objaverse", "659d2e4e1752459b874d63f2a8302aa1", "Objaverse candidate preview: kettle"),
    "RF08": ("objaverse", "861959a12e4a4da8ac7407c2be517794", "Objaverse candidate preview: Panela cookware"),
    "RF09": ("objaverse", "ffb0d644238b4c679658aa0ee46ac6da", "Expanded Objaverse metadata + preview: watering can"),
    "RF10": ("objaverse", "15b103f822ef4d92a386e95746a9aebe", "Objaverse candidate preview: bowl"),
    "RF11": ("objaverse", "8afccda6a86743368429e4271348ddc7", "Objaverse candidate preview: barrel/canister"),
    "RF12": ("objaverse", "487fdbc316324db08b61bb962b6c6838", "Objaverse candidate preview: Hydroflask"),
    "EN01": ("objaverse", "137107a9a148431ea6bd255a5a5a4cd2", "Objaverse candidate preview: vase with roses"),
    "EN02": ("objaverse", "aff09df52ea84c318ba591c0f70815e6", "Expanded Objaverse metadata + preview: vase"),
    "EN03": ("objaverse", "ba11f0bcd7c44f4cadafa4a91f964214", "Objaverse candidate preview: medieval jug"),
    "EN04": ("objaverse", "7d22e8fbe5a54519b2661fea300797fa", "Expanded Objaverse metadata + preview: cooking pot"),
    "EN05": ("objaverse", "57766f8721f24d368e531768349a5487", "Objaverse candidate preview: teapot"),
    "EN06": ("objaverse", "49868f0b7cc84b93a7aeff01fcd64546", "Expanded Objaverse metadata + preview: moose sculpture"),
    "TR01": ("objaverse", "43b3aa28587f49d8b320d27ef9cd2f6c", "Objaverse candidate preview: teddy bear"),
    "TR02": ("objaverse", "0dd0e116105e4daaad459d03f3d8de8e", "Objaverse candidate preview: rabbit"),
    "TR03": ("objaverse", "727eaad7ae154904a7f3d12a0affe14d", "Objaverse candidate preview: toy animatronic model"),
    "TR04": ("objaverse", "81cd003c9464476bbb6caa756559d8ae", "Objaverse candidate preview: horse"),
    "TR05": ("objaverse", "0df79b5bf0f74582a1473374c24dc326", "Objaverse candidate preview: elephant"),
    "TR06": ("objaverse", "bef0ada62b64410ba523923cf6ac56ce", "Objaverse candidate preview: plush doll"),
    "TR07": ("objaverse", "7ea984312db64f0aa77a40c305126953", "Objaverse candidate preview: distinct toy vehicle"),
    "TR08": ("objaverse", "f6230d8eff1f4de5b367a12a593018f8", "Objaverse candidate preview: folded bird"),
    "TR09": ("objaverse", "c6a90117cc6c4568b21d77a36a9295bf", "Objaverse candidate preview: owl figurine"),
    "TR10": ("objaverse", "93d7c9a0f0ba45eeb16897931d636720", "Objaverse candidate preview: wood cat"),
    "TR11": ("polyhaven", "modular_fort_01", "Objaverse toy-castle search had no suitable asset; Poly Haven modular fort fallback"),
    "TR12": ("nasa", "explorer_jupiter_c_rocket", "Objaverse and the searched GSO/Poly Haven catalogs had no suitable rocket; NASA single-model fallback"),
    "TR13": ("objaverse", "202828523d24472b9a309ac3bbed9a17", "Expanded Objaverse metadata + preview: camera tripod"),
    "TR14": ("objaverse", "ab2c61765bfb4081b05c51a54087ddb8", "Expanded Objaverse metadata + preview: newspaper display stand"),
    "TR15": ("objaverse", "2fa777c32afe4e048ef4067635790c24", "Expanded Objaverse metadata + preview: dolls house bedroom"),
    "OC01": ("objaverse", "f8e67bafae6f4f38a45619edf080b8c8", "Expanded Objaverse metadata + preview: necklace with linked chain"),
    "OC02": ("objaverse", "c3ef2c8fad514b69bd3d7180b84cb25f", "Expanded Objaverse metadata + preview: distinct necklace"),
    "OC03": ("polyhaven", "modular_electric_cables", "Objaverse loop search found only straight rope/wire; Poly Haven cable kit has bends and separable segments"),
    "OC04": ("polyhaven", "dry_branches_medium_01", "Objaverse tree search returned conifer/holiday trees; Poly Haven bare branches fallback"),
    "OC05": ("polyhaven", "pachira_aquatica_01", "Objaverse tree search had no suitable leafy branch; Poly Haven leafy shrub/tree fallback"),
    "OC06": ("polyhaven", "fern_02", "Objaverse fern search had no suitable fern frond; Poly Haven fern fallback"),
    "OC07": ("objaverse", "4ec9181f5f274cacb2138714202020d1", "Objaverse candidate preview: open woven vintage basket"),
    "OC08": ("objaverse", "84e0b1cc83f84b7698532cd982f02b4f", "Expanded Objaverse metadata + preview: net cage"),
    "OC09": ("objaverse", "cbef71e3235e4b99acb81d452cfc4019", "Expanded Objaverse metadata + preview: wires; a second cable asset will be interleaved"),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_glb_container(path: Path) -> int:
    """Validate the GLB 2.0 header and return its actual byte length.

    Objaverse metadata can report stale glb_size values for a UID. The HTTP
    response length and GLB's declared container length establish transfer
    integrity; Blender import and the asset inspector provide content checks.
    """
    actual = path.stat().st_size
    with path.open("rb") as source:
        header = source.read(12)
    if len(header) != 12 or header[:4] != b"glTF":
        raise RuntimeError(f"invalid Objaverse GLB header: {path.name}")
    version, declared = struct.unpack_from("<II", header, 4)
    if version != 2 or declared != actual:
        raise RuntimeError(
            f"invalid Objaverse GLB container: version={version}, declared={declared}, actual={actual}"
        )
    return actual


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_index() -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in ("objaverse_candidates.json", "objaverse_expanded_candidates.json"):
        doc = load_json(RUN / "selection" / filename)
        for row in doc.get("candidates", []):
            key = ("objaverse", str(row.get("uid")))
            old = result.get(key)
            if old is None or (not old.get("thumbnail") and row.get("thumbnail")):
                result[key] = dict(row)
    for row in load_json(RUN / "selection" / "polyhaven" / "candidates.json").get("candidates", []):
        if row.get("slug"):
            result[("polyhaven", str(row["slug"]))] = dict(row)
    for row in load_json(RUN / "selection" / "gso" / "candidates.json").get("candidates", []):
        if row.get("uid"):
            result[("gso", str(row["uid"]))] = dict(row)
    rf01_path = RUN / "selection" / "gso" / "rf01_white_bowl_candidates.json"
    if rf01_path.exists():
        for row in load_json(rf01_path).get("candidates", []):
            result[("gso", str(row["uid"]))] = dict(row)
    for sid, row in load_json(RUN / "selection" / "gso" / "selected_archives.json").get("selected", {}).items():
        result[("gso", str(row["uid"]))] = dict(row)
    return result


def metadata_file(row: dict[str, Any]) -> Path:
    shard = str(row["metadata_shard"])
    local = RUN / "assets" / "metadata" / f"{shard}.json.gz"
    cached = Path.home() / ".objaverse" / "hf-objaverse-v1" / "metadata" / f"{shard}.json.gz"
    return local if local.exists() else cached


def ensure_objaverse_preview(row: dict[str, Any], downloader: BudgetedDownloader) -> dict[str, Any]:
    old = row.get("thumbnail")
    if old and Path(old).is_file():
        return {"path": old, "sha256": row.get("thumbnail_sha256") or sha256(Path(old)), "reused": True}
    source = metadata_file(row)
    if not source.exists():
        source = RUN / "assets" / "metadata" / f"{row['metadata_shard']}.json.gz"
        transfer = downloader.fetch(f"{OBJAVERSE_BASE}/metadata/{row['metadata_shard']}.json.gz", source,
            purpose="selected_objaverse_metadata_shard", asset_id=f"metadata:{row['metadata_shard']}", max_file_bytes=16 * 1024 * 1024)
        source = Path(transfer["path"])
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        annotation = json.load(handle)[row["uid"]]
    images = ((annotation.get("thumbnails") or {}).get("images") or [])
    images = [item for item in images if item.get("url")]
    images.sort(key=lambda item: (int(item.get("width") or 512) > 720, abs(int(item.get("width") or 512) - 512)))
    if not images:
        raise RuntimeError(f"selected Objaverse model has no preview thumbnail: {row['uid']}")
    preview_root = RUN / "selection" / "selected_thumbnails"
    preview_root.mkdir(parents=True, exist_ok=True)
    path = preview_root / f"{row['scene_id']}_{row['uid']}.jpg"
    transfer = downloader.fetch(images[0]["url"], path, purpose="selected_asset_thumbnail",
        asset_id=f"thumbnail:{row['scene_id']}:{row['uid']}", max_file_bytes=8 * 1024 * 1024)
    return {"path": transfer["path"], "sha256": transfer["sha256"], "reused": transfer.get("reused", False)}


def selected_rows(downloader: BudgetedDownloader) -> dict[str, dict[str, Any]]:
    index = candidate_index()
    if set(TARGETS) != {task.scene_id for task in SCENES}:
        raise RuntimeError("curated target table does not match the fixed 60-root contract")
    obj_uids = [uid for dataset, uid, _ in TARGETS.values() if dataset == "objaverse"]
    if len(obj_uids) != len(set(obj_uids)):
        raise RuntimeError("a primary Objaverse UID is repeated across scene roots")
    selected: dict[str, dict[str, Any]] = {}
    for sid, (dataset_key, uid, reason) in TARGETS.items():
        row = index.get((dataset_key, uid))
        if row is None and dataset_key != "nasa":
            raise RuntimeError(f"curated asset not found in pre-download candidates: {sid} {dataset_key}:{uid}")
        if dataset_key == "objaverse":
            if row.get("selection_status") not in {"eligible_thumbnail_review", "eligible_candidate_review", "eligible"}:
                raise RuntimeError(f"Objaverse candidate did not pass metadata gates: {sid} {row.get('selection_status')} {uid}")
            if str(row.get("license", "")).lower() not in {"cc0", "by", "cc0-1.0", "cc-by-4.0"}:
                raise RuntimeError(f"Objaverse candidate license is outside allowlist: {sid} {row.get('license')}")
        if dataset_key == "polyhaven" and uid not in {"modular_fort_01", "modular_electric_cables", "dry_branches_medium_01", "pachira_aquatica_01", "fern_02"}:
            raise RuntimeError(f"unreviewed Poly Haven fallback: {uid}")
        if dataset_key == "gso" and not str(row.get("license", row.get("license_name", ""))).lower().startswith("creative commons attribution 4.0"):
            raise RuntimeError(f"GSO candidate is missing CC BY 4.0 listing metadata: {sid}:{uid}")
        selected[sid] = {"scene_id": sid, "dataset_key": dataset_key, "uid": uid,
                         "selection_reason": reason, "candidate": row or {}}
    return selected


def fetch_previews(selected: dict[str, dict[str, Any]], downloader: BudgetedDownloader) -> None:
    ph_root = RUN / "selection" / "polyhaven"
    for sid, entry in selected.items():
        key, uid = entry["dataset_key"], entry["uid"]
        row = entry["candidate"]
        if key == "objaverse":
            preview = ensure_objaverse_preview(row, downloader)
        elif key == "polyhaven":
            path = row.get("thumbnail")
            if not path or not Path(path).is_file():
                raise RuntimeError(f"selected Poly Haven model has no reviewed preview: {uid}")
            preview = {"path": path, "sha256": row.get("thumbnail_sha256") or sha256(Path(path)), "reused": True}
        elif key == "gso":
            path = row.get("thumbnail")
            if not path or not Path(path).is_file():
                raise RuntimeError(f"selected GSO model has no reviewed preview: {uid}")
            preview = {"path": path, "sha256": row.get("thumbnail_sha256") or sha256(Path(path)), "reused": True}
        else:
            path = RUN / "selection" / "nasa" / "explorer_jupiter_c_rocket.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            t = downloader.fetch(NASA_PREVIEW_URL, path, purpose="targeted_nasa_asset_thumbnail",
                asset_id=f"nasa-thumbnail:{uid}", max_file_bytes=8 * 1024 * 1024)
            preview = {"path": t["path"], "sha256": t["sha256"], "reused": t.get("reused", False)}
        entry["preview"] = preview
        entry["candidate_source_scene"] = row.get("scene_id") if row else "NASA 3D Resources"
    write_json(RUN / "selection" / "curated_targets.json", {
        "experiment_id": EXPERIMENT_ID,
        "selection_policy": "Objaverse first; only use an alternate 3D model source when reviewed Objaverse candidates did not fit the declared object role",
        "model_downloads_completed": False,
        "targets": selected,
        "download_budget": downloader.status(),
    })
    make_selected_sheets(selected)


def make_selected_sheets(selected: dict[str, dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw
    items = list(selected.items())
    tile_w, tile_h = 210, 180
    out_dir = RUN / "selection"
    for page in range(3):
        subset = items[page * 20:(page + 1) * 20]
        sheet = Image.new("RGB", (tile_w * 4, tile_h * 5), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (sid, record) in enumerate(subset):
            x = (i % 4) * tile_w; y = (i // 4) * tile_h
            p = Path(record["preview"]["path"])
            image = Image.open(p).convert("RGB"); image.thumbnail((tile_w - 8, tile_h - 31))
            sheet.paste(image, (x + (tile_w - image.width) // 2, y + 22 + (tile_h - 31 - image.height) // 2))
            draw.text((x + 4, y + 3), f"{sid} [{record['dataset_key']}]", fill="black")
            draw.text((x + 4, y + tile_h - 9), str(record['candidate'].get('name') or record['uid'])[:30], fill="#333333")
        sheet.save(out_dir / f"curated_contact_sheet_{page + 1:02d}.jpg", quality=86)


def polyhaven_asset(slug: str, row: dict[str, Any], downloader: BudgetedDownloader) -> dict[str, Any]:
    ph_root = RUN / "selection" / "polyhaven"
    info_path = ph_root / "info" / f"{slug}.json"
    files_path = ph_root / "files" / f"{slug}.json"
    for purpose, path, url in (
        ("targeted_polyhaven_asset_metadata", info_path, POLYHAVEN_INFO.format(urllib.parse.quote(slug, safe=""))),
        ("selected_polyhaven_file_manifest", files_path, POLYHAVEN_FILES.format(urllib.parse.quote(slug, safe=""))),
    ):
        if not path.is_file():
            t = downloader.fetch(url, path, purpose=purpose, asset_id=f"{purpose}:{slug}", max_file_bytes=2 * 1024 * 1024)
        else:
            t = {"path": str(path), "sha256": sha256(path), "reused": True}
    info = load_json(info_path); files = load_json(files_path)
    if info.get("type") != 2:
        raise RuntimeError(f"Poly Haven file is not an individual model: {slug}")
    gltf = files.get("gltf", {}).get("1k", {}).get("gltf")
    if not gltf or not isinstance(gltf.get("include"), dict):
        raise RuntimeError(f"Poly Haven 1K GLTF + dependency manifest missing: {slug}")
    model_root = RUN / "assets" / "polyhaven" / slug
    main_path = model_root / f"{slug}_1k.gltf"
    downloads = [(PurePosixPath(f"{slug}_1k.gltf"), gltf)]
    for rel, spec in gltf["include"].items():
        rel_path = PurePosixPath(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise RuntimeError(f"unsafe dependency in Poly Haven manifest: {slug}:{rel}")
        downloads.append((rel_path, spec))
    file_records = []
    for rel_path, spec in downloads:
        destination = model_root.joinpath(*rel_path.parts)
        transfer = downloader.fetch(spec["url"], destination,
            purpose="selected_polyhaven_model_file", asset_id=f"polyhaven-model:{slug}:{rel_path.as_posix()}",
            max_file_bytes=int(spec["size"]) + 1)
        if destination.stat().st_size != int(spec["size"]):
            raise RuntimeError(f"Poly Haven advertised byte length mismatch: {slug}:{rel_path}")
        actual_md5 = md5(destination)
        if actual_md5.lower() != str(spec["md5"]).lower():
            raise RuntimeError(f"Poly Haven MD5 mismatch: {slug}:{rel_path}")
        file_records.append({"path": str(destination), "relative_path": rel_path.as_posix(),
                             "bytes": destination.stat().st_size, "md5": actual_md5, "sha256": sha256(destination),
                             "source_url": spec["url"]})
    return {"dataset": "Poly Haven", "uid": slug, "name": info.get("name", slug),
        "local_path": str(main_path), "source_url": f"https://polyhaven.com/a/{slug}",
        "model_url": gltf["url"], "license": "CC0", "author": info.get("authors", {}),
        "tags": info.get("tags", []), "polycount_metadata": info.get("polycount"),
        "max_resolution": info.get("max_resolution"), "metadata_path": str(info_path),
        "metadata_sha256": sha256(info_path), "file_manifest_path": str(files_path),
        "file_manifest_sha256": sha256(files_path), "file_resolution": "1k",
        "downloaded_files": file_records,
        "source_texture_policy": "all 1K GLTF manifest dependencies retained; original maps are not recolored"}


def extract_gso_candidate(sid: str, row: dict[str, Any], downloader: BudgetedDownloader) -> dict[str, Any]:
    import os
    import shlex
    import stat
    import zipfile
    uid = str(row["uid"])
    archive_root = RUN / "assets" / "gso_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{uid}.zip"
    model_url = str(row["model_url"])
    advertised = int(row.get("filesize") or 0)
    if not model_url.endswith(".zip") or advertised <= 0:
        raise RuntimeError(f"GSO single-model catalog entry lacks an exact ZIP URL/size: {uid}")
    transfer = downloader.fetch(model_url, archive_path, purpose="selected_gso_model_archive",
        asset_id=f"gso-model:{uid}", max_file_bytes=advertised + 1)
    if archive_path.stat().st_size != advertised:
        raise RuntimeError(f"GSO model archive size mismatch for {uid}: catalog={advertised}, downloaded={archive_path.stat().st_size}")
    extracted_root = RUN / "assets" / "gso" / uid
    extracted_root.mkdir(parents=True, exist_ok=True)
    members = []
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            rel = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if rel.is_absolute() or ".." in rel.parts or stat.S_ISLNK(mode):
                raise RuntimeError(f"unsafe member in selected GSO archive {uid}: {info.filename}")
            members.append({"name": rel.as_posix(), "bytes": info.file_size})
            if info.is_dir():
                continue
            target = extracted_root.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with zf.open(info) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    objs = sorted(extracted_root.rglob("*.obj"))
    if len(objs) != 1:
        raise RuntimeError(f"expected one OBJ in selected GSO model {uid}, found {len(objs)}")
    obj_path = objs[0]
    mtls = []
    for line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = shlex.split(line.strip())
        if fields and fields[0].lower() == "mtllib":
            mtls.extend((obj_path.parent / item).resolve() for item in fields[1:])
    texture_paths: set[str] = set()
    for mtl in mtls:
        if not mtl.is_file():
            raise RuntimeError(f"GSO OBJ references missing material file: {mtl}")
        for line in mtl.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = shlex.split(line.strip())
            if not fields or not fields[0].lower().startswith("map_") or len(fields) < 2:
                continue
            rel = Path(fields[-1])
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"unsafe source texture path in GSO MTL: {fields[-1]}")
            target = (mtl.parent / rel).resolve()
            if not target.is_file():
                options = [x for x in extracted_root.rglob(rel.name) if x.is_file()]
                if len(options) != 1:
                    raise RuntimeError(f"GSO texture dependency is missing/ambiguous: {uid}:{rel}")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(options[0], target)
            texture_paths.add(str(target))
    record = {"experiment_id": EXPERIMENT_ID, "scene_id": sid, "uid": uid,
        "name": row.get("name", uid), "dataset": "Google Scanned Objects", "owner": row.get("owner", "GoogleResearch"),
        "license": row.get("license_name") or row.get("license"), "license_url": row.get("license_url"),
        "description": row.get("description"), "source_url": row["source_url"], "model_url": model_url,
        "filesize": advertised, "archive_path": str(archive_path), "archive_sha256": sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size, "local_path": str(obj_path),
        "extracted_root": str(extracted_root), "mtl_paths": [str(x) for x in mtls],
        "texture_paths": sorted(texture_paths), "zip_members": members,
        "thumbnail_url": row.get("thumbnail_url"), "thumbnail": row.get("thumbnail"),
        "thumbnail_sha256": row.get("thumbnail_sha256"), "selection_status": "selected_downloaded"}
    return record


def download_all(selected: dict[str, dict[str, Any]], downloader: BudgetedDownloader) -> None:
    selected_archive_path = RUN / "selection" / "gso" / "selected_archives.json"
    selected_archive_doc = load_json(selected_archive_path)
    archive_doc = selected_archive_doc.get("selected", {})
    archive_by_uid = {str(value.get("uid")): value for value in archive_doc.values()}
    records: dict[str, dict[str, Any]] = {}
    downloaded_rows: list[dict[str, Any]] = []
    for sid, entry in selected.items():
        key, uid, row = entry["dataset_key"], entry["uid"], entry["candidate"]
        candidate_path = row.get("object_path")
        try:
            if key == "objaverse":
                dest = RUN / "assets" / "objaverse" / f"{uid}.glb"
                relative = row["object_path"]
                url = f"{OBJAVERSE_BASE}/{relative}"
                transfer = downloader.fetch(url, dest, purpose="selected_objaverse_model",
                    asset_id=f"objaverse-model:{uid}", max_file_bytes=512 * 1024 * 1024)
                metadata_size = int(row["glb_size"])
                actual = validate_glb_container(dest)
                record = {"dataset": "Objaverse 1.0", "uid": uid, "name": row.get("name", uid),
                    "local_path": str(dest), "object_path": relative,
                    "source_url": row.get("source_url") or row.get("viewer_url"), "model_url": url,
                    "license": row.get("license"), "author": row.get("author"), "tags": row.get("tags", []),
                    "lvis_category": row.get("lvis_category"), "metadata_shard": row.get("metadata_shard"),
                    "metadata_sha256": row.get("metadata_sha256"), "metadata_face_count": row.get("face_count"),
                    "metadata_glb_size_bytes": metadata_size, "metadata_size_matches_download": metadata_size == actual,
                    "expected_bytes": transfer.get("expected_bytes"), "downloaded_bytes": actual,
                    "candidate_preview": entry["preview"], "sha256": sha256(dest)}
            elif key == "gso":
                archive = archive_by_uid.get(uid)
                if not archive or not Path(archive.get("archive_path", "")).is_file() or not Path(archive.get("local_path", "")).is_file():
                    archive = extract_gso_candidate(sid, row, downloader)
                    archive_by_uid[uid] = archive
                    archive_doc[sid] = archive
                else:
                    archive["scene_id"] = sid
                selected_archive_doc["selected"] = archive_doc
                selected_archive_doc["budget"] = downloader.status()
                write_json(selected_archive_path, selected_archive_doc)
                dest = Path(archive["local_path"])
                record = {"dataset": "Google Scanned Objects", "uid": uid, "name": archive["name"],
                    "local_path": str(dest), "source_url": archive["source_url"], "model_url": archive["model_url"],
                    "license": archive["license"], "license_url": archive["license_url"], "author": archive["owner"],
                    "tags": ["white bowl" if sid == "RF01" else "toy blocks"], "archive_path": archive["archive_path"],
                    "archive_sha256": archive["archive_sha256"], "archive_bytes": archive["archive_bytes"],
                    "texture_paths": archive["texture_paths"], "candidate_preview": entry["preview"],
                    "source_texture_policy": "original material maps referenced by MTL retained; missing relative path repaired with a hard link"}
            elif key == "polyhaven":
                record = polyhaven_asset(uid, row, downloader)
            else:
                dest = RUN / "assets" / "nasa" / "Explorer_Jupiter-C_Rocket.glb"
                dest.parent.mkdir(parents=True, exist_ok=True)
                downloader.fetch(NASA_MODEL_URL, dest, purpose="selected_nasa_model",
                    asset_id=f"nasa-model:{uid}", max_file_bytes=512 * 1024 * 1024)
                record = {"dataset": "NASA 3D Resources", "uid": uid, "name": "Explorer Jupiter-C Rocket",
                    "local_path": str(dest), "source_url": NASA_MODEL_PAGE, "model_url": NASA_MODEL_URL,
                    "license": "NASA public-use guidance; model page states free to download and use",
                    "author": "NASA/Michael D. Carbajal", "tags": ["rocket", "3D model"],
                    "candidate_preview": entry["preview"], "downloaded_bytes": dest.stat().st_size,
                    "source_texture_policy": "GLB materials retained as supplied", "sha256": sha256(dest)}
            record.update({"asset_id": uid, "scene_id": sid, "selection_reason": entry["selection_reason"],
                           "candidate_source_scene": entry.get("candidate_source_scene"),
                           "acquisition_status": "downloaded"})
            records[sid] = record
            downloaded_rows.append({"scene_id": sid, "dataset": record["dataset"], "uid": uid,
                "path": record["local_path"], "sha256": record.get("sha256"),
                "model_url": record.get("model_url"), "license": record.get("license"),
                "selection_reason": entry["selection_reason"], "status": "downloaded"})
        except ExperimentBudgetExceeded:
            raise
        except Exception as exc:
            records[sid] = {"scene_id": sid, "dataset": row.get("dataset", key), "uid": uid,
                "name": row.get("name", uid), "local_path": None, "requested_local_path": candidate_path,
                "source_url": row.get("source_url"), "model_url": row.get("model_url"),
                "license": row.get("license") or row.get("license_name"), "author": row.get("author"),
                "candidate_preview": entry.get("preview"), "selection_reason": entry["selection_reason"],
                "acquisition_status": "failed", "acquisition_error": str(exc)}
            downloaded_rows.append({"scene_id": sid, "dataset": row.get("dataset", key), "uid": uid,
                "path": candidate_path, "license": row.get("license") or row.get("license_name"),
                "selection_reason": entry["selection_reason"], "status": "failed", "error": str(exc)})
    write_json(RUN / "selection" / "selected_assets_preinspection.json", {
        "experiment_id": EXPERIMENT_ID, "selected": records,
        "download_budget": downloader.status(),
        "note": "Only the exact candidate models listed here were fetched; Poly Haven downloads use exact 1K GLTF manifests."})
    write_jsonl(RUN / "selection" / "selected_asset_downloads.jsonl", downloaded_rows)
    downloader.export_jsonl(RUN / "asset_download_ledger.jsonl")


def inspect_all() -> None:
    doc_path = RUN / "selection" / "selected_assets_preinspection.json"
    doc = load_json(doc_path); selected = doc["selected"]
    inspect_root = RUN / "assets" / "inspections"
    inspect_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    for sid, record in selected.items():
        source = Path(record["local_path"]) if record.get("local_path") else None
        if source is None or not source.is_file():
            if record.get("acquisition_status") != "failed":
                record["acquisition_status"] = "failed"
                record["acquisition_error"] = "selected asset file is missing after acquisition"
            failures.append({"scene_id": sid, "uid": record.get("uid", ""), "error": record.get("acquisition_error", "missing model file")})
            record["local_path"] = None
            continue
        output = inspect_root / f"{sid}_{record['uid']}.json"
        command = [str(BLENDER_BIN), "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(PROJECT_ROOT / "blender" / "inspect_asset.py"), "--",
                   "--source", str(source), "--output", str(output)]
        try:
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr[-2500:])
            inspection = load_json(output)
            if not inspection["qa"]["has_mesh"] or not inspection["qa"]["finite_geometry"]:
                raise RuntimeError("selected asset geometry QA failed")
            if inspection["qa"]["missing_dependency_count"]:
                raise RuntimeError("selected asset has missing source material/map dependencies")
            if not inspection["qa"]["triangle_limit_ok"]:
                raise RuntimeError("selected asset exceeds 1M triangle inspection gate")
            record.update({"normalization": inspection["normalization"],
                "component_count": inspection["component_count"], "geometry_fingerprint": inspection["geometry_fingerprint"],
                "sha256": inspection["source_sha256"], "source_bytes": inspection["source_bytes"],
                "triangle_count": inspection["triangle_count"], "vertex_count": inspection["vertex_count"],
                "image_count": inspection["image_count"], "image_texture_node_count": inspection["image_texture_node_count"],
                "base_color_image_links": inspection["base_color_image_links"],
                "external_dependencies": inspection["external_dependencies"],
                "inspection_path": str(output), "inspection_qa": inspection["qa"], "acquisition_status": "validated"})
        except Exception as exc:
            record["acquisition_status"] = "inspection_failed"
            record["acquisition_error"] = str(exc)
            record["requested_local_path"] = record.get("local_path")
            record["local_path"] = None
            failures.append({"scene_id": sid, "uid": record.get("uid", ""), "error": str(exc)})
    fingerprint_to_scene: dict[str, str] = {}
    for sid, record in selected.items():
        fingerprint = record.get("geometry_fingerprint")
        if not fingerprint or not record.get("local_path"):
            continue
        if fingerprint in fingerprint_to_scene:
            record["acquisition_status"] = "inspection_failed"
            record["acquisition_error"] = f"same primary geometry fingerprint as {fingerprint_to_scene[fingerprint]}"
            record["requested_local_path"] = record["local_path"]
            record["local_path"] = None
            failures.append({"scene_id": sid, "uid": record.get("uid", ""), "error": record["acquisition_error"]})
        else:
            fingerprint_to_scene[fingerprint] = sid
    final_path = RUN / "selection" / "selected_assets.json"
    status = BudgetedDownloader(RUN / "download_budget.sqlite", RUN).status()
    write_json(final_path, {"experiment_id": EXPERIMENT_ID, "selected": selected,
        "download_budget": status, "unique_primary_count": sum(bool(x.get("local_path")) for x in selected.values()),
        "unique_geometry_fingerprint_count": len(fingerprint_to_scene),
        "source_appearance_policy": "preserve original materials, base colors, and textures; only material channels declared in a scene recipe may be varied",
        "asset_validation_failures": failures})
    doc["selected"] = selected; doc["download_budget"] = status
    write_json(doc_path, doc)
    if failures:
        write_json(RUN / "selection" / "asset_validation_failures.json", failures)
    BudgetedDownloader(RUN / "download_budget.sqlite", RUN).export_jsonl(RUN / "asset_download_ledger.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previews-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    downloader = BudgetedDownloader(RUN / "download_budget.sqlite", RUN, budget_bytes=TOTAL_DOWNLOAD_BYTES)
    selected = selected_rows(downloader)
    if args.previews_only or not (RUN / "selection" / "curated_targets.json").exists():
        fetch_previews(selected, downloader)
    if args.download:
        fetch_previews(selected, downloader)
        download_all(selected, downloader)
    if args.inspect:
        inspect_all()
    print(json.dumps(downloader.status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
