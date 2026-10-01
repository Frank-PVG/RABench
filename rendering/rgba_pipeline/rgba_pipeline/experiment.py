from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "exp60_20260930_v1"
TOTAL_DOWNLOAD_BYTES = 100 * 1024**3
EXPECTED_VARIANTS = ("L1_C1", "L1_C2", "L2_C1", "L2_C2")
CLASS_COUNTS = {
    "simple_control": 6,
    "contact_shadow": 12,
    "reflection_color_spill": 12,
    "environmental_influence": 6,
    "translucent_overlap": 15,
    "interleaved_occlusion": 9,
}


@dataclass(frozen=True)
class SceneTask:
    scene_id: str
    category: str
    primary_name: str
    primary_search: tuple[str, ...]
    composition: str
    auxiliary_search: tuple[str, ...] = ()


def _task(scene_id: str, category: str, name: str, query: str, composition: str,
          auxiliary: tuple[str, ...] = ()) -> SceneTask:
    return SceneTask(scene_id, category, name, tuple(query.split("|")), composition, auxiliary)


# This is a fixed acquisition contract. Search can select different UIDs within
# these categories, but cannot replace a missing interaction class with another.
SCENES: tuple[SceneTask, ...] = (
    _task("SC01", "simple_control", "toy car", "toy_car|car_(automobile)|toy", "toy car and ball, visibly separated", ("ball|sphere",)),
    _task("SC02", "simple_control", "camera", "camera|camcorder", "camera and closed book, visibly separated", ("book|hardback_book",)),
    _task("SC03", "simple_control", "shoe", "shoe|sandal_(type_of_shoe)", "shoe and bottle, visibly separated", ("bottle|water_bottle",)),
    _task("SC04", "simple_control", "apple", "apple", "apple and banana, visibly separated", ("banana",)),
    _task("SC05", "simple_control", "alarm clock", "alarm_clock|clock", "alarm clock and teacup, visibly separated", ("teacup|cup|mug",)),
    _task("SC06", "simple_control", "bird figurine", "bird|figurine|toy", "bird figurine and spool, visibly separated", ("spool|thread_spool|reel",)),

    _task("CT01", "contact_shadow", "mug", "mug|cup", "mug contacts a finite cloth surface", ("tablecloth|towel|fabric",)),
    _task("CT02", "contact_shadow", "teacup", "teacup|cup", "teacup rests on a saucer", ("saucer|plate|dish",)),
    _task("CT03", "contact_shadow", "teapot", "teapot|teakettle", "teapot rests on a shallow tray", ("tray|serving_tray",)),
    _task("CT04", "contact_shadow", "bottle", "bottle|water_bottle", "bottle contacts a small wood-colored plinth", ()),
    _task("CT05", "contact_shadow", "sculpture", "statue_(sculpture)|sculpture|figurine", "sculpture contacts a shallow light pedestal", ()),
    _task("CT06", "contact_shadow", "toy truck", "toy_car|car_(automobile)|toy", "toy truck tires contact a finite platform", ()),
    _task("CT07", "contact_shadow", "toy building block", "block|brick|building_block|toy", "block contacts a plinth and short upright block", ()),
    _task("CT08", "contact_shadow", "chess piece", "chess_piece|chessman|chessboard|figurine", "chess piece contacts a finite checkerboard plinth", ()),
    _task("CT09", "contact_shadow", "candle holder", "candle_holder|candlestick", "candle holder rests on a shallow tray", ("tray|plate",)),
    _task("CT10", "contact_shadow", "seashell", "seashell|shell", "seashell contacts a small finite platform", ()),
    _task("CT11", "contact_shadow", "lidded jar", "jar|canister|pottery", "lidded jar rests on a closed book", ("book|hardback_book",)),
    _task("CT12", "contact_shadow", "wooden toy part", "wooden_toy|block|brick|toy", "two distinct wooden toy parts are stacked in contact", ("toy|block|brick",)),

    _task("RF01", "reflection_color_spill", "white bowl", "bowl", "white bowl near a red block; red indirect spill lands on the bowl", ("toy|block|brick",)),
    _task("RF02", "reflection_color_spill", "white bust", "bust|statue_(sculpture)|sculpture", "white bust near a blue bottle; blue spill lands on the bust", ("bottle|water_bottle",)),
    _task("RF03", "reflection_color_spill", "white jar", "jar|canister", "white jar near a green book; green spill lands on the jar", ("book|hardback_book",)),
    _task("RF04", "reflection_color_spill", "white pitcher", "pitcher_(vessel_for_liquid)|cream_pitcher", "white pitcher near an orange sphere; orange spill lands on the pitcher", ("ball|sphere",)),
    _task("RF05", "reflection_color_spill", "white animal figurine", "figurine|statue_(sculpture)", "white animal figurine near a magenta block; color spill is visible", ("toy|block|brick",)),
    _task("RF06", "reflection_color_spill", "white cup", "cup|mug|teacup", "white cup near a yellow block; yellow spill is visible", ("toy|block|brick",)),
    _task("RF07", "reflection_color_spill", "metal kettle", "teakettle|kettle|teapot", "metal kettle reflects a red neighboring object", ("toy|block|brick",)),
    _task("RF08", "reflection_color_spill", "metal pot", "pot|cookware", "metal pot reflects a blue cup", ("cup|mug",)),
    _task("RF09", "reflection_color_spill", "metal watering can", "watering_can", "metal watering can reflects a green bottle", ("bottle|water_bottle",)),
    _task("RF10", "reflection_color_spill", "metal bowl", "bowl", "metal bowl reflects a striped neighboring object", ("toy|block|brick",)),
    _task("RF11", "reflection_color_spill", "metal canister", "canister|thermos_bottle", "metal canister reflects a toy car", ("toy_car|car_(automobile)|toy",)),
    _task("RF12", "reflection_color_spill", "metal flask", "thermos_bottle|flask|canteen", "metal flask reflects a high-contrast colored object", ("toy|block|brick",)),

    _task("EN01", "environmental_influence", "slender vase", "vase|flowerpot", "camera-invisible window frame casts directional light on the vase", ()),
    _task("EN02", "environmental_influence", "round vase", "vase|pottery", "camera-invisible divided window creates a distinct lighting pattern", ()),
    _task("EN03", "environmental_influence", "ceramic water jug", "pitcher_(vessel_for_liquid)|jug|pottery", "warm camera-invisible side wall casts indirect color on the jug", ()),
    _task("EN04", "environmental_influence", "metal cookware", "pot|kettle|cookware", "warm camera-invisible wall reflects onto the cookware", ()),
    _task("EN05", "environmental_influence", "metal teapot", "teapot|teakettle", "narrow bright camera-invisible reflector forms a band on the teapot", ()),
    _task("EN06", "environmental_influence", "metal ornament", "ornament|figurine|sculpture", "asymmetric camera-invisible reflectors alter the ornament appearance", ()),

    _task("TR01", "translucent_overlap", "teddy bear", "teddy_bear|bear", "single neutral woven veil covers the teddy bear and extends beyond it"),
    _task("TR02", "translucent_overlap", "rabbit toy", "rabbit|toy", "single neutral woven veil covers the rabbit and extends beyond it"),
    _task("TR03", "translucent_overlap", "robot toy", "robot|toy", "single neutral woven veil covers the robot and extends beyond it"),
    _task("TR04", "translucent_overlap", "horse figurine", "horse|figurine|toy", "two offset neutral woven veils cover the horse and extend beyond it"),
    _task("TR05", "translucent_overlap", "elephant toy", "elephant|toy", "two offset neutral woven veils cover the elephant and extend beyond it"),
    _task("TR06", "translucent_overlap", "doll", "doll|rag_doll", "two offset neutral woven veils cover the doll and extend beyond it"),
    _task("TR07", "translucent_overlap", "toy vehicle", "toy_car|car_(automobile)|toy", "two offset neutral woven veils cover the vehicle and extend beyond it"),
    _task("TR08", "translucent_overlap", "bird toy", "bird|toy", "three offset neutral woven veils cover the bird and extend beyond it"),
    _task("TR09", "translucent_overlap", "owl figurine", "owl|figurine|toy", "three offset neutral woven veils cover the owl and extend beyond it"),
    _task("TR10", "translucent_overlap", "cat figurine", "cat|figurine|toy", "three offset neutral woven veils cover the cat and extend beyond it"),
    _task("TR11", "translucent_overlap", "toy castle", "toy_castle|castle|toy", "single neutral no-refraction film covers the toy castle"),
    _task("TR12", "translucent_overlap", "toy rocket", "toy_rocket|rocket|toy", "two overlapping neutral no-refraction films cover the rocket"),
    _task("TR13", "translucent_overlap", "tripod stand", "tripod|stand|camera", "two overlapping neutral no-refraction films cover the stand"),
    _task("TR14", "translucent_overlap", "display stand", "display_stand|stand|rack", "three overlapping neutral no-refraction films cover the stand"),
    _task("TR15", "translucent_overlap", "toy house", "dollhouse|toy_house|birdhouse", "three overlapping neutral no-refraction films cover the toy house"),

    _task("OC01", "interleaved_occlusion", "round linked chain", "chain|chain_link|necklace", "two original separable chain links interlock with alternating depth", ("ring|chain|necklace",)),
    _task("OC02", "interleaved_occlusion", "oval linked chain", "chain|chain_link|necklace", "distinct oval original chain links interlock with alternating depth", ("ring|chain|necklace",)),
    _task("OC03", "interleaved_occlusion", "rope loop", "rope|cord|string|cable", "existing rope loop interlocks with a dataset wooden ring or branch", ("ring|branch|tree",)),
    _task("OC04", "interleaved_occlusion", "bare branch", "branch|tree|twig", "branch alternates in depth through an existing sparse woven grid"),
    _task("OC05", "interleaved_occlusion", "leafy branch", "branch|plant|tree|twig", "leafy branch and existing grid alternate foreground depth"),
    _task("OC06", "interleaved_occlusion", "fern frond", "fern|plant|leaf", "fern frond and existing fine grid alternate foreground depth"),
    _task("OC07", "interleaved_occlusion", "wire basket", "basket|wire_basket", "wire basket and a dataset branch create multiple alternating overlaps", ("branch|tree|twig",)),
    _task("OC08", "interleaved_occlusion", "folded net object", "net|mesh|fishing_net|basket", "existing folded net and dataset handle alternate foreground depth", ("racket|handle|tool|tripod|camera_stand|stand",)),
    _task("OC09", "interleaved_occlusion", "curved cable assembly", "cable|cord|rope|string|chain", "two original separable cables or closed loops create alternating overlaps", ("cable|cord|rope|string|wire",)),
)


def verify_scene_contract() -> None:
    if len(SCENES) != 60 or len({x.scene_id for x in SCENES}) != 60:
        raise ValueError("expansion suite must contain 60 unique scene IDs")
    got: dict[str, int] = {}
    for scene in SCENES:
        got[scene.category] = got.get(scene.category, 0) + 1
    if got != CLASS_COUNTS:
        raise ValueError(f"scene quotas mismatch: {got}")


def stable_seed(scene_id: str, variant: str = "root") -> int:
    raw = hashlib.sha256(f"{EXPERIMENT_ID}:{scene_id}:{variant}".encode()).digest()
    return int.from_bytes(raw[:4], "big") % 2_147_483_648


def initialize_run(root: Path, *, source_snapshot: dict[str, Any], environment: dict[str, Any]) -> None:
    verify_scene_contract()
    root.mkdir(parents=True, exist_ok=True)
    (root / "recipes").mkdir(exist_ok=True)
    (root / "samples").mkdir(exist_ok=True)
    (root / "selection").mkdir(exist_ok=True)
    (root / "assets").mkdir(exist_ok=True)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": "experiment_1.1",
        "total_download_budget_bytes": TOTAL_DOWNLOAD_BYTES,
        "expected_scenes": len(SCENES),
        "expected_variants_per_scene": list(EXPECTED_VARIANTS),
        "class_counts": CLASS_COUNTS,
        "resolution": 1024,
        "preserve_source_appearance": True,
        "source_snapshot": source_snapshot,
        "environment": environment,
    }
    manifest = root / "scene_plan.jsonl"
    if manifest.exists():
        existing = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(existing) != 60 or [x["scene_id"] for x in existing] != [x.scene_id for x in SCENES]:
            raise RuntimeError(f"Refusing to overwrite incompatible experiment manifest: {manifest}")
    else:
        lines = []
        for scene in SCENES:
            lines.append({
                "experiment_id": EXPERIMENT_ID,
                "scene_id": scene.scene_id,
                "category": scene.category,
                "primary_name": scene.primary_name,
                "primary_search": list(scene.primary_search),
                "composition": scene.composition,
                "auxiliary_search": list(scene.auxiliary_search),
                "status": "planned",
                "stable_seed": stable_seed(scene.scene_id),
            })
        manifest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    config_path = root / "experiment.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"Refusing to overwrite unrelated experiment config: {config_path}")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
