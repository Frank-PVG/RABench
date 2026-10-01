from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from rgba_pipeline.effects import _erode, _intervention_recipe, _polygon_mask, _project_cover_polygon
from rgba_pipeline.io import canonical_hash


def base_recipe(category: str, *, film: bool = False) -> dict:
    cover = {
        "id": "cover_1", "role": "procedural",
        "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
        "geometry": {"type": "thin_film", "width": .3, "height": .3, "thickness": .0002},
        "material": {"preset": "neutral_thin_coverage", "coverage": .2},
    } if film else {
        "id": "cover_1", "role": "procedural",
        "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
        "geometry": {"type": "veil", "width": .3, "height": .3, "coverage": .35},
        "material": {"preset": "opaque_principled"},
    }
    objects = [
        {"id": "subject", "role": "asset", "material_policy": "preserve_source",
         "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
         "material": {"preset": "opaque_principled"}, "surface_overrides": {}},
        {"id": "companion", "role": "procedural", "transform": {"location": [.3, 0, 0]},
         "material": {"preset": "opaque_principled", "base_color": [.5, .5, .5, 1]}},
        {"id": "support", "role": "procedural", "geometry": {"type": "rounded_box", "size": [.3, .3, .02]},
         "transform": {"location": [0, 0, 0]}, "material": {"preset": "opaque_principled"}},
        {"id": "neighbor", "role": "procedural", "transform": {"location": [.2, 0, 0]},
         "material": {"preset": "opaque_principled", "base_color": [1, 0, 0, 1]}},
        {"id": "panel", "role": "procedural", "transform": {"location": [0, .2, 0]},
         "material": {"preset": "opaque_principled"}},
        cover,
    ]
    return {
        "identity": {"sample_id": "T_L1_C1", "scene_root_id": "T", "category": category,
                     "recipe_hash": "parent-hash"},
        "objects": objects,
        "sets": {"TARGET": ["subject", "companion", "support", "neighbor", "cover_1"], "ENVIRONMENT": ["panel"]},
        "relations": [{"type": "contact", "subject": "subject", "support": "support", "tolerance_m": .002}],
        "diagnostics": {"independent_order": ["subject", "companion", "support", "neighbor", "cover_1"],
                        "intervention": {"object": "subject", "parameter": "material.roughness", "value": .95}},
    }


@pytest.mark.parametrize("category,kind,object_id", [
    ("simple_control", "color", "companion"),
    ("contact_shadow", "translate", "subject"),
    ("reflection_color_spill", "neutral_color", "neighbor"),
    ("environmental_influence", "remove_environment", "environment"),
    ("translucent_overlap", "coverage", "cover_1"),
    ("interleaved_occlusion", "translate", "subject"),
])
def test_intervention_is_category_specific_and_hash_linked(category, kind, object_id):
    recipe = base_recipe(category)
    derived = _intervention_recipe(recipe)
    directive = derived["diagnostics"]["intervention"]
    assert directive["kind"] == kind
    assert directive["object"] == object_id
    assert derived["identity"]["parent_recipe_hash"] == recipe["identity"]["recipe_hash"]
    unhashed = deepcopy(derived)
    unhashed["identity"].pop("recipe_hash")
    assert derived["identity"]["recipe_hash"] == canonical_hash(unhashed)
    assert derived["identity"]["recipe_hash"] != recipe["identity"]["recipe_hash"]


def test_thin_film_intervention_increases_per_face_mix():
    derived = _intervention_recipe(base_recipe("translucent_overlap", film=True))
    assert derived["diagnostics"]["intervention"]["value"] == pytest.approx(.35)


def test_projected_cover_envelope_is_centered_and_erodes_by_requested_pixels():
    cover = {"geometry": {"type": "veil", "width": .3, "height": .3},
             "transform": {"location": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}}
    camera = {"location": [0, -1, .4], "look_at": [0, 0, 0], "sensor_width_mm": 36, "lens_mm": 50}
    polygon = _project_cover_polygon(cover, camera, 64, 64)
    mask = _polygon_mask([polygon], (64, 64))
    eroded = _erode(mask, 2)
    assert mask[32, 32]
    assert eroded.sum() < mask.sum()
    assert eroded.sum() > 0
