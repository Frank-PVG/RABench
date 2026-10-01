import json
from pathlib import Path

from rgba_pipeline.render import variant_output_complete
from rgba_pipeline.recipes import contact


def test_independent_resume_requires_every_semantic_layer(tmp_path: Path):
    recipe = contact("C01", {"asset_id": "a", "uid": "u", "prepared_path": "/tmp/a.glb", "lineage_id": "objaverse:u"}, 1, 1)
    root = tmp_path / "sample"
    for object_id in recipe["diagnostics"]["independent_order"]:
        layer = root / "variants" / "independent" / "independent" / object_id
        layer.mkdir(parents=True)
        for name in ("linear_premult.exr", "alpha.exr", "straight_rgba.png"):
            (layer / name).write_bytes(b"complete")
    assert variant_output_complete(root, recipe, "independent", None)
    (root / "variants" / "independent" / "independent" / "subject" / "alpha.exr").unlink()
    assert not variant_output_complete(root, recipe, "independent", None)


def test_recipe_hash_is_stable_for_same_explicit_values():
    a = contact("C01", {"asset_id": "a", "uid": "u", "prepared_path": "/tmp/a.glb", "lineage_id": "objaverse:u"}, 1, 1)
    b = json.loads(json.dumps(a))
    from rgba_pipeline.io import canonical_hash
    assert canonical_hash(a) == canonical_hash(b)
