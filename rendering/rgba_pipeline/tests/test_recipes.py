from rgba_pipeline.recipes import contact, validate_recipe


def test_recipe_has_explicit_sets_and_is_valid():
    asset = {"asset_id": "asset_a", "uid": "uid", "prepared_path": "/tmp/a.glb", "lineage_id": "objaverse:uid"}
    recipe = contact("C01", asset, 1, 1)
    assert validate_recipe(recipe) == []
    assert recipe["sets"]["TARGET"] == ["subject", "pedestal"]
    assert recipe["background_protocol"]["mode"] == "core"
