from rgba_pipeline.runner import _group_rows_by_scene


def test_group_rows_by_scene_keeps_scene_and_combination_order():
    rows = [
        {"scene_id": "SC01", "sample_id": "SC01_L1_C1"},
        {"scene_id": "SC01", "sample_id": "SC01_L1_C2"},
        {"scene_id": "SC02", "sample_id": "SC02_L1_C1"},
        {"scene_id": "SC01", "sample_id": "SC01_L2_C1"},
    ]
    grouped = _group_rows_by_scene(rows)
    assert [x["scene_id"] for x in grouped] == ["SC01", "SC02"]
    assert [x["sample_id"] for x in grouped[0]["rows"]] == ["SC01_L1_C1", "SC01_L1_C2", "SC01_L2_C1"]
