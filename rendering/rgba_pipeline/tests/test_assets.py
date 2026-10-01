from rgba_pipeline.assets import Candidate


def test_candidate_is_explicit_and_provenance_rich():
    c = Candidate("asset_a", "abc", "https://example.test/a.glb", "CC-BY-4.0", "https://example.test/a")
    assert c.uid == "abc"
