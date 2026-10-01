from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rgba_pipeline.budget import BudgetedDownloader, ExperimentBudgetExceeded, _safe_url
from rgba_pipeline.experiment import CLASS_COUNTS, SCENES, TOTAL_DOWNLOAD_BYTES, verify_scene_contract
from rgba_pipeline.expansion import _recipe
from rgba_pipeline.recipes import validate_recipe


class FixtureHandler(BaseHTTPRequestHandler):
    payload = b"selected objaverse metadata or asset" * 16
    error_body = b"not found" * 3

    def log_message(self, fmt, *args):
        pass

    def do_HEAD(self):
        if self.path in {"/known", "/redirect", "/fail"}:
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/known")
                self.end_headers()
                return
            data = self.error_body if self.path == "/fail" else self.payload
            self.send_response(404 if self.path == "/fail" else 200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        if self.path == "/unknown":
            self.send_response(200)
            self.end_headers()
            return
        self.send_error(404)

    def do_GET(self):
        if self.path == "/redirect":
            body = b"redirect body counted"
            self.send_response(302)
            self.send_header("Location", "/known")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/unknown":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.payload)
            return
        if self.path == "/fail":
            self.send_response(404)
            self.send_header("Content-Length", str(len(self.error_body)))
            self.end_headers()
            self.wfile.write(self.error_body)
            return
        if self.path == "/known":
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        self.send_error(404)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        thread.join()
        srv.server_close()


def test_expansion_scene_manifest_has_exact_quotas_and_unique_roots():
    verify_scene_contract()
    assert len(SCENES) == 60
    counts = {}
    for scene in SCENES:
        counts[scene.category] = counts.get(scene.category, 0) + 1
    assert counts == CLASS_COUNTS
    assert len({scene.scene_id for scene in SCENES}) == 60
    assert TOTAL_DOWNLOAD_BYTES == 107_374_182_400


def test_budgeted_download_counts_selected_asset_and_redirect_body(tmp_path: Path, server: str):
    output = tmp_path / "downloads"
    meter = BudgetedDownloader(tmp_path / "ledger.sqlite", output, budget_bytes=2 * 1024 * 1024)
    result = meter.fetch(server + "/redirect", output / "picked.glb", purpose="model",
                         asset_id="scene:S01", max_file_bytes=1024 * 1024)
    assert Path(result["path"]).read_bytes() == FixtureHandler.payload
    assert result["bytes"] == len(FixtureHandler.payload) + len(b"redirect body counted")
    status = meter.status()
    assert status["received_bytes"] == result["bytes"]
    assert status["reserved_bytes"] == 0


def test_unknown_length_is_streamed_with_a_pre_reserved_limit(tmp_path: Path, server: str):
    output = tmp_path / "downloads"
    meter = BudgetedDownloader(tmp_path / "ledger.sqlite", output, budget_bytes=1024 * 1024)
    result = meter.fetch(server + "/unknown", output / "meta.json", purpose="metadata",
                         asset_id="shard:test", max_file_bytes=1024 * 1024)
    assert Path(result["path"]).read_bytes() == FixtureHandler.payload
    assert meter.status()["received_bytes"] == len(FixtureHandler.payload)


def test_failed_transfer_is_still_charged_and_retry_cannot_reset_ledger(tmp_path: Path, server: str):
    output = tmp_path / "downloads"
    meter = BudgetedDownloader(tmp_path / "ledger.sqlite", output, budget_bytes=2 * 1024 * 1024)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        meter.fetch(server + "/fail", output / "missing.glb", purpose="model",
                    asset_id="failed:selected", max_file_bytes=1024 * 1024, retries=2)
    assert meter.status()["received_bytes"] == 2 * len(FixtureHandler.error_body)
    assert meter.status()["reserved_bytes"] == 0
    resumed = BudgetedDownloader(tmp_path / "ledger.sqlite", output, budget_bytes=2 * 1024 * 1024)
    with pytest.raises(RuntimeError, match="attempt limit already used"):
        resumed.fetch(server + "/fail", output / "still-missing.glb", purpose="model",
                      asset_id="failed:selected", max_file_bytes=1024 * 1024, retries=2)
    assert resumed.status()["received_bytes"] == 2 * len(FixtureHandler.error_body)


def test_unknown_response_cannot_cross_remaining_budget(tmp_path: Path, server: str):
    output = tmp_path / "downloads"
    meter = BudgetedDownloader(tmp_path / "ledger.sqlite", output, budget_bytes=32)
    with pytest.raises(ExperimentBudgetExceeded):
        meter.fetch(server + "/unknown", output / "too-big", purpose="metadata",
                    asset_id="oversize:selected", max_file_bytes=1024)
    assert meter.status()["received_bytes"] <= 32


def test_download_ledger_removes_temporary_url_query_parameters():
    assert _safe_url("https://cdn.example/model.glb?token=temporary&expires=123") == "https://cdn.example/model.glb"


def test_schema_11_four_combinations_share_one_scene_content_identity():
    task = next(item for item in SCENES if item.scene_id == "SC01")
    selected = {"SC01": {"uid": "unique-scene-asset", "local_path": "/assets/car.glb",
                          "name": "toy car", "lineage_id": "objaverse:unique-scene-asset",
                          "sha256": "a" * 64, "glb_size": 1024}}
    combinations = [_recipe(task, selected["SC01"], selected, camera, light)
                    for light in (1, 2) for camera in (1, 2)]
    assert len({x["identity"]["scene_root_id"] for x in combinations}) == 1
    assert len({x["identity"]["scene_content_hash"] for x in combinations}) == 1
    assert len({x["identity"]["recipe_hash"] for x in combinations}) == 4
    assert all(x["schema_version"] == "1.1" and not validate_recipe(x) for x in combinations)
    assert all(x["objects"][0]["material_policy"] == "preserve_source" for x in combinations)
