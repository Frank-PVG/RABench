from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .experiment import EXPERIMENT_ID, TOTAL_DOWNLOAD_BYTES


def _safe_url(value: str | None) -> str | None:
    if not value:
        return value
    parts = urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _safe_error(value: str | None) -> str | None:
    if not value:
        return value
    return re.sub(r"https?://[^\s]+", lambda match: _safe_url(match.group(0)) or "", value)


class ExperimentBudgetExceeded(RuntimeError):
    """The experiment has exhausted (or cannot safely reserve) its byte budget."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BudgetedDownloader:
    """Serial, resumable, byte-metered HTTP body downloader for one experiment.

    An expected-length request reserves its full body before GET. Unknown-length
    requests reserve all remaining bytes. A crash leaves the reservation in the
    database, intentionally failing closed instead of reopening an unmetered gap.
    HEAD response bodies are zero-length and are never used as asset data.
    """

    def __init__(self, db_path: Path, output_root: Path,
                 *, budget_bytes: int = TOTAL_DOWNLOAD_BYTES):
        self.db_path = Path(db_path)
        self.output_root = Path(output_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = int(budget_bytes)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=FULL")
            con.execute("""CREATE TABLE IF NOT EXISTS transfers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                url TEXT NOT NULL,
                final_url TEXT,
                status TEXT NOT NULL,
                reserved_bytes INTEGER NOT NULL DEFAULT 0,
                received_bytes INTEGER NOT NULL DEFAULT 0,
                sha256 TEXT,
                error TEXT,
                started_at REAL NOT NULL,
                completed_at REAL
            )""")

    def _usage(self, con: sqlite3.Connection) -> dict[str, int]:
        row = con.execute("""SELECT COALESCE(SUM(received_bytes),0) AS received,
                    COALESCE(SUM(reserved_bytes),0) AS reserved FROM transfers""").fetchone()
        return {"received_bytes": int(row["received"]), "reserved_bytes": int(row["reserved"]),
                "budget_bytes": self.budget_bytes}

    def status(self) -> dict[str, int]:
        with self._connect() as con:
            return self._usage(con)

    def _head(self, url: str) -> tuple[int | None, str | None]:
        opener = urllib.request.build_opener(_NoRedirect)
        current = url
        for _ in range(8):
            req = urllib.request.Request(current, method="HEAD", headers={"User-Agent": "rgba-pipeline-exp60/1.1"})
            try:
                with opener.open(req, timeout=30) as response:
                    length = response.headers.get("Content-Length")
                    return (int(length) if length else None, response.geturl())
            except urllib.error.HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                    current = urllib.parse.urljoin(current, exc.headers["Location"])
                    continue
                return None, current
            except (urllib.error.URLError, ValueError):
                return None, current
        return None, current

    def fetch(self, url: str, destination: Path, *, purpose: str, asset_id: str,
              max_file_bytes: int, retries: int = 3) -> dict[str, Any]:
        destination = Path(destination)
        try:
            destination.resolve().relative_to(self.output_root.resolve())
        except ValueError as exc:
            raise ValueError(f"download destination must be inside {self.output_root}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if destination.exists() and destination.stat().st_size:
                return {"path": str(destination), "reused": True, "bytes": destination.stat().st_size,
                        "sha256": self._hash(destination)}
            length, head_final = self._head(url)
            if length is not None and length > max_file_bytes:
                raise ValueError(f"selected file exceeds per-file limit ({length} > {max_file_bytes}): {asset_id}")
            if length == 0:
                raise IOError(f"server reports an empty asset: {asset_id}")

            last_error = "unknown network error"
            with self._connect() as con:
                used = int(con.execute("SELECT COUNT(*) FROM transfers WHERE experiment_id=? AND asset_id=? AND url=?",
                                       (EXPERIMENT_ID, asset_id, _safe_url(url))).fetchone()[0])
            remaining_attempts = max(0, int(retries) - used)
            if remaining_attempts == 0:
                raise RuntimeError(f"transfer attempt limit already used ({retries}) for {asset_id}")
            for attempt in range(1, remaining_attempts + 1):
                part = destination.with_name(destination.name + ".part")
                received = 0
                transfer_id: int | None = None
                current_url = url
                try:
                    transfer_id, reserve = self._reserve(purpose, asset_id, url, length, max_file_bytes)
                    received, current_url = self._get_body(
                        transfer_id, current_url, part, reserve, max_file_bytes, expected=length,
                    )
                    actual_length = part.stat().st_size
                    if actual_length <= 0:
                        raise IOError(f"empty download: {asset_id}")
                    if length is not None and actual_length != length:
                        raise IOError(f"Content-Length mismatch for {asset_id}: expected {length}, got {actual_length}")
                    digest = self._hash(part)
                    part.replace(destination)
                    self._finish(transfer_id, received, digest, None, current_url)
                    return {"path": str(destination), "reused": False, "bytes": received,
                            "sha256": digest, "source_url": _safe_url(url), "final_url": _safe_url(current_url),
                            "expected_bytes": length, "attempt": attempt}
                except ExperimentBudgetExceeded:
                    raise
                except Exception as exc:
                    last_error = str(exc)
                    if transfer_id is not None:
                        received = max(received, self._last_received(transfer_id))
                        self._finish(transfer_id, received, None, last_error, current_url)
                    if part.exists():
                        part.unlink()
                    if attempt == retries:
                        break
            raise RuntimeError(f"failed to fetch selected asset {asset_id} after {retries} attempts: {last_error}")

    def _reserve(self, purpose: str, asset_id: str, url: str,
                 expected: int | None, max_file_bytes: int) -> tuple[int, int]:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            usage = self._usage(con)
            remaining = self.budget_bytes - usage["received_bytes"] - usage["reserved_bytes"]
            safe_cap = int(max_file_bytes) + 8 * 64 * 1024
            reserve = (int(expected) + 8 * 64 * 1024 if expected is not None
                       else min(remaining, safe_cap))
            if reserve <= 0 or reserve > remaining:
                con.rollback()
                raise ExperimentBudgetExceeded(
                    f"cannot safely reserve next body; remaining={remaining}, requested={reserve}, "
                    f"limit={self.budget_bytes}; experiment stopped before GET"
                )
            cur = con.execute("""INSERT INTO transfers
                (experiment_id,purpose,asset_id,url,status,reserved_bytes,started_at)
                VALUES (?,?,?,?,?,?,?)""",
                (EXPERIMENT_ID, purpose, asset_id, _safe_url(url), "reserved", reserve, time.time()))
            return int(cur.lastrowid), reserve

    def _get_body(self, transfer_id: int, url: str, part: Path, reserve: int,
                  max_file_bytes: int, *, expected: int | None) -> tuple[int, str]:
        opener = urllib.request.build_opener(_NoRedirect)
        current = url
        received = 0
        file_hash = hashlib.sha256()
        with part.open("wb") as output:
            for _ in range(8):
                req = urllib.request.Request(current, headers={"User-Agent": "rgba-pipeline-exp60/1.1"})
                try:
                    response = opener.open(req, timeout=90)
                except urllib.error.HTTPError as exc:
                    if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                        received += self._account_error_body(transfer_id, exc, received, reserve)
                        current = urllib.parse.urljoin(current, exc.headers["Location"])
                        continue
                    received += self._account_error_body(transfer_id, exc, received, reserve)
                    raise RuntimeError(f"HTTP {exc.code} for {current}") from exc
                with response:
                    current = response.geturl()
                    advertised = response.headers.get("Content-Length")
                    advertised_len = int(advertised) if advertised else None
                    if advertised_len is not None and advertised_len > max_file_bytes:
                        raise ValueError(f"response exceeds per-file limit ({advertised_len} > {max_file_bytes})")
                    if expected is not None and advertised_len is not None and advertised_len != expected:
                        raise IOError(f"redirected response length mismatch: HEAD={expected}, GET={advertised_len}")
                    if advertised_len is not None and advertised_len > reserve:
                        raise ExperimentBudgetExceeded("GET response exceeds its pre-reserved byte length")
                    while True:
                        remaining = reserve - received
                        if remaining <= 0:
                            # Stop at the exact reserved byte count. No extra read
                            # is issued to probe for EOF, so an unknown body cannot
                            # push the aggregate beyond its hard ceiling.
                            raise ExperimentBudgetExceeded(
                                "unknown-length response exhausted experiment budget reservation; GET aborted"
                            )
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            self._progress(transfer_id, received, current)
                            return received, current
                        output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                        received += len(chunk)
                        self._progress(transfer_id, received, current)
                        if received > max_file_bytes:
                            raise ValueError(f"response exceeds per-file limit ({max_file_bytes}): {current}")
                        file_hash.update(chunk)
            raise RuntimeError("too many HTTP redirects")

    def _account_error_body(self, transfer_id: int, response: Any, received: int,
                            reserve: int) -> int:
        start = received
        while True:
            remaining = reserve - received
            if remaining <= 0:
                raise ExperimentBudgetExceeded("HTTP error body reached reserved download ceiling")
            chunk = response.read(min(1024 * 1024, remaining))
            if not chunk:
                return received - start
            received += len(chunk)
            self._progress(transfer_id, received, response.geturl())
        return received

    def _progress(self, transfer_id: int, received: int, final_url: str) -> None:
        with self._connect() as con:
            con.execute("UPDATE transfers SET received_bytes=?, final_url=? WHERE id=?",
                        (received, _safe_url(final_url), transfer_id))

    def _last_received(self, transfer_id: int) -> int:
        with self._connect() as con:
            row = con.execute("SELECT received_bytes FROM transfers WHERE id=?", (transfer_id,)).fetchone()
            return int(row[0]) if row else 0

    def _finish(self, transfer_id: int, received: int, digest: str | None,
                error: str | None, final_url: str | None) -> None:
        with self._connect() as con:
            con.execute("""UPDATE transfers SET status=?,reserved_bytes=0,
                received_bytes=?,sha256=?,error=?,final_url=?,completed_at=? WHERE id=?""",
                ("complete" if error is None else "failed", received, digest, _safe_error(error),
                 _safe_url(final_url), time.time(), transfer_id))

    @staticmethod
    def _hash(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def export_jsonl(self, path: Path) -> None:
        with self._connect() as con:
            rows = [dict(row) for row in con.execute("SELECT * FROM transfers ORDER BY id")]
        Path(path).write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
