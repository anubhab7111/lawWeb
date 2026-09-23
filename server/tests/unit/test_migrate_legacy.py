import json

from app.ingest.migrate_legacy import migrate


def _legacy(tmp_path):
    src = tmp_path / "data"
    (src / "bare_acts" / "criminal").mkdir(parents=True)
    (src / "bare_acts" / "criminal" / "IPC.pdf").write_bytes(b"%PDF ipc")
    (src / "mappings").mkdir()
    (src / "mappings" / "table.pdf").write_bytes(b"%PDF map")
    (src / "case_law").mkdir()
    (src / "case_law" / "kesavananda.json").write_text("{}")
    (src / "case_law" / "ignored.txt").write_text("x")
    (src / "faiss_index").mkdir()
    (src / "faiss_index" / "keep.bin").write_bytes(b"idx")
    return src


def test_migrate_preserves_relative_paths_and_verifies(tmp_path):
    src, dst = _legacy(tmp_path), tmp_path / "drive"
    dst.mkdir()
    result = migrate(src, dst)
    assert result["files"] == 3 and not result["failures"]
    assert (dst / "statutes" / "bare_acts" / "criminal" / "IPC.pdf").read_bytes() == b"%PDF ipc"
    assert (dst / "statutes" / "mappings" / "table.pdf").exists()
    assert (dst / "case_law" / "curated" / "kesavananda.json").exists()
    rows = [json.loads(line) for line in (dst / "manifest" / "legacy_migration.jsonl").open()]
    assert all(r["verified"] and len(r["sha256"]) == 64 for r in rows)
    assert (src / "bare_acts" / "criminal" / "IPC.pdf").exists()  # copy, not move


def test_delete_source_removes_only_migrated_files(tmp_path):
    src, dst = _legacy(tmp_path), tmp_path / "drive"
    dst.mkdir()
    result = migrate(src, dst, delete_source=True)
    assert result["removed"] == 3
    assert not (src / "bare_acts" / "criminal" / "IPC.pdf").exists()
    assert (src / "faiss_index" / "keep.bin").exists()
    assert (src / "case_law" / "ignored.txt").exists()


def test_rerun_is_idempotent(tmp_path):
    src, dst = _legacy(tmp_path), tmp_path / "drive"
    dst.mkdir()
    migrate(src, dst)
    again = migrate(src, dst)
    assert again["files"] == 3 and not again["failures"]
