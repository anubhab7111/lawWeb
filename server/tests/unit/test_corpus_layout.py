import hashlib
import json

from app.ingest.corpus_layout import init_layout, verify


def _sc_year(root, year, names):
    pdf = root / "judgments" / "sc" / "pdf" / f"year={year}"
    pdf.mkdir(parents=True)
    rows = []
    for n in names:
        data = f"%PDF {n}".encode()
        (pdf / n).write_bytes(data)
        rows.append({"name": n, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = root / "judgments" / "sc" / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)
    (manifest / f"year={year}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_init_tidies_root_strays(tmp_path):
    (tmp_path / "download.sh").write_text("#!/bin/sh")
    (tmp_path / "1950").mkdir()
    (tmp_path / "1951").mkdir()
    (tmp_path / "1951" / "keep.pdf").write_bytes(b"%PDF")
    (tmp_path / "myenv").mkdir()
    actions = init_layout(tmp_path)
    assert (tmp_path / "_legacy" / "download.sh").exists()
    assert actions["removed"] == ["1950"] and actions["kept"] == ["1951"]
    assert (tmp_path / "README.md").exists() and (tmp_path / "myenv").exists()


def test_verify_clean_layout(tmp_path):
    init_layout(tmp_path)
    (tmp_path / "myenv").mkdir()
    (tmp_path / "myenv" / "x").write_text("x")
    for d in ("_legacy", "tools", "builds", "iltur", "indiankanoon", "quarantine"):
        (tmp_path / d / "keep.txt").write_text("x")
    for d in ("statutes", "case_law", "manifest"):
        (tmp_path / d / "keep.txt").write_text("x")
    _sc_year(tmp_path, 2001, ["a.pdf", "b.pdf"])
    assert verify(tmp_path, deep=True) == []


def test_verify_flags_problems(tmp_path):
    init_layout(tmp_path)
    (tmp_path / "stray_dir").mkdir()
    (tmp_path / "statutes" / "bare_acts" / "criminal").mkdir(parents=True)
    _sc_year(tmp_path, 2001, ["a.pdf", "b.pdf"])
    pdf = tmp_path / "judgments" / "sc" / "pdf" / "year=2001"
    (pdf / "b.pdf").unlink()
    (pdf / "extra.pdf").write_bytes(b"%PDF")
    (pdf / "a.pdf").write_bytes(b"%PDF corrupted")
    problems = "\n".join(verify(tmp_path, deep=True))
    assert "unexpected top-level entry: stray_dir" in problems
    assert "missing on disk: b.pdf" in problems
    assert "not in manifest: extra.pdf" in problems
    assert "sha256 mismatch: a.pdf" in problems
    assert "empty directory" in problems
