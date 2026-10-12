import gzip

from import_case_law import _slug, record_from_file

ENTRY = {
    "case_name": "Fateh Chand v. Balkishan Dass (1963)",
    "doctrines": ["damages for breach"],
    "statutes_cited": [["Indian Contract", "74"]],
}


def _record(path, **overrides):
    args = dict(
        title="Fateh Chand vs Balkishan Das on 15 January, 1963",
        citation="[1964] 1 S.C.R. 515",
        court="Supreme Court of India",
        date="1963-01-15",
        bench_size=5,
        url="",
    )
    return record_from_file(ENTRY, path, **{**args, **overrides})


def test_record_from_gzipped_official_text(tmp_path):
    path = tmp_path / "1964_1_515_534.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(
            "1 S.C.R.\n\n\n\nFATEH CHAND v. BALKlSHAN DAS\n\nContract-Compensation for breach"
        )
    record = _record(path)
    assert record["text"].startswith(
        "Fateh Chand vs Balkishan Das on 15 January, 1963\n\nEquivalent citations: [1964]"
    )
    assert "\n\n\n" not in record["text"]
    assert record["doctrines"] == ["damages for breach"]
    assert record["statutes_cited"] == [["Indian Contract", "74"]]
    assert record["court_rank"] == 4


def test_court_rank_and_text_cap(tmp_path):
    path = tmp_path / "judgment.txt"
    path.write_text("x" * 100000, encoding="utf-8")
    assert _record(path, bench_size=2)["court_rank"] == 3
    assert _record(path, court="Delhi High Court")["court_rank"] == 2
    assert len(_record(path)["text"]) == 60000


def test_slug_matches_existing_record_names():
    assert (
        _slug("Percept D'Mark v. Zaheer Khan (2006)")
        == "percept_d_mark_v_zaheer_khan_2006"
    )
