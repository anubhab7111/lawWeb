import gzip
import json

import pandas as pd
import pytest

from app.config import get_settings
from app.ingest.process_judgments import dedupe_across_years, keep_year


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("LAWWEB_CORPUS_ROOT", str(tmp_path))
    get_settings.cache_clear()
    root = tmp_path / "judgments" / "sc"
    yield root
    monkeypatch.undo()
    get_settings.cache_clear()


def _year(root, year, stems):
    chunks = root / "chunks" / f"year={year}"
    text = root / "text" / f"year={year}"
    chunks.mkdir(parents=True)
    text.mkdir(parents=True)
    with gzip.open(chunks / "chunks.jsonl.gz", "wt") as f:
        for stem in stems:
            for i in range(2):
                f.write(json.dumps({"doc_id": f"sc-{stem}", "chunk_id": f"sc-{stem}#{i:03d}", "year": year}) + "\n")
    (chunks / "_DONE.json").write_text(json.dumps({"year": year, "chunks": 2 * len(stems)}))
    pd.DataFrame({"stem": stems, "year": year, "status": "ok", "reason": None}).to_parquet(
        text / "_extraction.parquet", index=False
    )


def _doc_ids(root, year):
    with gzip.open(root / "chunks" / f"year={year}" / "chunks.jsonl.gz", "rt") as f:
        return {json.loads(line)["doc_id"] for line in f}


def test_keep_year_prefers_the_report_year_in_the_stem():
    assert keep_year("1951_1_1_51", [1950, 1951]) == 1951
    assert keep_year("S_1996_2_866_868", [1995, 1996]) == 1996
    assert keep_year("2001_5_10_20", [2002, 2003]) == 2002  # no folder matches: earliest


def test_duplicates_are_dropped_from_the_non_kept_year_and_recorded(corpus):
    _year(corpus, 1950, ["1951_1_1_51", "1950_1_5_9"])
    _year(corpus, 1951, ["1951_1_1_51", "1951_1_60_70"])
    result = dedupe_across_years([1950, 1951])
    assert result["dropped_docs"] == 1
    assert _doc_ids(corpus, 1950) == {"sc-1950_1_5_9"}
    assert _doc_ids(corpus, 1951) == {"sc-1951_1_1_51", "sc-1951_1_60_70"}
    ledger = [json.loads(l) for l in (corpus.parent.parent / "manifest" / "sc_cross_year_duplicates.jsonl").open()]
    assert ledger == [{"doc_id": "sc-1951_1_1_51", "kept_year": 1951, "dropped_year": 1950}]
    ext = pd.read_parquet(corpus / "text" / "year=1950" / "_extraction.parquet").set_index("stem")
    assert ext.loc["1951_1_1_51", "status"] == "duplicate"
    assert json.loads((corpus / "chunks" / "year=1950" / "_DONE.json").read_text())["chunks"] == 2


def test_dedupe_is_idempotent(corpus):
    _year(corpus, 1950, ["1951_1_1_51"])
    _year(corpus, 1951, ["1951_1_1_51"])
    dedupe_across_years([1950, 1951])
    again = dedupe_across_years([1950, 1951])
    assert again["dropped_docs"] == 0
    assert _doc_ids(corpus, 1951) == {"sc-1951_1_1_51"}
