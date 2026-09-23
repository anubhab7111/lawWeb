import pytest

from app.ingest.judgment_chunker import (
    ChunkValidationError,
    chunk_judgment,
    split_sections,
    validate_chunk,
)

META = {
    "title": "SONA BALA BORA AND ORS. versus JYOTIRINDRA BHATACHARJEE",
    "judge": "RUMA PAL",
    "citation": "[2005] 3 S.C.R. 454",
    "decision_date": "11-04-2005",
    "disposal_nature": "Appeal(s) allowed",
}


def words(n, word="word"):
    return " ".join([word] * n)


CLEAN = "\n\n".join(
    [
        "SONA BALA BORA AND ORS.",
        "v.",
        "JYOTIRINDRA BHATACHARJEE",
        "APRIL 11, 2005 [RUMA PAL AND C.K. THAKKER, JJ.]",
        "Contract Act, 1872: ss. 11 and 12 - Contract with a person of unsound mind - Held, capacity "
        "must be judged on conduct. Penal Code: s. 302 IPC considered.",
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1234 of 2001.",
        "Ranjit Kumar and Ms. Anita for the Appellant.",
        "The Judgment of the Court was delivered by",
        "RUMA PAL, J. " + words(200) + " Section 302 of the Indian Penal Code applies.",
        words(150, "alpha") + ".",
        words(500, "beta") + ". " + words(100, "gamma") + ".",
        "Accordingly, the appeal is allowed and the impugned order is set aside.",
    ]
)


def test_split_sections_separates_headnote_counsel_and_body():
    parts = split_sections(CLEAN.split("\n\n"))
    assert parts["headnote"][0].startswith("Contract Act, 1872")
    assert len(parts["headnote"]) == 1
    assert parts["body"][0].startswith("RUMA PAL, J.")
    assert not any("for the Appellant" in p for p in parts["body"])


def test_chunks_carry_metadata_roles_and_citations():
    chunks = chunk_judgment("sc-2005_3_454_465", 2005, CLEAN, META)
    assert chunks[0]["role"] == "headnote"
    assert chunks[0]["case_title"] == META["title"]
    assert "IPC:302" in chunks[0]["sections_cited"]
    assert chunks[-1]["role"] == "order"
    assert all(c["doc_id"] == "sc-2005_3_454_465" and c["year"] == 2005 for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_chunk_sizes_respect_the_budget_and_long_paragraphs_are_split_with_overlap():
    chunks = chunk_judgment("sc-x", 2005, CLEAN, META)
    body = [c for c in chunks if c["role"] != "headnote"]
    assert all(c["n_words"] <= 400 for c in chunks)
    assert max(c["n_words"] for c in body) >= 100
    long_para_pieces = [c for c in body if "beta" in c["text"]]
    assert len(long_para_pieces) >= 2  # the 600-word paragraph was split


def test_document_without_markers_is_all_body():
    paragraphs = [words(120, "delta") + "." for _ in range(4)]
    chunks = chunk_judgment("sc-y", 1970, "\n\n".join(paragraphs), {"title": "A v. B"})
    assert chunks and all(c["role"] in ("body", "order") for c in chunks)


def test_validation_rejects_bad_chunks():
    good = chunk_judgment("sc-2005_3_454_465", 2005, CLEAN, META)[0]
    validate_chunk(good)
    for bad in (
        {**good, "text": ""},
        {**good, "role": "weird"},
        {**good, "text": "short text"},
        {**good, "text": good["text"] + "\f"},
        {**good, "doc_id": ""},
    ):
        with pytest.raises(ChunkValidationError):
            validate_chunk(bad)
