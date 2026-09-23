import numpy as np

from app.ingest.decontaminate import (
    SHINGLE_WORDS,
    TestShingles,
    contaminated_cases,
    shingle_hashes,
    tokenize,
)

FACTS = (
    "the prosecution case in brief is that the deceased was married to the appellant five years "
    "ago and after marriage she was treated with cruelty by her husband and in laws over demand "
    "of dowry and for this reason she was found dead in the matrimonial home on the night of the incident"
)


def test_entity_masks_break_shingles_instead_of_matching_them():
    tokens = tokenize("the <ENTITY> was married to <ENTITY> five years ago in the village")
    assert "\x00" in tokens
    hashes = shingle_hashes(tokens, n=3)
    assert len(hashes) == len([i for i in range(len(tokens) - 2) if "\x00" not in tokens[i : i + 3]])


def test_verbatim_overlap_is_flagged_and_boilerplate_is_not():
    index = TestShingles.from_texts({"case-1": FACTS, "case-2": "unrelated facts about a property dispute over land " * 10})
    judgment = "Some judgment preface. " + FACTS + " The court then considered the appeal at length."
    assert contaminated_cases(index, tokenize(judgment), min_shingles=10) == {"case-1": len(tokenize(FACTS)) - SHINGLE_WORDS + 1}
    boilerplate = "it is well settled that the burden of proof lies on the prosecution and the appeal is dismissed"
    assert contaminated_cases(index, tokenize(boilerplate), min_shingles=10) == {}


def test_a_few_shared_shingles_stay_below_the_threshold():
    index = TestShingles.from_texts({"case-1": FACTS})
    partial = " ".join(FACTS.split()[:14]) + " and then something else entirely different follows here for a while"
    hits = contaminated_cases(index, tokenize(partial), min_shingles=1)
    assert hits.get("case-1", 0) < 10
    assert contaminated_cases(index, tokenize(partial), min_shingles=10) == {}


def test_index_round_trips_through_disk(tmp_path):
    index = TestShingles.from_texts({"case-1": FACTS})
    index.save(tmp_path / "shingles.npz")
    loaded = TestShingles.load(tmp_path / "shingles.npz")
    assert np.array_equal(loaded.hashes, index.hashes)
    assert loaded.case_ids == ["case-1"]
