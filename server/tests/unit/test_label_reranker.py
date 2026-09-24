from app.tools import label_reranker as lr


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def test_statute_fixture_has_all_labels_with_text():
    s = lr.statutes()
    assert len(s) == 100 and all(s.values())
    assert lr.statute_text("Section 302").startswith("Section 302. Punishment for murder")


def test_fact_segments_cover_the_whole_narrative_evenly():
    segs = lr.fact_segments(FakeTokenizer(), " ".join(["w"] * 1000), max_segments=3, seg_tokens=100)
    assert len(segs) == 3
    assert segs[0].startswith("t0 ") and segs[-1].endswith("t999")


def test_short_facts_are_one_segment():
    assert lr.fact_segments(FakeTokenizer(), "a b c", max_segments=3, seg_tokens=100) == ["t0 t1 t2"]
