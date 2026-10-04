from app.tools.unified_rerank import Candidate, rerank, segments


def keyword_predict(word):
    """Fake cross-encoder: relevance = count of `word` in the text (as a probability)."""
    def predict(pairs):
        return [min(1.0, 0.05 + 0.3 * text.lower().count(word)) for _q, text in pairs]
    return predict


def test_segments_split_long_text_without_losing_the_tail():
    text = "\n\n".join(f"paragraph {i} " + "x" * 500 for i in range(10))
    segs = segments(text, size=1400, limit=4)
    assert len(segs) == 4
    assert segs[0].startswith("paragraph 0") and "paragraph 9" in segs[-1]
    assert segments("short") == ["short"] and segments("  ") == []


def test_all_sources_are_ranked_on_one_scale():
    cands = [
        Candidate("statute", "ipc302", "punishment for murder murder"),
        Candidate("judgment", "sc-1#3", "the appellant was convicted of murder"),
        Candidate("case_law", "bachan", "sentencing principles, rarest of rare"),
    ]
    out = rerank("murder", cands, keyword_predict("murder"), top_k=5, min_relative=0.0)
    assert [c.key for c in out] == ["ipc302", "sc-1#3", "bachan"]
    assert out[0].score == 1.0 and 0 < out[-1].score < out[1].score


def test_relevant_text_late_in_a_long_passage_is_found():
    long_text = ("background facts about land records. " * 60) + "\n\nthe accused committed murder."
    cands = [Candidate("judgment", "long", long_text), Candidate("judgment", "other", "land records only")]
    out = rerank("murder", cands, keyword_predict("murder"), min_relative=0.5)
    assert out[0].key == "long"


def test_weak_items_are_dropped_and_sources_capped():
    cands = [Candidate("judgment", f"j{i}", "murder murder murder") for i in range(5)]
    cands.append(Candidate("statute", "s", "murder murder"))
    cands.append(Candidate("statute", "noise", "unrelated"))
    out = rerank("murder", cands, keyword_predict("murder"), top_k=10, min_relative=0.3,
                 per_source_cap={"judgment": 2})
    keys = [c.key for c in out]
    assert sum(k.startswith("j") for k in keys) == 2
    assert "s" in keys and "noise" not in keys


def test_raw_logits_are_normalised():
    cands = [Candidate("statute", "a", "x"), Candidate("statute", "b", "y")]
    out = rerank("q", cands, lambda pairs: [3.0, -1.0], min_relative=0.0)
    assert out[0].key == "a" and 0 < out[1].score < 1


def test_one_passage_per_judgment():
    cands = [
        Candidate("judgment", "sc-1#2", "murder murder murder", group="sc-1"),
        Candidate("judgment", "sc-1#5", "murder murder", group="sc-1"),
        Candidate("judgment", "sc-2#1", "murder", group="sc-2"),
    ]
    out = rerank("murder", cands, keyword_predict("murder"), top_k=5, min_relative=0.0)
    assert [c.key for c in out] == ["sc-1#2", "sc-2#1"]
