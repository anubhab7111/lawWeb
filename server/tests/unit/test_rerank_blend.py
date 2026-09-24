from app.tools.base_legal_rag import LegalChunk, _rerank_text, blend_rerank_order


def test_pure_cross_encoder_order_at_weight_one():
    scored = [("a", 0.2), ("b", 1.0), ("c", 0.5)]
    assert [c for c, _ in blend_rerank_order(scored, ["a", "b", "c"], 1.0)] == ["b", "c", "a"]


def test_fused_order_at_weight_zero():
    scored = [("a", 0.2), ("b", 1.0), ("c", 0.5)]
    assert [c for c, _ in blend_rerank_order(scored, ["a", "c", "b"], 0.0)] == ["a", "c", "b"]


def test_blend_keeps_a_candidate_both_retrievers_agreed_on():
    # fused rank 1, cross-encoder rank 3 vs a candidate only the cross-encoder likes
    scored = [("agreed", 0.6), ("ce_only", 1.0), ("x", 0.8)]
    order = [c for c, _ in blend_rerank_order(scored, ["agreed", "x", "ce_only"], 0.5)]
    assert order[0] in ("agreed", "x") and order.index("agreed") < 2


def test_scores_are_preserved_for_filtering():
    scored = [("a", 0.2), ("b", 1.0)]
    assert dict(blend_rerank_order(scored, ["a", "b"], 0.5)) == {"a": 0.2, "b": 1.0}


def test_rerank_text_carries_act_and_section():
    c = LegalChunk("id", "criminal", "Indian Penal Code", "302", "Punishment for murder", "Whoever commits murder...", "ipc.pdf")
    assert _rerank_text(c).startswith("Indian Penal Code, Section 302: Punishment for murder. Whoever")
    art = LegalChunk("id", "constitutional", "Constitution of India", "Article 21", "Protection of life", "No person...", "c.pdf")
    assert _rerank_text(art).startswith("Constitution of India, Article 21:")
