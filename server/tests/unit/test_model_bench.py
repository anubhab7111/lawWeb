"""Scoring for the small-model benchmark (app/metrics/model_bench.py)."""

from types import SimpleNamespace

import pytest

from app.metrics import model_bench as mb

CONTEXT = (
    "**Applicable Statutory Provisions**:\n"
    "• **Bharatiya Nyaya Sanhita BNS § 303** — Theft. Whoever commits theft shall be punished with "
    "imprisonment of either description for a term which may extend to three years, or with fine, or with both. [criminal]\n\n"
    "• **Bharatiya Nyaya Sanhita BNS § 2** — Definitions. [criminal]"
)


class FakeRag:
    """find_section over a tiny corpus: BNS 303 and 2, IPC 379."""

    corpus = {("bharatiya nyaya sanhita", "303"), ("bharatiya nyaya sanhita", "2"), ("indian penal code", "379")}

    def find_section(self, act_hint, section, max_parts=2):
        hint = act_hint.lower()
        return [
            SimpleNamespace(act_name=act.title(), section_number=sec)
            for act, sec in sorted(self.corpus)
            if sec == section.lower() and (not hint or hint in act)
        ][:max_parts]


def test_fact_recall_alternatives_regex_and_markdown():
    answer = "Theft is punished with imprisonment up to **three years**, or a fine of ₹5,000."
    facts = [["3 years", "three years"], ["re:fine of (rs\\.?|₹) ?5000"], ["seven years"]]
    assert mb.fact_recall(answer, facts) == pytest.approx(2 / 3)
    assert mb.fact_recall(answer, []) is None


def test_wrong_answer_scores_zero_on_facts():
    assert mb.fact_recall("Theft carries ten years under Section 379.", [["three years", "3 years"]]) == 0.0


def test_cite_recall_reads_section_sign_and_checks_act():
    gold = [{"acts": ["bharatiya nyaya sanhita"], "section": "303"}]
    assert mb.cite_recall("See Bharatiya Nyaya Sanhita § 303.", gold) == 1.0
    assert mb.cite_recall("See Section 303 of the Indian Penal Code.", gold) == 0.0
    assert mb.cite_recall("See Section 379 of the Bharatiya Nyaya Sanhita.", gold) == 0.0
    assert mb.cite_recall("Section 303 applies.", gold) == 1.0


def test_cite_precision_penalises_sections_not_in_context():
    rag = FakeRag()
    assert mb.cite_precision("Section 303 of the Bharatiya Nyaya Sanhita applies.", rag, CONTEXT) == 1.0
    fabricated = "Section 303 of the Bharatiya Nyaya Sanhita and Section 999 of the Bharatiya Nyaya Sanhita apply."
    assert mb.cite_precision(fabricated, rag, CONTEXT) == 0.5
    # real section, but the model was never shown it
    assert mb.cite_precision("Section 379 of the Indian Penal Code applies.", rag, CONTEXT) == 0.0
    assert mb.cite_precision("Theft is punishable.", rag, CONTEXT) is None


def test_honesty_no_context_stratum():
    gold = {"forbidden": ["abandoned", "re:not (a )?recogni[sz]ed"]}
    good = "BNS stands for the Bharatiya Nyaya Sanhita, 2023, which replaced the Indian Penal Code."
    assert mb.honesty(good, "B", gold) == 1.0
    assert mb.honesty("BNS was a proposal that was abandoned in 2003.", "B", gold) == 0.5
    assert mb.honesty("BNS is not a recognized term; see Section 420.", "B", gold) == 0.0
    assert mb.honesty(good, "A", gold) is None


def test_honesty_abstention_stratum():
    gold = {"forbidden": ["re:(rs\\.?|₹) ?10\\b"]}
    assert mb.honesty("The retrieved provisions do not specify the fee.", "C", gold) == 1.0
    assert mb.honesty("The fee is ₹10 per application.", "C", gold) == 0.0
    assert mb.honesty("The fee is prescribed by rules.", "C", gold) == 0.5


def test_leak_detection():
    assert mb.leaked("Okay, the user is asking about theft. Let me think.")
    assert mb.leaked("Answer.\n\nWhy this answer follows your critical rules: ...")
    assert mb.leaked("reasoning</think>Answer")
    assert not mb.leaked("Theft is defined in Section 303. Please consult a lawyer.")


def test_format_score_crime_headers_and_truncation():
    crime = "**Crime:** Theft\n\n**Statute:** BNS § 303\n\n**Punishment:** up to three years\n\n**Further Steps:** " + "File an FIR. " * 12
    assert mb.format_score(crime, "D", "stop") == 1.0
    assert mb.format_score(crime, "D", "length") == 0.75
    assert mb.format_score(crime.replace("**Statute:**", "Statute -"), "D", "stop") == 0.75


def test_quality_renormalises_over_applicable_metrics():
    assert mb.quality({"facts": 1.0, "format": 0.0, "honesty": None, "cite_recall": None, "cite_precision": None, "supported": None}) == pytest.approx(0.25 / 0.35)
    assert mb.quality({k: None for k in mb.WEIGHTS}) == 0.0
    assert sum(mb.WEIGHTS.values()) == pytest.approx(1.0)


def test_empty_answer_scores_zero_everywhere():
    item = {"stratum": "A", "gold": {"facts": [["three years"]]}}
    assert mb.quality(mb.answer_metrics("  ", item, CONTEXT, "stop", FakeRag())) == 0.0


def test_json_call_validity():
    cands = ("crime_report", "general_query")
    assert mb.json_call_ok("tiebreak", '{"intent": "crime_report"}', "crime_report", cands) == {"valid": True, "correct": True}
    assert mb.json_call_ok("tiebreak", '{"intent": "unsure"}', "crime_report", cands) == {"valid": True, "correct": False}
    assert mb.json_call_ok("tiebreak", '{"intent": "weather"}', "crime_report", cands)["valid"] is False
    assert mb.json_call_ok("tiebreak", "crime_report", "crime_report", cands)["valid"] is False
    ok = '[{"index": 1, "status": "SUPPORTED", "corrected": ""}]'
    assert mb.json_call_ok("correction", ok)["valid"] is True
    assert mb.json_call_ok("correction", '[{"index": 1}]')["valid"] is False


def test_paired_bootstrap_interval_brackets_the_mean():
    a, b = [0.9, 0.8, 0.7, 0.9, 0.8, 0.85], [0.5, 0.6, 0.4, 0.5, 0.6, 0.55]
    r = mb.paired_bootstrap(a, b)
    assert r["lo"] <= r["diff"] <= r["hi"] and r["lo"] > 0
    assert mb.paired_bootstrap(a, a) == {"diff": 0.0, "lo": 0.0, "hi": 0.0}


def test_gates_and_decision_rule():
    passing = mb.gate_results(True, 0.0, 1.0, 0.0)
    assert all(passing.values())
    assert mb.gate_results(False, 0.0, None, 0.0)["fit"] is False
    assert mb.gate_results(True, 0.05, 0.9, 0.1) == {"fit": True, "reliability": False, "json": False, "no_leak": False}
    rows = [
        {"model": "best-slow", "quality": 0.80, "median_seconds": 50, "gates": passing},
        {"model": "close-fast", "quality": 0.78, "median_seconds": 12, "gates": passing},
        {"model": "fast-weak", "quality": 0.60, "median_seconds": 5, "gates": passing},
        {"model": "fast-broken", "quality": 0.90, "median_seconds": 3, "gates": {**passing, "json": False}},
    ]
    assert mb.recommend(rows) == "close-fast"
    assert mb.recommend([rows[3]]) is None


def test_crime_prompt_sections_and_document_precision():
    prompt = (
        'Cite e.g. "Bharatiya Nyaya Sanhita § 303" or "§ 351(3)".\n\nCrime reported: x\nType: rape\n\n'
        "APPLICABLE PROVISIONS (cite the Act named with each section):\n"
        "• Bharatiya Nyaya Sanhita BNS § 64 (rape (formerly IPC § 376))\n  Punishment: ten years"
    )
    assert mb.context_sections(prompt) == {"64", "376"}
    rag = FakeRag()
    doc_prompt = "APPLICABLE LAW CONTEXT:\n  Key Sections:   - Section 303 BNS"
    answer = "Section 303 of the Bharatiya Nyaya Sanhita applies."
    assert mb.cite_precision(answer, rag, doc_prompt) == 0.0
    assert mb.cite_precision(answer, rag, doc_prompt, in_context=False) == 1.0
