from fetch_case_law import _pick_best_result, _search_queries, suspect_reason

BHAJAN_LAL = "State of Haryana v. Bhajan Lal (1992)"


def doc(title, court, date, tid=1):
    return {"title": title, "docsource": court, "publishdate": date, "tid": tid}


def test_same_titled_judgment_from_another_year_is_rejected():
    docs = [doc("Bhajan Lal vs State Of Haryana", "Punjab-Haryana High Court", "2009-05-14")]
    assert _pick_best_result(BHAJAN_LAL, docs) is None


def test_supreme_court_judgment_of_the_cited_year_wins():
    docs = [
        doc("Bhajan Lal vs State Of Haryana", "Punjab-Haryana High Court", "1992-03-01", tid=1),
        doc("State Of Haryana And Others vs Bhajan Lal And Others", "Supreme Court of India", "1990-11-21", tid=2),
    ]
    assert _pick_best_result(BHAJAN_LAL, docs)["tid"] == 2


def test_searches_supreme_court_first_within_the_cited_year():
    sc, any_court = _search_queries(BHAJAN_LAL)
    assert "doctypes: supremecourt" in sc and "fromdate: 1-1-1990 todate: 31-12-1993" in sc
    assert "doctypes" not in any_court and "fromdate: 1-1-1990" in any_court


def test_audit_flags_wrong_year_and_non_supreme_court_records():
    assert "2009" in suspect_reason({"case_name": BHAJAN_LAL, "date": "2009-05-14", "court": "Supreme Court of India"})
    assert "High Court" in suspect_reason({"case_name": "X v. Y (2022)", "date": "2023-08-31", "court": "Karnataka High Court"})
    assert suspect_reason({"case_name": BHAJAN_LAL, "date": "1990-11-21", "court": "Supreme Court of India"}) is None


def test_supreme_court_document_with_other_parties_is_rejected():
    docs = [doc("Beghar Foundation vs Justice K.S.Puttaswamy(Retd)", "Supreme Court of India", "2021-01-11")]
    assert _pick_best_result("S.G. Vombatkere v. Union of India (2022)", docs) is None


def test_abbreviated_party_names_still_match():
    docs = [doc("M/S. Kailash Nath Associates vs Delhi Development Authority & Anr", "Supreme Court of India", "2015-01-09")]
    assert _pick_best_result("Kailash Nath Associates v. DDA (2015)", docs) is not None


def test_procedural_orders_lose_to_the_judgment():
    docs = [
        doc("Rit Foundation vs The Union Of India", "Delhi High Court - Orders", "2022-03-02", tid=1),
        doc("Rit Foundation vs The Union Of India", "Delhi High Court", "2022-05-11", tid=2),
    ]
    assert _pick_best_result("RIT Foundation v. Union of India (2022)", docs)["tid"] == 2


def test_exact_year_supreme_court_listing_beats_another_case_between_the_parties():
    docs = [
        doc("Dr.<b>Subramanian</b> <b>Swamy</b> vs Director, Cbi & Anr", "Supreme Court of India", "2014-05-06", tid=1),
        doc("<b>Subramanian</b> <b>Swamy</b> vs Union Of India, Min. Of Law", "Supreme Court - Daily Orders", "2016-05-13", tid=2),
        doc("Subramanian Swamy vs Union Of India", "Supreme Court - Daily Orders", "2015-07-02", tid=3),
    ]
    assert _pick_best_result("Subramanian Swamy v. Union of India (2016)", docs)["tid"] == 2


def test_high_court_orders_alone_give_no_match():
    docs = [doc("Rit Foundation vs The Union Of India", "Delhi High Court - Orders", "2022-03-02")]
    assert _pick_best_result("RIT Foundation v. Union of India (2022)", docs) is None
