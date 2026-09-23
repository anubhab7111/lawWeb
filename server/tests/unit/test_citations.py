from app.ingest.citations import extract_citations


def test_single_section_full_act_name():
    assert extract_citations("convicted under Section 302 of the Indian Penal Code") == ["IPC:302"]


def test_number_lists_with_slash_and_abbreviations():
    assert extract_citations("Ss. 302/34 IPC were framed") == ["IPC:302", "IPC:34"]


def test_list_with_read_with_and_dotted_abbreviation():
    text = "sections 148, 149 and 302 read with section 34 of the I.P.C."
    assert extract_citations(text) == ["IPC:148", "IPC:149", "IPC:302", "IPC:34"]


def test_crpc_variants_and_subclauses_stripped():
    assert extract_citations("application under Section 438(1) Cr.P.C.") == ["CrPC:438"]
    assert extract_citations("s. 482 of the Code of Criminal Procedure") == ["CrPC:482"]


def test_letter_suffix_and_evidence_act():
    assert extract_citations("section 65B of the Evidence Act") == ["IEA:65B"]
    assert extract_citations("Section 304-B IPC and Section 498A, IPC") == ["IPC:304B", "IPC:498A"]


def test_other_acts_keep_their_name_and_year():
    assert extract_citations("section 10 of the Contract Act, 1872") == ["contract act 1872:10"]


def test_constitution_articles():
    assert extract_citations("Article 21 of the Constitution") == ["Constitution:21"]


def test_unresolved_act_is_skipped_and_dedup_preserves_order():
    assert extract_citations("section 5 of the Act and section 302 IPC; s. 302 IPC again") == ["IPC:302"]


def test_bns_and_bnss_are_not_confused():
    assert extract_citations("Section 103 of the BNS and Section 482 BNSS") == ["BNS:103", "BNSS:482"]
