from app.ingest.judgment_clean import clean_judgment_text, text_quality

RAW = """    556
A
                         SONA BALA BORA AND ORS.
                                  v.
                     JYOTIRINDRA BHATACHARJEE

B                     [RUMA PAL AND C.K. THAKKER, JJ.]

          Held, the appellant was not guilty of any miscon-
c   duct and the order is set aside. The Court restated the
    position clearly.
          The second paragraph begins here and runs across
D   two lines with a margin marker.

    556        SUPREME COURT REPORTS       [1985] 2 S.C.R.

          Third paragraph.                                     E
"""


def test_margin_letters_page_numbers_and_running_headers_are_removed():
    cleaned = clean_judgment_text(RAW)
    assert "SUPREME COURT REPORTS" not in cleaned
    assert not any(line.strip() in {"556", "A", "B", "c", "D"} for line in cleaned.splitlines())
    assert "SONA BALA BORA AND ORS." in cleaned
    assert "[RUMA PAL AND C.K. THAKKER, JJ.]" in cleaned


def test_hyphenated_line_breaks_are_rejoined():
    assert "misconduct and the order is set aside" in clean_judgment_text(RAW)


def test_paragraphs_are_reconstructed_from_indentation():
    paragraphs = clean_judgment_text(RAW).split("\n\n")
    assert any(p.startswith("Held, the appellant") and p.endswith("clearly.") for p in paragraphs)
    assert any(p.startswith("The second paragraph") and p.endswith("margin marker.") for p in paragraphs)
    assert paragraphs[-1] == "Third paragraph."


def test_text_quality_flags_garbled_and_empty_text():
    good = "The appellant was convicted under Section 302 of the Indian Penal Code. " * 40
    bad = "La1v-Mi£Conduct en1p[uy1ne11t di~sm1ssal ti!ne ra1se f;ffect " * 40
    assert text_quality(good, pages=1)["ok"]
    assert not text_quality(bad, pages=1)["ok"]
    assert not text_quality("", pages=5)["ok"]
    assert text_quality("x" * 50, pages=10)["reason"] == "no text layer"


def test_varying_indentation_does_not_split_wrapped_lines():
    raw = (
        "            The appellant contended that the provision, read in the manner\n"
        "    suggested by the respondent, would defeat the object of the statute\n"
        "            and render Section 22 nugatory in its application to universities.\n"
        "\n"
        "            The respondent replied that the Court should not read the\n"
        "    provision narrowly.\n"
    )
    paragraphs = clean_judgment_text(raw).split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0].endswith("to universities.")


def test_numbered_paragraphs_start_new_paragraphs():
    raw = "    6.1. First point is made here.\n    6.2. The second point follows.\n"
    assert clean_judgment_text(raw).split("\n\n") == [
        "6.1. First point is made here.",
        "6.2. The second point follows.",
    ]


def test_running_case_name_headers_are_dropped_but_party_block_is_kept():
    raw = (
        "    The Collector passed an order under the Act on that date and\n"
        "    BALKRISHNA v. SWADESHI POLYTEX 855\n"
        "    the matter went to the High Court.\n"
        "\n"
        "    STATE OF PUNJAB v. AJAIB SINGH\n"
    )
    cleaned = clean_judgment_text(raw)
    assert "855" not in cleaned
    assert "on that date and the matter went to the High Court." in cleaned
    assert "STATE OF PUNJAB v. AJAIB SINGH" in cleaned
