import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.base_legal_rag import _choose_extraction, _split_word_count

BODY = "The Magistrate may order the arrest. The Magistrate shall record reasons. A Magistrate may grant bail. "


def _parse(text):
    return [SimpleNamespace(section_number=n) for n in re.findall(r"^(\d+)\. ", text, re.M)]


def test_split_words_counted_against_vocabulary():
    vocab = {"magistrate", "the", "may"}
    assert _split_word_count("the Magi strate may", vocab) == 1
    assert _split_word_count("the Magistrate may", vocab) == 0


def test_layout_chosen_when_sections_match_and_fewer_splits():
    plain = f"1. Arrest\n{BODY}Magi strate.\n2. Bail\n{BODY}Magi strate."
    layout = f"1. Arrest\n{BODY}Magistrate.\n2. Bail\n{BODY}Magistrate."
    text, chunks = _choose_extraction(plain, layout, _parse)
    assert text is layout
    assert [c.section_number for c in chunks] == ["1", "2"]


def test_plain_kept_when_layout_loses_sections():
    plain = f"1. Arrest\n{BODY}Magi strate.\n2. Bail\n{BODY}"
    layout = f"Arrest\n{BODY}Magistrate.\n2. Bail\n{BODY}"
    text, _ = _choose_extraction(plain, layout, _parse)
    assert text is plain


def test_plain_kept_when_layout_not_cleaner_or_missing():
    plain = f"1. Arrest\n{BODY}"
    assert _choose_extraction(plain, plain + " ", _parse)[0] is plain
    assert _choose_extraction(plain, None, _parse)[0] is plain
