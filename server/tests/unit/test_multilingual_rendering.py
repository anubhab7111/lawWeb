"""Hindi answers kept their citations but rendered badly: IndicTrans2 splits ``**`` into ``* *``,
drops list indentation, and now and then corrupts a sentinel (``QZXQAQXZQ`` came back as ``QZXQXZQ``)."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.multilingual import entity_guard
from app.multilingual.translation import (
    IndicTrans2Service,
    repair_markdown,
    split_line_prefix,
)

LINE = "- **Procedural requirement (CrPC § 55):** see Section 41 and BNSS § 35(1)."


def test_mask_unmask_round_trip():
    masked, mapping = entity_guard.mask(LINE)
    assert entity_guard.unmask(masked, mapping) == LINE


def test_sentinels_are_numbered_in_text_order():
    _, mapping = entity_guard.mask("Under CrPC § 41 and BNSS § 35, arrest is allowed.")
    assert list(mapping.values()) == ["CrPC", "§ 41", "BNSS", "§ 35"]


def test_observed_corruption_gets_its_citation_back():
    masked, mapping = entity_guard.mask("(CrPC § 55): a written order is needed.")
    tokens = list(mapping)
    corrupted = masked.replace(tokens[1], "QZXQXZQ")
    assert entity_guard.unmask(corrupted, mapping) == "(CrPC § 55): a written order is needed."


def test_two_corrupted_sentinels_are_restored_in_order():
    masked, mapping = entity_guard.mask("Read § 41 with § 35 and § 50.")
    tokens = list(mapping)
    corrupted = masked.replace(tokens[0], "QZXQXZQ").replace(tokens[2], "ZXQQXZQ")
    assert entity_guard.unmask(corrupted, mapping) == "Read § 41 with § 35 and § 50."


def test_junk_remnant_is_dropped_when_nothing_is_missing():
    masked, mapping = entity_guard.mask("Under § 41 arrest is allowed.")
    out = entity_guard.unmask(masked + " QZXQXZQ", mapping)
    assert "QZX" not in out and "§ 41" in out


def test_is_complete_detects_a_dropped_citation():
    masked, mapping = entity_guard.mask("Under § 41 and § 35.")
    dropped = masked.replace(list(mapping)[1], "")
    assert entity_guard.is_complete(entity_guard.unmask(masked, mapping), mapping)
    assert not entity_guard.is_complete(entity_guard.unmask(dropped, mapping), mapping)


def test_repair_rejoins_split_bold_markers():
    assert (
        repair_markdown("* * CrPC § 41 * * और * * BNSS § 35 * *: अनुमति")
        == "**CrPC § 41** और **BNSS § 35**: अनुमति"
    )


def test_repair_leaves_adjacent_intact_bold_spans_alone():
    assert repair_markdown("**A** **B**") == "**A** **B**"


def test_repair_does_not_tighten_an_odd_number_of_markers():
    assert repair_markdown("* * a * * और * * b") == "** a ** और ** b"


def test_line_prefix_keeps_bullets_and_nesting_out_of_translation():
    assert split_line_prefix("  - nested item") == ("  - ", "nested item")
    assert split_line_prefix("* bullet") == ("* ", "bullet")
    assert split_line_prefix("**Bold heading**") == ("", "**Bold heading**")
    assert split_line_prefix("12. step") == ("12. ", "step")


def _service(transform):
    """A service whose 'model' applies ``transform`` to each masked segment, like IndicTrans2 would."""
    svc = IndicTrans2Service()
    svc._processor = SimpleNamespace(
        preprocess_batch=lambda texts, src_lang, tgt_lang: [f"{src_lang} {tgt_lang} {t}" for t in texts],
        postprocess_batch=lambda texts, lang: texts,
    )
    sp = SimpleNamespace(encode=lambda text, out_type=str: text.split(" "), decode=lambda toks: " ".join(toks))

    def translate_batch(batch, **_):
        return [SimpleNamespace(hypotheses=[transform(" ".join(toks[2:])).split(" ")]) for toks in batch]

    direction = SimpleNamespace(translator=SimpleNamespace(translate_batch=translate_batch), sp_src=sp, sp_tgt=sp)
    svc._indic_en = svc._en_indic = direction
    return svc


def test_translation_keeps_nested_list_indentation():
    svc = _service(lambda s: " ".join(s.split()))  # the model normalises whitespace
    text = "- **Key provisions:**\n  - CrPC § 41 applies\n  - BNSS § 35 applies"
    assert svc._translate_blocking(text, "eng_Latn", "hin_Deva") == text


def test_translation_repairs_bold_split_by_the_model():
    svc = _service(lambda s: s.replace("**", "* *"))
    out = svc._translate_blocking("- **CrPC § 41** and **BNSS § 35**: arrest", "eng_Latn", "hin_Deva")
    assert out == "- **CrPC § 41** and **BNSS § 35**: arrest"


def test_segment_that_loses_a_citation_stays_in_english():
    def drop_second_sentinel(s):
        tokens = [w for w in s.split(" ") if w.startswith("QZXQ")]
        return s.replace(tokens[1], "").upper() if len(tokens) > 1 else s.upper()

    svc = _service(drop_second_sentinel)
    out = svc._translate_blocking("Read § 41 with § 35.\nPolice may arrest.", "eng_Latn", "hin_Deva")
    assert out.split("\n") == ["Read § 41 with § 35.", "POLICE MAY ARREST."]
