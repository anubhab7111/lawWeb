from app.ingest.section_maps import base_section, build_maps, rows_to_pairs


def test_base_section_drops_subclauses_and_rejects_non_sections():
    assert base_section("3(5)") == "3"
    assert base_section(" 121 A ") == "121A"
    assert base_section("-") is None
    assert base_section("Omitted") is None
    assert base_section(None) is None


def test_rows_to_pairs_skips_headers_and_unmapped_rows():
    rows = [
        ["Indian Penal Code, 1860", "", "Bharatiya Nyaya Sanhita, 2023", ""],
        ["Section", "Heading", "Section", "Heading"],
        ["302", "Punishment for murder", "103", "Punishment for murder"],
        ["34", "Common intention", "3(5)", "Acts done by several persons"],
        ["309", "Attempt to commit suicide", "-", "Omitted"],
    ]
    assert rows_to_pairs(rows) == [("302", "103"), ("34", "3")]


def test_maps_are_bidirectional_and_keep_one_to_many():
    maps = build_maps([("2", "1"), ("3", "1"), ("302", "103")])
    assert maps["old_to_new"]["302"] == ["103"]
    assert maps["new_to_old"]["1"] == ["2", "3"]
