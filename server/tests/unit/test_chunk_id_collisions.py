from app.tools.unified_legal_rag import UnifiedLegalRAGSystem

TEXT = (
    "\n34. Acts done by several persons in furtherance of common intention.--"
    "When a criminal act is done by several persons in furtherance of the common intention "
    "of all, each of such persons is liable for that act in the same manner as if it were "
    "done by him alone.\n"
    "\n35. Whenever an act, which is criminal only by reason of its being done with a "
    "criminal knowledge or intention, is done by several persons.\n"
)


def _ids(rag, source_file, domain="criminal"):
    return {c.section_number: c.chunk_id for c in rag._parse_pdf(TEXT, source_file, domain)}


def test_acts_sharing_a_name_prefix_get_distinct_chunk_ids():
    # IPC and the Evidence Act both start "INDIAN"; the three Bharatiya acts start "BHARAT".
    # Colliding ids made the build keep one and silently drop the other's section.
    rag = UnifiedLegalRAGSystem()
    ipc = _ids(rag, "Indian_Penal_Code_1860.pdf")
    iea = _ids(rag, "Indian_Evidence_Act_1872.pdf")
    bns = _ids(rag, "Bharatiya_Nyaya_Sanhita_BNS_2023.pdf")
    bnss = _ids(rag, "Bharatiya_Nagarik_Suraksha_Sanhita_BNSS_2023.pdf")
    assert "34" in ipc
    assert len({ipc["34"], iea["34"], bns["34"], bnss["34"]}) == 4
