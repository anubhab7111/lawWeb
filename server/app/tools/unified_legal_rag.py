"""
unified_legal_rag.py — Unified All-Domain Legal RAG System

Indexes EVERY bare-act PDF under app/data/bare_acts/ (all domain
subdirectories) plus the prose corpora (notifications/, explanatory/) into
one merged FAISS + BM25 index at app/data/faiss_index/unified/.

Key design decisions
--------------------
* One merged index, domain as metadata. Keyword-based domain routing was
  the weakest link of the per-domain design: a query about "dowry death"
  spans family + criminal, "GST fraud" spans indirect_tax + criminal.
  With a merged index, hybrid retrieval + the cross-encoder reranker do the
  routing; callers that DO know the domain pass `domains=[...]` to
  retrieve() as a filter.

* PDF dedup at build time. Several acts live in more than one domain
  directory (e.g. Representation of the People Act in constitutional/ and
  election/). Duplicate files are detected by content hash and by
  normalized act name; the copy with the best extracted text is indexed
  and every source domain is recorded in the chunk's `domains` metadata.

* Per-domain parser dispatch. Constitutional PDFs go through the
  Article-aware parser, criminal PDFs keep the per-Act punishment-clause
  filter, prose corpora get sliding-window chunking, and everything else
  uses the generic section parser. Chunk IDs are re-prefixed with a unique
  per-domain code (FAM_, TAX_, …) so IDs from all domains coexist without
  collision.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.tools.base_legal_rag import BaseLegalRAGSystem, LegalChunk
from app.tools.civil_rag import CivilRAGSystem
from app.tools.constitutional_rag import ConstitutionalRAGSystem
from app.tools.criminal_rag import CriminalRAGSystem


class _GenericDomainParser(BaseLegalRAGSystem):
    """Minimal concrete system used only for its generic section parser."""

    def __init__(self, domain: str):
        self._domain = domain
        super().__init__()

    @property
    def domain_name(self) -> str:
        return self._domain

    @property
    def pdf_subdir(self) -> str:
        return self._domain


def _normalized_act_key(stem: str) -> str:
    """Normalize a PDF filename stem for duplicate-act detection."""
    key = re.sub(r"[^a-z0-9]", "", stem.lower())
    if key.startswith("the"):
        key = key[3:]
    return key


class UnifiedLegalRAGSystem(BaseLegalRAGSystem):
    """
    Unified legal RAG: one index over all bare_acts domains + prose corpora,
    with per-chunk `domains` metadata enabling optional query-time filtering.
    """

    # Unique 3-letter chunk-ID prefix per domain (plain domain[:3] collides:
    # constitutional/consumer/consumer_cyber_ip would all be "CON").
    DOMAIN_CODES: Dict[str, str] = {
        "civil": "CIV",
        "commercial": "CML",
        "constitutional": "CON",
        "consumer": "CNS",
        "consumer_cyber_ip": "CCI",
        "corporate": "CRP",
        "criminal": "CRI",
        "election": "ELE",
        "environment": "ENV",
        "family": "FAM",
        "human_rights": "HUM",
        "indirect_tax": "TAX",
        "labour": "LAB",
        "property": "PRP",
        "notifications": "NTF",
        "explanatory": "EXP",
    }

    # Prose corpora outside bare_acts/ (enforcement notifications, judiciary
    # explainers). These aren't section-numbered statutes: they get sliding-
    # window chunking instead of the section parser.
    PROSE_DIRS = ("notifications", "explanatory")

    def __init__(self, data_dir: str = "app/data"):
        super().__init__(data_dir=data_dir)
        self._domain_parsers: Dict[str, BaseLegalRAGSystem] = {}

    @property
    def domain_name(self) -> str:
        return "unified"

    @property
    def pdf_subdir(self) -> str:
        # Empty subdir → _bare_acts_dir is bare_acts/ itself, so the base
        # class's rglob (used by _should_rebuild) walks every domain.
        return ""

    # ── Build-time hooks ────────────────────────────────────────

    def _collect_pdf_sources(self) -> List[Tuple[Path, str, List[str]]]:
        """
        Walk all domain subdirectories plus the prose corpora, dedupe PDFs
        that appear in several places (by content hash, then by normalized
        act name — including one filename being a suffix of another, e.g.
        two RERA copies), and record every source domain of each surviving
        file.
        """
        located: List[Tuple[Path, str]] = []  # (path, domain)
        for pdf in sorted(self._bare_acts_dir.rglob("*.pdf")):
            rel = pdf.relative_to(self._bare_acts_dir)
            if len(rel.parts) < 2:
                continue  # skip files not inside a domain subdirectory
            located.append((pdf, rel.parts[0]))

        # Prose corpora live beside bare_acts/ under data_dir
        for domain in self.PROSE_DIRS:
            prose_dir = self.data_dir / domain
            if prose_dir.exists():
                for pdf in sorted(prose_dir.rglob("*.pdf")):
                    located.append((pdf, domain))

        # Group by content hash first (exact duplicates), keyed to one act key
        by_key: Dict[str, List[Tuple[Path, str]]] = {}
        hash_to_key: Dict[str, str] = {}
        for pdf, domain in located:
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            key = hash_to_key.get(digest) or _normalized_act_key(pdf.stem)
            hash_to_key[digest] = key
            by_key.setdefault(key, []).append((pdf, domain))

        # Merge groups whose normalized keys contain one another (catches
        # "rera_real_estate_..." vs "real_estate_..." filename variants).
        keys = sorted(by_key, key=len)
        merged_into: Dict[str, str] = {}
        for i, short in enumerate(keys):
            if len(short) < 20:
                continue
            for long in keys[i + 1 :]:
                if short != long and short in long:
                    target = merged_into.get(short, short)
                    merged_into[long] = target
        for src, dst in merged_into.items():
            if src in by_key:
                by_key.setdefault(dst, []).extend(by_key.pop(src))

        sources: List[Tuple[Path, str, List[str]]] = []
        for key, entries in sorted(by_key.items()):
            if len(entries) == 1:
                best_path, best_domain = entries[0]
            else:
                # Pick the copy with the best extracted text, not the biggest
                # file — a scanned/OCR copy is often larger on disk but yields
                # far worse text than a born-digital one.
                best_path, best_domain = max(
                    entries, key=lambda e: self._text_quality(e[0])
                )
            domains = sorted({domain for _, domain in entries})
            if len(entries) > 1:
                print(
                    f"  Dedup: {best_path.name} indexed once for domains "
                    f"{domains} ({len(entries)} copies found)"
                )
            sources.append((best_path, best_domain, domains))
        return sources

    @staticmethod
    def _text_quality(pdf_path: Path) -> int:
        """Alphanumeric character count of the PDF's extracted text."""
        try:
            from langchain_community.document_loaders import PyPDFLoader

            pages = PyPDFLoader(str(pdf_path)).load()
            text = "\n".join(p.page_content for p in pages)
            return sum(ch.isalnum() for ch in text)
        except Exception:
            return pdf_path.stat().st_size  # crude fallback

    def _parser_for_domain(self, domain: str) -> BaseLegalRAGSystem:
        """Reuse the specialized domain parsers; generic parser elsewhere."""
        if domain not in self._domain_parsers:
            if domain == "constitutional":
                self._domain_parsers[domain] = ConstitutionalRAGSystem()
            elif domain == "criminal":
                self._domain_parsers[domain] = CriminalRAGSystem()
            elif domain == "civil":
                self._domain_parsers[domain] = CivilRAGSystem()
            else:
                self._domain_parsers[domain] = _GenericDomainParser(domain)
        return self._domain_parsers[domain]

    def _parse_pdf(
        self, full_text: str, source_file: str, primary_domain: str
    ) -> List[LegalChunk]:
        if primary_domain in self.PROSE_DIRS:
            return self._parse_prose_pdf(full_text, source_file, primary_domain)
        parser = self._parser_for_domain(primary_domain)
        chunks = parser._parse_legal_sections(full_text, source_file)
        code = self.DOMAIN_CODES.get(primary_domain, primary_domain[:3].upper())
        for c in chunks:
            # Replace the parser's domain prefix with the unified unique code
            rest = c.chunk_id.split("_", 1)[1] if "_" in c.chunk_id else c.chunk_id
            c.chunk_id = f"{code}_{rest}"
            c.domain = primary_domain
        return chunks

    def _parse_prose_pdf(
        self, full_text: str, source_file: str, domain: str
    ) -> List[LegalChunk]:
        """Sliding-window chunking for non-statute documents."""
        from app.tools.base_legal_rag import _infer_act_name, _split_long_text

        title = _infer_act_name(source_file)
        text = re.sub(r"\s+", " ", full_text).strip()
        if len(text) < 100:
            return []
        code = self.DOMAIN_CODES.get(domain, domain[:3].upper())
        prefix = re.sub(r"[^A-Z0-9]", "", title.upper())[:6]
        # "part N", not a bare number: these are window indices, and must
        # neither read like statutory citations in prompts nor collide with
        # real section numbers during evaluation.
        return [
            LegalChunk(
                chunk_id=f"{code}_{prefix}_{n}",
                domain=domain,
                act_name=title,
                section_number=f"part {n}",
                title=title,
                text=part,
                source_file=source_file,
            )
            for n, part in enumerate(_split_long_text(text), 1)
        ]

    def _current_pdf_fingerprint(self) -> Dict[str, int]:
        fingerprint = super()._current_pdf_fingerprint()
        for domain in self.PROSE_DIRS:
            prose_dir = self.data_dir / domain
            if prose_dir.exists():
                for pdf in prose_dir.rglob("*.pdf"):
                    fingerprint[f"{domain}/{pdf.relative_to(prose_dir)}"] = (
                        pdf.stat().st_size
                    )
        return fingerprint

    # ── Query-time hook ─────────────────────────────────────────

    def _preprocess_query(self, query: str) -> str:
        """
        Aggregate the specialized domains' vocabulary expansions. Each
        expander only fires on its own trigger keywords, so combining them
        adds targeted synonyms without cross-domain noise.
        """
        extras: List[str] = []
        for domain in ("criminal", "civil", "constitutional"):
            expanded = self._parser_for_domain(domain)._preprocess_query(query)
            if len(expanded) > len(query):
                extras.append(expanded[len(query) :].strip())
        if extras:
            return query + " " + " ".join(extras)
        return query


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────

_unified_rag: Optional[UnifiedLegalRAGSystem] = None


def get_unified_rag_system() -> UnifiedLegalRAGSystem:
    """Get or create the UnifiedLegalRAGSystem singleton."""
    global _unified_rag
    if _unified_rag is None:
        _unified_rag = UnifiedLegalRAGSystem()
    return _unified_rag
