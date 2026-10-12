"""unified_legal_rag.py — Unified All-Domain Legal RAG System Indexes EVERY bare-act PDF under app/data/bare_acts/ (all domain subdirectories) plus the prose corpora..."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.tools.base_legal_rag import BaseLegalRAGSystem, LegalChunk, file_fingerprint
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
    """Unified legal RAG: one index over all bare_acts domains + prose corpora, with per-chunk `domains` metadata enabling optional query-time filtering."""


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
        "mappings": "MAP",
        "rules": "RUL",
        "guides": "GDE",
    }


    PROSE_DIRS = ("notifications", "explanatory", "mappings", "rules", "guides")

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir=data_dir)
        self._domain_parsers: Dict[str, BaseLegalRAGSystem] = {}

    @property
    def domain_name(self) -> str:
        return "unified"

    @property
    def pdf_subdir(self) -> str:


        return ""


    def _collect_pdf_sources(self) -> List[Tuple[Path, str, List[str]]]:
        """Walk all domain subdirectories plus the prose corpora, dedupe PDFs that appear in several places (by content hash, then by normalized act name — including one filename being a..."""
        located: List[Tuple[Path, str]] = []
        for pdf in sorted(self._bare_acts_dir.rglob("*.pdf")):
            rel = pdf.relative_to(self._bare_acts_dir)
            if len(rel.parts) < 2:
                continue
            located.append((pdf, rel.parts[0]))


        for domain in self.PROSE_DIRS:
            prose_dir = self._statutes_dir / domain
            if prose_dir.exists():
                for pdf in sorted(prose_dir.rglob("*.pdf")):
                    located.append((pdf, domain))


        by_key: Dict[str, List[Tuple[Path, str]]] = {}
        hash_to_key: Dict[str, str] = {}
        for pdf, domain in located:
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            key = hash_to_key.get(digest) or _normalized_act_key(pdf.stem)
            hash_to_key[digest] = key
            by_key.setdefault(key, []).append((pdf, domain))


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
            return pdf_path.stat().st_size

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
        prefix = re.sub(r"[^A-Z0-9]", "", title.upper())


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

    def _current_pdf_fingerprint(self) -> Dict[str, str]:
        fingerprint = super()._current_pdf_fingerprint()
        for domain in self.PROSE_DIRS:
            prose_dir = self._statutes_dir / domain
            if prose_dir.exists():
                for pdf in prose_dir.rglob("*.pdf"):
                    fingerprint[f"{domain}/{pdf.relative_to(prose_dir)}"] = (
                        file_fingerprint(pdf)
                    )
        return fingerprint


    def _preprocess_query(self, query: str) -> str:
        """Aggregate the specialized domains' vocabulary expansions."""
        extras: List[str] = []
        for domain in ("criminal", "civil", "constitutional"):
            expanded = self._parser_for_domain(domain)._preprocess_query(query)
            if len(expanded) > len(query):
                extras.append(expanded[len(query) :].strip())
        if extras:
            return query + " " + " ".join(extras)
        return query


_unified_rag: Optional[UnifiedLegalRAGSystem] = None


def get_unified_rag_system() -> UnifiedLegalRAGSystem:
    """Get or create the UnifiedLegalRAGSystem singleton."""
    global _unified_rag
    if _unified_rag is None:
        _unified_rag = UnifiedLegalRAGSystem()
    return _unified_rag
