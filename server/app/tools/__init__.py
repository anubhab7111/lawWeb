"""Tools package initialization."""

from .document_extractor import DocumentExtractor, get_document_extractor
from .crime_reporter import detect_crime_type, is_complex_crime, CRIME_TYPES
from .lawyer_recommender import (
    recommend_lawyers,
    format_lawyer_results,
    LEGAL_SPECIALIZATIONS,
)
from .indian_kanoon import IndianKanoonTool, get_indian_kanoon_tool
from .document_classifier import DocumentClassifier, get_document_classifier
from .statutory_validator import StatutoryValidator, get_statutory_validator
from .indian_law_rag import IndianLawRAGTool, get_indian_law_rag
from .legal_defect_analyzer import LegalDefectAnalyzer, get_legal_defect_analyzer

# Multi-domain RAG modules (replaces monolithic crime_rag.py)
from .base_legal_rag import BaseLegalRAGSystem, LegalChunk, LegalContext
from .criminal_rag import CriminalRAGSystem, get_criminal_rag_system
from .civil_rag import CivilRAGSystem, get_civil_rag_system
from .constitutional_rag import ConstitutionalRAGSystem, get_constitutional_rag_system
from .unified_legal_rag import UnifiedLegalRAGSystem, get_unified_rag_system

__all__ = [
    "DocumentExtractor",
    "get_document_extractor",
    "detect_crime_type",
    "is_complex_crime",
    "CRIME_TYPES",
    "recommend_lawyers",
    "format_lawyer_results",
    "LEGAL_SPECIALIZATIONS",
    "IndianKanoonTool",
    "get_indian_kanoon_tool",
    "DocumentClassifier",
    "get_document_classifier",
    "StatutoryValidator",
    "get_statutory_validator",
    "IndianLawRAGTool",
    "get_indian_law_rag",
    "LegalDefectAnalyzer",
    "get_legal_defect_analyzer",
    # Multi-domain RAG
    "BaseLegalRAGSystem",
    "LegalChunk",
    "LegalContext",
    "CriminalRAGSystem",
    "get_criminal_rag_system",
    "CivilRAGSystem",
    "get_civil_rag_system",
    "ConstitutionalRAGSystem",
    "get_constitutional_rag_system",
    "UnifiedLegalRAGSystem",
    "get_unified_rag_system",
]
