"""
State definitions for the LangGraph legal chatbot.
"""

from typing import TypedDict, Literal, Optional, List, Any
from dataclasses import dataclass, field


class Message(TypedDict):
    """A single message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class LawyerInfo(TypedDict):
    """Information about a lawyer."""

    name: str
    specialization: str
    location: str
    contact: str
    rating: Optional[float]
    experience_years: Optional[int]


class DocumentInfo(TypedDict):
    """Extracted document information."""

    text: str
    summary: str
    key_points: List[str]
    document_type: str
    parties_involved: List[str]
    important_dates: List[str]
    legal_implications: str


class CrimeReportInfo(TypedDict):
    """Crime report information and guidance."""

    crime_type: str


class DocumentValidationInfo(TypedDict):
    """Result of the 3-layer document validation pipeline with ReAct reasoning."""

    # Layer 1: Classification
    classified_type: str
    classification_confidence: float
    sub_type: Optional[str]
    jurisdiction_hints: List[str]

    # Layer 2: Statutory Validation
    compliance_score: float
    total_checks: int
    passed: int
    failed: int
    missing_elements: List[Any]
    present_elements: List[Any]
    non_compliance: List[Any]

    # Layer 3: Legal Analysis (ReAct)
    llm_analysis: str
    applicable_acts: List[str]
    applicable_sections: List[str]
    precedent_notes: List[str]
    state_specific_notes: List[str]

    # ReAct Reasoning Trace
    reasoning_trace: Optional[dict]  # {"think": str, "observe": str, "analyze": str}


class ChatState(TypedDict):
    """Main state for the chatbot graph."""

    # Conversation
    messages: List[Message]
    current_input: str
    conversation_context: Optional[str]  # Summary of recent conversation

    # Intent classification
    intent: Optional[
        Literal[
            "document_analysis",
            "crime_report",
            "find_lawyer",
            "general_query",
            "non_legal",
        ]
    ]

    # Enhanced routing metadata
    routing_confidence: Optional[float]  # Confidence score for the routing decision
    routing_reasoning: Optional[str]  # Explanation for the routing decision
    is_ambiguous: Optional[bool]  # Flag for low-confidence routing
    secondary_intents: Optional[List[str]]  # For multi-intent queries
    extracted_entities: Optional[List[str]]  # Legal terms/acts extracted from query

    # Tool selection
    selected_tools: Optional[
        List[str]
    ]  # Tools to be used: ["indian_kanoon", "crime_rag", "lawyer_finder", etc.]
    active_document_context: Optional[
        bool
    ]  # Flag when document is actively being discussed

    # Document analysis
    document_content: Optional[str]
    document_type: Optional[str]  # Type of document: pdf, image_ocr, docx, etc.
    document_info: Optional[DocumentInfo]

    # Document validation (3-layer pipeline)
    document_validation: Optional[DocumentValidationInfo]

    # Crime reporting
    crime_details: Optional[str]
    crime_report: Optional[CrimeReportInfo]

    # Lawyer search
    lawyer_query: Optional[str]
    lawyers_found: Optional[List[LawyerInfo]]

    # Response
    response: Optional[str]

    # Metadata
    session_id: Optional[str]
    error: Optional[str]
