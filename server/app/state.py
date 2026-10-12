"""State definitions for the LangGraph legal chatbot."""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class Message(TypedDict):
    """A single message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class LawyerInfo(TypedDict):
    """Information about a lawyer."""

    id: str
    name: str
    specialization: str
    location: str
    contact: Optional[str]
    rating: Optional[float]
    experience_years: Optional[int]
    hourly_rate: Optional[int]
    success_rate: Optional[int]
    bio: Optional[str]


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


    classified_type: str
    classification_confidence: float
    sub_type: Optional[str]
    jurisdiction_hints: List[str]


    compliance_score: float
    total_checks: int
    passed: int
    failed: int
    missing_elements: List[Any]
    present_elements: List[Any]
    non_compliance: List[Any]


    llm_analysis: str
    applicable_acts: List[str]
    applicable_sections: List[str]
    precedent_notes: List[str]
    state_specific_notes: List[str]


    reasoning_trace: Optional[dict]


class ChatState(TypedDict):
    """Main state for the chatbot graph."""


    messages: List[Message]
    current_input: str
    conversation_context: Optional[str]


    retrieval_query: Optional[str]


    intent: Optional[
        Literal[
            "document_analysis",
            "crime_report",
            "find_lawyer",
            "general_query",
            "non_legal",
        ]
    ]


    routing_confidence: Optional[float]
    routing_reasoning: Optional[str]
    is_ambiguous: Optional[bool]
    secondary_intents: Optional[List[str]]
    extracted_entities: Optional[List[str]]


    selected_tools: Optional[
        List[str]
    ]
    domain_hint: Optional[
        Literal["criminal"]
    ]
    active_document_context: Optional[
        bool
    ]


    document_content: Optional[str]
    document_type: Optional[str]
    document_info: Optional[DocumentInfo]


    document_validation: Optional[DocumentValidationInfo]


    crime_details: Optional[str]
    crime_report: Optional[CrimeReportInfo]


    lawyer_query: Optional[str]
    lawyers_found: Optional[List[LawyerInfo]]


    sub_questions: Optional[List[str]]
    tool_results: Optional[Dict[str, Any]]
    retrieved_context: Optional[str]
    retrieved_sections: Optional[Any]
    rag_succeeded: Optional[bool]
    retrieval_confidence: Optional[float]
    retrieval_grade: Optional[str]
    retrieval_attempts: Optional[int]
    regen_count: Optional[int]
    regen_pending: Optional[bool]
    regen_feedback: Optional[str]
    extra_queries: Optional[List[str]]
    grounding_score: Optional[float]
    clarification: Optional[bool]
    followup_answered: Optional[bool]


    response: Optional[str]


    session_id: Optional[str]
    error: Optional[str]
    started_at: Optional[float]


    trace: Optional[Dict[str, Any]]
