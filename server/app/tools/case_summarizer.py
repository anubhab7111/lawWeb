"""
AI summaries for My Cases — reuses the existing Ollama models (no new model
loaded), invoked on-demand from app/routers/cases.py and from the scheduled
sync job (app/jobs/_case_sync.py) whenever a new order event appears.
"""

from app.chatbot import get_fast_llm_prose, invoke_llm_safely, strip_reasoning_tags


async def summarize_case_event(case_title: str, event_title: str, event_detail: str) -> str:
    prompt = (
        "Summarize this court case update in 2-3 plain-language sentences for "
        "someone without a legal background. Do not add information not present below.\n\n"
        f"Case: {case_title}\n"
        f"Update: {event_title}\n"
        f"Detail: {event_detail or '(no further detail provided)'}"
    )
    try:
        summary = await invoke_llm_safely(get_fast_llm_prose(), prompt)
        return strip_reasoning_tags(summary or "")
    except Exception as e:
        print(f"[CaseSummarizer] summary generation failed: {e}")
        return ""
