"""
LLM-as-a-Judge Module  (Gemini Edition)
========================================
Uses Google Gemini 2.5 Flash (reasoning model) to evaluate RAG pipeline
quality across four dimensions:

  • Faithfulness      — Are all claims in the answer supported by the retrieved
                        context?  (anti-hallucination measure)
  • Answer Relevance  — Does the answer actually address the user's question?
  • Context Precision — Is the retrieved context useful / relevant to the query?
  • Context Recall    — Does the answer cover the key facts from a reference
                        answer / the retrieved documents?

Design principles
-----------------
  1. Every judge call returns a normalised float in [0, 1] plus a short
     natural-language reasoning string so that failures are debuggable.
  2. Prompts are deliberately simple and deterministic (temperature=0).
  3. JSON is parsed with a regex fallback so that even a slightly chatty model
     still yields a usable score.
  4. All calls are async to integrate cleanly with the LangGraph pipeline.
  5. A lightweight retry with exponential back-off handles transient API
     timeouts without blocking the test suite indefinitely.

Requirements
------------
    pip install google-genai

Environment / config
--------------------
    Set GEMINI_API_KEY in your environment or in app.config settings.
    Default model : gemini-2.5-flash   (reasoning variant, best quality/speed)
    Fallback model: gemini-2.0-flash   (faster, slightly lower quality)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import google.generativeai as genai

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
GEMINI_REASONING_MODEL = "gemini-2.5-flash"   # latest reasoning model
GEMINI_FALLBACK_MODEL  = "gemini-2.0-flash"   # fast fallback

# ---------------------------------------------------------------------------
# Result container  (unchanged from original)
# ---------------------------------------------------------------------------
@dataclass
class JudgeScore:
    """Structured output from a single LLM judge call."""

    score: float        # [0.0 – 1.0]
    reasoning: str      # brief natural-language explanation
    metric: str         # which metric this score belongs to
    raw_response: str = field(default="", repr=False)   # raw LLM output
    latency_s: float  = field(default=0.0)              # wall-clock time

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))

    @classmethod
    def failure(cls, metric: str, reason: str) -> "JudgeScore":
        """Return a neutral score (0.5) when the judge cannot run."""
        return cls(score=0.5, reasoning=f"[JUDGE FAILED] {reason}", metric=metric)

    @property
    def label(self) -> str:
        """Human-readable quality band."""
        if self.score >= 0.85:
            return "Excellent"
        if self.score >= 0.70:
            return "Good"
        if self.score >= 0.50:
            return "Fair"
        if self.score >= 0.30:
            return "Poor"
        return "Very Poor"


# ---------------------------------------------------------------------------
# Prompt templates  (unchanged from original)
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """\
You are a strict legal evaluation judge. Your only job is to determine whether
every factual claim in the ANSWER is explicitly supported by the RETRIEVED CONTEXT.

RETRIEVED CONTEXT:
\"\"\"
{context}
\"\"\"

ANSWER:
\"\"\"
{answer}
\"\"\"

INSTRUCTIONS:
1. Identify each distinct factual claim in the Answer.
2. For each claim, check whether it is directly supported by the Retrieved Context.
3. Compute:  score = (number of supported claims) / (total claims)
4. If the Answer makes NO claims (e.g. it is a refusal), return score = 1.0.
5. If the Retrieved Context is empty, any non-trivial claim is unsupported; return score = 0.0.

Respond with ONLY a valid JSON object on a single line, like this:
{{"score": <float between 0 and 1>, "reasoning": "<one sentence>"}}
"""

_ANSWER_RELEVANCE_PROMPT = """\
You are an expert evaluator of legal question-answering systems.

USER QUESTION:
\"\"\"
{question}
\"\"\"

ANSWER:
\"\"\"
{answer}
\"\"\"

INSTRUCTIONS:
1. Decide how directly and completely the Answer addresses the User Question.
2. Score rubric:
   - 1.0 : Answer directly and completely addresses the question.
   - 0.75: Answer mostly addresses the question with minor gaps.
   - 0.5 : Answer partially addresses the question; significant gaps remain.
   - 0.25: Answer is tangentially related but does not really answer the question.
   - 0.0 : Answer does not address the question at all.
3. Interpolate freely between these anchor points.

Respond with ONLY a valid JSON object on a single line, like this:
{{"score": <float between 0 and 1>, "reasoning": "<one sentence>"}}
"""

_CONTEXT_PRECISION_PROMPT = """\
You are a search-quality judge for a legal retrieval system.

USER QUERY:
\"\"\"
{query}
\"\"\"

RETRIEVED CONTEXT (all chunks concatenated):
\"\"\"
{context}
\"\"\"

INSTRUCTIONS:
1. Determine what fraction of the retrieved text is actually relevant to the
   User Query (i.e., would help answer it).
2. Score rubric:
   - 1.0 : All retrieved text is highly relevant.
   - 0.75: Mostly relevant; one or two slightly off-topic chunks.
   - 0.5 : About half is relevant; noticeable noise.
   - 0.25: Mostly irrelevant; only small fragments are useful.
   - 0.0 : Completely irrelevant retrieval.
3. If no context was retrieved, return score = 0.0.

Respond with ONLY a valid JSON object on a single line, like this:
{{"score": <float between 0 and 1>, "reasoning": "<one sentence>"}}
"""

_CONTEXT_RECALL_PROMPT = """\
You are a coverage evaluator for a legal question-answering system.

REFERENCE ANSWER (gold standard):
\"\"\"
{reference}
\"\"\"

MODEL ANSWER (to be evaluated):
\"\"\"
{answer}
\"\"\"

INSTRUCTIONS:
1. Identify each distinct key fact / legal concept present in the Reference Answer.
2. For each key fact, check whether the Model Answer also covers it (even if
   phrased differently).
3. Compute: score = (facts covered by model) / (total facts in reference)
4. If the Reference Answer is empty, return score = 1.0 (nothing to recall).

Respond with ONLY a valid JSON object on a single line, like this:
{{"score": <float between 0 and 1>, "reasoning": "<one sentence>"}}
"""


# ---------------------------------------------------------------------------
# JSON extraction helpers  (unchanged from original)
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract the first JSON object from an LLM response.
    Handles:
      • Clean JSON: {"score": 0.8, "reasoning": "..."}
      • JSON embedded in markdown code fences
      • JSON with a brief preamble sentence before it
      • Gemini reasoning models sometimes wrap in <think>...</think> tags
    """
    # Strip Gemini thinking blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "").strip()

    # 1. Try full parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Regex: first {...} containing "score"
    pattern = re.compile(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', re.DOTALL)
    match = pattern.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Looser: any {...}
    pattern2 = re.compile(r"\{.*?\}", re.DOTALL)
    for m in pattern2.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if "score" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    return None


def _parse_score(raw: str, metric: str) -> JudgeScore:
    """Parse raw LLM text into a JudgeScore, falling back gracefully."""
    parsed = _extract_json(raw)
    if parsed and "score" in parsed:
        try:
            score = float(parsed["score"])
            reasoning = str(parsed.get("reasoning", "No reasoning provided."))
            return JudgeScore(
                score=score,
                reasoning=reasoning,
                metric=metric,
                raw_response=raw,
            )
        except (TypeError, ValueError):
            pass

    # Last resort: bare float in the text
    float_match = re.search(r"\b(0\.\d+|1\.0|1)\b", raw)
    if float_match:
        return JudgeScore(
            score=float(float_match.group(1)),
            reasoning="Score extracted from unstructured response.",
            metric=metric,
            raw_response=raw,
        )

    logger.warning(
        "LLM judge could not parse score for metric=%s. raw=%r", metric, raw[:200]
    )
    return JudgeScore.failure(metric, f"Unparseable response: {raw[:120]}")


# ---------------------------------------------------------------------------
# Gemini client factory  (module-level, configured once)
# ---------------------------------------------------------------------------
def _build_gemini_model(model_name: str, api_key: str) -> genai.GenerativeModel:
    """Configure the google-genai client and return a GenerativeModel."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.types.GenerationConfig(
            # Strict JSON output — Gemini honours this natively
            response_mime_type="application/json",
            temperature=0.0,          # deterministic
            max_output_tokens=512,    # score + one-sentence reasoning only
        ),
        # Optional safety settings — relax for legal content that might
        # mention violence / harm in a purely academic context.
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        },
    )


# ---------------------------------------------------------------------------
# Core judge class
# ---------------------------------------------------------------------------
class LLMJudge:
    """
    Async LLM-as-a-judge that scores RAG pipeline outputs using Google's
    Gemini 2.5 Flash reasoning model.

    Drop-in replacement for the original Ollama-backed LLMJudge — the public
    API (faithfulness / answer_relevance / context_precision / context_recall /
    score_all) is identical.

    Usage::
        judge = LLMJudge()
        score = await judge.faithfulness(answer=..., context=...)
        print(score.score, score.reasoning)

    Configuration (in order of precedence)
    ---------------------------------------
    1. Constructor kwargs
    2. app.config.get_settings()  fields:
         - gemini_api_key   (str)   — your Google AI Studio key
         - gemini_model     (str)   — override default model name
    3. Defaults: model=gemini-2.5-flash
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        retry_delay_s: float = 2.0,
        timeout_s: float = 60.0,
    ) -> None:
        settings = get_settings()

        # Resolve API key: kwarg → settings → env var (google-genai picks
        # up GOOGLE_API_KEY automatically, so passing None is also fine if
        # the env var is set)
        resolved_key: Optional[str] = (
            api_key
            or getattr(settings, "gemini_api_key", None)
        )

        # Resolve model name
        self._model_name: str = (
            model
            or getattr(settings, "gemini_model", None)
            or GEMINI_REASONING_MODEL
        )

        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._timeout_s = timeout_s

        # Build the Gemini model (configures genai globally with the key)
        if resolved_key:
            genai.configure(api_key=resolved_key)

        self._gemini = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=512,
            ),
            safety_settings={
                "HARM_CATEGORY_HARASSMENT":        "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH":       "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            },
        )

        logger.info("LLMJudge initialised with model=%s", self._model_name)

    # ------------------------------------------------------------------
    # Internal async invoke with retry
    # ------------------------------------------------------------------
    async def _invoke(self, prompt: str, metric: str) -> JudgeScore:
        """
        Call Gemini with retry/back-off and return a parsed JudgeScore.

        The google-genai SDK's generate_content_async is awaitable, so we
        use it directly without run_in_executor.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            t0 = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._gemini.generate_content_async(prompt),
                    timeout=self._timeout_s,
                )
                raw = response.text  # str; Gemini returns text even in JSON mode
                elapsed = time.perf_counter() - t0

                score = _parse_score(raw, metric)
                score.latency_s = elapsed

                logger.debug(
                    "metric=%s  score=%.3f  latency=%.2fs",
                    metric, score.score, elapsed,
                )
                return score

            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "Gemini judge timed out (attempt %d/%d) metric=%s",
                    attempt + 1, self._max_retries + 1, metric,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Gemini judge error (attempt %d/%d) metric=%s: %s",
                    attempt + 1, self._max_retries + 1, metric, exc,
                )

            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay_s * (attempt + 1))

        logger.error(
            "Gemini judge exhausted retries for metric=%s: %s", metric, last_exc
        )
        return JudgeScore.failure(metric, str(last_exc))

    # ------------------------------------------------------------------
    # Public metric methods  (same signatures as original)
    # ------------------------------------------------------------------
    async def faithfulness(
        self,
        answer: str,
        context: str,
    ) -> JudgeScore:
        """
        Faithfulness / Groundedness.
        Measures what fraction of the claims in *answer* are explicitly
        supported by *context*.  Score of 1.0 → zero hallucination.
        """
        if not answer.strip():
            return JudgeScore(
                score=0.0,
                reasoning="Empty answer — nothing to evaluate.",
                metric="faithfulness",
            )
        if not context.strip():
            return JudgeScore(
                score=0.0,
                reasoning="No context retrieved; cannot verify groundedness.",
                metric="faithfulness",
            )
        prompt = _FAITHFULNESS_PROMPT.format(
            context=context[:3000],
            answer=answer[:2000],
        )
        return await self._invoke(prompt, "faithfulness")

    async def answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> JudgeScore:
        """
        Answer Relevance.
        Measures how directly and completely the *answer* addresses the
        *question*.
        """
        if not answer.strip():
            return JudgeScore(
                score=0.0,
                reasoning="Empty answer — cannot assess relevance.",
                metric="answer_relevance",
            )
        prompt = _ANSWER_RELEVANCE_PROMPT.format(
            question=question[:1000],
            answer=answer[:2000],
        )
        return await self._invoke(prompt, "answer_relevance")

    async def context_precision(
        self,
        query: str,
        context: str,
    ) -> JudgeScore:
        """
        Context Precision.
        Measures what fraction of the *retrieved context* is actually
        relevant to the *query*.
        """
        if not context.strip():
            return JudgeScore(
                score=0.0,
                reasoning="No context retrieved — precision is undefined (scored 0).",
                metric="context_precision",
            )
        prompt = _CONTEXT_PRECISION_PROMPT.format(
            query=query[:800],
            context=context[:3000],
        )
        return await self._invoke(prompt, "context_precision")

    async def context_recall(
        self,
        reference_answer: str,
        model_answer: str,
    ) -> JudgeScore:
        """
        Context Recall.
        Measures what fraction of the key facts in the *reference_answer*
        are also present in the *model_answer*.
        """
        if not reference_answer.strip():
            return JudgeScore(
                score=1.0,
                reasoning="No reference answer provided — recall trivially satisfied.",
                metric="context_recall",
            )
        if not model_answer.strip():
            return JudgeScore(
                score=0.0,
                reasoning="Empty model answer — no facts recalled.",
                metric="context_recall",
            )
        prompt = _CONTEXT_RECALL_PROMPT.format(
            reference=reference_answer[:2000],
            answer=model_answer[:2000],
        )
        return await self._invoke(prompt, "context_recall")

    # ------------------------------------------------------------------
    # Convenience: score all four metrics concurrently
    # ------------------------------------------------------------------
    async def score_all(
        self,
        query: str,
        answer: str,
        context: str,
        reference_answer: str = "",
    ) -> dict[str, JudgeScore]:
        """
        Run all four judge metrics concurrently and return a dict keyed by
        metric name.

        Returns::
            {
                "faithfulness":      JudgeScore,
                "answer_relevance":  JudgeScore,
                "context_precision": JudgeScore,
                "context_recall":    JudgeScore,
            }
        """
        faith, relevance, precision, recall = await asyncio.gather(
            self.faithfulness(answer=answer, context=context),
            self.answer_relevance(question=query, answer=answer),
            self.context_precision(query=query, context=context),
            self.context_recall(reference_answer=reference_answer, model_answer=answer),
            return_exceptions=False,
        )
        return {
            "faithfulness":      faith,
            "answer_relevance":  relevance,
            "context_precision": precision,
            "context_recall":    recall,
        }


# ---------------------------------------------------------------------------
# Module-level singleton  (mirrors get_chatbot() pattern)
# ---------------------------------------------------------------------------
_judge: Optional[LLMJudge] = None


def get_judge() -> LLMJudge:
    """Return the module-level LLMJudge singleton (lazy init)."""
    global _judge
    if _judge is None:
        _judge = LLMJudge()
    return _judge