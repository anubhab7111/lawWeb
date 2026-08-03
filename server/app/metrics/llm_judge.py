"""
LLM-as-a-Judge Module  (OpenRouter Edition)
============================================
Uses a free OpenRouter model to evaluate RAG pipeline quality across four
dimensions:

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
  6. Three quota-protection layers keep evaluation runs inside OpenRouter's
     free-tier limits (20 req/min, 50 req/day — 1000/day once the account has
     $10+ in lifetime credit purchases):
       - an on-disk response cache keyed by (model, prompt), so re-running
         the same test suite after an unrelated code change costs nothing;
       - a process-wide rate limiter that spaces real API calls out to stay
         under the per-minute cap;
       - an on-disk daily call counter that refuses new calls (returning a
         neutral JudgeScore.failure) once the configured daily budget is hit,
         instead of hammering the API into a 429.

Requirements
------------
    Uses `httpx` (already a project dependency) directly against
    OpenRouter's OpenAI-compatible /chat/completions endpoint — no extra
    SDK needed.

Environment / config
--------------------
    Set OPENROUTER_API_KEY in server/.env (get one free at openrouter.ai/keys).
    OPENROUTER_MODEL        default: openai/gpt-oss-20b:free
    OPENROUTER_DAILY_LIMIT  default: 50  (raise to 1000 after a $10+ topup)
    See https://openrouter.ai/models?max_price=0 for the current free-model
    catalog — it rotates, so the default above may need updating over time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
OPENROUTER_DEFAULT_MODEL = "openai/gpt-oss-20b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Quota-protection state files (git-ignored; see server/.gitignore)
_STATE_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _STATE_DIR / ".judge_cache.json"
_USAGE_PATH = _STATE_DIR / ".judge_usage.json"

# OpenRouter free-tier hard cap is 20 req/min; stay comfortably under it.
_MIN_SECONDS_BETWEEN_CALLS = 3.2


# ---------------------------------------------------------------------------
# Result container  (unchanged from original)
# ---------------------------------------------------------------------------
@dataclass
class JudgeScore:
    """Structured output from a single LLM judge call."""

    score: float  # [0.0 – 1.0]
    reasoning: str  # brief natural-language explanation
    metric: str  # which metric this score belongs to
    raw_response: str = field(default="", repr=False)  # raw LLM output
    latency_s: float = field(default=0.0)  # wall-clock time

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

# Batched rubric: all four RAG-triad metrics in ONE call, so a full evaluation
# run costs one API request per query (~18/run) instead of four — keeping it
# inside OpenRouter's free-tier 50/day cap. Same rubrics as the single-metric
# prompts above, just scored together.
_RAG_TRIAD_PROMPT = """\
You are a strict evaluation judge for a legal question-answering system.
Score the MODEL ANSWER on four independent metrics, each a float in [0, 1].

USER QUERY:
\"\"\"
{query}
\"\"\"

RETRIEVED CONTEXT (chunks the system retrieved):
\"\"\"
{context}
\"\"\"

MODEL ANSWER:
\"\"\"
{answer}
\"\"\"

REFERENCE ANSWER (gold standard):
\"\"\"
{reference}
\"\"\"

METRICS:
- faithfulness: fraction of the Model Answer's factual claims that are explicitly
  supported by the Retrieved Context. If context is empty, non-trivial claims are
  unsupported (score 0). A pure refusal scores 1.
- answer_relevance: how directly and completely the Model Answer addresses the
  User Query (1 = fully, 0 = not at all).
- context_precision: fraction of the Retrieved Context that is relevant to the
  User Query (1 = all relevant, 0 = all noise / empty).
- context_recall: fraction of the key facts in the Reference Answer that the Model
  Answer also covers (1 = all covered; if reference is empty, score 1).

Respond with ONLY a valid JSON object on a single line, exactly like this:
{{"faithfulness": {{"score": <float>, "reasoning": "<one sentence>"}}, \
"answer_relevance": {{"score": <float>, "reasoning": "<one sentence>"}}, \
"context_precision": {{"score": <float>, "reasoning": "<one sentence>"}}, \
"context_recall": {{"score": <float>, "reasoning": "<one sentence>"}}}}
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
      • Reasoning models (e.g. gpt-oss, nemotron) sometimes wrap chain-of-thought
        in <think>...</think> tags before the JSON
    """
    # Strip reasoning/thinking blocks if present
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


def _extract_batched_json(text: str) -> Optional[dict]:
    """
    Parse the batched RAG-triad response (a JSON object whose values are
    themselves ``{"score", "reasoning"}`` objects). Unlike ``_extract_json``,
    this keeps the OUTER object rather than grabbing the first inner
    ``{"score": ...}``.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Greedy outermost {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _score_from_metric_block(text: str, metric: str) -> Optional[tuple]:
    """Regex fallback: pull one metric's (score, reasoning) out of a batched
    response the JSON parser couldn't handle. Returns None if not found."""
    m = re.search(
        rf'"{metric}"\s*:\s*\{{[^{{}}]*?"score"\s*:\s*([\d.]+)'
        r'(?:[^{}]*?"reasoning"\s*:\s*"([^"]*)")?',
        text,
        re.DOTALL,
    )
    if not m:
        return None
    return float(m.group(1)), (m.group(2) or "No reasoning provided.")


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
# Quota protection: on-disk response cache + daily budget + rate limiter
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data))
    except OSError as exc:
        logger.warning("Could not write judge state file %s: %s", path, exc)


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()


def _cache_get(model: str, prompt: str) -> Optional[dict]:
    return _load_json(_CACHE_PATH).get(_cache_key(model, prompt))


def _cache_put(model: str, prompt: str, entry: dict) -> None:
    cache = _load_json(_CACHE_PATH)
    cache[_cache_key(model, prompt)] = entry
    _save_json(_CACHE_PATH, cache)


def _usage_count_today() -> int:
    usage = _load_json(_USAGE_PATH)
    if usage.get("date") != date.today().isoformat():
        return 0
    return int(usage.get("count", 0))


def _usage_increment() -> int:
    today = date.today().isoformat()
    usage = _load_json(_USAGE_PATH)
    count = int(usage.get("count", 0)) + 1 if usage.get("date") == today else 1
    _save_json(_USAGE_PATH, {"date": today, "count": count})
    return count


# Process-wide throttle so concurrent judge calls (evaluator.py runs several
# queries in parallel) still serialize down to OpenRouter's 20 req/min cap.
_rate_lock = asyncio.Lock()
_last_call_monotonic = 0.0


async def _throttle() -> None:
    global _last_call_monotonic
    async with _rate_lock:
        wait = _MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_monotonic)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_monotonic = time.monotonic()


# ---------------------------------------------------------------------------
# Core judge class
# ---------------------------------------------------------------------------
class LLMJudge:
    """
    Async LLM-as-a-judge that scores RAG pipeline outputs using a free
    OpenRouter-hosted model over its OpenAI-compatible chat/completions API.

    Drop-in replacement for the original Ollama/Gemini-backed LLMJudge — the
    public API (faithfulness / answer_relevance / context_precision /
    context_recall) is unchanged, so callers don't need to change.

    Usage::
        judge = LLMJudge()
        score = await judge.faithfulness(answer=..., context=...)
        print(score.score, score.reasoning)

    Configuration (in order of precedence)
    ---------------------------------------
    1. Constructor kwargs
    2. app.config.get_settings()  fields:
         - openrouter_api_key     (str) — from openrouter.ai/keys
         - openrouter_model       (str) — default: openai/gpt-oss-20b:free
         - openrouter_base_url    (str) — default: https://openrouter.ai/api/v1
         - openrouter_daily_limit (int) — default: 50 (free-tier daily cap)
    3. Defaults above.

    Every real API call is cached on disk by (model, prompt) — rerunning the
    same evaluation suite unchanged costs zero quota after the first pass.
    Once ``openrouter_daily_limit`` calls have been made today, further calls
    short-circuit to a neutral ``JudgeScore.failure`` instead of hitting the
    API, so a long test run degrades gracefully rather than erroring out.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        daily_limit: Optional[int] = None,
        max_retries: int = 2,
        retry_delay_s: float = 2.0,
        timeout_s: float = 60.0,
    ) -> None:
        settings = get_settings()

        self._api_key: str = api_key or getattr(settings, "openrouter_api_key", "")
        self._model_name: str = (
            model or getattr(settings, "openrouter_model", None) or OPENROUTER_DEFAULT_MODEL
        )
        self._base_url: str = (
            base_url or getattr(settings, "openrouter_base_url", None) or OPENROUTER_BASE_URL
        )
        self._daily_limit: int = int(
            daily_limit if daily_limit is not None else getattr(settings, "openrouter_daily_limit", 50)
        )

        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._timeout_s = timeout_s

        if not self._api_key:
            logger.warning(
                "OPENROUTER_API_KEY is not set — LLM judge calls will return "
                "neutral (0.5) scores. Get a free key at https://openrouter.ai/keys "
                "and set it in server/.env."
            )

        logger.info(
            "LLMJudge initialised with model=%s daily_limit=%d",
            self._model_name,
            self._daily_limit,
        )

    # ------------------------------------------------------------------
    # Internal async invoke with cache, budget guard, throttle, and retry
    # ------------------------------------------------------------------
    async def _post_chat(
        self, prompt: str, max_tokens: int = 512
    ) -> tuple[Optional[str], float, Optional[str]]:
        """
        One OpenRouter chat call with the daily-budget guard, rate throttle,
        and retry/back-off. Returns ``(raw_content, latency_s, failure_reason)``
        — ``raw_content`` is None (and ``failure_reason`` set) when the call
        could not be completed. Does NOT touch the response cache; callers own
        caching and parsing.
        """
        if not self._api_key:
            return None, 0.0, "OPENROUTER_API_KEY not configured."

        if _usage_count_today() >= self._daily_limit:
            logger.warning(
                "OpenRouter daily judge budget (%d calls) reached; skipping call.",
                self._daily_limit,
            )
            return (
                None,
                0.0,
                f"Daily OpenRouter budget of {self._daily_limit} calls reached. "
                "Raise OPENROUTER_DAILY_LIMIT after a $10+ topup (unlocks "
                "1000/day), or rerun after the daily reset.",
            )

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            await _throttle()
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "X-Title": "LawWeb RAG Evaluation",
                        },
                        json={
                            "model": self._model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0,
                            "max_tokens": max_tokens,
                        },
                    )
                elapsed = time.perf_counter() - t0
                _usage_increment()  # counts against quota whether it parsed or not

                response.raise_for_status()
                data = response.json()
                raw = data["choices"][0]["message"]["content"] or ""
                return raw, elapsed, None

            except httpx.HTTPStatusError as exc:
                last_exc = RuntimeError(
                    f"OpenRouter HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            logger.warning(
                "OpenRouter judge error (attempt %d/%d): %s",
                attempt + 1,
                self._max_retries + 1,
                last_exc,
            )
            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay_s * (attempt + 1))

        logger.error("OpenRouter judge exhausted retries: %s", last_exc)
        return None, 0.0, str(last_exc)

    async def _invoke(self, prompt: str, metric: str) -> JudgeScore:
        """Call OpenRouter with cache + quota protection; return a JudgeScore."""
        cached = _cache_get(self._model_name, prompt)
        if cached is not None and "score" in cached:
            return JudgeScore(
                score=float(cached["score"]),
                reasoning=f"{cached.get('reasoning', '')} [cached]",
                metric=metric,
                raw_response=cached.get("raw", ""),
            )

        raw, elapsed, reason = await self._post_chat(prompt)
        if raw is None:
            return JudgeScore.failure(metric, reason or "unknown error")

        score = _parse_score(raw, metric)
        score.latency_s = elapsed
        _cache_put(
            self._model_name,
            prompt,
            {"score": score.score, "reasoning": score.reasoning, "raw": raw},
        )
        return score

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
    # Batched: all four RAG-triad metrics in a single API call
    # ------------------------------------------------------------------
    _TRIAD_METRICS = (
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    )

    async def evaluate_rag(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        reference: str,
    ) -> Dict[str, JudgeScore]:
        """
        Score all four RAG-triad metrics in ONE OpenRouter call.

        Returns a dict keyed by metric name. This keeps a full evaluation run
        (one call per query) inside the free-tier daily cap, where calling the
        four single-metric methods separately would cost 4x the quota.
        """
        if not answer.strip():
            return {
                m: JudgeScore(0.0, "Empty answer — nothing to evaluate.", m)
                for m in self._TRIAD_METRICS
            }

        prompt = _RAG_TRIAD_PROMPT.format(
            query=query[:800],
            context=context[:3000],
            answer=answer[:2000],
            reference=reference[:2000],
        )

        cached = _cache_get(self._model_name, prompt)
        raw = cached.get("raw") if cached else None
        elapsed = 0.0
        if not raw:  # not cached, or a previous empty response we won't trust
            # gpt-oss is a reasoning model; the 4-metric prompt needs a generous
            # token budget so hidden reasoning doesn't crowd out the JSON answer.
            raw, elapsed, reason = await self._post_chat(prompt, max_tokens=2048)
            if not raw:
                return {
                    m: JudgeScore.failure(m, reason or "empty judge response")
                    for m in self._TRIAD_METRICS
                }
            _cache_put(self._model_name, prompt, {"raw": raw})

        parsed = _extract_batched_json(raw) or {}
        results: Dict[str, JudgeScore] = {}
        for m in self._TRIAD_METRICS:
            sub = parsed.get(m)
            score_reason = None
            if isinstance(sub, dict) and "score" in sub:
                try:
                    score_reason = (
                        float(sub["score"]),
                        str(sub.get("reasoning", "No reasoning provided.")),
                    )
                except (TypeError, ValueError):
                    score_reason = None
            if score_reason is None:  # per-metric regex fallback
                score_reason = _score_from_metric_block(raw, m)
            if score_reason is None:
                results[m] = JudgeScore.failure(
                    m, "metric missing/unparseable in batched judge response"
                )
            else:
                results[m] = JudgeScore(
                    score=score_reason[0],
                    reasoning=score_reason[1],
                    metric=m,
                    raw_response=raw,
                    latency_s=elapsed,
                )
        return results


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
