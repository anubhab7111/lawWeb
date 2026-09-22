# Chatbot Guardrails Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two citation-verification gaps in the chatbot's post-generation guardrail gate — statute citations go unchecked whenever this query's retrieval failed, and case-name citations are never checked at all — without adding any new LLM call or measurable latency.

**Architecture:** Both fixes extend the existing deterministic verification stack (`citation_verifier.py`, `grounding_verifier.py`, called from `chatbot.py`'s `_verify_response_citations`/`gq_verify`). B1 adds a `citation_only` mode to `_verify_response_citations` that runs statute-citation-existence checking even when this query's own retrieval failed (the check queries the whole persistent corpus index, not what was retrieved for this query). B2 adds an analogous case-name-existence check via a new `find_case()` lookup on `CaseLawRAGSystem` and a new `case_citation_verifier.py` module, wired into both the citation-only and full verification paths.

**Tech Stack:** Python 3, LangGraph, pytest, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-22-chatbot-agentic-optimization-design.md`

## Global Constraints

- No new LLM calls anywhere in this phase (from the spec's Non-goals).
- No new retrieval and no regeneration triggered by either new check — both are advisory-footer only.
- Every new check fails open (falls back silently) exactly like the existing verifiers: a bug here must never break chat.
- Formal citation-string (AIR/SCC/SCR reporter number) matching is explicitly out of scope — case-name matching only (spec B2, narrowed after inspecting real corpus data).
- Run all Python commands via the `legal_chatbot_env` conda environment: `conda run -n legal_chatbot_env python -m pytest ...` (or `conda activate legal_chatbot_env` first), per `CLAUDE.md`.
- Comment sparingly — only where the logic is genuinely non-obvious (matches this codebase's existing dense-but-purposeful comment style in `chatbot.py`/`citation_verifier.py`/`grounding_verifier.py`).
- All commands below assume cwd `server/` (the plan's file paths are relative to `server/`).

---

## Task 1: Verify statute citations on the ungrounded/no-retrieval path (B1)

**Files:**
- Modify: `app/chatbot.py` (`_verify_response_citations`, ~line 1345-1409; `gq_verify`, ~line 1940-2033)
- Modify: `tests/unit/test_chatbot.py` (`Rig.__init__`'s inner `verify` fake, ~line 160-165; `test_no_retrieval_streams_disclaimer_first_and_skips_verification`, ~line 543-550)

**Interfaces:**
- Consumes: `citation_verifier.verify_citations(answer: str, rag, retrieved_sections=None) -> VerificationReport` (existing), `citation_verifier.verification_footer(report: VerificationReport) -> str` (existing), `unified_legal_rag.get_unified_rag_system()` (existing, has `.initialized: bool` and `.find_section(...)`).
- Produces: `_verify_response_citations(..., citation_only: bool = False) -> tuple[str, Optional[GroundingReport]]` — the new `citation_only` kwarg. Later tasks (Task 4) extend this same function's body; nothing outside this task calls it with `citation_only=True` yet except `gq_verify`, updated in this task.

- [ ] **Step 1: Write the failing direct-unit test for `_verify_response_citations(citation_only=True)`**

Add to `tests/unit/test_chatbot.py`, near the other `_verify_response_citations`-adjacent tests (after `test_verification_corrections_reach_the_client_before_done`, ~line 708):

```python
class _StubUnifiedRag:
    initialized = True

    def find_section(self, act_hint, section, max_parts=1):
        if section == "420":
            return [SimpleNamespace(text="420. Cheating.", act_name="IPC")]
        return []


class _UninitializedRag:
    initialized = False

    def find_section(self, *a, **k):
        raise AssertionError("must not be called when uninitialized")


def test_citation_only_mode_checks_existence_without_llm_or_grounding(monkeypatch):
    monkeypatch.setattr(
        "app.tools.unified_legal_rag.get_unified_rag_system", lambda: _StubUnifiedRag()
    )
    text, report = run(cb._verify_response_citations(
        "Under Section 420 of the IPC this is cheating. Section 999 of the IPC also applies.",
        citation_only=True,
    ))
    assert report is None
    assert "Section 999" in text
    assert "could not be verified against the indexed corpus" in text


def test_citation_only_mode_is_silent_when_every_citation_verifies(monkeypatch):
    monkeypatch.setattr(
        "app.tools.unified_legal_rag.get_unified_rag_system", lambda: _StubUnifiedRag()
    )
    answer = "Under Section 420 of the IPC this is cheating."
    text, report = run(cb._verify_response_citations(answer, citation_only=True))
    assert text == answer and report is None


def test_citation_only_mode_no_ops_when_rag_uninitialized(monkeypatch):
    monkeypatch.setattr(
        "app.tools.unified_legal_rag.get_unified_rag_system", lambda: _UninitializedRag()
    )
    answer = "Section 420 of the IPC applies."
    text, report = run(cb._verify_response_citations(answer, citation_only=True))
    assert text == answer and report is None
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_chatbot.py -k citation_only -v`
Expected: FAIL — `_verify_response_citations()` raises `TypeError: ...unexpected keyword argument 'citation_only'`.

- [ ] **Step 3: Add `citation_only` to `_verify_response_citations`**

Replace the current body of `_verify_response_citations` (chatbot.py, the function starting `async def _verify_response_citations(`) with:

```python
async def _verify_response_citations(
    response_text: str,
    retrieved_sections=None,
    retrieved_context_text: str = "",
    llm_invoke: Optional[Callable[[str], Awaitable[str]]] = None,
    citation_only: bool = False,
) -> tuple:
    """
    Post-generation grounding gate.

    citation_only=True runs ONLY the deterministic statute-citation-existence
    check, never the claim-level grounding pass — used by gq_verify when
    this query's own retrieval failed but generation still produced a real
    answer. Claim-level grounding needs retrieved context text to judge
    claims against (none exists on that path), but citation existence is
    checkable regardless: verify_citations() queries the whole persistent
    corpus index (rag.find_section()), not what this specific query
    retrieved. Always returns report=None in this mode — nothing here
    should ever trigger a regeneration, since there is no better context to
    retarget retrieval at.

    citation_only=False (default, unchanged from before) runs the full
    two-layer gate:

    1. citation_verifier (unchanged): does every 'Section N of the X Act' /
       'Article N' in the answer exist in the indexed corpus under the
       cited act, and was it among what was actually retrieved for this
       query?
    2. grounding_verifier: goes past the citation token to the claim built
       around it — splits the answer into sentences, checks each cited or
       high-risk-absolute claim against its evidence text, and (only for
       what's flagged) uses one batched LLM call to rewrite the unsupported
       part from evidence alone. Supported sentences are never touched.

    Returns (text, report): the answer with advisory footers from both
    layers (silent when everything checks out) and the GroundingReport the
    caller can act on — or (response_text, None) when the gate was skipped,
    citation_only was requested, or an error occurred. Never raises — a
    verifier bug must not break chat.
    """
    try:
        from app.tools.grounding_verifier import ground_and_correct, grounding_footer
        from app.tools.citation_verifier import verify_citations, verification_footer
        from app.tools.unified_legal_rag import get_unified_rag_system

        rag = get_unified_rag_system()

        if citation_only:
            if not rag.initialized:
                return response_text, None
            citation_report = verify_citations(response_text, rag, retrieved_sections)
            if citation_report.checks:
                logger.info(
                    "CitationVerify (no-retrieval path): %s/%s citations verified",
                    len(citation_report.verified),
                    len(citation_report.checks),
                )
            return response_text + verification_footer(citation_report), None

        if not rag.initialized:
            return response_text, None

        corrected_text, report = await ground_and_correct(
            response_text,
            rag,
            retrieved_sections=retrieved_sections,
            retrieved_context_text=retrieved_context_text,
            llm_invoke=llm_invoke,
        )
        if report.citation_report.checks:
            logger.info(
                "CitationVerify: %s/%s citations verified",
                len(report.citation_report.verified),
                len(report.citation_report.checks),
            )
        if report.claim_sentences:
            corrected_count = sum(
                1 for s in report.sentences if s.outcome == "corrected"
            )
            logger.info(
                "GroundingGate: confidence=%.2f flagged=%s/%s corrected=%s",
                report.overall_score,
                len(report.flagged),
                len(report.claim_sentences),
                corrected_count,
            )
        return (
            corrected_text
            + verification_footer(report.citation_report)
            + grounding_footer(report),
            report,
        )
    except Exception:
        logger.exception("Citation verification skipped")
        return response_text, None
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_chatbot.py -k citation_only -v`
Expected: PASS (all 3 new tests).

- [ ] **Step 5: Write the failing test for `gq_verify`'s orchestration change**

`gq_verify` is only exercised indirectly through `Rig` in this test file. First update the `Rig` fake to observe the new `citation_only` argument — in `tests/unit/test_chatbot.py`, replace the `Rig.__init__` body's `verify_calls`/`verify` definitions (~line 150 and ~line 160-165):

```python
        self.verify_calls = 0
        self.citation_only_calls = []
```

```python
        async def verify(text, retrieved_sections=None, retrieved_context_text="",
                          llm_invoke=None, citation_only=False):
            self.verify_calls += 1
            self.citation_only_calls.append(citation_only)
            if not self._reports:
                return text, None
            report = self._reports.pop(0) if len(self._reports) > 1 else self._reports[0]
            return text, report
```

Then replace `test_no_retrieval_streams_disclaimer_first_and_skips_verification` (~line 543-550) with:

```python
def test_no_retrieval_runs_citation_only_verification_not_full_grounding(monkeypatch):
    empty = _statute(ok=False, chunks=0, conf=0.0)
    rig = Rig(monkeypatch, FakeLLM(["General answer."]), _classification("general_query"), [empty])
    events = run(rig.stream())
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens[0] == GROUNDING_UNAVAILABLE_DISCLAIMER
    assert events[-1]["response"].startswith(GROUNDING_UNAVAILABLE_DISCLAIMER)
    assert rig.verify_calls == 1
    assert rig.citation_only_calls == [True]
    assert events[-1]["trace"]["grounding"]["reason"] == "no_retrieval_citation_only"


def test_successful_retrieval_runs_full_verification_not_citation_only(monkeypatch):
    rig = Rig(monkeypatch, FakeLLM(["Answer."]), _classification("general_query"),
              [_statute()], [_report(1.0)])
    run(rig.stream())
    assert rig.citation_only_calls == [False]
```

- [ ] **Step 6: Run to verify the new/changed tests fail for the right reason**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_chatbot.py -k "no_retrieval or successful_retrieval_runs_full" -v`
Expected: FAIL — `rig.verify_calls == 0` (current behavior skips verification entirely on the no-retrieval path) and `trace["grounding"]["reason"] == "no_retrieval"` (old value), not `"no_retrieval_citation_only"`.

- [ ] **Step 7: Update `gq_verify`**

Replace the body of `gq_verify` in `chatbot.py` (from `async def gq_verify(state: ChatState) -> ChatState:` through its closing `return {...}`) with:

```python
async def gq_verify(state: ChatState) -> ChatState:
    """Post-generation grounding gate. Corrects unsupported sentences in
    place; if the answer is still poorly supported it loops back once to
    regenerate with retrieval aimed at the unsupported citations.

    When this query's retrieval failed (rag_succeeded is False) but
    generation still produced a real answer, runs a citation-only pass
    instead of skipping verification altogether — see
    _verify_response_citations(citation_only=True). No regeneration is
    ever triggered on that path: there is no better context to retarget
    retrieval at.
    """
    settings = get_settings()
    response = state.get("response") or ""
    finished = {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": response}],
    }

    # Generation itself failed: the text is a canned apology, nothing to
    # check against anything.
    if state.get("error") == "generation_failed":
        return {
            **finished,
            "trace": _merge_trace(state, grounding={"verified": False, "reason": "generation_failed"}),
        }

    citation_only = not state.get("rag_succeeded")

    await emit_event(
        "status", stage="verifying", label="Checking citations against the statutes…"
    )

    async def _grounding_correction_invoke(prompt: str) -> str:
        raw = await invoke_llm_safely(
            get_grounding_correction_llm(), prompt, stream=False
        )
        return strip_reasoning_tags(raw)

    final_text, report = await _verify_response_citations(
        response,
        state.get("retrieved_sections"),
        retrieved_context_text=state.get("retrieved_context") or "",
        llm_invoke=_grounding_correction_invoke,
        citation_only=citation_only,
    )

    score = report.overall_score if report is not None else None
    # citation_only is also checked explicitly here (not just relied on via
    # report is None): defense in depth so a future change to
    # _verify_response_citations's return contract can't silently reopen
    # regeneration on the no-retrieval path.
    can_regenerate = (
        not citation_only
        and settings.grounding_retry_enabled
        and report is not None
        and report.llm_succeeded
        and report.confirmed_flagged
        and score is not None
        and score < settings.grounding_retry_threshold
        and (state.get("regen_count") or 0) < 1
        and _time_elapsed(state) < settings.request_budget_seconds
    )
    grounding_trace = {
        "verified": report is not None,
        "adjudicated": bool(report is not None and report.llm_succeeded),
        "score": score,
        "flagged": len(report.flagged) if report is not None else 0,
        "regenerated": bool(state.get("regen_count")),
    }
    if citation_only:
        grounding_trace["reason"] = "no_retrieval_citation_only"
    if can_regenerate:
        feedback, queries = _regeneration_plan(report)
        logger.info(
            "Grounding score %.2f < %.2f — regenerating once (targeted: %s)",
            score,
            settings.grounding_retry_threshold,
            queries,
        )
        return {
            **state,
            "regen_pending": True,
            "regen_count": 1,
            "regen_feedback": feedback,
            "extra_queries": queries,
            "retrieval_attempts": 1,
            "grounding_score": score,
            "trace": _merge_trace(state, grounding={**grounding_trace, "regenerating": True}),
        }

    if final_text != response:
        # Corrections and footers land in the client immediately, not only
        # once the terminal event arrives.
        await emit_event("replace", content=final_text)
    return {
        **state,
        "response": final_text,
        "grounding_score": score,
        "messages": state["messages"] + [{"role": "assistant", "content": final_text}],
        "trace": _merge_trace(state, grounding=grounding_trace),
    }
```

- [ ] **Step 8: Run the full chatbot test file to verify everything passes**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_chatbot.py -v`
Expected: PASS — all tests, including the two new/renamed ones from Step 5 and every previously-passing test (in particular `test_low_grounding_score_regenerates_once_with_targeted_retrieval`, `test_unadjudicated_low_score_does_not_regenerate`, `test_generation_failure_is_not_verified`, which exercise `citation_only=False` and must be unaffected).

- [ ] **Step 9: Commit**

```bash
git add app/chatbot.py tests/unit/test_chatbot.py
git commit -m "Verify statute citations even when this query's retrieval failed

gq_verify previously skipped all citation/grounding verification
whenever rag_succeeded was False - exactly the path most likely to be
the model answering from parametric memory. citation_verifier checks
citations against the whole persistent corpus index, independent of
this query's own retrieval, so it can still run there at zero added
cost (no LLM call, no I/O). Claim-level grounding still requires
retrieved context and stays skipped on this path; no regeneration is
ever triggered from it."
```

---

## Task 2: `find_case()` lookup on `CaseLawRAGSystem` (B2, part 1)

**Files:**
- Modify: `app/tools/case_law_rag.py` (add near `normalize_act`/`_acts_match`, ~line 128-141; add method on `CaseLawRAGSystem`, near `retrieve()`, ~line 557)
- Modify: `tests/unit/test_case_law.py`

**Interfaces:**
- Consumes: `CaseRecord` (existing dataclass, `case_law_rag.py`).
- Produces: `normalize_case_name(name: str) -> str`, `find_case_in_cases(cases: Dict[str, CaseRecord], name: str) -> Optional[CaseRecord]`, and `CaseLawRAGSystem.find_case(self, name: str) -> Optional[CaseRecord]`. Task 3 calls `find_case()` (via the `case_rag` object passed to `verify_case_citations`); Task 3's tests call `normalize_case_name` directly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_case_law.py`:

```python
from app.tools.case_law_rag import (
    CaseRecord,
    _statute_overlap_score,
    find_case_in_cases,
    normalize_act,
    normalize_case_name,
)


def _case(case_id, name):
    return CaseRecord(
        case_id=case_id, case_name=name, citation="", court="",
        bench_size=1, court_rank=1, date="", status="reported",
    )


def test_normalize_case_name_ignores_year_vs_spelling_and_whitespace():
    assert normalize_case_name("Ghose v. Mugneeram Bangur (1954)") == normalize_case_name(
        "ghose   vs  mugneeram bangur"
    )
    assert normalize_case_name("Anvar P.V. v. P.K. Basheer (2014)") == normalize_case_name(
        "Anvar P.V. versus P.K. Basheer"
    )


def test_find_case_exact_match_after_normalization():
    cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    found = find_case_in_cases(cases, "shreya singhal vs union of india")
    assert found is not None and found.case_id == "1"


def test_find_case_fuzzy_match_tolerates_a_dropped_word():
    cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    found = find_case_in_cases(cases, "Shreya Singhal v Union India")
    assert found is not None and found.case_id == "1"


def test_find_case_returns_none_for_an_unrelated_name():
    cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    assert find_case_in_cases(cases, "Kesavananda Bharati v. State of Kerala") is None


def test_find_case_returns_none_on_empty_corpus():
    assert find_case_in_cases({}, "Shreya Singhal v. Union of India") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_case_law.py -v`
Expected: FAIL — `ImportError: cannot import name 'find_case_in_cases'` (and `normalize_case_name`).

- [ ] **Step 3: Implement `normalize_case_name` and `find_case_in_cases`**

In `app/tools/case_law_rag.py`, add after `_acts_match` (which ends ~line 141) and before `_statute_overlap_score`:

```python
_CASE_NAME_YEAR_RE = re.compile(r"\(\d{4}\)\s*$")
_CASE_NAME_VS_RE = re.compile(r"\s+(?:v\.?|vs\.?|versus)\s+", re.IGNORECASE)
_CASE_NAME_WS_RE = re.compile(r"\s+")
_CASE_NAME_MATCH_THRESHOLD = 0.7  # false negatives (real case, flagged unverified)
                                   # are safe; false positives (wrong case validated
                                   # as real) are not — bias conservative.


def normalize_case_name(name: str) -> str:
    """Canonical case-name key: drops a trailing "(year)", folds v./vs./versus
    to a single separator, lowercases, collapses whitespace — so "Ghose v.
    Mugneeram Bangur (1954)" == "ghose vs mugneeram bangur"."""
    name = _CASE_NAME_YEAR_RE.sub("", name).strip()
    name = _CASE_NAME_VS_RE.sub(" v ", name)
    return _CASE_NAME_WS_RE.sub(" ", name).strip().lower()


def _case_name_tokens(name: str) -> Set[str]:
    return {t for t in normalize_case_name(name).split() if t != "v"}


def find_case_in_cases(cases: Dict[str, "CaseRecord"], name: str) -> Optional["CaseRecord"]:
    """Look up an indexed case by name: exact match on the normalized name
    first, then a bounded fuzzy (token-Jaccard) match above
    _CASE_NAME_MATCH_THRESHOLD. Returns None (never raises) when nothing
    clears the bar, including on an empty corpus — never call rag.initialize()
    here, this must stay a pure lookup with no I/O."""
    target = normalize_case_name(name)
    if not target:
        return None
    by_norm = {normalize_case_name(c.case_name): c for c in cases.values()}
    if target in by_norm:
        return by_norm[target]
    target_toks = _case_name_tokens(name)
    if not target_toks:
        return None
    best, best_score = None, 0.0
    for c in cases.values():
        toks = _case_name_tokens(c.case_name)
        if not toks:
            continue
        score = len(target_toks & toks) / len(target_toks | toks)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= _CASE_NAME_MATCH_THRESHOLD else None
```

- [ ] **Step 4: Run to verify the module-level tests pass**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_case_law.py -v`
Expected: PASS (all tests including the pre-existing `test_normalize_act`/`test_statute_overlap_matches_index_names`).

- [ ] **Step 5: Add the `find_case` method to `CaseLawRAGSystem` and a test for it**

Add to `tests/unit/test_case_law.py`:

```python
from app.tools.case_law_rag import CaseLawRAGSystem


def test_case_law_rag_system_find_case_delegates_to_the_cases_dict():
    system = CaseLawRAGSystem()
    system._cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    assert system.find_case("Shreya Singhal vs Union of India").case_id == "1"
    assert system.find_case("Nonexistent Case v. Nobody") is None
```

Run it (expect FAIL — `AttributeError: 'CaseLawRAGSystem' object has no attribute 'find_case'`), then add the method to `CaseLawRAGSystem` in `case_law_rag.py`, immediately after `retrieve()` (which ends with `return out`, ~line 557):

```python
    def find_case(self, name: str) -> Optional[CaseRecord]:
        """Look up an indexed case by name — see find_case_in_cases(). A pure
        in-memory lookup, no I/O: callers must check .initialized themselves
        (this returns None either way on an empty/uninitialized corpus, but
        callers should not treat that as "verified nothing exists")."""
        return find_case_in_cases(self._cases, name)
```

- [ ] **Step 6: Run the full test file to verify it passes**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_case_law.py -v`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add app/tools/case_law_rag.py tests/unit/test_case_law.py
git commit -m "Add find_case() lookup to CaseLawRAGSystem

Exact-then-fuzzy (token-Jaccard) match on normalized case names,
biased toward false negatives over false positives. Pure in-memory
lookup - no I/O, no initialization side effects."
```

---

## Task 3: `case_citation_verifier.py` — extraction and verification (B2, part 2)

**Files:**
- Create: `app/tools/case_citation_verifier.py`
- Create: `tests/unit/test_case_citation_verifier.py`

**Interfaces:**
- Consumes: `case_law_rag.normalize_case_name(name: str) -> str` (Task 2). A `case_rag` object exposing `find_case(name: str) -> Optional[CaseRecord]` (Task 2's `CaseLawRAGSystem.find_case`, or a test stub with the same method).
- Produces: `iter_case_citations(answer: str) -> List[Tuple[str, str]]` (list of `(raw_text, normalized_display_name)`), `verify_case_citations(answer: str, case_rag) -> CaseCitationReport`, `case_verification_footer(report: CaseCitationReport) -> str`. Task 4 calls `verify_case_citations` and `case_verification_footer` from `chatbot.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_case_citation_verifier.py`:

```python
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.case_citation_verifier import (
    case_verification_footer,
    iter_case_citations,
    verify_case_citations,
)
from app.tools.case_law_rag import normalize_case_name


class StubCaseRag:
    """find_case as case_citation_verifier uses it."""

    def __init__(self, known_names):
        self._known = list(known_names)

    def find_case(self, name):
        target = normalize_case_name(name)
        for known in self._known:
            if normalize_case_name(known) == target:
                return SimpleNamespace(case_name=known)
        return None


def test_extracts_name_and_year():
    citations = iter_case_citations(
        "As held in Shreya Singhal v. Union of India (2015), restrictions must be reasonable."
    )
    assert citations == [
        ("Shreya Singhal v. Union of India (2015)", "Shreya Singhal v. Union of India")
    ]


def test_does_not_extract_without_a_year():
    # Deliberate precision-over-recall choice: real citations almost always
    # carry a year; requiring one keeps ordinary "X v. Y" prose from being
    # flagged as an unverifiable case citation.
    assert iter_case_citations("Ghose v. Mugneeram Bangur applies here.") == []


def test_deduplicates_repeated_citations():
    text = (
        "Shreya Singhal v. Union of India (2015) held X. Later, "
        "Shreya Singhal v. Union of India (2015) was followed."
    )
    assert len(iter_case_citations(text)) == 1


def test_extracts_through_markdown_bold():
    citations = iter_case_citations(
        "**Shreya Singhal v. Union of India** (2015) is the leading case."
    )
    assert citations[0][1] == "Shreya Singhal v. Union of India"


def test_extracts_names_with_lowercase_connector_words():
    citations = iter_case_citations("See Attorney General for India v. Satish (2021).")
    assert citations[0][1] == "Attorney General for India v. Satish"


def test_a_cited_case_in_the_corpus_is_verified():
    report = verify_case_citations(
        "Shreya Singhal v. Union of India (2015) is the leading case.",
        StubCaseRag(["Shreya Singhal v. Union of India (2015)"]),
    )
    assert report.verified and not report.unverified


def test_a_fabricated_case_is_unverified_and_produces_a_footer():
    report = verify_case_citations(
        "Sharma v. Fictional Union Authority (2019) established this rule.",
        StubCaseRag(["Shreya Singhal v. Union of India (2015)"]),
    )
    assert report.unverified
    footer = case_verification_footer(report)
    assert "Sharma v. Fictional Union Authority" in footer
    assert "could not be verified" in footer


def test_no_citations_produces_no_footer():
    report = verify_case_citations(
        "This is general commentary with no case citation.", StubCaseRag([])
    )
    assert case_verification_footer(report) == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_case_citation_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.tools.case_citation_verifier'`.

- [ ] **Step 3: Implement `case_citation_verifier.py`**

Create `app/tools/case_citation_verifier.py`:

```python
"""
case_citation_verifier.py — post-generation case-name citation check (no LLM).

Extracts every "<Party> v. <Party> (<Year>)"-shaped case-name citation from
a generated answer and checks the name against the indexed landmark-judgment
corpus (case_law_rag.py):

  - verified:   the name matches an indexed case (exact or close fuzzy match)
  - unverified: not found in the indexed corpus (may be a real judgment
                outside this curated database — flagged, not called false)

Formal citation strings (AIR/SCC/SCR reporter numbers) are NOT checked: the
corpus's `citation` field is a heterogeneous blob of many alternate reporter
citations per case in inconsistent formats (inspected directly against the
real corpus during design), unlike the clean, consistent `case_name` field.
Matching those reliably needs its own normalization effort — tracked as
future work, not attempted half-reliably here.

Extraction requires a trailing "(<Year>)" — Indian case citations in this
corpus's style almost always carry one, and requiring it keeps this from
flagging ordinary "X versus Y" prose that has nothing to do with a court
judgment. This is a deliberate precision-over-recall choice: a missed
real citation without a year is a false negative (safe, per this module's
family of checks); flagging unrelated prose as an unverifiable case would
be a false positive (not safe — see citation_verifier.py's docstring for
the same principle applied to statute citations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from app.tools.case_law_rag import normalize_case_name

# A party phrase: a capitalized word, optionally followed by more capitalized
# words or a small set of lowercase connectors ("Union of India", "Attorney
# General for India", "State of Punjab").
_CONN = r"(?:of|for|the|and|de|van|von)"
_WORD = r"[A-Z][A-Za-z.&'’-]*"
_PARTY_RE = rf"{_WORD}(?:\s+(?:{_WORD}|{_CONN})){{0,6}}"

# "\*{0,2}" between the second party and the year tolerates a generated
# answer bold-wrapping the case name ("**X v. Y** (2015)"), the format
# CASE_LAW_CONTEXT_BLOCK's retrieved entries already use.
_CASE_CITE_RE = re.compile(
    rf"(?P<p1>{_PARTY_RE})\s+(?:v\.?|vs\.?|versus)\s+(?P<p2>{_PARTY_RE})"
    rf"\*{{0,2}}\s*\((?P<year>\d{{4}})\)"
)


@dataclass
class CaseCitationCheck:
    raw: str  # e.g. "Shreya Singhal v. Union of India (2015)"
    name: str  # "Shreya Singhal v. Union of India" (year stripped)
    status: str  # verified | unverified


@dataclass
class CaseCitationReport:
    checks: List[CaseCitationCheck] = field(default_factory=list)

    @property
    def verified(self) -> List[CaseCitationCheck]:
        return [c for c in self.checks if c.status == "verified"]

    @property
    def unverified(self) -> List[CaseCitationCheck]:
        return [c for c in self.checks if c.status == "unverified"]


def iter_case_citations(answer: str) -> List[Tuple[str, str]]:
    """Every distinct "<Party> v. <Party> (<Year>)" case-name citation found
    in `answer`, in reading order, deduplicated by normalized name. Returns
    (raw_text, display_name) pairs — display_name has the year stripped."""
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for m in _CASE_CITE_RE.finditer(answer):
        name = f"{m.group('p1')} v. {m.group('p2')}"
        key = normalize_case_name(name)
        if key in seen:
            continue
        seen.add(key)
        out.append((m.group(0), name))
    return out


def verify_case_citations(answer: str, case_rag) -> CaseCitationReport:
    """Check every case-name citation in `answer` against `case_rag.find_case`.
    Caller must confirm case_rag.initialized before calling this — it never
    initializes the RAG system itself (initialization does real I/O and must
    not happen inside a verification gate)."""
    report = CaseCitationReport()
    for raw, name in iter_case_citations(answer):
        found = case_rag.find_case(name)
        report.checks.append(
            CaseCitationCheck(raw=raw, name=name, status="verified" if found else "unverified")
        )
    return report


def case_verification_footer(report: CaseCitationReport) -> str:
    """Advisory footer for unverified case-name citations. Silent when
    everything verified or nothing was cited — same no-noise-on-good-answers
    convention as citation_verifier.verification_footer."""
    if not report.unverified:
        return ""
    lines = ["", "---", "⚠️ **Case citation check** (against the indexed case-law corpus):"]
    for c in report.unverified:
        lines.append(
            f"- {c.name}: could not be verified against the indexed case-law corpus. "
            f"It may be a real judgment outside this database — please verify."
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_case_citation_verifier.py -v`
Expected: PASS (all tests). If `test_extracts_names_with_lowercase_connector_words` or any bold-markdown test fails on a regex edge case, adjust `_CONN`/`_PARTY_RE` and re-run — do not weaken `test_does_not_extract_without_a_year`, which is load-bearing for precision.

- [ ] **Step 5: Commit**

```bash
git add app/tools/case_citation_verifier.py tests/unit/test_case_citation_verifier.py
git commit -m "Add case_citation_verifier: extract and verify case-name citations

New module, no LLM. Extracts '<Party> v. <Party> (<Year>)' citations
(year required, to keep ordinary prose from being flagged) and checks
each against the case-law corpus via CaseLawRAGSystem.find_case().
Formal citation-string matching (AIR/SCC/SCR) is out of scope - see
the module docstring for why."
```

---

## Task 4: Wire case-citation verification into the chat pipeline (B2, part 3)

**Files:**
- Modify: `app/chatbot.py` (`_verify_response_citations`, as left by Task 1)
- Modify: `tests/unit/test_chatbot.py`

**Interfaces:**
- Consumes: `case_citation_verifier.verify_case_citations(answer, case_rag) -> CaseCitationReport` and `case_verification_footer(report) -> str` (Task 3); `case_law_rag.get_case_law_rag_system() -> CaseLawRAGSystem` (existing).
- Produces: no new public interface — `_verify_response_citations`'s existing return contract (`tuple[str, Optional[GroundingReport]]`) is unchanged; case-check footers are now appended to the returned text on both the `citation_only` and full paths.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_chatbot.py`, near the Task 1 direct-unit tests:

```python
class _StubCaseRag:
    def __init__(self, initialized, known_names=()):
        self.initialized = initialized
        self._known = list(known_names)

    def find_case(self, name):
        from app.tools.case_law_rag import normalize_case_name
        target = normalize_case_name(name)
        for known in self._known:
            if normalize_case_name(known) == target:
                return SimpleNamespace(case_name=known)
        return None


def test_citation_only_mode_also_flags_a_fabricated_case_name(monkeypatch):
    monkeypatch.setattr(
        "app.tools.unified_legal_rag.get_unified_rag_system", lambda: _UninitializedRag()
    )
    monkeypatch.setattr(
        "app.tools.case_law_rag.get_case_law_rag_system",
        lambda: _StubCaseRag(True, ["Shreya Singhal v. Union of India (2015)"]),
    )
    answer = "As in Fabricated Party v. Other Party (2019), this rule applies."
    text, report = run(cb._verify_response_citations(answer, citation_only=True))
    assert report is None
    assert "Fabricated Party v. Other Party" in text
    assert "could not be verified against the indexed case-law corpus" in text


def test_citation_only_mode_no_ops_case_check_when_case_rag_uninitialized(monkeypatch):
    monkeypatch.setattr(
        "app.tools.unified_legal_rag.get_unified_rag_system", lambda: _UninitializedRag()
    )
    monkeypatch.setattr(
        "app.tools.case_law_rag.get_case_law_rag_system", lambda: _StubCaseRag(False)
    )
    answer = "As in Fabricated Party v. Other Party (2019), this rule applies."
    text, report = run(cb._verify_response_citations(answer, citation_only=True))
    assert text == answer and report is None
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_chatbot.py -k case_rag -v`
Expected: FAIL — the fabricated-case footer is absent (case checking not wired in yet), so the first test's `assert "Fabricated Party..." in text` fails.

- [ ] **Step 3: Wire the case check into `_verify_response_citations`**

In `chatbot.py`, inside `_verify_response_citations`'s `try` block (as left by Task 1), add the two new imports alongside the existing ones, and layer in the case check on both branches:

```python
    try:
        from app.tools.grounding_verifier import ground_and_correct, grounding_footer
        from app.tools.citation_verifier import verify_citations, verification_footer
        from app.tools.case_citation_verifier import (
            case_verification_footer,
            verify_case_citations,
        )
        from app.tools.unified_legal_rag import get_unified_rag_system
        from app.tools.case_law_rag import get_case_law_rag_system

        rag = get_unified_rag_system()
        case_rag = get_case_law_rag_system()

        if citation_only:
            text = response_text
            if rag.initialized:
                citation_report = verify_citations(response_text, rag, retrieved_sections)
                if citation_report.checks:
                    logger.info(
                        "CitationVerify (no-retrieval path): %s/%s citations verified",
                        len(citation_report.verified),
                        len(citation_report.checks),
                    )
                text += verification_footer(citation_report)
            if case_rag.initialized:
                text += case_verification_footer(verify_case_citations(response_text, case_rag))
            return text, None

        if not rag.initialized:
            return response_text, None

        corrected_text, report = await ground_and_correct(
            response_text,
            rag,
            retrieved_sections=retrieved_sections,
            retrieved_context_text=retrieved_context_text,
            llm_invoke=llm_invoke,
        )
        if report.citation_report.checks:
            logger.info(
                "CitationVerify: %s/%s citations verified",
                len(report.citation_report.verified),
                len(report.citation_report.checks),
            )
        if report.claim_sentences:
            corrected_count = sum(
                1 for s in report.sentences if s.outcome == "corrected"
            )
            logger.info(
                "GroundingGate: confidence=%.2f flagged=%s/%s corrected=%s",
                report.overall_score,
                len(report.flagged),
                len(report.claim_sentences),
                corrected_count,
            )
        text = (
            corrected_text
            + verification_footer(report.citation_report)
            + grounding_footer(report)
        )
        if case_rag.initialized:
            text += case_verification_footer(verify_case_citations(corrected_text, case_rag))
        return text, report
    except Exception:
        logger.exception("Citation verification skipped")
        return response_text, None
```

This replaces the equivalent block Task 1 left in place (the `if citation_only:` branch and the `return (corrected_text + ... , report)` line in particular).

- [ ] **Step 4: Run to verify the new tests pass**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_chatbot.py -k case_rag -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run the full test suite for the touched modules**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/test_chatbot.py tests/unit/test_case_law.py tests/unit/test_case_citation_verifier.py tests/unit/test_grounding_gate.py -v`
Expected: PASS — every test in all four files, confirming Tasks 1-4 compose correctly and nothing in the existing grounding-gate/case-law suites regressed.

- [ ] **Step 6: Commit**

```bash
git add app/chatbot.py tests/unit/test_chatbot.py
git commit -m "Wire case-name citation verification into the chat pipeline

_verify_response_citations now also checks case-name citations (via
case_citation_verifier, Task 3) on both the citation-only and full
verification paths, gated on case_law_rag being initialized. Zero new
LLM calls; fails open like every other check in this stack."
```

---

## Final verification

- [ ] **Run the complete unit test suite**

Run: `conda run -n legal_chatbot_env python -m pytest tests/unit/ -v`
Expected: PASS — no regressions anywhere in the unit suite (this does not run `tests/test_chatbot.py`, the slow live-Ollama eval sweep, which is out of scope for a guardrail-only change and does not need Ollama running for this phase).

- [ ] **Review the diff against the spec's Non-goals**

Confirm: no changes to `app/routers/`, no new entries in `app/config.py`, no new LLM instantiation anywhere in `git diff main`, no changes to retrieval/routing/prompt files. If `git diff --stat` shows anything outside `app/chatbot.py`, `app/tools/case_law_rag.py`, `app/tools/case_citation_verifier.py` (new), and the four touched/created test files, stop and reconcile against the spec before proceeding.
