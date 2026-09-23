# Chatbot agentic-workflow optimization — design

**Date:** 2026-09-22
**Status:** Phase 1 approved for implementation. Phases 2-3 scoped, not yet approved in detail.

## Context

The chatbot (`server/app/chatbot.py`, LangGraph orchestration over `qwen3:4b`
on a 15GB RAM / 4GB VRAM box) already runs a disciplined agentic pipeline:
embedding-based routing with an LLM tie-break only on near-ties, a bounded
retrieval-grade-rewrite loop (one retry), a bounded post-generation
grounding-verify-regenerate loop (one retry), a give-up retry on stalled
generation, a circuit breaker, and a "concise first" fast path for simple
well-retrieved questions. Every one of these additions is deterministic-first
and reaches for an LLM call only as a last resort — the right shape for this
hardware, and the reason the system already outperforms what its model size
would suggest (see `[[lawweb-rag-architecture]]`, `[[chatbot-llm-reliability-gotchas]]`
memory).

The request driving this doc: find optimizations/additions to reduce agent
latency, improve response quality, and add guardrails against hallucination,
while handling the full breadth of Indian-law queries — without regressing
the deterministic-first discipline above (confirmed with the user: no new
always-on LLM hops, no self-critique/multi-agent-debate machinery).

User-set priorities (2026-09-22): **guardrails win over latency when the two
trade off**; latency changes are **measured before tuned**, not guessed;
work lands as **separate PRs per phase**; **no fixed deadline**.

## Phasing

1. **Phase 1 — Guardrails** (this doc's implementation scope; approved).
   Two citation-verification gaps, both zero-latency-cost fixes.
2. **Phase 2 — Latency: measure, then tune.** Instrument four currently-
   unmeasured variables driving the ~116s mean latency (PR#27 baseline, see
   `[[eval-harness-state]]`), run a small eval slice, then make evidence-based
   threshold changes. Not detailed in this doc beyond the variables to measure
   — the tuning itself depends on the numbers.
3. **Phase 3 — Query-coverage investigation.** Driven by eval failure modes
   once Phases 1-2 land and the harness is re-run clean, plus already-known
   corpus gaps (`[[lawweb-corpus-gaps]]`: pre-2000 Evidence Act missing
   §65B — now fixed per that memory but worth re-confirming; BNSS glued
   words). Not detailed in this doc.

Each phase is its own branch/PR so it can be reviewed and reverted
independently.

## Phase 1 — Guardrails (implementation scope)

### B1. Verify statute citations on the ungrounded path

**Problem.** `gq_verify` (`chatbot.py:1953`) returns immediately, skipping
*all* citation/grounding verification, whenever `state["rag_succeeded"]` is
False — regardless of whether the LLM still produced a real answer (as
opposed to `_INCOMPLETE_GENERATION_NOTE`). This is exactly the path where the
model is most likely answering from parametric memory rather than retrieved
text, and today it gets only the `GROUNDING_UNAVAILABLE_DISCLAIMER` prefix —
no check that any statute number it invents actually exists.

**Why it's fixable at zero cost.** `citation_verifier.verify_citations()`
checks each extracted citation via `rag.find_section(act_hint, section)`,
which queries the *persistent, whole-corpus* chunk index
(`get_unified_rag_system()`) — not this query's retrieved passages. It works
identically whether or not retrieval succeeded for this specific query, and
costs only regex extraction + dict lookups (no LLM, no I/O).

**Design.**
- In `gq_verify`, split the current combined early-return condition
  (`chatbot.py:1953`) into two cases:
  - `error == "generation_failed"` (canned apology text) → skip everything,
    unchanged.
  - `not rag_succeeded` but generation produced a real answer → run
    `_verify_response_citations` in a **citation-only mode**: call
    `citation_verifier.verify_citations(response, rag, retrieved_sections=None)`
    (no `grounding_verifier.assess_grounding`, since that needs
    `retrieved_context_text` to judge claims against — none exists on this
    path) and append `verification_footer(report)` if non-empty.
  - Do **not** attempt regeneration on this path — there is no better
    context to retarget retrieval at, so `can_regenerate` must stay False
    here regardless of any other condition.
- Guard identically to the existing pattern: if `get_unified_rag_system()`
  is not `.initialized`, no-op (matches `_verify_response_citations`'s
  existing guard).
- Trace: record `grounding={"verified": <ran>, "reason": "no_retrieval_citation_only"}`
  so this path is distinguishable from the full verify path in logs/eval.

**Testing.** Add a case to whatever test covers `gq_verify` (or a focused
unit test on the new branch) that: (a) retrieval failed, (b) the generated
answer cites a nonexistent section, (c) asserts the footer fires. A second
case where the cited section is real should produce a silent (no-footer)
pass, confirming no false positives.

### B2. Verify case-name/citation references, not just statute sections

**Problem.** `citation_verifier.py`'s docstring states case names are "NOT
checked (no case-law corpus yet — Phase 4)". A case-law corpus now exists
(`case_law_rag.py`, `CaseRecord` with `case_name`, `citation`, `court`,
`doctrines`, fully loaded in memory as `self._cases: Dict[str, CaseRecord]`),
and `CASE_LAW_CONTEXT_BLOCK` explicitly instructs the model to cite case
names — so every case citation the model produces is currently unverified,
including outright-fabricated case names (a well-documented LLM failure
mode for legal citations).

**Design.** Inspecting a real sample of `case_law/*.json` while planning
this task showed `case_name` is clean and consistent for every case
("`<Party> v. <Party> (<Year>)`"), but `citation` is a comma-joined blob of
5-50+ alternate reporter citations per case in inconsistent formats (e.g.
one record mixes "AIR 1980 SUPREME COURT 898" and "(1980) 2 SCC 684" and
"1980 (2) SCC 684" for the same judgment). Reliably parsing and matching
formal citation strings against that blob would need its own normalization
effort, not a bounded zero-cost fix — **Phase 1 scopes B2 to case-name
matching only**; formal-citation-string verification (AIR/SCC/SCR number
matching) is deferred as future work, tracked here rather than attempted
half-reliably.

- New extraction regex (sibling to `_CITE_RE` in `citation_verifier.py`, or
  a new `case_citation_verifier.py` module — prefer the latter to keep
  `citation_verifier.py`'s existing statute-only scope and tests
  unentangled) recognizing name-form citations: `<Party> v.? <Party>` /
  `<Party> vs\.? <Party>`, optionally followed by `(<Year>)` (reuse the
  capitalization/word-boundary heuristics already proven in
  `citation_verifier.py`'s sentence-boundary handling and
  `grounding_verifier.py`'s `_ABBREVIATIONS`/`_split_sentences` handling of
  "v."/"vs." to avoid false positives on ordinary "X versus Y" prose or
  mis-splitting a case name at its own abbreviation).
- New lookup on `CaseLawRAGSystem` (`case_law_rag.py`): `find_case(name)
  -> Optional[CaseRecord]`. Normalize (lowercase, strip "v."/"vs."/"versus"
  variants and the trailing "(year)", collapse whitespace) both the query
  and every indexed `case_name`; exact match first, then a bounded fuzzy
  match (token-Jaccard, reusing the pattern `_issue_overlap_score` already
  establishes in this same file) above a conservative threshold — false
  negatives (flagged as unverified when the case is real) are safe, false
  positives (validating a wrong case) are not, so bias the threshold
  accordingly.
- New `verify_case_citations(answer, case_rag) -> CaseCitationReport`
  (mirroring `VerificationReport`'s shape: verified / unverified), called
  from `_verify_response_citations` alongside the existing statute check,
  gated on `case_law_rag.initialized` (fail-open, matching every other
  verifier in this stack).
- Footer: extend `verification_footer` (or add a sibling
  `case_verification_footer`) with the same silent-on-success,
  advisory-on-failure convention already used throughout this module —
  "X v. Y: could not be verified against the indexed case-law corpus. It
  may be a real judgment outside this database — please verify." (never
  "this case does not exist" — the corpus is a curated subset, not
  exhaustive, same epistemic caveat `unverified` already carries for
  statutes).

**Testing.** Cases: (a) a cited case that exists verbatim (name only) in
`case_law/*.json` → silent pass; (b) a fabricated case name → footer fires;
(c) a real case name written with minor formatting variation (extra
whitespace, "vs" instead of "v.", missing the "(year)") → still matches
(exercises the fuzzy-match path) so cosmetic variation doesn't produce a
false positive footer.

## Non-goals for Phase 1

- No changes to retrieval, routing, generation prompts, or the `gq_*` graph
  topology beyond the one-line split in `gq_verify` described in B1.
- No new LLM calls anywhere in this phase.
- No regeneration triggered by either new check — both are advisory-footer
  only, consistent with `unverified`'s existing epistemic caveat (a
  corpus gap not a confirmed hallucination) and with the "no new latency"
  constraint agreed with the user.
- No formal-citation-string (AIR/SCC/SCR reporter number) verification —
  deferred; the `citation` field's format is too heterogeneous across the
  corpus to match reliably without its own normalization effort (see B2).

## Phase 2 preview — variables to instrument (not yet approved in detail)

For the record, so Phase 2 doesn't have to re-derive these: `_prefers_concise`'s
actual hit rate on real queries (`chatbot.py:1818-1836`, four conjunctive
gates); the wall-clock cost of the pre-retrieval `parse_legal_query_llm` hop
in `gq_retrieve` (`chatbot.py:1584-1586`, runs before every primary statute
query); how often `gq_verify`'s `can_regenerate` branch actually fires
(`chatbot.py:1986-1995`, each firing costs roughly a second full generation);
and whether Ollama reuses the KV cache across `_build_answer_prompt`'s
prefix given that `conversation_context` currently varies ahead of the
static instruction template (`chatbot.py:1810-1814`) — needs a live test
against the running Ollama instance, not an assumption.

## Phase 2 findings (2026-09-23)

Measured live on qwen3:4b / Ollama 0.23.0 (n=6 then n=8 queries, in-memory
chat state, real RAG stack). **Caveat:** the box was shared with other
sessions' Ollama/CPU load, which made the first pass ~2x slower than the
second; only the relative stage composition is trustworthy, not absolutes.
Small samples — this is directional evidence, not an eval.

**Instrumentation added** (trace only, no behavior change):
`trace.generation.concise_gates`, `trace.retrieval.query_parse_llm_seconds`,
`trace.grounding.verify_seconds` and `trace.grounding.adjudication`.
Regeneration firing was already visible as `trace.grounding.regenerating`.

**Where the time goes** (second pass):

| Query type | Total | Retrieval | Generation | Verify |
|---|---|---|---|---|
| Simple (6 of 8) | 28-56s | 8-19s | 17-30s | 0-3s |
| Multi-part (2 of 8) | 108-138s | 31-51s | 70-86s | 0-6s |

**The four variables:**
1. *Concise fast path:* fired on 6/6 simple queries; on the 2 multi-part
   queries the only failing gate was `single_part`. No gate is mis-tuned on
   this sample. Loosening `single_part` would trade answer depth (a 300-word
   cap on a 3-part question) for ~3x faster generation — not recommended
   without a quality eval.
2. *Query-parse LLM hop:* fires only when the deterministic ontology parse
   finds no doctrine and no pinned section (4 of 34 ground-truth queries, plus
   multi-part queries); costs **6-18s** when it does. It is a recall assist,
   so it should not be removed.
3. *Regeneration:* 0/14 queries. Not a latency driver on this sample.
4. *KV-cache prefix reuse:* real (prefix-shared prompts evaluated ~9x faster
   per token), but prompt evaluation is ~1-3% of a request (generation is
   the bottleneck), and the bulk of the prompt (retrieved context) is
   query-specific and cannot be a shared prefix. Not worth reordering the
   prompt.

**Unplanned finding — retrieval is CPU-reranker bound.** A warm statute
search is 9-12s, ~98% of it in three CPU cross-encoder passes (main rerank
~20 pairs, second pass 9-63 short pairs, case-law rerank 15 pairs x ~1300
chars, ~3s). The comment in `base_legal_rag.py` that CPU reranking costs
"~1-3s/query" is stale. The reranker cannot move to the GPU (qwen3:4b at
num_ctx 8192 already occupies ~3.65GB of 4GB).

**Tried and rejected (do not retry without new information):**
- *int8 dynamic quantization of the reranker:* only ~12% faster
  (1.17s to 1.03s per batch), scores drift, and `quantize_dynamic` is
  deprecated. The obvious implementation (`reranker.model = quantize_dynamic(...)`)
  also breaks `predict` and the caller **silently falls back to un-reranked
  order**, which makes a retrieval eval look fine while measuring nothing.
- *Schema-constrained decoding for the doctrine pick:* 0.75s vs 6-18s, but it
  returns the first enum entries for every query (also with an in-schema
  "issue" sentence first). The free-text call's thinking does the actual work.
- *A "verify-stage LLM echo" fix:* two 61s/93s verify outliers in the first
  pass did not reproduce (adjudication took 1.8-6.3s), and are most likely
  contention with other sessions. No change made.

**Candidates for follow-up (need their own eval, none implemented):**
1. Embedding-first doctrine matching with the LLM as fallback: on 6 queries,
   embedding top-1 matched the LLM's pick in 4/4 cases where it found one
   (scores 0.56-0.82, ~0.09s vs 6-9s), but the "LLM said none" cases scored
   0.57-0.58, so no clean threshold exists on this sample. A wrong doctrine
   pins irrelevant sections, so this needs a proper eval before use.
2. Fewer/shorter case-law rerank candidates (15 pairs x ~1300 chars is ~3s of
   every retrieval for a corpus of 78 cases and k=3).
3. Per-node latency tracing as a standing part of the eval harness, so
   contention noise can be separated from real regressions.
