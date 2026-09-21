# Legal Chatbot — Workflow Analysis, Bugs & Production-Readiness Report

**Scope:** `server/app/chatbot.py` (1936 lines) and the modules it orchestrates
(`app/state.py`, `app/tool_dispatch.py`, `app/intent_classifier.py`,
`app/routers/chat.py`), plus the client-side stream consumer
(`client/src/api/index.ts`, `client/src/components/AskAI.tsx`) where it determines
whether a server-side defect is user-visible.

**Date:** 2026-09-21

Each finding carries a verification status:

- **Confirmed** — reproduced, or read directly off the code with no inference.
- **Inferred** — follows from control flow that was read but not executed.

---

## 0. Resolution status (2026-09-21, after implementation)

The sections below are the original analysis, kept as written (with one
correction, marked in B10). This section records what was done about it.

### Bugs

| # | Status | What changed | Guarded by |
|---|---|---|---|
| B1 | **Fixed** | `stream_chat`'s `finally` now cancels the graph task when the consumer is gone. `stream_chat` is a thin wrapper that closes the inner generator explicitly, so this is deterministic rather than GC-timed. | `test_disconnect_cancels_generation_and_frees_slot` (mutation-checked: fails with the cancel removed) |
| B2 | **Fixed, and wider than reported** | Five other endpoints had the same positional-argument bug, not just `/upload`. One `_persist_chat_result` helper now serves all six call sites; `_persist_turn_sync`'s message arguments are keyword-only. | `test_persist_chat_result_stores_canonical_english`, `test_persist_turn_message_arguments_are_keyword_only` |
| B3 | **Fixed** | `stream_chat` runs the compiled graph; `handler_map` and the inline router are gone. Streaming is a contextvar side channel (`emit_event` / `emit_text` / `invoke_llm_safely(stream=True)`). | `test_stream_runs_the_compiled_graph_and_streams_tokens`, `test_chat_and_stream_agree` |
| B4 | **Fixed** | Non-streaming calls use `llm.ainvoke`; `wait_for` now cancels the request itself instead of stranding an executor thread. | `test_timeout_cancels_the_in_flight_request` |
| B5 | **Fixed at the source** | `recommend_lawyers` moves its blocking query to `asyncio.to_thread`, so the `/find-lawyer` endpoint benefits too. Row access in the chat handler now happens inside the session. | (no DB-backed unit test) |
| B6 | **Fixed** | The grounding-unavailable disclaimer streams *before* the answer; corrections arrive as a `replace` event; the client warns when an answer was cut off before verification. | `test_no_retrieval_streams_disclaimer_first_and_skips_verification`, `test_verification_corrections_reach_the_client_before_done` |
| B7 | **Fixed** | `classify_intent` reroutes a document turn with no document to `general_query`, so the reported intent matches the flow that ran. | `test_document_intent_without_document_reroutes_to_general_query` |
| B8 | **Fixed** | Every routing signal is now consumed: `is_ambiguous`/`secondary_intents` drive clarification, `selected_tools` drives what handlers run, and the rest go into the returned `trace`. | clarification tests, `test_select_tools_policy` |
| B9 | **Fixed structurally** | `invoke_llm_safely` defaults to `stream=False`; only the four answer-generating calls pass `stream=True`. | `test_non_streaming_is_the_default_and_never_touches_the_queue` |
| B10 | **Fixed** | Chunk-boundary truncation; canned error text is no longer verified; `max_sessions` enforced on new sessions (with an off-by-one caught by a test); `clear_session` cancels in-flight work; `lawyer_query` clamped; document fences now neutralise `</document>` and the validation path is fenced. | `test_truncate_block_...`, `test_in_memory_fallback_respects_max_sessions`, `test_clear_session_...`, `test_sanitize_...` |

### Agentic improvements

| # | Status | Notes |
|---|---|---|
| A1 unify paths | **Done** | See B3. |
| A2 clarification | **Done, narrowly scoped** | Fires only when an *action* intent (find a lawyer / report a crime) is among near-tied contenders, never with a document, on long messages, or twice in a row. In the live run the router was confident (margin 0.076 vs the 0.03 threshold) on the example I expected to be ambiguous, so expect this to fire rarely; `CLARIFY_ON_AMBIGUOUS=false` disables it. |
| A3 retrieval grading | **Done** | Weak (<3 provisions or mean score < `retrieval_min_confidence`) or empty retrieval triggers one LLM-rewritten, unfiltered retry. The 0.45 threshold comes from a **7-query** sample (in-corpus 0.58-1.00, off-topic 0.34-0.40) and should be re-tuned against the eval set. |
| A4 grounding retry | **Done; live once the correction bug below was fixed** | One targeted regeneration, only on an *adjudicated* report with at least one *corroborated* flag (see Follow-up 3). It was silently unreachable until the wrapper bug described in the last follow-up section was fixed (my earlier note here blamed the model; that was wrong). |
| A5 decomposition | **Done, deterministic** | Splits on question marks / enumerations; the full query is always searched too, so it can only add recall. No LLM call: a 4B model spends seconds on even a trivial split. |
| A6 real tool selection | **Done (option 1)** | `select_tools()` decides what handlers run. `bare_act_lookup` is still not wired to any handler. |
| A7 LangGraph checkpointer | **Done** (follow-up, see below) | Conversation memory now lives in Postgres checkpoints. |
| A8 reasoning trace | **Done** (follow-up, see below) | Returned in the `done` event and `chat()` result, logged, and persisted in `chat_messages.trace`. |

### Production-readiness gaps

Ollama circuit breaker (fail fast for 30s after 3 consecutive failures, single
half-open probe); structured logging with a per-request id for `chatbot.py` and
`tool_dispatch.py` (other tool modules still `print()`); per-caller rate limit
(20/min, `CHAT_RATE_LIMIT_PER_MINUTE`) and a global concurrency cap
(`CHAT_MAX_CONCURRENT`, 503 when busy); unit tests in `test_chatbot.py`, `test_chat_api.py`,
`test_chat_trace.py` and `test_chat_memory.py` (see the follow-up below for the current count) that run the real compiled graph against a fake LLM.

### What the live run showed (real Ollama qwen3:4b, real retrieval, real router)

Four queries, ~13 minutes including cold-start index loading (~20 minutes for
the very first query in a fresh process, ~4.5 minutes of it case-law embedding).

- **The workflow works end to end.** Non-legal short-circuits in 17s with no
  retrieval; the multi-part question decomposed into 2 sub-questions, ran
  retrieval, generated, verified, regenerated once, sent `reset` then `replace`,
  and finished in 190s with every status event arriving in order.
- **Pre-existing model failure, hit 1 time in 3 real generations:** on
  "Is anticipatory bail available for economic offences?" qwen3:4b never closed
  its `<think>` block (`thinking exceeded 20000 chars without closing`) and the
  user got the "please try again" note after ~2 minutes. The code documents this
  failure mode as fixed for the 18-query eval; it is evidently not gone. This
  run also exposed a defect in *my* first implementation, since fixed: it ran
  verification on that canned note and reported `verified: true, score 1.0`.
- **Claim-level correction was not working (diagnosed wrongly at first).** Both
  verification passes logged `LLM correction skipped: No JSON array found`, and I
  reported that the correction LLM "does not return JSON on qwen3:4b". That was
  wrong: the model returns valid JSON in ~3s, and `invoke_llm_safely` discarded
  it (see the last follow-up section). My first implementation also regenerated
  on the deterministic score alone, costing ~60s for a signal `grounding_footer`
  documents as too blunt to act on; it now requires an adjudicated report.
- **Not exercised live:** document analysis/validation, lawyer search, the
  clarification path (the router never called anything ambiguous), the
  disconnect path against a real Ollama stream (covered by the unit test only),
  and the client UI (no browser available; type-checked and bundled only).

### Follow-up: checkpointer, persisted trace and give-up retry (owner approved all three)

**Give-up retry (open question 3).** When the model never closes `<think>`,
`gq_generate` retries once before the user sees "please try again". A bare
re-run is only latency, so the retry differs: statute and case-law blocks only
(~1,400 tokens instead of the full budget), an explicit "answer directly, under
300 words" instruction, and temperature 0.4 instead of 0.1. Qwen3's `/no_think`
soft switch was tried first and does **not** work on this Ollama build (no
closing `</think>` at all in 1,500 tokens). The give-up note is withheld from
the stream on the first attempt, so a successful retry is seamless. It has its
own time budget (`LLM_GIVEUP_RETRY_MAX_ELAPSED_SECONDS`, 330s) because a give-up
itself takes 2-3 minutes; `LLM_GIVEUP_RETRY_ENABLED=false` turns it off.
Measured on the query that gave up live (real retrieval, qwen3:4b, **n=3 each,
so directional**): the normal prompt gave up 1/3 and averaged **123s**; the
concise prompt gave up 0/3 and averaged **21s** (answers ~1,400 chars vs
3,900-5,400). The retry is therefore also ~6x cheaper than the pass it
replaces, which suggests trying the concise prompt earlier for simple queries.
Crime reports and lawyer search have no retry (small prompts; not observed to
give up).

**Persisted trace (open question 2).** Nullable `chat_messages.trace JSONB` on
assistant rows only (`schema.sql`, an idempotent `ensure_chat_messages_trace_column`
migration, the model, and the history endpoint). Migrations otherwise run only
via `python -m app.db.init_db`, and the model now maps the column, so startup
also runs the idempotent `ALTER ... IF NOT EXISTS`; without it an un-migrated
database would fail every logged-in user's chat insert. Sets in the trace are
converted before they reach JSONB.

**Checkpointer (open question 1), one line: LangGraph checkpoints in Postgres are
now the source of truth for the live 20-message conversation window;
`chat_messages` stays the durable, user-visible transcript and re-seeds a thread
that has no checkpoint.** Design decisions, and why:

- *Only message history is checkpointed.* A one-node outer graph holds the
  history; the workflow runs inside it, compiled with `checkpointer=False`. I
  round-tripped the real graph state through LangGraph's serializer first:
  `messages` are fine; a `ToolInvocationResult` survives only while its `raw` is
  plain data (and already warns it "will be blocked in a future version"); it
  **raises** once `raw` holds a real retriever or Indian Kanoon object; and
  exceptions are silently flattened to strings. Uploaded documents (up to 10MB)
  would also have been checkpointed every step. `checkpointer=False` matters: a
  nested graph compiled with `None` *inherits* its parent's saver
  (mutation-checked: the test fails without it).
- *Per-turn inputs and the result travel in a contextvar,* not state, so none of
  it is persisted.
- *Isolation:* `thread_id` is the existing `_memory_key`
  (`user:<id>:<session>` / `guest:<session>`), so guessing a session id still
  cannot read another account's conversation (tested).
- *Degrades, doesn't fail:* if Postgres is unreachable at startup the app logs
  the error and falls back to an in-process saver with the old TTL / size cap
  (guest chat never needed the database).
- *Retention:* every run stamps `last_active` into the checkpoint metadata; a
  6-hourly scheduled job deletes threads idle over `CHAT_THREAD_RETENTION_DAYS`
  (7). Authenticated users lose nothing (re-seeded from `chat_messages`).
- *Behaviour change:* a stopped turn saves the partial text the user saw as the
  assistant turn (as before); a disconnect or an error records the question with
  no reply, and an error message is no longer written into memory.
- New dependencies `langgraph-checkpoint-postgres` and `psycopg-pool` (added to
  `requirements.txt`); the checkpoint tables are created by the saver's own
  idempotent `setup()` at startup, not by `schema.sql`. The chat memory API
  (`has_session`, `seed_session`, `get_session_history`, `clear_session`) is now
  async, and the history/clear endpoints with it.

Tests: 94 in the unit suite (61 new across the work), including real-Postgres
tests on the scratch database for durability across a simulated restart, idle
reaping, `init_checkpointer`, and re-seeding from `chat_messages`.

### Follow-up 2: concise-first for simple queries, and the grounding-correction bug

**Concise prompt by default for simple queries (owner approved).** The first
attempt uses the concise prompt when retrieval is graded *good*, the question has
one part (no decomposition), is at most 30 words, and is not a multi-offense
scenario (`_prefers_concise`). Everything else keeps the full prompt, since depth
is what the concise answer costs. `CONCISE_FIRST_ENABLED=false` reverts; a
give-up on the concise pass still gets the one retry.

Measured on three single-part in-corpus questions (real retrieval, qwen3:4b,
answers scored by the real grounding gate; **n=3 questions, so directional**):

| Question | Full prompt | Concise @0.1 | Concise @0.4 |
|---|---|---|---|
| Anticipatory bail, economic offences | 91s, 2,940 chars | 19s, 1,632 | 16s, 1,183 |
| WhatsApp chats as evidence | 49s, 3,939 | 20s, 1,580 | 19s, 1,426 |
| Oral agreement | 68s, 4,965 | 39s, 1,134 | 22s, 1,853 |

Concise is 2-4x faster and its grounding scores are in the same noisy range
(0.50-1.00; concise answers contain only 1-3 checkable claims, so a single
flagged claim moves the score by 0.5). I kept temperature 0.1 for the first
attempt (deterministic legal text; the difference at 0.4 is within noise) and
left 0.4 as the give-up retry. Through the real graph with everything live, the
two simple questions finished in **51s and 29s** with no give-up; before, a full
pass took 90-130s and gave up roughly 1 time in 3.

**Grounding correction: root cause, and a correction to this report.** I had
written that the correction LLM "does not return JSON on qwen3:4b". Probing the
raw output showed that was wrong. The model returns valid JSON in ~3s: grammar
constrained decoding (`format=schema`) skips the think block entirely. But
`invoke_llm_safely` treated "no `</think>`" as a give-up and returned the
give-up note, so the parser never saw the JSON, `llm_succeeded` stayed `False`,
and neither claim correction nor the regeneration trigger could ever fire. This
was pre-existing (the non-streaming path had always required `</think>` when
`llm_thinking` is on). Fix: a call whose model has `format` set is not held to
the think-block rule (three regression tests, including that an unconstrained
reply without `</think>` is still a give-up).

Consequences, observed live: correction runs (~2-6s per check) and rewrites
unsupported claims from the evidence; the regeneration loop is now real. On the
multi-part question the adjudicated score was 0.27 (15/17 claims flagged, 8
corrected), which triggered the regeneration, whose draft scored 0.53. Two
caveats: the gate still flags a large share of claims on long answers (12/19
after the redraft), so its strictness on long, paraphrased answers deserves a
look; and the regeneration adds ~60s on the full-prompt path (it is not
concise-eligible when the question is multi-part).

### Follow-up 3: the grounding gate was too strict, and it was rewriting correct answers

**What was wrong.** Once correction actually ran (previous section), the gate
flagged 37 of 67 claims across 15 real answers and rewrote 28 sentences. Reading
every flag against the evidence, the problem was not just noise: on the Hindu
Marriage Act answer it replaced six correct, near-verbatim statutory statements
with *"The retrieved evidence does not confirm this claim and recommends
consulting a lawyer."* Causes, each verified in the code and reproduced:

1. **The dropped-qualifier check ran against the whole retrieved-context blob.**
   That always contains "except"/"unless"/"provided that" somewhere, so any
   uncited sentence containing `must` or `cannot` was CONTRADICTED. It also fired
   on faithful restatements: Article 21's "no person shall be deprived … *except*
   according to procedure" restated as "cannot be deprived without procedure".
2. **Wrong evidence for the adjudicator.** For an uncited claim it was the first
   800 characters of the context (the header and first provision); for an
   act-less citation ("Section 56") it was the first Act with that section
   number, i.e. the **Copyright Act** for a Contract Act claim.
3. **Non-claims graded as claims:** markdown headings, table cells, and the
   "I don't have specific references" hedges the prompt *instructs* the model to
   write. The sentence splitter also cut at "B. " and "Ghose v. Mugneeram",
   creating fragments (one became `*Satyabrata Ghose v. The retrieved sources do
   not confirm…`).
4. **Numbers were invisible.** Word overlap looks only at alphabetic words of
   4+ letters, so "maximum punishment of two years" for a provision saying seven
   passed as SUPPORTED. The gate was too lenient exactly where an error is worst.
5. **A 4B model rewriting a sentence from an 800-char window flips meaning in both
   directions**: "§53 is *not* applicable" became "§53 *is* applicable", and
   "marital rape is *not* criminalized for adult spouses" (what the retrieved text
   says) became "*criminalizes*".

**What changed** (`app/tools/grounding_verifier.py`, `app/chatbot.py`):

- A claim is graded against the best-matching *passage* (its cited provision,
  case law, or, if uncited, any retrieved passage), and the adjudicator is shown
  that passage's best-matching 800-char window. An act-less citation resolves to
  the provision that was actually retrieved; a citation to a section that was not
  retrieved is ungrounded, and says so.
- Near-verbatim statute (word-trigram match against the context) is grounded,
  and matching is suffix-insensitive ("established"/"establishes").
- The contradiction check compares against one passage, uses word boundaries, and
  the dropped-qualifier rule now needs an exceptionless marker (`always`, `never`,
  `absolute`, `under any circumstances`), not `must`/`cannot`.
- **New quantity check:** punishments, periods and amounts ("seven years",
  "24 hours", "₹5 lakh") must appear in the evidence; number words fold to digits.
- Headings, table cells, hedges and list lead-ins are not claims; the splitter
  no longer breaks on enumerators and abbreviations.
- Cited claims that are not near-verbatim are sent to the adjudicator, since
  overlap cannot tell a faithful paraphrase from a fabricated rule about the same
  section. A "SUPPORTED" verdict never rewrites text.
- **Two signals must agree before the system acts visibly.** A flag is
  *confirmed* only if the deterministic pass also condemned the claim
  (`GroundingReport.confirmed_flagged`). Only confirmed claims are rewritten,
  trigger the regeneration, or are itemised in the footer. An LLM-only objection
  still lowers the score and gets one honest summary line ("N other statement(s)
  could not be confirmed"); it never edits text.

**Measured** (same 15 real answers from 8 questions, plus 9 hand-written faults:
fabricated sections and amounts, reversed rules, an invented condition):

| | Before | After |
|---|---|---|
| Claims graded (non-claims removed) | 67 | 48 |
| Claims flagged | 37 | 22 (advisory) |
| **Sentences rewritten** | **28** | **2** |
| Answers that would regenerate | 3 | 0 |
| Injected faults caught | 7/9 | **8/9** |

Through the real graph, the multi-part question that previously scored 0.27 on
noisy flags, regenerated and took 187s now takes 144s with no regeneration and no
rewrites. Reading the 22 residual flags by hand, roughly half are real problems
(e.g. "privacy remains protected even during emergencies unless overridden",
which is wrong because Article 21 cannot be suspended; "IPC §376 explicitly
defines marital rape"; an unverifiable quoted Karnataka amendment); the rest are
mild noise (a correct WhatsApp summary; a paraphrase of §55). They are advisory
only now. 19 new tests (`test_grounding_gate.py`); four mutants of the key fixes
are all killed, which exposed one missing test (a mis-cited section riding on a
neighbouring provision's wording), now added.

**Limits, stated plainly.** The corpus is small (15 answers, 8 questions), the
faults are 9 and hand-written by me, and I judged the flags myself: treat the
numbers as directional, not a benchmark. The one missed fault ("divorce solely
because one spouse earns less", citing a real section) is the inherent limit of a
lexical gate: fabricated content in familiar vocabulary needs the adjudicator,
which is non-deterministic run to run (flag counts moved by ±1-3 between
identical runs) and is therefore never trusted to edit on its own. Thresholds
(`_CITED_REVIEW_BELOW` 0.6, `_VERBATIM_FRACTION` 0.6) are heuristics tuned on this
corpus. Not done: pronoun-style back-references ("This provision establishes…")
do not inherit the previous sentence's citation.

### Still open

1. **Not exercised against real Ollama:** the checkpointer with a real generation
   (the real app lifespan *was* booted against Postgres in a test, with the model
   warmup stubbed, and a chat turn against a fake LLM used the real Postgres
   saver), document/lawyer flows, and the browser UI.

---

## 1. How the workflow actually runs today

```
                     ┌─────────────────────────────────────────┐
  POST /api/chat ───►│ LegalChatbot.chat()                     │
                     │   preprocess_query()  (detect+translate)│
                     │   graph.ainvoke()  ◄── compiled LangGraph
                     │   postprocess_response()                │
                     └─────────────────────────────────────────┘

                     ┌─────────────────────────────────────────┐
  POST /api/chat/    │ LegalChatbot.stream_chat()              │
       stream    ───►│   preprocess_query()                    │
                     │   classify_intent()      ◄── called DIRECTLY
                     │   route_by_intent()      ◄── called DIRECTLY
                     │   handler_map[intent]()  ◄── called DIRECTLY
                     │   postprocess_response()                │
                     └─────────────────────────────────────────┘
```

The graph itself is a single hop — `START → classify_intent → [route] → handler → END`:

| Node | Handler | Tools (hard-wired via `INTENT_TOOL_MAP`) |
|---|---|---|
| `document_analysis` | `handle_document_analysis` | Indian Kanoon + criminal RAG, or the 3-layer validation pipeline |
| `crime_report` | `handle_crime_report` | `crime_sections` (IPC/BNS via FAISS, k=2) |
| `general_query` | `handle_general_query` | `indian_kanoon` + `statute_context`, in parallel |
| `find_lawyer` | `handle_find_lawyer` | pgvector lawyer search + optional Indian Kanoon |
| `non_legal` | `handle_non_legal_query` | none (canned refusal) |

Routing is embedding nearest-centroid over 5 classes (`classify_intent_embedding`),
with a history-aware query rewrite (`_rewrite_query_for_retrieval`) for follow-up
turns. `handle_general_query` is the only path with a post-generation grounding
gate (`_verify_response_citations` → citation verifier + grounding verifier with
LLM claim correction).

The engineering around the LLM itself is genuinely careful and well-documented —
the `num_ctx`/`num_predict` sizing, the `</think>` handling, `_fit_context_blocks`,
and the compulsory-RAG disclaimer policy all reflect real measured failure modes.
The problems below are concentrated in **lifecycle, state, and observability**, not
in prompt or retrieval engineering.

---

## 2. Bugs — severity ordered

### B1. Generation is orphaned (and becomes uncancellable) on client disconnect — **Confirmed, reproduced**

`stream_chat` runs the handler as a detached task and registers it for
cancellation, then de-registers it in a `finally`:

- `app/chatbot.py:1688` — `task = asyncio.create_task(run_handler())`
- `app/chatbot.py:1725` — `finally: if self._active_stream_tasks.get(session_id) is task: ... pop(...)`

When the browser aborts the fetch (tab close, navigation, network drop), Starlette
closes the `StreamingResponse` generator, which `aclose()`s `stream_chat`. That runs
the `finally` — **removing the task from `_active_stream_tasks` without cancelling
it**. The handler keeps generating against Ollama, and `stop_stream()`
(`app/chatbot.py:1795`) can no longer find it.

Reproduced with a structurally identical standalone asyncio script:

```
[main] after aclose: active={'s1': (done=False, cancelled=False)}
[stream_chat] FINALLY ran -> popping task from active dict
[main] 0.6s later: handler still running? [True]  active dict=[]
[main] 1.1s later: handler still running? [True]
[handler] FINALLY (cancelled or finished)     <-- ran to completion, never cancelled
```

**Impact.** On a single-worker, 4GB-VRAM box where one generation can hold the GPU
for 180s, every abandoned tab pins the LLM for the full `_LLM_TIMEOUT_SECONDS`.
A user who reloads the page three times has three concurrent, unkillable
generations queued ahead of their fourth. This is the highest-severity item in the
report. The explicit Stop button works (`AskAI.tsx` calls `stopChatStream()`
alongside `abort()`); it is only the implicit disconnect that leaks.

**Fix.** Cancel the task in the `finally` when the generator is closing without
having consumed the terminal event:

```python
finally:
    if self._active_stream_tasks.get(session_id) is task:
        self._active_stream_tasks.pop(session_id, None)
        if not task.done():
            task.cancel()          # client went away — stop burning GPU
    else:
        superseded = True
```

Guard it so a normal completion (where `task` is already done) is unaffected.

---

### B2. `/upload` persists translated text as canonical English, corrupting reseeded history — **Confirmed**

`app/routers/chat.py:471`:

```python
await _persist_turn(session, user, session_id, message, result.get("response", ""))
```

`_persist_turn_sync`'s signature (`app/routers/chat.py:180-188`) is
`(session, user, session_id, user_message, assistant_message, language="en",
user_message_display=None, assistant_message_display=None)`. So this call:

- stores `result["response"]` — the **display** text, already translated out of
  English by `postprocess_response()` — into `content`, the column documented as
  "canonical English text stored in `content` (memory is language-independent)";
- stores the raw `message` (the user's original language) as canonical English;
- leaves `language="en"` and both `*_display` columns `NULL`.

The `/chat` endpoint does this correctly (`chat.py:307-317`, using `response_en` /
`query_en`), as does `/stream` (`chat.py:365-377`). `/upload` is the outlier.

**Impact.** For any non-English document-upload turn the DB row is mislabelled. On
a later turn, `_seed_from_db_if_needed` → `seed_session()` primes the in-memory
LangGraph history with Hindi/Bengali text that the whole downstream pipeline
(routing, retrieval, `_rewrite_query_for_retrieval`, `conversation_context`) treats
as English. Retrieval quality on the follow-up degrades silently, and the history
re-renders in the wrong language because `content_display` is empty.

**Fix.** Make the call keyword-based and mirror `/chat`:

```python
language = result.get("language", "en")
is_translated = language != "en"
await _persist_turn(
    session, user, session_id,
    user_message=result.get("query_en") or message,
    assistant_message=result.get("response_en") or result.get("response", ""),
    language=language,
    user_message_display=message if is_translated else None,
    assistant_message_display=result.get("response") if is_translated else None,
)
```

Worth auditing `/analyze-document`, `/crime-report` and the other `_persist_turn`
call sites for the same positional-argument drift, and changing
`_persist_turn_sync` to keyword-only (`*,` after `session_id`) so this class of bug
cannot recur.

---

### B3. The compiled LangGraph is dead code on the streaming path — **Confirmed**

`stream_chat` does not call `self.graph`. It re-implements the graph inline:

- `app/chatbot.py:1656` — `classified_state = await classify_intent(initial_state)`
- `app/chatbot.py:1657` — `intent = route_by_intent(classified_state)`
- `app/chatbot.py:1660` — a `handler_map` dict duplicating `build_legal_chatbot_graph`'s edges

Only `chat()` (`app/chatbot.py:1861`) actually runs `graph.ainvoke()`. Since the
client's primary UX is `sendChatMessageStream` (`AskAI.tsx:214`), **the LangGraph
is exercised almost exclusively by the non-streaming endpoint and the eval
harness** — i.e. by tests, not by users.

**Impact.** Two divergent execution paths that must be kept in sync by hand. Any
new node (a clarification step, a retrieval-grading step, a retry loop — all
recommended in §3) added to `build_legal_chatbot_graph` will silently not run for
real traffic. This is the structural blocker for every agentic improvement below.

**Fix.** Make streaming a property of the graph rather than a bypass of it. Two
viable routes:

1. Use `graph.astream_events(...)` / `astream(..., stream_mode="custom")` so a
   single compiled graph serves both endpoints, and drop `handler_map` entirely.
2. If the contextvar-queue mechanism is kept, at minimum derive `handler_map` from
   the same dict the graph is built from, so the two cannot drift.

---

### B4. A timed-out non-streaming LLM call leaks its worker thread for up to 210s — **Confirmed by inspection**

`app/chatbot.py:387`:

```python
response = await asyncio.wait_for(
    loop.run_in_executor(None, lambda: llm.invoke([HumanMessage(content=prompt)])),
    timeout=_LLM_TIMEOUT_SECONDS,   # 180
)
```

`asyncio.wait_for` cancels the *future*, not the OS thread. The `llm.invoke` call
keeps blocking in the default `ThreadPoolExecutor` until `ChatOllama`'s own
`timeout=210.0` fires — deliberately set *above* the enforced cap
(`get_llm()`, `chatbot.py:119`).

**Impact.** The default executor has `min(32, cpu_count + 4)` workers, shared
process-wide. Every timeout parks one for an extra 30s beyond the point the
coroutine gave up. The non-streaming path is used by `_invoke_fast_text` (query
rewrite, legal-query parsing), the grounding-correction LLM, the document pipeline
and the defect analyzer — i.e. several calls per request. Under a burst of slow
generations the pool saturates and *unrelated* `run_in_threadpool` work in the
router queues behind it.

**Fix.** Use a bounded, dedicated executor for LLM calls rather than the shared
default, and bring `ChatOllama`'s `timeout` at or below `_LLM_TIMEOUT_SECONDS` so
the thread unwinds when the coroutine does. Longer term, prefer `llm.ainvoke()` —
`ChatOllama` has a native async path, which removes the thread entirely.

---

### B5. Blocking DB I/O on the event loop in `handle_find_lawyer` — **Confirmed**

`app/chatbot.py:924`:

```python
with DBSession(get_engine()) as session:
    lawyers = await recommend_lawyers_core(session, problem_description=lawyer_query, limit=5)
```

`recommend_lawyers` is `async def`, but its actual query is synchronous SQLAlchemy —
`session.exec(stmt).all()` (`app/tools/lawyer_recommender.py:113` and `:120`),
including a pgvector `cosine_distance` ORDER BY over the candidate pool. Nothing
awaits across it, so it blocks the single worker's event loop outright.

This is inconsistent with the router, which deliberately wraps every equivalent
call: *"The DB helpers above are plain blocking SQLAlchemy calls; run them in the
thread pool so a slow query never stalls the (single-worker) event loop"*
(`app/routers/chat.py:263`).

**Impact.** On a single worker, a slow vector scan stalls every other in-flight
request — including the token streams of other users.

**Fix.** Wrap it the same way the router does:

```python
lawyers = await run_in_threadpool(_recommend_lawyers_sync, lawyer_query)
```

*Latent, low priority:* the returned `Lawyer` ORM instances are read after the
`with` block closes the session. It works today (no commit → no expiry, no lazy
relationship touched), but it is one `lazy="select"` relationship away from a
`DetachedInstanceError`.

---

### B6. Grounding corrections and the "unverified" disclaimer do not survive a dropped connection — **Confirmed**

In streaming mode the tokens reach the user *before* any verification runs.
`handle_general_query` then post-processes the completed text:

- `app/chatbot.py:1210` — `final_response = disclaimer_prefix + final_response`
- `app/chatbot.py:1231` — `final_response = await _verify_response_citations(...)`

Both only reach the client inside the terminal `done` event.
`AskAI.tsx:225-229` does handle this (`content: meta.response || acc`), so on the
happy path the corrected text replaces the streamed text.

**Impact.** The failure mode, not the flash-replace, is the production concern: if
the `done` event never arrives — disconnect (see B1), an `error` event, or a
`superseded` event, all of which the client treats as terminal and all of which
fall back to `acc` — the user is left reading the **uncorrected, unverified,
undisclaimed** answer as final. For a legal-information product, the
"grounding unavailable — do not rely on these citations" disclaimer is precisely
the content that must not be the first thing lost when a connection drops.

Secondarily, on the happy path the user watches an answer stream in and then get
silently rewritten under them, with no indication that a correction occurred.

**Fix.** Emit the disclaimer *before* generation starts (it is known from
`rag_succeeded` the moment retrieval returns, well before the first token), and
send verification results as an incremental event the client can append, rather
than folding them into a terminal-only payload. A `{"type": "verification", ...}`
event would also let the UI show *that* a correction happened.

---

### B7. A rerouted document turn reports the wrong intent — **Confirmed**

`app/chatbot.py:668`: when `document_analysis` is selected but no document is
attached and the user is not asking about uploading, the handler delegates:

```python
return await handle_general_query(state)
```

`handle_general_query` returns `{**state, ...}` without touching `intent`, so
`state["intent"]` stays `"document_analysis"`. `stream_chat` then reports
`result.get("intent") or intent` → `"document_analysis"` in the `done` event, and
`/chat` returns the same in `ChatResponse.intent`.

**Impact.** The answer is a general-query answer; the telemetry, the client's
intent-specific UI affordances, and any future analytics all say otherwise. It
also carries `selected_tools=["indian_kanoon"]` (the document set) instead of
general_query's `["indian_kanoon", "statute_context"]` — harmless only because
nothing reads `selected_tools` (see B8).

**Fix.** `return await handle_general_query({**state, "intent": "general_query"})`.

---

### B8. Six routing signals are computed and never read — **Confirmed**

`classify_intent` populates rich routing metadata (`chatbot.py:620-635`). A
whole-tree grep for each key outside `state.py` finds **writes only**:

| Field | Written | Read anywhere? |
|---|---|---|
| `is_ambiguous` | `chatbot.py:601, 629` | **No** |
| `secondary_intents` | `chatbot.py:630` | **No** |
| `routing_confidence` | `chatbot.py:599, 627` | **No** |
| `routing_reasoning` | `chatbot.py:600, 628` | **No** |
| `selected_tools` | `chatbot.py:602, 632` | **No** |
| `active_document_context` | `chatbot.py:604, 634` | **No** |
| `extracted_entities` | `chatbot.py:631` | Only to `print()` at `chatbot.py:1079` |
| `conversation_context` | always `None` (`chatbot.py:1640, 1845`) | **No** — `handle_general_query` builds its own local of the same name |

`intent_classifier.py` computes `AMBIGUITY_MARGIN = 0.03` and
`SECONDARY_INTENT_MARGIN = 0.05` to produce signals that are then discarded.

**Impact.** Not a crash, but it is the single clearest statement of the gap between
the intended agentic design and what runs: the system *knows* when it is unsure and
*knows* a query has a secondary intent, and acts on neither. `INTENT_TOOL_MAP` is
described in its own comment as "metadata for `state["selected_tools"]`" while each
handler calls its tools directly — so the tool-selection layer is documentation,
not control flow. See §3 for what to do with these signals.

---

### B9. `stream=True` is the wrong default for a shared helper — **Confirmed mechanism, no live leak**

`invoke_llm_safely(llm, prompt, stream=True)` routes output to whatever
`_stream_queue_var` holds. `run_handler` sets that contextvar
(`chatbot.py:1675`), and `asyncio.create_task` **copies the current context** — so
every child task spawned inside a handler inherits the user's live token queue.
Handlers do spawn such children: `chatbot.py:709-713` (document analysis) and
`chatbot.py:1329` (document validation).

Today nothing leaks, because every callee reachable from a chat handler happens to
pass `stream=False` (`document_analysis_pipeline.py:315`,
`legal_defect_analyzer.py:314`). The four helpers that use the `stream=True`
default — `case_summarizer.py:19`, `case_firac_extractor.py:102`,
`firac_extractor.py:80`, `bare_act_explorer.py:110` — are reached only from their
own routers or the APScheduler job, each of which runs in a separate task context
where the contextvar is unset. (`sync_case_events` *is* called from request
handlers at `routers/cases.py:105` and `:168`, but those are different requests,
not children of a chat handler.)

**Impact.** The safety is incidental, not structural. Adding one `await
summarize_case_event(...)` inside a chat handler would dump case summaries into a
user's answer stream, with no type error and no test to catch it.

**Fix.** Invert the default to `stream: bool = False` and make the ~3 streaming call
sites in `chatbot.py` pass `stream=True` explicitly. Streaming to the user is the
rare, deliberate case; it should be the one that is spelled out.

---

### B10. Smaller correctness and robustness items — **Confirmed**

- **Unclamped prompt inputs.** `_MAX_QUERY_CHARS` (8000) is applied in
  `handle_crime_report` and `handle_general_query`, but not to `lawyer_query` in
  `LAWYER_SEARCH_PROMPT.format(...)` (`chatbot.py:952`). A long free-text problem
  description goes into the prompt unbounded and is front-truncated by Ollama —
  exactly the failure `_fit_context_blocks` was written to prevent elsewhere.
- **`_fit_context_blocks` truncates mid-sentence.** `block[: remaining * 4]`
  (`chatbot.py:149`) can cut a statutory provision mid-word, handing the model a
  half-quoted section it may then cite as if complete. Truncating on a chunk
  boundary (the `•`-delimited entries `_budget_context` already produces) would be
  safer.
- **Verification runs on error text.** If generation raises,
  `final_response = GENERAL_QUERY_ERROR` (`chatbot.py:1206`), and because
  `rag_succeeded` is `True` the code still calls `_verify_response_citations` on a
  386-char apology — a wasted LLM correction round-trip on the exact path that is
  already degraded.
- **`_add_message` bypasses the session cap.** It creates a missing session dict
  directly (`chatbot.py:1596`, cap at `:1602`) without calling `_evict_stale_sessions`, so
  `max_sessions` is enforced only on the read path.
- **`clear_session` does not cancel in-flight work.** It pops `_sessions` and
  `_session_last_access` but leaves `_active_stream_tasks[session_id]` running.
- **Prompt-injection surface (corrected).** This bullet originally claimed
  `DOCUMENT_ANALYSIS_PROMPT` had "no delimiting, no instruction-hierarchy
  reminder". That was wrong: I had read the `.format()` call, not the prompt.
  `DOCUMENT_ANALYSIS_PROMPT` and the analysis-pipeline prompt both fence the
  document in `<document>` tags with an explicit "treat as data, never as
  instructions" line. The real gaps were narrower: `REACT_OBSERVE_PROMPT` (the
  3-layer *validation* path) put the document between bare `---` markers with
  no such instruction, and none of the prompts stopped a file from containing a
  literal `</document>` to close the fence early. Both are now fixed
  (`sanitize_untrusted_document`, and the OBSERVE prompt is fenced).

---

## 3. Improving the agentic workflow

The current graph is a **classify-then-dispatch router**, not an agent: one
classification, one fixed tool set, one generation, one optional post-hoc
correction. There is no point at which the system can look at what it retrieved or
what it wrote and decide to do something different. Below, in the order I would
implement them.

### A1. Unify the execution path first (prerequisite)

Resolve **B3**. Every item below is a new node or edge, and while `stream_chat`
bypasses the graph, each one must be written twice or it will not run for real
users. Nothing else here is worth doing until the streaming path goes through
`graph.astream_events(...)`.

### A2. Act on `is_ambiguous` — add a clarification node

`AMBIGUITY_MARGIN = 0.03` already identifies queries the router cannot separate,
and `classify_intent` currently resolves them by silently defaulting to
`general_query` (`chatbot.py:586-589`). Add a `clarify` node on that edge that asks
one targeted question ("Are you looking for the law on this, or for a lawyer who
handles it?") and short-circuits to `END`.

For a legal product this is a correctness feature, not a UX nicety: answering the
wrong question confidently is the failure users cannot detect. Cost is one extra
turn on a small minority of queries, and it makes an existing, discarded signal
load-bearing.

### A3. Grade retrieval before generating (CRAG)

Today `rag_succeeded` is a pure emptiness check — `bool(results)` in
`tool_dispatch.py`. Non-empty but *irrelevant* retrieval is indistinguishable from
good retrieval, and produces a confidently-grounded answer citing the wrong
provisions.

Add a `grade_retrieval` node between the tools and generation that scores each
retrieved chunk for relevance to the query (the reranker scores are already
computed — `retrieve_statutes` returns `context.confidence` but only `print`s it at
`tool_dispatch.py:265-268`). Then branch:

- **good** → generate as now;
- **partial** → widen: re-run with a higher `k`, or drop `domain_hint`;
- **poor** → rewrite the query and retry once, then fall back to the
  no-context prompt with the existing `GROUNDING_UNAVAILABLE_PROMPT_WARNING`.

This turns the `confidence` number the retriever already produces into a decision.

### A4. Close the loop on the grounding verifier

`_verify_response_citations` already computes `report.overall_score` and a list of
flagged sentences, and repairs them with a single LLM rewrite pass
(`chatbot.py:1023-1044`). But the outcome is terminal — a low score produces a
footer, never a retry.

Add a conditional edge: if `overall_score` is below a threshold *and* the flagged
claims cite provisions that were never retrieved, loop back to retrieval with those
section numbers as an explicit query, then regenerate once. Cap at one retry. This
is the highest-value agentic change for answer quality, because the detection
machinery is already built and merely under-used.

### A5. Decompose multi-part questions

`is_multi_offense` (`chatbot.py:1094`) currently expresses multi-part complexity as
`k=10` instead of `k=8` — one number. Yet the `num_ctx` comments record that
multi-part questions ("privacy limits, FIR quashing, marital rape, force majeure")
are exactly where the model exhausts its budget re-drafting inside `<think>`.

Bumping `k` makes that worse: more context, same budget. Decomposing instead —
split into sub-questions, retrieve per sub-question, answer each with a tight
context block, then synthesise — directly targets the documented failure mode.
`secondary_intents` (currently unread, B8) is the natural trigger signal.

### A6. Replace `INTENT_TOOL_MAP` with real tool selection

`selected_tools` is written and ignored (B8); handlers call their tools directly.
Two options, in increasing order of ambition:

1. **Make the map real** — have handlers iterate `state["selected_tools"]` through
   `RAG_TOOL_REGISTRY`, so tool choice becomes data. This alone would make
   `invoke_bare_act_lookup` (registered at `tool_dispatch.py:399` but unreachable
   from any handler) usable.
2. **Bind the registry to the LLM** as proper tools and let a ReAct-style loop
   choose, with a cap of 2-3 iterations. More capable, but on qwen3:4b at 30-35
   tok/s each extra hop costs real seconds — see the latency note below.

Given the hardware, option 1 is the pragmatic choice now and a prerequisite for
option 2 later.

### A7. Move session state into a LangGraph checkpointer

`_sessions`, `_session_last_access` and `_active_stream_tasks` are process-local
dicts with hand-rolled TTL and LRU eviction, re-seeded from Postgres via
`seed_session()`. This works only because the deployment is pinned to one worker,
and it is the reason history is capped at 20 messages in two places
(`chatbot.py:1602`, `:1923`) that must agree.

A `PostgresSaver` checkpointer against the existing database gives durable state,
thread-scoped history, and multi-worker safety for free, and is what makes any
multi-node loop (A3-A5) resumable rather than restart-from-scratch.

### A8. Persist the reasoning trace

`DocumentValidationInfo` already has a `reasoning_trace` field
(`state.py:74`) populated by the ReAct defect analyzer. Nothing equivalent exists
for the main chat path, and none of it is persisted.

For a legal product, "which provisions were retrieved, which were cited, what the
grounding score was, what was corrected" is the audit trail that makes an answer
defensible after the fact. It is also the only way to debug a bad answer in
production. Store it alongside `ChatMessage`.

---

## 4. Production-readiness gaps beyond the chatbot logic

**Observability.** There are 32 `print()` calls in `chatbot.py`, no `import logging`,
and more `print()` across `tool_dispatch.py` — no levels, no structured fields, no
request or trace id. In a single-worker process serving concurrent streams, the
interleaved output cannot be attributed to a request. Replace with `logging` and
thread a correlation id through `ChatState`; the existing print sites map almost
one-to-one onto structured events (`router.decision`, `rag.retrieved`,
`grounding.flagged`).

**Test coverage.** `tests/unit/` has 10 test files covering calendar sync, case law,
checkout, matching, push, translation, the validator and the vault — and **nothing
for `chatbot.py`**. The only coverage is `tests/test_chatbot.py`, an accuracy sweep
that needs Ollama and takes ~515s/query per the project's own notes. Every bug in
§2 is unit-testable without a model: B1 with a fake queue and a cancelled
generator, B2 by asserting on the persisted row, B7 by asserting the returned
intent, B8 by asserting the fields are consumed. Add a `tests/unit/test_chatbot.py`
with the LLM and RAG stubbed.

**No rate limiting.** `/api/chat/stream` accepts unauthenticated requests
(`get_current_user_optional`) and each one can occupy the GPU for 180s. There is no
limiter middleware in `main.py`. On a 4GB-VRAM single-worker box this is a
trivially exploitable resource exhaustion, and combined with B1 it does not even
require the attacker to keep the connection open. Add per-IP and per-session
concurrency caps — at minimum, reject a new stream while one is already in flight
for the same memory key.

**No retry or circuit breaker on Ollama.** A failed `invoke_llm_safely` surfaces as
`GENERAL_QUERY_ERROR`. If Ollama is down or reloading a model, every request pays
the full 180s timeout before failing. A fast health probe plus a short-circuit on
consecutive failures would turn a 180s hang into an immediate, honest error.

**Config staleness.** `get_llm()` and friends are `@lru_cache()`d over
`get_settings()`, so nothing picks up settings changes without a process restart.
Acceptable given `reload=False` is deliberate, but worth knowing.

**Incidental:** `CLAUDE.md` says Ollama must be running with `qwen3:14b`;
`config.py:18` has defaulted to `qwen3:4b` since the VRAM measurements were done.
Docs nit, not a code issue.

---

## 5. Suggested order of work

| # | Item | Why first |
|---|---|---|
| 1 | **B1** orphaned generation | Resource leak; ~6-line fix; worst impact on this hardware |
| 2 | **B2** `/upload` persistence | Silent multilingual data corruption; ~10-line fix |
| 3 | **B5**, **B4** blocking I/O and thread leak | Stalls every concurrent user on a single worker |
| 4 | Rate limiting + structured logging | Cannot operate or diagnose the service without them |
| 5 | `tests/unit/test_chatbot.py` | Locks in 1-3 and makes the rest safe to attempt |
| 6 | **B3** unify graph and streaming | Prerequisite for every agentic change |
| 7 | **A2**, **A4** clarification + grounding retry | Best quality-per-line: both reuse signals already computed |
| 8 | **A3**, **A5**, **A7** CRAG, decomposition, checkpointer | Larger, dependent on 6 |

Items 1-3 are small, localised and independently shippable. Items 6-8 are a
refactor and should be planned as one.

---

## Appendix: verification notes

- **B1** reproduced with a standalone asyncio script mirroring the
  `event_generator → stream_chat → create_task` structure; output quoted inline.
  Not reproduced against a live Ollama instance.
- **B8** verified by grepping each state key across `app/` excluding `state.py` and
  `__pycache__`; the table reports that grep's complete output.
- **B9** verified by tracing every static caller of the four `stream=True`-default
  helpers. The contextvar-inheritance mechanism is standard
  `asyncio.create_task` behaviour, stated rather than re-tested.
- **B6** verified end-to-end including the client (`AskAI.tsx:225-229`), which is
  what establishes that the happy path is unaffected and only the terminal-event
  failure modes are.
- All other findings are read directly off the cited lines.
