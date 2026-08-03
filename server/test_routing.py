"""
Routing-accuracy eval for the embedding-based intent classifier
(app.intent_classifier.classify_intent_embedding) and the embedding-based
domain-hint classifier (app.intent_classifier.classify_domain_hint_embedding).

Cheap by design: pure embedding calls, no Ollama/generation dependency.
Calls the classifier and domain-hint function directly rather than the
full LangGraph chat() — this is a fast regression gate for routing alone,
meant to run on every change to the router.

IMPORTANT: the queries below are independently paraphrased, never copied
verbatim from app.intent_classifier.INTENT_REFERENCE_EXAMPLES (the
classifier's seed set). Testing a nearest-neighbor classifier against its
own seed strings would be close to meaningless — it would trivially score
~100% on exact matches while telling you nothing about generalization.

Usage: python test_routing.py
"""

import asyncio
import time

from app.intent_classifier import classify_domain_hint_embedding, classify_intent_embedding

ROUTING_GROUND_TRUTH = [
    # -- document_analysis --
    {
        "query": "Take a look at this lease I uploaded and tell me if anything's off.",
        "has_document": True,
        "expected_intent": "document_analysis",
    },
    {
        "query": "I attached my will, is it legally sound?",
        "has_document": True,
        "expected_intent": "document_analysis",
    },
    {
        "query": "What do you think of the termination clause in the contract I just sent?",
        "has_document": True,
        "expected_intent": "document_analysis",
    },
    {
        "query": "Can you point out any compliance gaps in the attached NDA?",
        "has_document": True,
        "expected_intent": "document_analysis",
    },
    # -- crime_report --
    {
        "query": "Two guys grabbed my bag and ran off with my wallet on the street.",
        "has_document": False,
        "expected_intent": "crime_report",
    },
    {
        "query": "My ex has been sending me threatening messages every day.",
        "has_document": False,
        "expected_intent": "crime_report",
    },
    {
        "query": "I paid an online store for a laptop that never arrived and they stopped replying.",
        "has_document": False,
        "expected_intent": "crime_report",
    },
    {
        "query": "Someone hacked my email and is blackmailing me with private photos.",
        "has_document": False,
        "expected_intent": "crime_report",
    },
    # -- find_lawyer --
    {
        "query": "Can you point me to a good divorce attorney in Chennai?",
        "has_document": False,
        "expected_intent": "find_lawyer",
    },
    {
        "query": "I want to talk to someone who handles corporate disputes.",
        "has_document": False,
        "expected_intent": "find_lawyer",
    },
    {
        "query": "who handles property partition cases in hyderabad",
        "has_document": False,
        "expected_intent": "find_lawyer",
    },
    {
        "query": "I need someone to represent me in a labor tribunal hearing.",
        "has_document": False,
        "expected_intent": "find_lawyer",
    },
    # -- general_query, including the domain-isolation edge cases --
    {
        "query": "If my landlord wants to evict me without notice, what are my rights as a tenant?",
        "has_document": False,
        "expected_intent": "general_query",
        "expected_domain_hint": None,
    },
    {
        "query": "Can the government restrict my freedom of speech online under public order grounds?",
        "has_document": False,
        "expected_intent": "general_query",
        "expected_domain_hint": None,
    },
    {
        "query": "What IPC sections apply if someone commits forgery and criminal trespass together?",
        "has_document": False,
        "expected_intent": "general_query",
        "expected_domain_hint": "criminal",
    },
    {
        "query": "Is anticipatory bail available for someone accused of cheating under IPC?",
        "has_document": False,
        "expected_intent": "general_query",
        "expected_domain_hint": "criminal",
    },
    {
        "query": "What's required for an oral contract to be enforceable in India?",
        "has_document": False,
        "expected_intent": "general_query",
        "expected_domain_hint": None,
    },
    # -- deliberately terse/ambiguous, exercises the no-LLM ambiguity default --
    {
        "query": "law",
        "has_document": False,
        "expected_intent": "general_query",
    },
]


async def run() -> None:
    print("=" * 70)
    print(f"  Routing accuracy eval — {len(ROUTING_GROUND_TRUTH)} cases")
    print("=" * 70)

    latencies_ms = []
    intent_correct = 0
    domain_hint_checked = 0
    domain_hint_correct = 0
    failures = []

    for i, case in enumerate(ROUTING_GROUND_TRUTH, 1):
        query = case["query"]
        has_document = case["has_document"]

        t0 = time.time()
        result = await classify_intent_embedding(query, has_document)
        intent = "general_query" if result.is_ambiguous else result.primary_intent
        domain_hint = await classify_domain_hint_embedding(query)
        latency_ms = (time.time() - t0) * 1000
        latencies_ms.append(latency_ms)

        intent_pass = intent == case["expected_intent"]
        if intent_pass:
            intent_correct += 1

        domain_pass = True
        if "expected_domain_hint" in case:
            domain_hint_checked += 1
            domain_pass = domain_hint == case["expected_domain_hint"]
            if domain_pass:
                domain_hint_correct += 1

        overall_pass = intent_pass and domain_pass
        status = "PASS" if overall_pass else "FAIL"
        if not overall_pass:
            failures.append((i, case, intent, domain_hint))

        print(
            f"[{i:2d}/{len(ROUTING_GROUND_TRUTH)}] {status}  {latency_ms:6.1f}ms  "
            f"{query[:55]!r:58} -> intent={intent:18} "
            f"(expected={case['expected_intent']}) domain_hint={domain_hint!r}"
        )

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(
        f"Intent accuracy:       {intent_correct}/{len(ROUTING_GROUND_TRUTH)} "
        f"({intent_correct / len(ROUTING_GROUND_TRUTH):.0%})"
    )
    if domain_hint_checked:
        print(
            f"Domain-hint accuracy:  {domain_hint_correct}/{domain_hint_checked} "
            f"({domain_hint_correct / domain_hint_checked:.0%})"
        )
    print(
        f"Latency (excl. cold start): p50={sorted(latencies_ms)[len(latencies_ms) // 2]:.1f}ms "
        f"max={max(latencies_ms):.1f}ms"
    )

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for i, case, intent, domain_hint in failures:
            print(
                f"  [{i}] {case['query'][:60]!r} -> got intent={intent}, domain_hint={domain_hint} "
                f"| expected intent={case['expected_intent']}, "
                f"domain_hint={case.get('expected_domain_hint', '(unchecked)')}"
            )
    else:
        print("\nAll cases passed.")


if __name__ == "__main__":
    asyncio.run(run())
