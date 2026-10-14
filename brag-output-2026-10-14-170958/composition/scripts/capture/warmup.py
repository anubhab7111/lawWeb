"""Warm the live stack (models, indices, IndicTrans2) with throwaway questions and report timings."""
import json
import sys
import time
import urllib.request

QUESTIONS = sys.argv[1:] or [
    "What is the punishment for criminal trespass?",
    "दहेज मांगने पर क्या सजा है?",
]
for i, q in enumerate(QUESTIONS):
    body = json.dumps({"message": q, "session_id": f"warmup-{i}-{int(time.time())}"}).encode()
    req = urllib.request.Request("http://localhost:8000/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    print(f"--- {time.time() - t0:.1f}s lang={data.get('language')} ({data.get('language_confidence')}) intent={data.get('intent')}")
    print("Q:", q)
    print("A:", (data.get("response") or "")[:700])
    print()
