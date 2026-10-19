"""Regenerate the narration: one Kokoro clip per sentence into assets/vo/.

Run from the composition directory. `hyperframes tts` needs a Python with
kokoro-onnx + soundfile; point HYPERFRAMES_PYTHON at it, e.g.
    uv venv --python 3.12 ~/.cache/hyperframes-kokoro-venv
    uv pip install --python ~/.cache/hyperframes-kokoro-venv/bin/python kokoro-onnx soundfile
    HYPERFRAMES_PYTHON=~/.cache/hyperframes-kokoro-venv/bin/python python scripts/gen_vo.py [key ...]

Clip start times live in index.html; if a clip's length changes, re-time it there.
"""
import json
import os
import subprocess
import sys

OUT = "assets/vo"
VOICE, SPEED = "af_heart", "1.1"
LINES = [
    ("s1a", "LawWeb answers Indian legal questions, citing the exact section, on a four gigabyte laptop GPU."),
    ("s2a", "Here it is, live: about twelve seconds of search and grading on the CPU, then the first token in about a second, and the answer streams in."),
    ("s3a", "Everything runs locally."),
    ("s3b", "The model has the GPU; embeddings, reranking, classifiers and translation share the CPU."),
    ("s4a", "A LangGraph agent routes each message."),
    ("s4b", "Legal questions loop: retrieve, grade, widen the search if evidence is weak, then generate and verify."),
    ("s5a", "Search is hybrid: keyword and dense, fused and reranked."),
    ("s5b", "On our test questions, the right section is in the top three nine times in ten."),
    ("s6a", "Hindi and fifteen other Indian languages are translated in and out with IndicTrans2, with citations masked so they survive."),
    ("s7a", "On the IL-TUR benchmark, our statute classifier beats the published baselines by ten points."),
    ("s8a", "Small classifiers route messages, type documents and suggest crime sections."),
    ("s9a", "Document review plans once per document type and streams its analysis, more than twice as fast as before."),
    ("s10a", "Seven small models were benchmarked with the rule fixed in advance."),
    ("s10b", "Timed on the LLM call alone: first token in one second, not twelve, at equal quality."),
    ("s11a", "Every byte is budgeted, so the model never leaves the GPU."),
    ("s12a", "Every number here was measured."),
    ("s12b", "The code is on GitHub."),
]

only = set(sys.argv[1:])
os.makedirs(OUT, exist_ok=True)
for key, text in LINES:
    if only and key not in only:
        continue
    path = os.path.join(OUT, f"{key}.wav")
    proc = subprocess.run(
        ["npx", "--yes", "hyperframes@0.8.157", "tts", text, "--voice", VOICE, "--speed", SPEED, "--output", path, "--json"],
        capture_output=True, text=True, timeout=600,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    info = json.loads(line[-1]) if line else {"ok": False, "error": proc.stderr[-400:]}
    print(key, info.get("durationSeconds"), info.get("error", ""), flush=True)
