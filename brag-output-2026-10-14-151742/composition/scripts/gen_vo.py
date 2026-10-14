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
    ("s1a", "LawWeb answers Indian legal questions on a four gigabyte laptop GPU."),
    ("s2a", "Ask in plain language."),
    ("s2b", "Answers cite the exact section, and every citation is checked against the statutes."),
    ("s3a", "A LangGraph agent routes each message."),
    ("s3b", "Legal questions run a corrective loop: retrieve, grade, rewrite if the evidence is weak, then generate and verify."),
    ("s4a", "Retrieval is hybrid: keyword and dense search, fused, then reranked."),
    ("s4b", "The right section is in the top three, ninety-four percent of the time."),
    ("s5a", "Our statute classifier beats the published IL-TUR baselines by more than ten points."),
    ("s6a", "Then small trained classifiers took over routing, document typing, and crime sections."),
    ("s7a", "Document review now makes two model calls instead of three, and streams as it writes."),
    ("s7b", "About two hundred fifty seconds became ninety-nine."),
    ("s8a", "We benchmarked seven small models, with the decision rule fixed before any of them ran."),
    ("s8b", "The winner shows its first word in one second, not twelve."),
    ("s9a", "And every byte is budgeted, so the model never leaves the GPU."),
    ("s10a", "All of it, measured on held-out tests."),
    ("s11a", "LawWeb. An academic project, open on GitHub."),
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
