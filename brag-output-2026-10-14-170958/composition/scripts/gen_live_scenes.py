"""Write the two scenes that play live captures (s2-live, s6-multilingual) with their per-frame overlay data inlined.

Run from the composition directory after rebuilding assets/live/*.{mp4,json}. Inlining keeps the timeline
deterministic (no fetch at render time).
"""
import json
from pathlib import Path

LIVE = Path("assets/live")
OUT = Path("compositions")


def compact(name):
    d = json.loads((LIVE / f"{name}.json").read_text())
    statuses, status_idx, ollama, last = [], [], [], 0
    for f in d["frames"]:
        if f["status"] not in statuses:
            statuses.append(f["status"])
        status_idx.append(statuses.index(f["status"]))
        last = f["ollama"] or last  # a few apps samples miss the ollama row; carry the last reading
        ollama.append(last)
    return {
        "fps": d["fps"],
        "real": [f["real"] for f in d["frames"]],
        "speed": [f["speed"] for f in d["frames"]],
        "st": status_idx,
        "statuses": statuses,
        "used": [f["used"] for f in d["frames"]],
        "total": d["frames"][0]["total"],
        "util": [f["util"] for f in d["frames"]],
        "temp": [f["temp"] for f in d["frames"]],
        "ollama": ollama,
        "gpu": d["frames"][0]["name"],
        "first": d["first_text_real"],
        "done": d["done_real"],
    }


BROWSER_CSS = """
        .{p}-card {{ position: absolute; border-radius: 12px; overflow: hidden; background: var(--surface);
          border: 1px solid var(--border-2); box-shadow: 0 30px 70px rgba(30, 26, 17, 0.16), 0 2px 8px rgba(30, 26, 17, 0.06); }}
        .{p}-bar {{ position: absolute; left: 0; right: 0; top: 0; height: 44px; display: flex; align-items: center; gap: 9px;
          padding: 0 16px; background: var(--surface-alt); border-bottom: 1px solid var(--border-2); }}
        .{p}-dot {{ width: 12px; height: 12px; border-radius: 99px; background: #d6ccb8; }}
        .{p}-url {{ margin-left: 14px; flex: 1; height: 28px; border-radius: 6px; background: var(--surface); border: 1px solid var(--border);
          display: flex; align-items: center; padding: 0 12px; font: 500 17px/1 var(--font-mono); color: var(--text-2); }}
        .{p}-view {{ position: absolute; left: 0; right: 0; top: 44px; bottom: 0; overflow: hidden; }}
        .{p}-badge {{ display: inline-flex; align-items: center; padding: 8px 14px; border-radius: 99px;
          font: 600 20px/1 var(--font-body); letter-spacing: 0.02em; }}
        .{p}-badge.fast {{ background: var(--accent-tint); color: var(--accent-dark); border: 1px solid var(--accent-line); }}
        .{p}-badge.real {{ background: var(--green-bg); color: var(--green-fg); border: 1px solid #c6d4bf; }}
"""

OVERLAY_JS = """
          const D = {data};
          const N = D.real.length;
          const el = (id) => document.getElementById(id);
          const setFrame = (i) => {{
            i = Math.max(0, Math.min(N - 1, Math.round(i)));
            const real = D.real[i];
            el("{p}-timer").textContent = real < 0 ? "typing" : real.toFixed(1) + " s";
            const fast = D.speed[i] > 1;
            const badge = el("{p}-speed");
            badge.textContent = fast ? D.speed[i] + "\\u00d7 speed" : "real time";
            badge.className = "{p}-badge " + (fast ? "fast" : "real");
            {extra}
          }};
          setFrame(0);
          const prox = {{ i: 0 }};
          tl.fromTo(prox, {{ i: 0 }}, {{ i: N - 1, duration: N / D.fps, ease: "none", onUpdate: () => setFrame(prox.i) }}, {start});
"""

LIVE_EXTRA = """
            let stage = D.statuses[D.st[i]];
            if (!stage) stage = real < 0 ? "Typing the question" : D.first !== null && real >= D.first ? (real >= D.done ? "Done" : "Streaming the answer") : "Sending";
            el("lv-stage").textContent = stage;
            el("lv-vram").textContent = D.used[i] + " / " + D.total + " MiB";
            el("lv-util").textContent = D.util[i] + " %";
            el("lv-temp").textContent = D.temp[i] + " \\u00b0C";
            el("lv-ollama").textContent = D.ollama[i] + " MiB";
            el("lv-vram-bar").style.transform = "scaleX(" + (D.used[i] / D.total) + ")";
            el("lv-util-bar").style.transform = "scaleX(" + (D.util[i] / 100) + ")";"""


def live_scene(d):
    data = json.dumps(d, separators=(",", ":"))
    return f"""<!doctype html>
<html>
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      <style>
        #lv-root {{ position: absolute; inset: 0; overflow: hidden; }}
        #lv-stage-wrap {{ position: absolute; inset: 0; background: var(--bg); }}
{BROWSER_CSS.format(p="lv")}
        #lv-card {{ left: 120px; top: 226px; width: 900px; height: 780px; }}
        #lv-video {{ position: absolute; left: 0; top: -74px; width: 900px; height: 810px; }}
        #lv-panel {{ position: absolute; left: 1100px; top: 226px; width: 700px; }}
        .lv-k {{ font: 600 20px/1 var(--font-body); letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); }}
        #lv-timer-row {{ display: flex; align-items: center; justify-content: space-between; }}
        #lv-timer {{ font-size: 104px; line-height: 1; color: var(--text-strong); margin-top: 12px; letter-spacing: -0.02em; }}
        #lv-stage {{ font: 500 29px/1.25 var(--font-body); color: var(--text-strong); margin-top: 12px; min-height: 36px; }}
        #lv-smi {{ margin-top: 34px; background: var(--ink); border-radius: 12px; padding: 24px 28px; font: 400 22px/1.5 var(--font-mono); color: var(--ink-fg); }}
        #lv-smi .h {{ color: var(--ink-muted); font-size: 18px; }}
        #lv-smi .g {{ color: var(--accent-line); margin-bottom: 8px; }}
        .lv-row {{ display: flex; align-items: center; gap: 16px; }}
        .lv-row .k {{ width: 104px; color: var(--ink-muted); }}
        .lv-row .v {{ width: 260px; }}
        .lv-track {{ flex: 1; height: 10px; border-radius: 6px; background: #3a3327; overflow: hidden; }}
        .lv-fill {{ width: 100%; height: 100%; background: var(--accent-line); transform-origin: left center; }}
        #lv-note {{ margin-top: 22px; font: 400 21px/1.4 var(--font-body); color: var(--muted); }}
        #lv-sum {{ position: absolute; left: 1100px; top: 952px; width: 700px; font: 600 26px/1.2 var(--font-body); color: var(--text-strong); }}
        #lv-sum em {{ font-style: normal; color: var(--accent); }}
      </style>

      <div id="lv-root" data-composition-id="s2-live" data-width="1920" data-height="1080">
        <div id="lv-stage-wrap">
          <div class="fig-head" id="lv-head">
            <div class="fig-tag">Live capture · this laptop</div>
            <div class="fig-title">One question, <em>answered live</em></div>
          </div>
          <div class="lv-card" id="lv-card">
            <div class="lv-bar"><div class="lv-dot"></div><div class="lv-dot"></div><div class="lv-dot"></div>
              <div class="lv-url">localhost:5173/#/chat · LawWeb</div></div>
            <div class="lv-view">
              <video id="lv-video" src="assets/live/live-en.mp4" data-layout-allow-overflow data-hf-media-start-basis="local" data-start="0.5" data-duration="{len(d['real']) / d['fps']:.2f}" data-track-index="2" muted playsinline></video>
            </div>
          </div>
          <div id="lv-panel">
            <div id="lv-timer-row"><div class="lv-k">Elapsed since submit</div><div class="lv-badge real" id="lv-speed">real time</div></div>
            <div id="lv-timer" class="num">typing</div>
            <div class="lv-k" style="margin-top: 26px;">App status</div>
            <div id="lv-stage">Typing the question</div>
            <div id="lv-smi">
              <div class="h">nvidia-smi · sampled every 250 ms</div>
              <div class="g">{d['gpu']}</div>
              <div class="lv-row"><span class="k">VRAM</span><span class="v" id="lv-vram"></span><div class="lv-track"><div class="lv-fill" id="lv-vram-bar"></div></div></div>
              <div class="lv-row"><span class="k">GPU</span><span class="v" id="lv-util"></span><div class="lv-track"><div class="lv-fill" id="lv-util-bar"></div></div></div>
              <div class="lv-row"><span class="k">Temp</span><span class="v" id="lv-temp"></span></div>
              <div class="lv-row"><span class="k">ollama</span><span class="v" id="lv-ollama"></span></div>
            </div>
            <div id="lv-note">qwen3:4b-instruct via Ollama (PR #64, not merged). Waits shown sped up, streaming in real time.</div>
          </div>
          <div id="lv-sum">First text at <em>{d['first']} s</em> · done at <em>{d['done']} s</em></div>
        </div>
      </div>

      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
          // Fade through paper: hold hidden while the previous scene fades out, then fade in.
          tl.fromTo("#lv-stage-wrap", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.6, ease: (t) => (t < 0.5 ? 0 : 1 - Math.pow(2 - 2 * t, 2)) }}, 0);
          tl.fromTo("#lv-card", {{ y: 24 }}, {{ y: 0, duration: 0.8, ease: "power3.out" }}, 0.2);
          tl.fromTo("#lv-panel", {{ opacity: 0, x: 18 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: "power2.out" }}, 0.5);
          tl.fromTo("#lv-sum", {{ opacity: 0, y: 10 }}, {{ opacity: 1, y: 0, duration: 0.4, ease: "power2.out" }}, 8.9);
{OVERLAY_JS.format(p="lv", data=data, extra=LIVE_EXTRA, start=0.5)}
          tl.to("#lv-stage-wrap", {{ opacity: 0, duration: 0.3, ease: "power1.in" }}, 10.2);
          window.__timelines["s2-live"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
"""


def multilingual_scene(d):
    data = json.dumps({k: d[k] for k in ("fps", "real", "speed")}, separators=(",", ":"))
    steps = [
        ("Detect the language", "fastText lid.176, offline · 16 Indian languages"),
        ("Mask citations", "§ 41 · BNSS · AIR 1978 SC 597 → placeholders"),
        ("Translate to English", "IndicTrans2 200M via CTranslate2, on the CPU"),
        ("Answer in English", "the same LangGraph pipeline"),
        ("Translate back, restore", "citations and markdown put back (PR #65)"),
    ]
    step_html = "\n".join(
        f'            <div class="ml-step"><div class="ml-n num">{i + 1}</div><div><div class="ml-t">{t}</div>'
        f'<div class="ml-d">{dd}</div></div></div>'
        for i, (t, dd) in enumerate(steps)
    )
    return f"""<!doctype html>
<html>
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      <style>
        #ml-root {{ position: absolute; inset: 0; overflow: hidden; }}
        #ml-stage {{ position: absolute; inset: 0; background: var(--bg); }}
{BROWSER_CSS.format(p="ml")}
        #ml-steps {{ position: absolute; left: 120px; top: 262px; width: 700px; }}
        .ml-step {{ position: relative; display: flex; gap: 22px; align-items: flex-start; padding: 20px 0 22px; border-top: 1px solid var(--border-2); }}
        .ml-step:last-child {{ border-bottom: 1px solid var(--border-2); }}
        .ml-n {{ flex: none; width: 46px; height: 46px; border-radius: 99px; border: 1.5px solid var(--accent-line); color: var(--accent-dark);
          display: flex; align-items: center; justify-content: center; font-size: 24px; }}
        .ml-t {{ font: 600 28px/1.15 var(--font-body); color: var(--text-strong); }}
        .ml-d {{ font: 400 22px/1.35 var(--font-mono); color: var(--text-2); margin-top: 6px; }}
        #ml-card {{ left: 880px; top: 250px; width: 880px; height: 763px; }}
        #ml-video {{ position: absolute; left: 0; top: -73px; width: 880px; height: 792px; }}
        #ml-cap {{ position: absolute; left: 880px; top: 200px; width: 880px; display: flex; align-items: center; justify-content: space-between; }}
        #ml-cap-t {{ font: 600 21px/1 var(--font-body); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }}
        #ml-cap-r {{ display: flex; align-items: center; gap: 14px; }}
        #ml-timer {{ font-size: 30px; line-height: 1; color: var(--text-strong); min-width: 96px; text-align: right; }}
      </style>

      <div id="ml-root" data-composition-id="s6-multilingual" data-width="1920" data-height="1080">
        <div id="ml-stage">
          <div class="fig-head" id="ml-head">
            <div class="fig-tag">Fig. 4 · Multilingual</div>
            <div class="fig-title">Hindi in, Hindi out, <em>citations intact</em></div>
          </div>
          <div id="ml-steps">
{step_html}
          </div>
          <div id="ml-cap"><div id="ml-cap-t">Live · Hindi question · with PR #65</div>
            <div id="ml-cap-r"><div class="ml-badge real" id="ml-speed">real time</div><div id="ml-timer" class="num">typing</div></div></div>
          <div class="ml-card" id="ml-card">
            <div class="ml-bar"><div class="ml-dot"></div><div class="ml-dot"></div><div class="ml-dot"></div>
              <div class="ml-url">localhost:5173/#/chat · LawWeb</div></div>
            <div class="ml-view">
              <video id="ml-video" src="assets/live/live-hi.mp4" data-layout-allow-overflow data-hf-media-start-basis="local" data-start="0.5" data-duration="{len(d['real']) / d['fps']:.2f}" data-track-index="2" muted playsinline></video>
            </div>
          </div>
        </div>
      </div>

      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
          tl.fromTo("#ml-stage", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.6, ease: (t) => (t < 0.5 ? 0 : 1 - Math.pow(2 - 2 * t, 2)) }}, 0);
          tl.fromTo("#ml-head", {{ y: 16 }}, {{ y: 0, duration: 0.7, ease: "power2.out" }}, 0);
          tl.fromTo("#ml-card", {{ y: 24 }}, {{ y: 0, duration: 0.8, ease: "power3.out" }}, 0.2);
          tl.fromTo(".ml-step", {{ opacity: 0, x: -14 }}, {{ opacity: 1, x: 0, duration: 0.45, stagger: 0.75, ease: "power2.out" }}, 0.7);
{OVERLAY_JS.format(p="ml", data=data, extra="", start=0.5)}
          tl.to("#ml-stage", {{ opacity: 0, duration: 0.3, ease: "power1.in" }}, 10.0);
          window.__timelines["s6-multilingual"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
"""


(OUT / "s2-live.html").write_text(live_scene(compact("live-en")))
(OUT / "s6-multilingual.html").write_text(multilingual_scene(compact("live-hi")))
print("wrote s2-live.html and s6-multilingual.html")
