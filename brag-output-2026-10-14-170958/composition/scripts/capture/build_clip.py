"""Turn a capture into a 30 fps clip with honest speed segments, plus per-frame overlay data.

python build_clip.py <capture_dir> <out.mp4> <out.json> "<start>:<end>:<speed>,..."   (capture-relative seconds)
The JSON holds, per output frame: real seconds since submit, the speed of that stretch, the app's status
label, and nvidia-smi readings (VRAM used/total MiB, GPU util %, temp C, Ollama MiB) at that capture time.
"""
import bisect
import csv
import json
import os
import shutil
import subprocess
import sys

FPS = 30
CROP = "1600:1440:720:0"  # w:h:x:y on the 2560x1440 capture: the chat column, nav to composer

cap_dir, out_mp4, out_json, spec = sys.argv[1:5]
d = json.load(open(os.path.join(cap_dir, "capture.json")))
ev = {e["type"]: e["t"] for e in d["events"]}
t0 = ev["capture_start"]
submit = (ev["submit"] - t0) / 1000
segments = [tuple(float(x) for x in s.split(":")) for s in spec.split(",")]

frame_t = [(f["t"] - t0) / 1000 for f in d["frames"]]
frame_files = [f["file"] for f in d["frames"]]


def frame_at(t):
    i = bisect.bisect_right(frame_t, t) - 1
    return frame_files[max(i, 0)]


status = []
for s in d["timeline"]:
    lab = s["all"].replace("LAWWEB · COUNSEL", "").strip().split("\n")[-1] if s["proseLen"] == 0 else ""
    status.append(((s["t"] - t0) / 1000, lab if lab not in ("L", "") else ""))


def status_at(t):
    lab = ""
    for ts, l in status:
        if ts <= t:
            lab = l
    return lab


gpu = [(int(r[0]) - t0) / 1000 for r in csv.reader(open(os.path.join(cap_dir, "smi_gpu.csv")))]
gpu_rows = list(csv.reader(open(os.path.join(cap_dir, "smi_gpu.csv"))))
apps_rows = list(csv.reader(open(os.path.join(cap_dir, "smi_apps.csv"))))
apps_t = [(int(r[0]) - t0) / 1000 for r in apps_rows]


def smi_at(t):
    r = gpu_rows[max(bisect.bisect_right(gpu, t) - 1, 0)]
    i = max(bisect.bisect_right(apps_t, t) - 1, 0)
    # every app line logged at the same instant as the nearest one
    stamp = apps_rows[i][0]
    ollama = sum(int(a[3]) for a in apps_rows if a[0] == stamp and "ollama" in a[2])
    return {"name": r[1].strip(), "used": int(float(r[2])), "total": int(float(r[3])),
            "util": int(float(r[4])), "temp": int(float(r[5])), "ollama": ollama}


seq = os.path.join(os.path.dirname(out_mp4), os.path.basename(out_mp4) + ".seq")
shutil.rmtree(seq, ignore_errors=True)
os.makedirs(seq)
rows = []
n = 0
for start, end, speed in segments:
    count = round((end - start) / speed * FPS)
    for k in range(count):
        t = start + k * speed / FPS
        os.link(os.path.join(cap_dir, "frames", frame_at(t)), os.path.join(seq, f"{n:05d}.jpg"))
        rows.append({"real": round(t - submit, 2), "speed": speed, "status": status_at(t), **smi_at(t)})
        n += 1

subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
                "-i", os.path.join(seq, "%05d.jpg"), "-vf", f"crop={CROP}",
                "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out_mp4], check=True)
shutil.rmtree(seq)
json.dump({"fps": FPS, "frames": rows, "done_real": round((ev["done"] - ev["submit"]) / 1000, 1),
           "first_text_real": round((ev["first_text"] - ev["submit"]) / 1000, 1) if "first_text" in ev else None},
          open(out_json, "w"), ensure_ascii=False)
print(out_mp4, f"{n} frames = {n / FPS:.2f}s")
