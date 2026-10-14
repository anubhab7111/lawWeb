"""After `hyperframes render --output ../brag.mp4`: two-pass loudnorm to -16 LUFS / -1.5 dBTP, then bake a poster frame.

Run from the composition directory: python scripts/finish.py [poster_seconds]
"""
import json
import os
import re
import subprocess
import sys

VIDEO, POSTER = "../brag.mp4", "../brag.jpg"
poster_at = sys.argv[1] if len(sys.argv) > 1 else "6.0"


def ff(*args):
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", *args], capture_output=True, text=True)


measure = ff("-i", VIDEO, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-")
m = json.loads(re.findall(r"\{[^{}]+\}", measure.stderr)[-1])
af = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
      f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
ff("-i", VIDEO, "-c:v", "copy", "-af", af, "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
   "-movflags", "+faststart", "../brag.norm.mp4")
os.replace("../brag.norm.mp4", VIDEO)

ff("-ss", poster_at, "-i", VIDEO, "-frames:v", "1", "-q:v", "2", POSTER)
ff("-i", VIDEO, "-i", POSTER, "-filter_complex", "[0:v][1:v]overlay=0:0:enable='eq(n,0)'[v]",
   "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
   "-c:a", "copy", "-movflags", "+faststart", "../brag.poster.mp4")
os.replace("../brag.poster.mp4", VIDEO)

final = ff("-i", VIDEO, "-af", "ebur128=peak=true", "-f", "null", "-").stderr
print("before:", m["input_i"], "LUFS,", m["input_tp"], "dBTP")
print("after:", re.findall(r"I:\s+(-?[\d.]+) LUFS", final)[-1], "LUFS,", re.findall(r"Peak:\s+(-?[\d.]+) dBFS", final)[-1], "dBFS")
