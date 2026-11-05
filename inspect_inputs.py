"""
Save exactly what the model sees for a few videos (16 sampled frames, 112 x 112, greyscale),
plus timing of video decoding. Usage:
  python inspect_inputs.py /root/folder-docker/EchoNSTE-MI /root/folder-docker/inspect
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "EchoNSTE-MI")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "inspect"); out.mkdir(parents=True, exist_ok=True)
N, SIZE = 16, 112

def read_clip(path):                       # identical to train.py
    cap = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    idx = np.linspace(0, len(frames) - 1, N).round().astype(int)
    return np.stack([frames[i] for i in idx]), len(frames), idx

times = []
for view in ["2CH", "4CH", "PLAX"]:
    for cls in ["NSTEMI", "UAP"]:
        files = sorted((root / view / cls).glob("*.mp4"))
        for f in files[:2]:
            t0 = time.time(); clip, n, idx = read_clip(f); times.append(time.time() - t0)
            sheet = np.vstack([np.hstack(clip[:8]), np.hstack(clip[8:])])
            name = f"{view}_{cls}_{f.stem}.png"
            cv2.imwrite(str(out / name), sheet)
            print(f"{name:40s} frames in video: {n:4d} | sampled: {idx.tolist()}")
        for f in files[2:20]:                  # extra files for timing only
            t0 = time.time(); read_clip(f); times.append(time.time() - t0)

print(f"\nDecoding time per video: mean {np.mean(times)*1000:.0f} ms over {len(times)} videos")
print(f"Expected loading time per epoch (612 train+val videos, 4 workers): "
      f"~{612*np.mean(times)/4:.0f} s")
print(f"Contact sheets written to {out.resolve()}")
