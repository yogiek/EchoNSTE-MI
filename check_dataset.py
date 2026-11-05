"""Check the EchoNSTE-MI folder: file counts, readable videos, and 3 views per patient ID."""
import re, sys
from collections import defaultdict
from pathlib import Path

import cv2

root = Path(sys.argv[1] if len(sys.argv) > 1 else "EchoNSTE-MI")
views, classes = ["2CH", "4CH", "PLAX"], ["NSTEMI", "UAP"]
ids = defaultdict(set)          # (class, view) -> set of IDs
problems = 0

print(f"Dataset: {root.resolve()}\n")
print(f"{'view':6s}{'class':8s}{'files':>7s}")
for v in views:
    for c in classes:
        files = sorted((root / v / c).glob("*.mp4"))
        print(f"{v:6s}{c:8s}{len(files):7d}")
        for f in files:
            m = re.search(r"\d+", f.stem)
            if m is None:
                print("  no ID in file name:", f); problems += 1; continue
            if m.group() in ids[(c, v)]:
                print("  duplicate ID in folder:", f); problems += 1
            ids[(c, v)].add(m.group())

print()
for c in classes:
    all_ids = set().union(*(ids[(c, v)] for v in views))
    for v in views:
        missing = sorted(all_ids - ids[(c, v)], key=int)
        if missing:
            problems += 1
            print(f"{c}: {len(missing)} ID(s) without a {v} video, e.g. {missing[:10]}")
    print(f"{c}: {len(all_ids)} patient IDs")

# read a few videos to make sure OpenCV can decode them
bad = 0
for v in views:
    for c in classes:
        for f in sorted((root / v / c).glob("*.mp4"))[:3]:
            cap = cv2.VideoCapture(str(f))
            ok, _ = cap.read()
            if not ok:
                bad += 1; print("  cannot decode:", f)
problems += bad
print("\nRESULT:", "OK" if problems == 0 else f"{problems} problem(s) found - see above")
