#!/usr/bin/env bash
# Train and evaluate both baselines on all three views, then summarise.
set -eo pipefail
cd "$(dirname "$0")"          # run from the code folder
DATA=${DATA:-/root/folder-docker/echo/EchoNSTE-MI}
OUT=${OUT:-/root/folder-docker/echo/results}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-30}
mkdir -p "$OUT"

for VIEW in 2CH 4CH PLAX; do
  for MODEL in cnn2d cnn3d; do
    python train.py --data "$DATA" --out "$OUT" --model $MODEL --view $VIEW \
                    --epochs $EPOCHS --workers $WORKERS 2>&1 | tee "$OUT/${MODEL}_${VIEW}_stdout.txt"
  done
done
python summarize.py "$OUT"
