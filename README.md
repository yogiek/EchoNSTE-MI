# EchoNSTE-MI – baseline classification (2D CNN & 3D CNN)

Binary classification **NSTEMI (1) vs UAP (0)**, one model per view (2CH, 4CH, PLAX).

| Model | Backbone | Pretrained weights | Input |
|---|---|---|---|
| `cnn2d` | ResNet-18, frame-wise + temporal average pooling | ImageNet | 16 frames × 112 × 112 |
| `cnn3d` | R(2+1)D-18 | Kinetics-400 | 16 frames × 112 × 112 |

Common settings: greyscale frames replicated to 3 channels, 16 frames sampled uniformly,
Adam (lr 1e-4), batch 8, 30 epochs, BCE loss, mixed precision, seed 42.
Patient-level random split 70/15/15 (same patients in the same subset for all views);
best epoch chosen on validation AUC, evaluated once on the test set.

## Files

| File | Purpose |
|---|---|
| `check_env.py` | Checks Python packages, GPU, and shared memory in the container |
| `check_dataset.py` | Counts videos per folder, checks 3 views per ID, test-decodes videos |
| `train.py` | Trains and tests one model on one view |
| `run_all.sh` | Runs all 6 experiments and writes the summary table |
| `summarize.py` | Collects test metrics into `summary_test_metrics.csv` |
| `verify_results.py` | Re-computes test metrics from predictions, checks split leakage, bootstrap 95% CI of AUC |
| `inspect_inputs.py` | Saves the 16 frames the model sees for sample videos and measures decoding time |
| `Dockerfile` | Optional: build a dedicated image instead of using NGC 23.12 |

## Running in the NGC PyTorch 23.12 container

Host folder layout (`/home/user03/yogiek/` is mounted as `/root/folder-docker/`):
```
/home/user03/yogiek/
├── echo_baseline/
└── EchoNSTE-MI/{2CH,4CH,PLAX}/{NSTEMI,UAP}/*.mp4
```

```bash
# host: create the container (once)
docker run -t -d --name yogiekecho --runtime=nvidia --net=host --ipc=host \
  -v /home/user03/yogiek/:/root/folder-docker 1cff6923bda0

# host: enter the container
docker exec -it yogiekecho bash

# container: install missing packages (once), then check
cd /root/folder-docker/echo_baseline
pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84" "pandas==2.2.2" "scikit-learn==1.5.1" tabulate
python check_env.py
python check_dataset.py /root/folder-docker/EchoNSTE-MI

# container: quick test (1 epoch), then full run in the background
python train.py --data /root/folder-docker/EchoNSTE-MI --out /root/folder-docker/test_run \
                --model cnn3d --view 4CH --epochs 1
CUDA_VISIBLE_DEVICES=0 nohup bash run_all.sh > /root/folder-docker/run_all.log 2>&1 &
tail -f /root/folder-docker/run_all.log
```

Keep NumPy below 2: the PyTorch build in NGC 23.12 was compiled against NumPy 1.x.

Environment variables for `run_all.sh`: `DATA`, `OUT`, `EPOCHS` (default 30), `WORKERS` (default 4).
If `/dev/shm` is small and the run stops with a "bus error" or "DataLoader worker" error, use `WORKERS=0`.

## Output (`/root/folder-docker/results/`)

`split_patient_level.csv`, `<model>_<view>_log.csv`, `<model>_<view>_best.pt`,
`<model>_<view>_test_predictions.csv`, `<model>_<view>_test_metrics.json`,
and `summary_test_metrics.csv` (final table for the paper).

## Validity checks

```bash
python verify_results.py /root/folder-docker/results
python inspect_inputs.py /root/folder-docker/EchoNSTE-MI /root/folder-docker/inspect
# label-permutation test: test AUC must drop to about 0.5
python train.py --data /root/folder-docker/EchoNSTE-MI --out /root/folder-docker/results \
                --model cnn3d --view 4CH --shuffle_labels
```
