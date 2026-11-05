"""
Baseline video classification for the EchoNSTE-MI dataset (NSTEMI vs UAP).

Models
  - cnn2d : ResNet-18 (ImageNet-pretrained) applied to every frame, features
            averaged over time (temporal average pooling), linear classifier.
  - cnn3d : R(2+1)D-18 (Kinetics-400-pretrained) applied to the whole clip.

Protocol (standard classification)
  - One model per view (2CH, 4CH, PLAX).
  - Patient-level random split 70/15/15 (train/val/test) with a fixed seed.
    The same patients are assigned to the same subset in every view.
  - Best epoch selected on validation AUC; final metrics reported on the test set.

Expected folder structure
  EchoNSTE-MI/<VIEW>/<CLASS>/<file>.mp4   with CLASS in {NSTEMI, UAP}
  The patient key is <CLASS>_<first integer in the file name>
  (e.g. NSTEMI/cropped_42_2CH.mp4 -> NSTEMI_42), because the same number can
  occur in both class folders for different patients.
"""
import argparse, json, os, random, re, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

CLASSES = {"UAP": 0, "NSTEMI": 1}          # positive class = NSTEMI (infarct)
VIEWS = ["2CH", "4CH", "PLAX"]
MEAN = {"cnn2d": [0.485, 0.456, 0.406], "cnn3d": [0.43216, 0.394666, 0.37645]}
STD = {"cnn2d": [0.229, 0.224, 0.225], "cnn3d": [0.22803, 0.22145, 0.216989]}


# --------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def patient_id(path):
    m = re.search(r"\d+", Path(path).stem)
    if m is None:
        raise ValueError(f"No patient ID found in file name: {path}")
    return int(m.group())


def index_dataset(root):
    rows = []
    for view in VIEWS:
        for cls, y in CLASSES.items():
            for f in sorted((Path(root) / view / cls).glob("*.mp4")):
                # IDs may repeat across class folders, so the class is part of the key
                rows.append({"path": str(f), "view": view, "label_name": cls,
                             "label": y, "patient_id": f"{cls}_{patient_id(f)}"})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No videos found under {root}/<VIEW>/<CLASS>/*.mp4")
    return df


def make_split(df, seed, val_frac, test_frac, out_csv):
    """Random patient-level split, shared by all views."""
    pids = sorted(df.patient_id.unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    n_test = int(round(len(pids) * test_frac))
    n_val = int(round(len(pids) * val_frac))
    subset = {}
    for p in pids[:n_test]: subset[p] = "test"
    for p in pids[n_test:n_test + n_val]: subset[p] = "val"
    for p in pids[n_test + n_val:]: subset[p] = "train"
    labels = df.groupby("patient_id").label_name.first()
    split = pd.DataFrame({"patient_id": list(subset), "subset": list(subset.values())})
    split["label_name"] = split.patient_id.map(labels)
    split.sort_values("patient_id").to_csv(out_csv, index=False)
    return subset


# ------------------------------------------------------------------- dataset
def read_clip(path, n_frames, size):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)       # unify greyscale / sepia
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(g)
    cap.release()
    if not frames:
        raise RuntimeError(f"Cannot read video: {path}")
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)  # uniform sampling
    clip = np.stack([frames[i] for i in idx]).astype(np.float32) / 255.0  # T,H,W
    return np.repeat(clip[:, None], 3, axis=1)                            # T,3,H,W


class EchoDataset(Dataset):
    def __init__(self, df, model, n_frames, size, train):
        self.df = df.reset_index(drop=True)
        self.model, self.n_frames, self.size, self.train = model, n_frames, size, train
        self.mean = torch.tensor(MEAN[model]).view(1, 3, 1, 1)
        self.std = torch.tensor(STD[model]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = torch.from_numpy(read_clip(r.path, self.n_frames, self.size))
        if self.train and random.random() < 0.5:                 # light augmentation
            x = x * random.uniform(0.85, 1.15)                    # brightness jitter
        x = ((x.clamp(0, 1) - self.mean) / self.std)              # T,3,H,W
        if self.model == "cnn3d":
            x = x.permute(1, 0, 2, 3)                             # 3,T,H,W
        return x, torch.tensor(r.label, dtype=torch.float32)


# -------------------------------------------------------------------- models
class CNN2D(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()
        self.backbone, self.head = net, nn.Linear(512, 1)

    def forward(self, x):                      # B,T,3,H,W
        b, t = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(b, t, -1).mean(1)
        return self.head(f).squeeze(1)


class CNN3D(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net = net

    def forward(self, x):                      # B,3,T,H,W
        return self.net(x).squeeze(1)


def build_model(name, pretrained):
    return CNN2D(pretrained) if name == "cnn2d" else CNN3D(pretrained)


# ---------------------------------------------------------------- train/eval
def run_epoch(model, loader, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    crit = nn.BCEWithLogitsLoss()
    losses, probs, ys = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = crit(logits.float(), y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        losses.append(loss.item() * len(y))
        probs.append(torch.sigmoid(logits.float()).detach().cpu()); ys.append(y.cpu())
    p, y = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    return sum(losses) / len(y), y, p


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "Accuracy": accuracy_score(y, yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Sensitivity": recall_score(y, yhat, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "F1": f1_score(y, yhat, zero_division=0),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="EchoNSTE-MI")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", choices=["cnn2d", "cnn3d"], required=True)
    ap.add_argument("--view", choices=VIEWS, required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="sanity check: randomly permute TRAINING labels; test AUC should drop to ~0.5")
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = index_dataset(a.data)
    subset = make_split(df, a.seed, a.val_frac, a.test_frac, out / "split_patient_level.csv")
    df["subset"] = df.patient_id.map(subset)
    dv = df[df.view == a.view]
    parts = {s: dv[dv.subset == s] for s in ["train", "val", "test"]}
    if a.shuffle_labels:
        rng = np.random.RandomState(a.seed + 1)
        parts["train"] = parts["train"].assign(label=rng.permutation(parts["train"].label.values))
        print("!! SANITY CHECK: training labels randomly permuted (val/test labels unchanged)")
    print(f"[{a.model} | {a.view}] device={device} | " +
          " | ".join(f"{s}: {len(d)} (NSTEMI {int(d.label.sum())})" for s, d in parts.items()))

    loaders = {s: DataLoader(EchoDataset(d, a.model, a.frames, a.size, s == "train"),
                             batch_size=a.batch, shuffle=(s == "train"),
                             num_workers=a.workers, pin_memory=True)
               for s, d in parts.items()}

    model = build_model(a.model, not a.no_pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    # torch.amp.GradScaler exists from PyTorch 2.3; NGC 23.12 ships PyTorch 2.2
    Scaler = getattr(torch.amp, "GradScaler", None) or torch.cuda.amp.GradScaler
    scaler = Scaler(enabled=device.type == "cuda")
    tag = f"{a.model}_{a.view}" + ("_shuffled" if a.shuffle_labels else "")
    ckpt = out / f"{tag}_best.pt"

    best_auc, log = -1, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, loaders["train"], device, opt, scaler)
        va_loss, yv, pv = run_epoch(model, loaders["val"], device)
        mv = metrics(yv, pv)
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                    "val_auc": mv["AUC"], "val_acc": mv["Accuracy"]})
        print(f"ep {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
              f"AUC {mv['AUC']:.4f} | acc {mv['Accuracy']:.4f} | {time.time()-t0:.0f}s")
        if mv["AUC"] > best_auc:
            best_auc = mv["AUC"]; torch.save(model.state_dict(), ckpt)
    pd.DataFrame(log).to_csv(out / f"{tag}_log.csv", index=False)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, yt, pt = run_epoch(model, loaders["test"], device)
    res = {"model": a.model, "view": a.view, "n_test": len(yt), **metrics(yt, pt),
           "best_val_AUC": best_auc}
    pd.DataFrame({"path": parts["test"].path.values, "label": yt, "prob": pt}) \
        .to_csv(out / f"{tag}_test_predictions.csv", index=False)
    with open(out / f"{tag}_test_metrics.json", "w") as f:
        json.dump({**res, "args": vars(a)}, f, indent=2)
    print("TEST:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in res.items()}))


if __name__ == "__main__":
    main()
