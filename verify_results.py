"""
Independent checks of a finished run (no GPU needed):
  1. Patient-level split: no patient key in more than one subset; view coverage.
  2. Recompute every test metric from *_test_predictions.csv and compare with the JSON.
  3. Bootstrap 95% confidence interval of the test AUC (1000 resamples).
  4. Flag runs whose test predictions are almost constant (model ignoring the input).

Usage: python verify_results.py /root/folder-docker/results
"""
import glob, json, os, re, sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

out = sys.argv[1] if len(sys.argv) > 1 else "results"
ok = True

def patient_key(path):                      # same key as train.py: <CLASS>_<ID>
    cls = path.replace("\\", "/").split("/")[-2]
    num = re.search(r"\d+", os.path.basename(path)).group()
    return cls + "_" + num


# ---------------------------------------------------------------- 1. split
split = pd.read_csv(f"{out}/split_patient_level.csv")
dup = split.patient_id.duplicated().sum()
print(f"[split] {len(split)} patient keys | train {sum(split.subset=='train')}, "
      f"val {sum(split.subset=='val')}, test {sum(split.subset=='test')} | duplicated keys: {dup}")
if dup:
    ok = False
if len(split) != 720:
    print(f"  ! {len(split)} patient keys instead of 720: some IDs differ between view folders "
          f"(run check_dataset.py to find them)")

# ---------------------------------------------------------- 2-4. each run
rng = np.random.RandomState(0)
rows = []
for jf in sorted(glob.glob(f"{out}/*_test_metrics.json")):
    tag = os.path.basename(jf).replace("_test_metrics.json", "")
    res = json.load(open(jf))
    pred = pd.read_csv(f"{out}/{tag}_test_predictions.csv")
    y, p = pred.label.values.astype(int), pred.prob.values
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, p)
    same = (abs(auc - res["AUC"]) < 1e-6 and (tp, tn, fp, fn) ==
            (res["TP"], res["TN"], res["FP"], res["FN"]))
    ok &= same

    # test patients must all be in the "test" subset of the split
    keys = pred.path.map(patient_key)
    leak = (~keys.isin(split.loc[split.subset == "test", "patient_id"])).sum()
    ok &= leak == 0

    boots = []
    for _ in range(1000):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append({"run": tag, "AUC": round(auc, 3), "95% CI": f"{lo:.3f}–{hi:.3f}",
                 "prob_std": round(float(p.std()), 3), "pred_NSTEMI": int(yhat.sum()),
                 "metrics_match": same, "test_leak": int(leak)})

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
if (df.prob_std < 0.05).any():
    print("\n! some runs give almost the same probability for every video (prob_std < 0.05)")
print("\nRESULT:", "all checks passed" if ok else "check the rows above")
