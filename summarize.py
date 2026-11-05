"""Collect all *_test_metrics.json files into one table (CSV + Markdown)."""
import glob, json, sys
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/*_test_metrics.json"))
        if "_shuffled" not in f]
if not rows:
    sys.exit(f"No results found in {out}/")
df = pd.DataFrame(rows)
df["Model"] = df.model.map({"cnn2d": "ResNet-18 (2D CNN)", "cnn3d": "R(2+1)D-18 (3D CNN)"})
df["view"] = pd.Categorical(df.view, ["2CH", "4CH", "PLAX"], ordered=True)
cols = ["view", "Model", "AUC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
tab = df.sort_values(["view", "model"])[cols].rename(columns={"view": "View"})
tab.to_csv(f"{out}/summary_test_metrics.csv", index=False)
print(tab.to_markdown(index=False, floatfmt=".3f"))
