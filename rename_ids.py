"""
Rename EchoNSTE-MI videos from cropped_<oldID>_<VIEW>.mp4 to P<newID>_<VIEW>.mp4, in place.

  NSTEMI : old IDs sorted numerically -> P01 ... P311
  UAP    : old IDs sorted numerically -> P312 ... P720

Safety
  * Default is a DRY RUN (nothing is renamed); add --apply to rename.
  * Stops if a patient does not have exactly one video in each of 2CH, 4CH and PLAX,
    if a file name does not match the expected pattern, or if class sizes differ
    from --expect (default 311 / 409).
  * Writes a mapping CSV (old -> new) OUTSIDE the dataset folder. Keep it internal:
    it is the only link between the new IDs and the clinical Excel file.
  * --undo <mapping.csv> restores the original names.

Usage
  python rename_ids.py /root/folder-docker/echo/EchoNSTE-MI            # dry run
  python rename_ids.py /root/folder-docker/echo/EchoNSTE-MI --apply    # rename
  python rename_ids.py /root/folder-docker/echo/EchoNSTE-MI --undo EchoNSTE-MI_rename_map.csv
"""
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

VIEWS = ["2CH", "4CH", "PLAX"]
CLASSES = ["NSTEMI", "UAP"]                      # NSTEMI numbered first
PATTERN = re.compile(r"^cropped_(\d+)_(2CH|4CH|PLAX)\.mp4$", re.IGNORECASE)


def collect(root):
    """Return {class: {old_id_int: {view: Path}}} and a list of problems."""
    data, problems = {c: defaultdict(dict) for c in CLASSES}, []
    for c in CLASSES:
        for v in VIEWS:
            folder = root / v / c
            if not folder.is_dir():
                problems.append(f"missing folder: {folder}")
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() != ".mp4":
                    continue
                m = PATTERN.match(f.name)
                if not m:
                    problems.append(f"unexpected file name: {f}")
                    continue
                old_id, fview = int(m.group(1)), m.group(2).upper()
                if fview != v:
                    problems.append(f"view in name ({fview}) differs from folder ({v}): {f}")
                if v in data[c][old_id]:
                    problems.append(f"two files with ID {old_id} in {folder}: "
                                    f"{data[c][old_id][v].name}, {f.name}")
                data[c][old_id][v] = f
    for c in CLASSES:
        for old_id, views in data[c].items():
            missing = [v for v in VIEWS if v not in views]
            if missing:
                problems.append(f"{c} ID {old_id}: no video for {', '.join(missing)}")
    return data, problems


def plan(data, width):
    rows, new_id = [], 1
    for c in CLASSES:
        for old_id in sorted(data[c]):                       # numeric order: 1, 2, ..., 9, 10, 100
            new_name = f"P{new_id:0{width}d}"
            for v in VIEWS:
                src = data[c][old_id][v]
                dst = src.with_name(f"{new_name}_{v}.mp4")
                rows.append({"class": c, "view": v, "old_id": old_id, "new_id": new_name,
                             "old_file": src.name, "new_file": dst.name, "folder": str(src.parent)})
            new_id += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="path to EchoNSTE-MI")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    ap.add_argument("--width", type=int, default=2,
                    help="zero padding of the number: 2 -> P01..P720, 3 -> P001..P720")
    ap.add_argument("--expect", type=int, nargs=2, default=[311, 409],
                    metavar=("N_NSTEMI", "N_UAP"))
    ap.add_argument("--map", default=None, help="mapping CSV path (default: next to the dataset)")
    ap.add_argument("--undo", default=None, help="mapping CSV produced by a previous --apply")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    # ------------------------------------------------------------------ undo
    if a.undo:
        rows = list(csv.DictReader(open(a.undo, encoding="utf-8")))
        todo = [(Path(r["folder"]) / r["new_file"], Path(r["folder"]) / r["old_file"]) for r in rows]
        todo = [(s, d) for s, d in todo if s.exists()]        # also works after a partial run
        bad = [d for s, d in todo if d.exists()]
        if bad:
            sys.exit("Cannot undo, these original names already exist:\n  " +
                     "\n  ".join(map(str, bad[:20])))
        for s, d in todo:
            s.rename(d)
        print(f"Restored {len(todo)} files to their original names.")
        return

    # ---------------------------------------------------------------- checks
    data, problems = collect(root)
    sizes = {c: len(data[c]) for c in CLASSES}
    print(f"Dataset: {root}")
    print(f"Patients found: NSTEMI {sizes['NSTEMI']}, UAP {sizes['UAP']}")
    if [sizes["NSTEMI"], sizes["UAP"]] != a.expect:
        problems.append(f"expected {a.expect[0]} NSTEMI / {a.expect[1]} UAP patients")
    if problems:
        print("\nProblems found - nothing was renamed:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    rows = plan(data, a.width)
    clash = [r for r in rows if (Path(r["folder"]) / r["new_file"]).exists()]
    if clash:
        sys.exit(f"Target names already exist (e.g. {clash[0]['new_file']}); "
                 f"the folder may already be renamed.")

    # --------------------------------------------------------------- preview
    first = {c: next(r for r in rows if r["class"] == c and r["view"] == "2CH") for c in CLASSES}
    last = {c: [r for r in rows if r["class"] == c and r["view"] == "2CH"][-1] for c in CLASSES}
    print("\nExamples (2CH):")
    for c in CLASSES:
        print(f"  {c:6s} {first[c]['old_file']:24s} -> {first[c]['new_file']}")
        print(f"  {c:6s} {last[c]['old_file']:24s} -> {last[c]['new_file']}")
    print(f"\n{len(rows)} files will be renamed ({len(rows)//3} patients x 3 views).")

    map_path = Path(a.map) if a.map else root.parent / f"{root.name}_rename_map.csv"
    if not a.apply:
        print(f"\nDRY RUN - nothing renamed. Run again with --apply to rename.\n"
              f"The mapping will be saved to: {map_path}")
        return

    # ---------------------------------------------------------------- rename
    with open(map_path, "w", newline="", encoding="utf-8") as f:     # save mapping first
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    done = 0
    try:
        for r in rows:
            (Path(r["folder"]) / r["old_file"]).rename(Path(r["folder"]) / r["new_file"])
            done += 1
    except Exception as e:
        print(f"\nERROR after {done} files: {e}\nUndo with: python rename_ids.py {root} --undo {map_path}")
        raise
    print(f"\nRenamed {done} files. Mapping saved to: {map_path}  (keep this file internal)")


if __name__ == "__main__":
    main()
