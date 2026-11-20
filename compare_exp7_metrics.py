
import argparse
import csv
import json
from pathlib import Path

CLASSES = ["normal", "bacteria", "virus", "COVID-19"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp3", default="runs/exp3_focal/best_val_metrics.json")
    parser.add_argument("--exp7a", default="runs/exp7A_380_original_aug/best_val_metrics.json")
    parser.add_argument("--exp7b", default="runs/exp7B_380_stronger_aug/best_val_metrics.json")
    parser.add_argument("--output", default="runs/exp7_comparison.csv")
    args = parser.parse_args()
    paths = [("Exp3 300px original aug", args.exp3),
             ("Exp7A 380px original aug", args.exp7a),
             ("Exp7B 380px stronger aug", args.exp7b)]
    rows = []
    for label, file in paths:
        p = Path(file)
        if not p.is_file():
            print(f"SKIP missing: {p}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        row = {"experiment": label,
               "best_epoch": d["epoch"],
               "val_macro_f1": d.get("macro_f1", d.get("val_macro_f1"))}
        row.update({f"f1_{cls}": d["per_class"][cls]["f1"] for cls in CLASSES})
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No experiment metric files found")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['experiment']}: Val Macro-F1 = {float(row['val_macro_f1']):.6f} "
              f"| Virus F1 = {row['f1_virus']:.6f}")
    print("Saved:", out)
    print("Validation was reused for several experiments; do not treat as an independent test score.")


if __name__ == "__main__":
    main()
