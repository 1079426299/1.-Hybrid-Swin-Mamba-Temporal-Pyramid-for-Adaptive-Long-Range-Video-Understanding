import argparse
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute class-wise temporal motion variance")
    parser.add_argument("--csv", required=True, help="CSV columns: class_id,motion_scores; motion_scores are semicolon-separated floats")
    parser.add_argument("--output", default="outputs/temporal_variance.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    values = defaultdict(list)
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row["class_id"]
            scores = np.array([float(x) for x in row["motion_scores"].split(";") if x.strip()])
            if len(scores) > 1:
                values[cls].append(float(np.var(np.diff(scores))))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "temporal_motion_variance"])
        for cls, vals in sorted(values.items()):
            writer.writerow([cls, float(np.mean(vals))])
    print(f"Saved temporal variance to {args.output}")


if __name__ == "__main__":
    main()
