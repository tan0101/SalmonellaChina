import os
import glob
import pandas as pd
import sys

# -------- Configuration --------
results_folder = "Results"
name_dataset = sys.argv[1]

INPUT_PATTERN = (results_folder + "/" + name_dataset + "_AMR/*_summary_metrics.csv")
OUTPUT_CSV = results_folder + "/" + name_dataset + "_performance_summary.csv"

METRIC_COLUMNS = [
    "AUC",
    "PR_AUC",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision"
]

def parse_name_parts(filename: str):
    base = os.path.basename(filename)
    if not base.endswith("_summary_metrics.csv"):
        raise ValueError(f"Unexpected filename pattern: {base}")

    core = base[:-len("_summary_metrics.csv")]
    parts = core.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse dataset/antibiotic from: {base}")

    dataset = "_".join(parts[:-1])
    antibiotic = parts[-1]
    return dataset, antibiotic


def summarise_file(path: str):
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"File '{path}' is empty — skipping.")

    # normalise column names
    rename_map = {c: c.strip() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    missing = [c for c in METRIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"File '{path}' missing columns: {missing}")
    means = df[METRIC_COLUMNS].mean()
    stds = df[METRIC_COLUMNS].std(ddof=1)
    return means, stds


def fmt(mean_val, sd_val):
    return f"{mean_val:.3f}±{sd_val:.3f}"


def main():
    files = sorted(glob.glob(INPUT_PATTERN))
    if not files:
        raise SystemExit(f"No input files found matching pattern: {INPUT_PATTERN}")

    rows = []

    for f in files:
        try:
            dataset, antibiotic = parse_name_parts(f)
        except Exception as e:
            print(f"WARNING: {e} — skipping.", file=sys.stderr)
            continue

        try:
            means, stds = summarise_file(f)
        except Exception as e:
            print(f"WARNING: {e} — skipping file.", file=sys.stderr)
            continue

        name_col = f"{dataset}_{antibiotic}"

        row = {"name_dataset_{name_antibiotic}": name_col}

        for metric in METRIC_COLUMNS:
            row[metric] = fmt(means[metric], stds[metric])

        rows.append(row)

    if not rows:
        pd.DataFrame(columns=["name_dataset_{name_antibiotic}", *METRIC_COLUMNS]).to_csv(OUTPUT_CSV, index=False)
        return

    out_df = pd.DataFrame(rows, columns=["name_dataset_{name_antibiotic}",*METRIC_COLUMNS])
    out_df.sort_values(by="name_dataset_{name_antibiotic}", inplace=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {OUTPUT_CSV} with {len(out_df)} rows.")

if __name__ == "__main__":
    main()
