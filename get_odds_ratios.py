
import os
import glob
import pandas as pd
import sys

# -------- Configuration --------
results_folder="Results"
name_dataset=sys.argv[1]
INPUT_PATTERN = results_folder+"/" + name_dataset + "_AMR/Final_FullData_Model/*_FULL_odds_ratios_CVbasedCI.csv"
OUTPUT_CSV = results_folder+"/" + name_dataset + "_all_oddsratio_summaries.csv"
NEW_COL_NAME = "Dataset"

def parse_name_parts(filename: str):

    base = os.path.basename(filename)
    for tail in ("_FULL_odds_ratios_CVbasedCI.csv"):
        if base.endswith(tail):
            core = base[:-len(tail)]
            parts = core.split("_")
            if len(parts) < 2:
                raise ValueError(f"Cannot parse dataset/antibiotic from: {base}")
            dataset = "_".join(parts[:-1])
            antibiotic = parts[-1]
            return dataset, antibiotic
    raise ValueError(f"Unexpected filename pattern: {base}")

def load_and_tag(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    dataset, antibiotic = parse_name_parts(path)
    df[NEW_COL_NAME] = f"{dataset}_{antibiotic}"
    return df

def main():
    files = sorted(glob.glob(INPUT_PATTERN))
    if not files:
        raise SystemExit(f"No input files found matching pattern: {INPUT_PATTERN}")

    frames = []
    for f in files:
        try:
            frames.append(load_and_tag(f))
        except Exception as e:
            print(f"[WARN] Skipping '{f}' due to error: {e}")

    if not frames:
        raise SystemExit("No valid files could be loaded. Aborting.")

    out_df = pd.concat(frames, ignore_index=True)
    cols = out_df.columns.tolist()
    if NEW_COL_NAME in cols:
        cols.insert(0, cols.pop(cols.index(NEW_COL_NAME)))
        out_df = out_df[cols]
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote '{OUTPUT_CSV}' with {len(out_df)} rows from {len(frames)} files.")

if __name__ == "__main__":
    main()
