import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def parse_model_name(name):
    parts = name.split("_")

    out = {
        "model_name": name,
        "size": None,
        "tokens": None,
        "condition": None,
    }

    for p in parts:
        if p in {"1b", "8b"}:
            out["size"] = p.upper()
        if p in {"100b", "500b"}:
            out["tokens"] = p.upper()
        if p in {"standard", "perturbed"}:
            out["condition"] = p

    return out


def safe_mean(series):
    vals = [x for x in series if x is not None and not pd.isna(x)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="results/raw")
    parser.add_argument("--out_dir", default="results/summary")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(args.raw_dir) / "*.jsonl")))

    all_rows = []

    for path in files:
        rows = read_jsonl(path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        raise ValueError("No result rows found.")

    meta = pd.DataFrame([parse_model_name(x) for x in df["model_name"]])
    df = pd.concat([df.reset_index(drop=True), meta.drop(columns=["model_name"]).reset_index(drop=True)], axis=1)

    metric_cols = ["exact_containment", "token_f1", "lcs_ratio"]

    summary = (
        df.groupby(["model_name", "size", "tokens", "condition", "task_type"], dropna=False)[metric_cols]
        .agg(safe_mean)
        .reset_index()
    )

    summary_path = Path(args.out_dir) / "model_task_summary.csv"
    summary.to_csv(summary_path, index=False)

    pair_rows = []

    for size in sorted(df["size"].dropna().unique()):
        for tokens in sorted(df["tokens"].dropna().unique()):
            for task_type in sorted(df["task_type"].dropna().unique()):
                sub = df[(df["size"] == size) & (df["tokens"] == tokens) & (df["task_type"] == task_type)]

                std = sub[sub["condition"] == "standard"]
                per = sub[sub["condition"] == "perturbed"]

                if len(std) == 0 or len(per) == 0:
                    continue

                for metric in metric_cols:
                    merged = std[["id", metric]].merge(
                        per[["id", metric]],
                        on="id",
                        suffixes=("_standard", "_perturbed"),
                    ).dropna()

                    if len(merged) == 0:
                        continue

                    diff = merged[f"{metric}_perturbed"] - merged[f"{metric}_standard"]

                    if len(merged) >= 2:
                        stat, pval = ttest_rel(
                            merged[f"{metric}_perturbed"],
                            merged[f"{metric}_standard"],
                        )
                    else:
                        stat, pval = np.nan, np.nan

                    pair_rows.append({
                        "size": size,
                        "tokens": tokens,
                        "task_type": task_type,
                        "metric": metric,
                        "n": len(merged),
                        "standard_mean": float(merged[f"{metric}_standard"].mean()),
                        "perturbed_mean": float(merged[f"{metric}_perturbed"].mean()),
                        "perturbed_minus_standard": float(diff.mean()),
                        "paired_t": float(stat) if not pd.isna(stat) else np.nan,
                        "p_value": float(pval) if not pd.isna(pval) else np.nan,
                    })

    pair_df = pd.DataFrame(pair_rows)

    pair_path = Path(args.out_dir) / "standard_vs_perturbed.csv"
    pair_df.to_csv(pair_path, index=False)

    full_path = Path(args.out_dir) / "all_results.parquet"
    df.to_parquet(full_path, index=False)

    print(f"Saved summary to {summary_path}")
    print(f"Saved pair comparison to {pair_path}")
    print(f"Saved full parquet to {full_path}")


if __name__ == "__main__":
    main()
