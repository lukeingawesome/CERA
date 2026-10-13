"""Build the CERA-Bench leaderboard from comparator predictions.

Consumes the long-format parquets written by baselines.py plus CERA's scores, and
reports tau against each gold endpoint with anchor-blocked CIs, the criticality
margin, and paired superiority of one metric over the rest with Holm correction.

    python leaderboard.py --predictions runs/base --cera runs/eval/scores.parquet \
        --gold data/pairs.parquet --out runs/leaderboard.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from stats import holm, paired, tau

KEY = ["anchor_id", "variant_id"]
GOLD_COLS = {"total": "rad_error_count", "critical": "K_critical", "minor": "K_minor"}


def load_predictions(pred_dir, cera_scores=None):
    frames = []
    for p in sorted(Path(pred_dir).glob("predictions_*.parquet")):
        d = pd.read_parquet(p)
        if {"method", "score"} <= set(d.columns):
            frames.append(d[KEY + ["method", "score"]])
    if cera_scores:
        c = pd.read_parquet(cera_scores)
        for col, name in [("k_pred", "cera:error_count"),
                          ("k_pred_critical", "cera:criticality")]:
            if col in c.columns:
                f = c[KEY + [col]].rename(columns={col: "score"})
                f["method"] = name
                frames.append(f[KEY + ["method", "score"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(KEY + ["method"])


def orient(wide, method, gold):
    """Flip similarity metrics so higher always means more errors."""
    r = kendalltau(wide[gold], wide[method]).statistic
    return wide[method] * (1.0 if (r is not None and r >= 0) else -1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--cera", default=None)
    p.add_argument("--gold", required=True)
    p.add_argument("--out", default="leaderboard.csv")
    p.add_argument("--reference-style", default=None)
    p.add_argument("--superiority-of", default="cera:criticality")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()

    gold = pd.read_parquet(a.gold)
    if a.reference_style and "reference_style" in gold.columns:
        gold = gold[gold.reference_style == a.reference_style]
    gold = gold.drop_duplicates(KEY)
    keep = KEY + [c for c in GOLD_COLS.values() if c in gold.columns]
    gold = gold[keep]

    preds = load_predictions(a.predictions, a.cera)
    wide = preds.pivot_table(index=KEY, columns="method", values="score").reset_index()
    d = gold.merge(wide, on=KEY, how="inner")
    methods = [c for c in wide.columns if c not in KEY]
    print(f"{len(d)} pairs | {len(methods)} methods")

    for m in methods:
        d[m] = orient(d.dropna(subset=[m]), m, GOLD_COLS["total"]).reindex(d.index)

    rows = []
    for m in methods:
        row = {"method": m, "n": int(d[m].notna().sum())}
        for name, col in GOLD_COLS.items():
            if col not in d.columns:
                continue
            t, lo, hi = tau(d, m, col, n_boot=a.n_boot)
            row[f"tau_{name}"] = t
            if name == "total":
                row["ci_lo"], row["ci_hi"] = lo, hi
        if "tau_critical" in row and "tau_minor" in row:
            row["crit_minus_minor"] = row["tau_critical"] - row["tau_minor"]
        rows.append(row)
    lb = pd.DataFrame(rows).sort_values("tau_total", ascending=False)

    ours = a.superiority_of
    if ours in methods:
        comp = [m for m in methods if m != ours and not m.startswith("cera:")]
        sup = pd.DataFrame([dict(method=c, **paired(d, ours, c, GOLD_COLS["total"],
                                                    n_boot=a.n_boot)) for c in comp])
        sup["p_holm"] = holm(sup.p.values)
        lb = lb.merge(sup[["method", "delta", "se", "z", "p", "p_holm"]], on="method", how="left")
        print(f"\npaired superiority of {ours} over {len(comp)} comparators, "
              f"Holm-corrected; all significant: {(sup.p_holm < 0.05).all()}")

    lb.to_csv(a.out, index=False)
    cols = [c for c in ("method", "n", "tau_total", "ci_lo", "ci_hi", "tau_critical",
                        "crit_minus_minor", "p_holm") if c in lb.columns]
    print()
    print(lb[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
