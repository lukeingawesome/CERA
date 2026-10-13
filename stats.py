"""Statistical analysis: anchor-blocked bootstrap, paired superiority, Holm.

Agreement is Kendall tau-b against radiologist error counts. Pairs from the same
anchor are correlated, so every bootstrap resamples anchors, not rows. Exact P
comes from a normal approximation on the bootstrap distribution, because the
empirical tail is censored at 1/(B+1).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

N_BOOT = 4000


def _blocks(anchors):
    uniq = anchors.unique()
    idx = {a: np.flatnonzero(anchors.to_numpy() == a) for a in uniq}
    return uniq, idx


def tau(df, pred, gold, anchor="anchor_id", n_boot=N_BOOT, seed=0):
    """tau-b with an anchor-blocked bootstrap CI."""
    d = df[[pred, gold, anchor]].dropna()
    point = kendalltau(d[gold], d[pred]).statistic
    uniq, idx = _blocks(d[anchor])
    rng = np.random.default_rng(seed)
    pv, gv = d[pred].to_numpy(), d[gold].to_numpy()
    out = []
    for _ in range(n_boot):
        i = np.concatenate([idx[a] for a in rng.choice(uniq, len(uniq))])
        t = kendalltau(gv[i], pv[i]).statistic
        if not np.isnan(t):
            out.append(t)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, lo, hi


def paired(df, a_col, b_col, gold, anchor="anchor_id", n_boot=N_BOOT, seed=0):
    """tau(a) - tau(b) against the same gold, anchor-blocked."""
    d = df[[a_col, b_col, gold, anchor]].dropna()
    uniq, idx = _blocks(d[anchor])
    rng = np.random.default_rng(seed)
    av, bv, gv = d[a_col].to_numpy(), d[b_col].to_numpy(), d[gold].to_numpy()
    out = []
    for _ in range(n_boot):
        i = np.concatenate([idx[a] for a in rng.choice(uniq, len(uniq))])
        t1, t2 = kendalltau(gv[i], av[i]).statistic, kendalltau(gv[i], bv[i]).statistic
        if not (np.isnan(t1) or np.isnan(t2)):
            out.append(t1 - t2)
    o = np.array(out)
    se = o.std(ddof=1)
    z = o.mean() / se
    lo, hi = np.percentile(o, [2.5, 97.5])
    return dict(delta=o.mean(), se=se, z=z, ci_lo=lo, ci_hi=hi, p=2 * norm.sf(abs(z)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (len(p) - rank)))
        adj[i] = run
    return adj


def superiority(df, ours, comparators, gold, anchor="anchor_id"):
    """Paired superiority of `ours` over each comparator, Holm-corrected."""
    rows = [dict(comparator=c, **paired(df, ours, c, gold, anchor)) for c in comparators]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out.sort_values("p")


def auroc(df, pred, gold_binary):
    d = df[[pred, gold_binary]].dropna()
    return roc_auc_score(d[gold_binary], d[pred])


def multiset_f1(gold_counts, pred_counts, cats):
    """Count-aware micro-F1: a prediction can match a gold error at most once."""
    tp = gt = pr = 0
    for k in gold_counts:
        g, p = gold_counts[k], pred_counts.get(k, {})
        for c in cats:
            tp += min(p.get(c, 0), g.get(c, 0))
            gt += g.get(c, 0)
            pr += p.get(c, 0)
    prec = tp / pr if pr else 0.0
    rec = tp / gt if gt else 0.0
    return dict(f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                precision=prec, recall=rec, tp=tp, gold=gt, predicted=pr)


def format_stability(df, pred, gold, style="reference_style", key=("anchor_id", "variant_id")):
    """Does the score survive re-rendering the reference?"""
    w = df.pivot_table(index=list(key), columns=style, values=pred).dropna()
    styles = list(w.columns)
    if len(styles) != 2:
        raise ValueError(f"expected 2 reference styles, found {styles}")
    taus = {s: kendalltau(df[df[style] == s][gold], df[df[style] == s][pred]).statistic
            for s in styles}
    a, b = w[styles[0]], w[styles[1]]
    diff = b - a
    return dict(tau_by_style=taus,
                delta_tau=taus[styles[1]] - taus[styles[0]],
                spearman_between=spearmanr(a, b).statistic,
                frac_identical=float((diff == 0).mean()),
                shift=float(diff.mean()),
                wilcoxon_p=float(wilcoxon(diff).pvalue) if diff.abs().sum() else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="parquet with predictions and gold counts")
    p.add_argument("--pred", default="k_pred")
    p.add_argument("--gold", default="rad_error_count")
    p.add_argument("--comparators", nargs="*", default=[])
    a = p.parse_args()

    d = pd.read_parquet(a.scores)
    t, lo, hi = tau(d, a.pred, a.gold)
    print(f"{a.pred} vs {a.gold}: tau = {t:.4f} [{lo:.3f}, {hi:.3f}]  n={len(d)}")

    if "K_critical" in d.columns:
        t, lo, hi = tau(d, a.pred, "K_critical")
        print(f"{a.pred} vs K_critical: tau = {t:.4f} [{lo:.3f}, {hi:.3f}]")
    if "k_pred_critical" in d.columns and "K_critical" in d.columns:
        t, lo, hi = tau(d, "k_pred_critical", "K_critical")
        print(f"k_pred_critical vs K_critical: tau = {t:.4f} [{lo:.3f}, {hi:.3f}]")

    if a.comparators:
        print()
        print(superiority(d, a.pred, a.comparators, a.gold).to_string(index=False))

    if "reference_style" in d.columns and d.reference_style.nunique() == 2:
        print()
        for k, v in format_stability(d, a.pred, a.gold).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
