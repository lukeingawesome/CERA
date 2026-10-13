"""Score CERA-Bench with comparator metrics.

Writes long format: method, anchor_id, variant_id, score. One file per group so a
slow group can be run separately or sharded.

    pip install -r requirements-baselines.txt

    python baselines.py --group lexical --pairs data/pairs.parquet --out runs/base
    python baselines.py --group green   --pairs data/pairs.parquet --out runs/base
    python baselines.py --group crimson --pairs data/pairs.parquet --out runs/base
"""
import argparse
import time
from pathlib import Path

import pandas as pd

# RadEval metric groups. Lexical and labeller metrics are cheap; green is an LLM
# judge and is an order of magnitude slower, so it is its own group.
GROUPS = {
    "lexical": ["bleu", "rouge", "bertscore", "radeval_bertscore"],
    "labeler": ["f1chexbert", "f1radbert_ct", "srrbert", "temporal"],
    "graph": ["radgraph", "ratescore", "radcliq", "radgraph_radcliq"],
    "green": ["green"],
}


def trivial(pairs):
    rows = []
    for r in pairs.itertuples(index=False):
        ref, cand = str(r.reference_report), str(r.candidate_report)
        rows.append(dict(method="length_diff", anchor_id=r.anchor_id,
                         variant_id=r.variant_id, score=float(len(cand) - len(ref))))
        rows.append(dict(method="cand_length", anchor_id=r.anchor_id,
                         variant_id=r.variant_id, score=float(len(cand))))
    return pd.DataFrame(rows)


def radeval(pairs, metrics, api_key=None):
    from RadEval import RadEval
    kw = dict(metrics=metrics, per_sample=True, show_progress=True)
    if api_key:
        kw["openai_api_key"] = api_key
    scores = RadEval(**kw)(refs=pairs.reference_report.astype(str).tolist(),
                           hyps=pairs.candidate_report.astype(str).tolist())
    rows = []
    for i, r in enumerate(pairs.itertuples(index=False)):
        for key, v in scores.items():
            val = v[i] if isinstance(v, list) else v
            if isinstance(val, dict):
                for sub, sv in val.items():
                    rows.append(dict(method=f"radeval:{key}_{sub}", anchor_id=r.anchor_id,
                                     variant_id=r.variant_id, score=float(sv)))
            elif val is not None:
                rows.append(dict(method=f"radeval:{key}", anchor_id=r.anchor_id,
                                 variant_id=r.variant_id, score=float(val)))
    return pd.DataFrame(rows)


def crimson(pairs, model="rajpurkarlab/medgemma-4b-it-crimson", device="cuda"):
    from CRIMSON.generate_score import CRIMSONScore
    scorer = CRIMSONScore(model_name=model, device=device)
    rows = []
    for r in pairs.itertuples(index=False):
        out = scorer(str(r.reference_report), str(r.candidate_report))
        counts = out.get("error_counts", {}) or {}
        rows.append(dict(method="crimson:crimson_score", anchor_id=r.anchor_id,
                         variant_id=r.variant_id, score=float(out.get("score", 0.0))))
        rows.append(dict(method="crimson:total_predicted_errors", anchor_id=r.anchor_id,
                         variant_id=r.variant_id, score=float(sum(counts.values()))))
        for k, v in counts.items():
            rows.append(dict(method=f"crimson:{k}", anchor_id=r.anchor_id,
                             variant_id=r.variant_id, score=float(v)))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--group", required=True,
                   choices=list(GROUPS) + ["trivial", "crimson", "all"])
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(a.pairs)
    if a.limit:
        pairs = pairs.head(a.limit)
    if a.n_shards > 1:
        pairs = pairs.iloc[a.shard::a.n_shards]
    print(f"{len(pairs)} pairs | group {a.group}")

    key = open(a.api_key_file).read().strip() if a.api_key_file else None
    groups = list(GROUPS) + ["trivial", "crimson"] if a.group == "all" else [a.group]

    for g in groups:
        t0 = time.time()
        if g == "trivial":
            df = trivial(pairs)
        elif g == "crimson":
            df = crimson(pairs)
        else:
            df = radeval(pairs, GROUPS[g], api_key=key)
        suffix = f"_{a.shard}" if a.n_shards > 1 else ""
        path = out / f"predictions_{g}{suffix}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {g}: {df.method.nunique()} methods, {len(df)} rows, "
              f"{time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
