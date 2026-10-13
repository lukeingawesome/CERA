"""Score report pairs with a trained CERA adapter."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cera.prompt import messages, render

CATS = ["false_finding", "omission", "location", "severity",
        "spurious_comparison", "omitted_comparison", "comparison"]


def parse(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\[.*\]", t, flags=re.S)
        if not m:
            return [], False
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return [], False
    if not isinstance(obj, list):
        return [], False
    return [e for e in obj if isinstance(e, dict)], True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--pairs", required=True, help="parquet with reference_report, candidate_report")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.adapter)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload().eval()

    d = pd.read_parquet(a.pairs)
    if a.limit:
        d = d.head(a.limit)

    rows, errs, raw = [], [], []
    for i in range(0, len(d), a.batch_size):
        chunk = d.iloc[i:i + a.batch_size]
        prompts = [render(tok, messages(str(r.reference_report), str(r.candidate_report)))
                   for r in chunk.itertuples(index=False)]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, r in enumerate(chunk.itertuples(index=False)):
            text = tok.decode(new[j], skip_special_tokens=True)
            e, ok = parse(text)
            key = {"anchor_id": r.anchor_id, "variant_id": r.variant_id}
            raw.append({**key, "raw": text, "truncated": bool(len(new[j]) >= a.max_new_tokens)})
            rows.append({**key, "k_pred": float(len(e)), "parse_ok": ok,
                         "k_pred_critical": float(sum(
                             1 for x in e if str(x.get("criticality", "")).lower() == "critical"))})
            errs += [{**key, **{k: x.get(k) for k in
                                ("category", "region", "concept", "criticality")}} for x in e]
        print(f"{min(i + a.batch_size, len(d))}/{len(d)}", flush=True)

    pd.DataFrame(rows).to_parquet(out / "scores.parquet", index=False)
    pd.DataFrame(errs).to_parquet(out / "predicted_errors.parquet", index=False)
    with open(out / "raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    s = pd.DataFrame(rows)
    print(f"n={len(s)} parse_ok={s.parse_ok.mean():.3f} "
          f"mean k={s.k_pred.mean():.3f} mean k_critical={s.k_pred_critical.mean():.3f}")


if __name__ == "__main__":
    main()
