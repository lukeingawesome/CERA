# CERA-4B

Training and evaluation code for CERA-4B, a report-level evaluator for generated
chest-CT reports. Given a reference and a candidate report it returns a list of
individual errors, each with a category, anatomical region and clinical criticality,
rather than a single similarity score.

Benchmark: [CERA-Bench](https://huggingface.co/datasets/lukeingawesome/CERA-Bench)
Weights: [CERA-4B](https://huggingface.co/lukeingawesome/CERA-4B)

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
torchrun --nproc_per_node=4 train.py --data data/sft --out runs/cera-4b
```

`--data` holds `sft_train.jsonl` and `sft_dev.jsonl`, one JSON object per line:

```json
{"anchor_id": "...", "messages": [
  {"role": "system",    "content": "<schema>"},
  {"role": "user",      "content": "REFERENCE:\n...\n\nCANDIDATE:\n..."},
  {"role": "assistant", "content": "[{\"category\": \"omission\", ...}]"}]}
```

Loss is masked to completion tokens. Put benchmark anchor ids in
`<data>/gold_anchors.txt` to exclude them from checkpoint selection.

Defaults reproduce the released adapter: LoRA r=32, alpha=64, lr 1e-4 cosine,
2 epochs, effective batch 96 (4 GPUs x 4 x 6 accumulation), max_len 2048.

Two details matter on Qwen3.5. It is a 3:1 Gated-DeltaNet hybrid, so LoRA must
target `in_proj_*`/`out_proj` as well as `q/k/v/o_proj` — naming only the attention
modules adapts 6 of 24 layers, and `train.py` asserts both were hit. And its chat
template leaves `<think>` open, which lets the pretrained reasoning prior override
the adapter; `cera.prompt.render` closes it.

## Evaluate

```bash
python evaluate.py --adapter runs/cera-4b/adapter --pairs data/pairs.parquet --out runs/eval
```

`--pairs` needs `anchor_id`, `variant_id`, `reference_report`, `candidate_report`.
Writes `scores.parquet` (`k_pred`, `k_pred_critical`, `parse_ok`), per-error rows,
and raw generations.

Keep `--max-new-tokens 4096`. At 1280 roughly 12% of pairs truncate, and the
truncated ones are the high-error pairs, which biases any aggregate. Scores are
deterministic within a batch size but not across batch sizes — fix and report it.

## Comparator metrics

`baselines.py` scores the same pairs with published metrics and writes long format
(`method, anchor_id, variant_id, score`), one file per group:

```bash
pip install -r requirements-baselines.txt

python baselines.py --group lexical --pairs data/pairs.parquet --out runs/base
python baselines.py --group labeler --pairs data/pairs.parquet --out runs/base
python baselines.py --group graph   --pairs data/pairs.parquet --out runs/base
python baselines.py --group trivial --pairs data/pairs.parquet --out runs/base
python baselines.py --group green   --pairs data/pairs.parquet --out runs/base
python baselines.py --group crimson --pairs data/pairs.parquet --out runs/base
```

| group | metrics |
|---|---|
| `lexical` | BLEU, ROUGE, BERTScore, RadEvalBERTScore |
| `labeler` | F1CheXbert, F1RadBERT-CT, SRR-BERT, Temporal F1 |
| `graph` | RadGraph, RaTEScore, RadCliQ, RadGraph-RadCliQ |
| `green` | GREEN |
| `crimson` | CRIMSON (local MedGemma-4B) |
| `trivial` | candidate length, length difference — controls, not metrics |

`green` and `crimson` are LLM judges and roughly an order of magnitude slower, which
is why they are separate groups; both take `--shard i --n-shards n` to split across
devices. GREEN via a hosted model needs `--api-key-file`.

## Leaderboard

```bash
python leaderboard.py --predictions runs/base --cera runs/eval/scores.parquet \
    --gold data/pairs.parquet --out runs/leaderboard.csv
```

Reports tau against total, critical and minor error counts with anchor-blocked CIs,
the criticality margin (tau_critical - tau_minor), and paired superiority of one
metric over the rest with Holm correction.

Similarity metrics score higher when reports agree, error counts score higher when
they disagree, so every method is oriented against the gold before comparison. A
negative criticality margin means the metric tracks minor errors better than critical
ones.

## Statistics

`stats.py` holds the primitives and is importable:

| function | purpose |
|---|---|
| `tau` | Kendall tau-b with an anchor-blocked bootstrap CI |
| `paired` | tau difference between two metrics on the same gold |
| `superiority` | paired tests against many comparators, Holm-corrected |
| `holm` | Holm-Bonferroni adjustment |
| `auroc` | discrimination of pairs containing any critical error |
| `multiset_f1` | count-aware detection F1 |
| `format_stability` | does the score survive re-rendering the reference |

Pairs from one anchor are correlated, so every interval resamples anchors rather
than rows. Exact P values use a normal approximation on the bootstrap
distribution, since the empirical tail is censored at 1/(B+1).

## Prompts

`prompts/` holds the prompts used to build the data, verbatim:

| file | use |
|---|---|
| `error_injection_prompt.txt` | generate candidate reports with controlled injected errors |
| `error_training_system.txt`, `error_training_user.txt` | label the training corpus |
| `error_label_claude_prompt.txt` | LLM audit pass over the radiologist's benchmark labels |

## Layout

```
train.py                     LoRA SFT
evaluate.py                  inference and parsing
baselines.py                 comparator metrics
leaderboard.py               the comparison table
stats.py                     analysis primitives
cera/prompt.py               the prompt, and the thinking-block fix
cera/data.py                 dataset and collation
prompts/                     data-construction prompts
requirements.txt             training and evaluation
requirements-baselines.txt   comparator metrics
```

## Licence

Code: MIT. The adapter is trained on CT-RATE-derived text; see the model card
before redistributing weights or using them commercially.
