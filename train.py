"""LoRA SFT of Qwen3.5-4B as a chest-CT report evaluator."""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

from cera.data import PairSFT, collate

# Qwen3.5 is a 3:1 Gated-DeltaNet hybrid. Targeting only q/k/v/o_proj adapts
# 6 of 24 layers; both block types must be named.
LORA_TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def held_out(eval_path, gold_path):
    """Anchors in the eval split, minus any that appear in the benchmark gold."""
    anchors = {json.loads(l)["anchor_id"] for l in open(eval_path)}
    if gold_path.exists():
        excluded = {l.strip() for l in open(gold_path) if l.strip()}
        return anchors - excluded, anchors & excluded
    return anchors, set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dir with sft_train.jsonl and sft_dev.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    set_seed(a.seed)
    data = Path(a.data)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dev, excluded = held_out(data / "sft_dev.jsonl", data / "gold_anchors.txt")
    print(f"dev anchors {len(dev)} | gold anchors excluded from selection {len(excluded)}")

    train_ds = PairSFT(data / "sft_train.jsonl", tok, a.max_len)
    eval_ds = PairSFT(data / "sft_dev.jsonl", tok, a.max_len, keep_anchors=dev)
    print(f"train {len(train_ds):,} | dev {len(eval_ds):,}")

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.print_trainable_parameters()

    hit = {n.split(".")[-3] for n, _ in model.named_parameters() if "lora_A" in n}
    assert hit & {"in_proj_qkv", "out_proj"}, "LoRA missed the DeltaNet blocks"
    assert hit & {"q_proj", "o_proj"}, "LoRA missed the attention blocks"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=a.out, num_train_epochs=a.epochs,
            per_device_train_batch_size=a.batch_size,
            per_device_eval_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accum,
            learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
            bf16=True, logging_steps=25,
            eval_strategy="steps", eval_steps=200,
            save_strategy="steps", save_steps=200, save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, gradient_checkpointing=True,
            ddp_find_unused_parameters=False, report_to=[], seed=a.seed),
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()

    out = Path(a.out) / "adapter"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (Path(a.out) / "config.json").write_text(json.dumps(vars(a), indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
