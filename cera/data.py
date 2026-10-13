import json
import torch
from torch.utils.data import Dataset

from .prompt import render


class PairSFT(Dataset):
    """Chat pair -> JSON error list. Loss on completion tokens only."""

    def __init__(self, path, tok, max_len=2048, keep_anchors=None, limit=None):
        self.tok, self.max_len, self.rows = tok, max_len, []
        for line in open(path):
            r = json.loads(line)
            if keep_anchors is not None and r["anchor_id"] not in keep_anchors:
                continue
            self.rows.append(r)
            if limit and len(self.rows) >= limit:
                break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        prompt = render(self.tok, msgs[:-1])
        full = self.tok.apply_chat_template(msgs, tokenize=False)
        p = self.tok(prompt, add_special_tokens=False).input_ids
        f = self.tok(full, add_special_tokens=False).input_ids[: self.max_len]
        labels = list(f)
        for j in range(min(len(p), len(labels))):
            labels[j] = -100
        return {"input_ids": f, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, att = [], [], []
    for b in batch:
        d = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * d)
        labs.append(b["labels"] + [-100] * d)
        att.append([1] * len(b["input_ids"]) + [0] * d)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(att)}
