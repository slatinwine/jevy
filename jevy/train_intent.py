#!/usr/bin/env python3
"""Laya-style typed-decision student, distilled from the official Jev teacher.

Laya's ideas, miniaturized for a 4GB GPU:
  1. ENCODER, not decoder: one forward pass scores ALL options.
  2. Option markers: each candidate sits behind an [OPT] marker token; a linear
     head reads marker hidden states -> option logits. Dynamic candidate count.
  3. Soft-label distillation + post-hoc temperature calibration.

Round-2 upgrades: per-record candidate sets (synthetic 6-intent + real
10-queue domains mixed), hard-example oversampling (student-vs-teacher
disagreement), bigger multilingual backbone with Adafactor.
"""
import argparse
import json
import os
import urllib.request

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

MARKER = "[OPT]"
INSTRUCTIONS = "判断用户消息最符合哪个意图，选出最匹配的候选"


def encode(tok, state, instructions, options, max_len=256, ins_cap=48, opt_cap=12):
    mk = tok.convert_tokens_to_ids([MARKER])[0]
    ins = tok.encode(instructions, add_special_tokens=False)[:ins_cap]
    opt_ids = [tok.encode(o, add_special_tokens=False)[:opt_cap] for o in options]
    # per option the sequence gains marker + option + sep = len(o) + 2 tokens
    fixed = 2 + len(ins) + 1 + sum(len(o) + 2 for o in opt_ids)
    st = tok.encode(state, add_special_tokens=False)[:max(8, max_len - fixed)]
    ids = [tok.cls_token_id] + st + [tok.sep_token_id] + ins + [tok.sep_token_id]
    marker_pos = []
    for o in opt_ids:
        ids.append(mk)
        marker_pos.append(len(ids) - 1)
        ids += o + [tok.sep_token_id]
    ids = ids[:max_len]
    # markers whose slot got truncated away must be dropped too, or the
    # gather in MarkerHead indexes past the sequence and CUDA asserts
    return ids, [p for p in marker_pos if p < len(ids)]


class MarkerHead(nn.Module):
    def __init__(self, hidden, p_drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.out = nn.Linear(hidden, 1)

    def forward(self, hidden, marker_pos):
        b = torch.arange(hidden.size(0), device=hidden.device).unsqueeze(1)
        h = hidden[b, marker_pos]
        return self.out(self.drop(h)).squeeze(-1)


class Student(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone
        self.head = MarkerHead(backbone.config.hidden_size)

    def forward(self, ids, mask, marker_pos):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(h, marker_pos)


def student_agreement(endpoint, records, timeout=120):
    """Query the local student server; return {message_id: agrees_with_teacher}."""
    out = {}
    for k in range(0, len(records), 100):
        chunk = records[k:k + 100]
        body = json.dumps({"states": [
            {"id": r["id"], "state": r["message"],
             "questions": {"intent": {"type": "choice", "instructions": INSTRUCTIONS,
                                      "criteria": r["intents"]}}} for r in chunk]},
            ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(endpoint + "/api/v1/decide", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
        by_id = {s["id"]: s["answers"]["intent"] for s in result["states"]}
        for rec in chunk:
            a = by_id.get(rec["id"])
            if not a:
                continue
            probs = a["probabilities"]
            out[rec["id"]] = max(probs, key=probs.get) == rec["teacher"]["intent"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/distill_teacher_v2.jsonl"])
    ap.add_argument("--out", default="intent/model")
    ap.add_argument("--backbone", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--optim", default="adafactor", choices=["adafactor", "adamw"])
    ap.add_argument("--student-endpoint", default="http://127.0.0.1:8767",
                    help="难例挖掘：查当前学生与老师的分歧（不可用则跳过）")
    ap.add_argument("--oversample-disagree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records, seen = [], set()
    for path in args.data:
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("teacher") and r["id"] not in seen:
                    seen.add(r["id"])
                    records.append(r)
    print(f"{len(records)} teacher-labeled records | device={device}")

    backbone_path = snapshot_download(
        args.backbone,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "sentencepiece.bpe.model", "model.safetensors"])
    tok = AutoTokenizer.from_pretrained(backbone_path)
    num_added = tok.add_tokens([MARKER])
    backbone = AutoModel.from_pretrained(backbone_path)
    backbone.resize_token_embeddings(len(tok))
    model = Student(backbone).to(device)
    print(f"backbone: {args.backbone} | {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params "
          f"| {num_added} new token")

    # hard-example mining: oversample records where the CURRENT student misses
    dis = {}
    if args.oversample_disagree > 1 and args.student_endpoint:
        try:
            dis = student_agreement(args.student_endpoint, records)
            print(f"难例挖掘: {sum(1 for v in dis.values() if not v)}/{len(dis)} 分歧样本")
        except Exception as exc:
            print(f"跳过难例挖掘（学生服务不可用: {exc}）")

    def build(rec):
        options = list(rec["intents"].keys())
        ids, mp = encode(tok, rec["message"], rec.get("instructions", INSTRUCTIONS), options)
        soft = torch.tensor([rec["teacher"]["dist"][o] for o in options], dtype=torch.float)
        return ids, mp, soft

    rng = __import__("random").Random(args.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_dev = max(8, len(idx) // 7)  # ~14% held out
    dev_idx, train_idx = idx[:n_dev], idx[n_dev:]
    extra = []
    for i in train_idx:
        if dis.get(records[i]["id"]) is False:
            extra += [i] * (args.oversample_disagree - 1)
    train_idx += extra
    rng.shuffle(train_idx)
    print(f"train {len(train_idx)} (含难例过采样) / dev {len(dev_idx)}")

    def collate(batch):
        n = max(len(b[1]) for b in batch)
        Tm = max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), Tm), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), Tm), dtype=torch.long)
        mp = torch.zeros((len(batch), n), dtype=torch.long)
        soft = torch.zeros((len(batch), n))
        smask = torch.zeros((len(batch), n), dtype=torch.bool)
        for r, (b_ids, b_mp, b_soft) in enumerate(batch):
            ids[r, :len(b_ids)] = torch.tensor(b_ids)
            mask[r, :len(b_ids)] = 1
            mp[r, :len(b_mp)] = torch.tensor(b_mp)
            soft[r, :len(b_soft)] = b_soft
            smask[r, :len(b_soft)] = True
        return ids, mask, mp, soft, smask

    from torch.utils.data import DataLoader
    train = DataLoader([build(records[i]) for i in train_idx], batch_size=args.batch_size,
                       shuffle=True, collate_fn=collate)

    if args.optim == "adafactor":
        opts = [torch.optim.Adafactor(model.encoder.parameters(), lr=args.lr,
                                      eps=(None, 1e-3)),
                torch.optim.AdamW(model.head.parameters(), lr=args.head_lr)]
    else:
        opts = [torch.optim.AdamW([
            {"params": model.head.parameters(), "lr": args.head_lr},
            {"params": model.encoder.parameters(), "lr": args.lr},
        ], weight_decay=0.01)]
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    def run_batch(ids, mask, mp):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(ids, mask, mp)
        return logits.float()

    def dev_check(T=1.0):
        model.eval()
        agree, n = 0, 0
        with torch.no_grad():
            for i in dev_idx:
                ids, mp, soft = build(records[i])
                logits = model(torch.tensor([ids], device=device),
                               torch.ones(1, len(ids), dtype=torch.long, device=device),
                               torch.tensor([mp], device=device)).float()[0]
                probs = F.softmax(logits / T, dim=0)
                agree += int(probs.argmax().item() == soft.argmax().item())
                n += 1
        model.train()
        return agree, n

    print("--- training (KD soft labels) ---")
    best = {"agree": -1, "state": None, "epoch": -1}
    for epoch in range(args.epochs):
        tot, nb = 0.0, 0
        for ids, mask, mp, soft, smask in train:
            ids, mask, mp = ids.to(device), mask.to(device), mp.to(device)
            soft, smask = soft.to(device), smask.to(device)
            for o in opts:
                o.zero_grad(set_to_none=True)
            logits = run_batch(ids, mask, mp)
            logp = F.log_softmax(logits, dim=1)
            row_loss = -(soft * logp * smask).sum(1) / smask.sum(1).clamp(min=1)
            loss = row_loss.mean()
            scaler.scale(loss).backward()
            for o in opts:
                scaler.unscale_(o)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                scaler.step(o)
            scaler.update()
            tot += loss.item()
            nb += 1
        agree, n = dev_check()
        print(f"epoch {epoch + 1} | loss {tot / max(nb, 1):.4f} | "
              f"dev agree {agree}/{n} = {100 * agree / n:.0f}%")
        if (agree, -epoch) > (best["agree"], best["epoch"]):
            best = {"agree": agree, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"restored best checkpoint from epoch {best['epoch'] + 1} (dev agree {best['agree']}/{n})")

    # post-hoc temperature calibration on dev
    dev_logits = []
    with torch.no_grad():
        for i in dev_idx:
            ids, mp, soft = build(records[i])
            lg = model(torch.tensor([ids], device=device),
                       torch.ones(1, len(ids), dtype=torch.long, device=device),
                       torch.tensor([mp], device=device)).float()[0]
            dev_logits.append((lg.cpu(), soft))
    best_T, best_nll = 1.0, float("inf")
    for T in [0.5 + 0.05 * k for k in range(41)]:
        nll = sum(-(s * F.log_softmax(lg / T, dim=0)).sum().item() for lg, s in dev_logits)
        if nll < best_nll:
            best_nll, best_T = nll, T
    agree, n = dev_check(best_T)
    print(f"calibrated T={best_T:.2f} | dev agree {agree}/{n} = {100 * agree / n:.0f}%")

    os.makedirs(args.out, exist_ok=True)
    model.encoder.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    torch.save({"head": model.head.state_dict(), "temperature": best_T,
                "instructions": INSTRUCTIONS}, os.path.join(args.out, "head.pt"))
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"backbone": args.backbone, "records": len(records),
                   "dev": n, "dev_agree_pct": 100 * agree / n, "temperature": best_T,
                   "best_epoch": best["epoch"] + 1, "hard_mining": bool(dis),
                   "domains": sorted({d for r in records for d in [len(r['intents'])]})},
                   f, ensure_ascii=False, indent=2)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
