#!/usr/bin/env python3
"""Derive noul (boolean) and score (ordered-level) training rows from existing
choice-style distillation data.

Both types reduce to the marker head's choice form:
  noul  -> 2 options ["yes", "no"], target = teacher match-probability
  score -> level descriptions as options, target = annotated level (smoothed)

Output records carry their own "instructions" (the trainer reads them
per-record), so one model serves all three question types.

Usage:
  python make_typed_data.py --labeled data/train_real.jsonl [more.jsonl ...] \
      --tickets-meta data/real_tickets_sample.jsonl data/real_tickets_sample2.jsonl \
      --out-dir data/
"""
import argparse
import hashlib
import json
import random

YES, NO = "是", "否"
LEVELS = {"low": "低", "medium": "中", "high": "高"}


def rid(*parts):
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", nargs="+", required=True,
                    help="choice-format distillation jsonl files")
    ap.add_argument("--tickets-meta", nargs="+", required=True,
                    help="real_tickets_sample jsonl files (for the priority field)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- noul rows: two per record (top intent = yes, random other = mostly no)
    noul = []
    for path in args.labeled:
        for l in open(path, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            dist = r["teacher"]["dist"]
            top = r["teacher"]["intent"]
            others = [k for k in dist if k != top]
            if not others:
                continue
            other = rng.choice(others)
            for intent_id, target in ((top, dist[top]), (other, dist[other])):
                desc = r["intents"].get(intent_id, intent_id)
                # rank-based boolean target: raw normalized probs are too flat
                # (top often ~0.45), which teaches the head a 50/50 answer
                if intent_id == top:
                    p = 0.9
                else:
                    p = min(0.3, target * 1.5)
                noul.append({
                    "id": rid("noul", r["id"], intent_id),
                    "message": r["message"],
                    "instructions": f"这条消息是否符合以下意图：{desc}？",
                    "intents": {YES: "是", NO: "否"},
                    "teacher": {"intent": YES if intent_id == top else NO,
                                "dist": {YES: round(p, 4), NO: round(1 - p, 4)},
                                "model": r["teacher"]["model"] + " (derived)"},
                })
    rng.shuffle(noul)

    # ---- score rows: urgency levels from the ticket dataset's priority field
    meta = {}
    for path in args.tickets_meta:
        for l in open(path, encoding="utf-8"):
            if not l.strip():
                continue
            x = json.loads(l)
            if x.get("priority"):
                meta[x["message"]] = x["priority"]
    score = []
    for msg, priority in meta.items():
        if priority not in LEVELS:
            continue
        dist = {k: args.smoothing for k in LEVELS}
        dist[priority] = round(1 - args.smoothing * (len(LEVELS) - 1), 4)
        score.append({
            "id": rid("score", msg),
            "message": msg,
            "instructions": "评估这条消息的紧急程度。",
            "intents": dict(sorted(LEVELS.items(), key=lambda kv: ["low", "medium", "high"].index(kv[0]))),
            "teacher": {"intent": priority, "dist": dist,
                        "model": "dataset-priority"},
        })
    rng.shuffle(score)

    open(f"{args.out_dir}/typed_noul.jsonl", "w", encoding="utf-8").write(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in noul) + "\n")
    open(f"{args.out_dir}/typed_score.jsonl", "w", encoding="utf-8").write(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in score) + "\n")
    print(f"noul rows: {len(noul)} | score rows: {len(score)}")


if __name__ == "__main__":
    main()
