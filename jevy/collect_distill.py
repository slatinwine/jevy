#!/usr/bin/env python3
"""Label messages with the official Jev API to build distillation data.

For each message, the official jev teacher scores every candidate intent with
a 'noul' question in one request; the normalized score distribution is stored
as a soft label. Output records match the training format:

  {"id", "message", "intents", "teacher": {"intent", "dist", "model"}, "ts"}

Usage:
  python collect_distill.py --messages-file msgs.txt --intents-file intents.json \
      --api-key $TYPESAFE_API_KEY --out data/train_new.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

OFFICIAL_URL = "https://api.typesafe.ai/v1/systemone"


def decide_official(endpoint, api_key, state, intents, model="jev-latest", timeout=120):
    """One request scores all candidate intents; returns (normalized dist, model)."""
    questions = {qid: {"type": "noul", "instructions": f"Does this message match this intent: {desc}?"}
                 for qid, desc in intents.items()}
    body = json.dumps({"state": state, "model": model, "questions": questions},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or os.environ.get('TYPESAFE_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    answers = result.get("answers") or {}
    if not answers:
        raise ValueError(f"official API returned no answers: {str(result)[:120]}")
    scores = {qid: float(a.get("noul", 0.0) or 0.0) for qid, a in answers.items()}
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}, result.get("model", "jev")


def msg_id(message):
    return hashlib.sha1(message.strip().lower().encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--messages-file", required=True, help="UTF-8 text, one message per line")
    ap.add_argument("--intents-file", required=True, help='JSON: {"id": "description", ...}')
    ap.add_argument("--api-key", default=os.environ.get("TYPESAFE_API_KEY"))
    ap.add_argument("--endpoint", default=OFFICIAL_URL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    intents = json.load(open(args.intents_file, encoding="utf-8"))
    done = set()
    if os.path.exists(args.out):  # resumable: skip already-labeled messages
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["id"])

    messages = [l.strip() for l in open(args.messages_file, encoding="utf-8") if l.strip()]
    new, failed = 0, 0
    for msg in messages:
        mid = msg_id(msg)
        if mid in done:
            continue
        try:
            dist, model = decide_official(args.endpoint, args.api_key, msg, intents)
        except Exception as exc:
            failed += 1
            print(f"failed: {msg[:40]} -> {exc}", file=sys.stderr)
            time.sleep(1.0)
            continue
        best = max(dist, key=dist.get)
        rec = {"id": mid, "message": msg, "intents": intents,
               "teacher": {"intent": best,
                           "dist": {k: round(v, 4) for k, v in
                                    sorted(dist.items(), key=lambda kv: -kv[1])},
                           "model": f"{model} (official)"},
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done.add(mid)
        new += 1
        print(f"teacher={best} <- {msg[:48]}")
        time.sleep(args.sleep)
    print(f"\nlabeled {new} new ({failed} failed) | total {len(done)} | {args.out}")


if __name__ == "__main__":
    main()
