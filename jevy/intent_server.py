#!/usr/bin/env python3
"""Serve the distilled student model behind /api/v1/decide (port 8767).

Contract-compatible with NanoJev/Laya servers: {"state", "questions" of type
"choice"} in, per-option probability distribution out. bool/noul questions are
not implemented in this student (choice only)."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from train_intent import MarkerHead, encode  # reuse serialization

PORT = 8767
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
encoder = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()
_head_state = torch.load(os.path.join(MODEL_DIR, "head.pt"), map_location=device, weights_only=True)
head = MarkerHead(encoder.config.hidden_size).to(device)
head.load_state_dict(_head_state["head"])
head.eval()
TEMP = float(_head_state.get("temperature", 1.0))
YES, NO = "是", "否"  # must match make_typed_data.py option texts


@torch.no_grad()
def score_choice(state, instructions, criteria, option_texts=None):
    keys = list(criteria.keys())
    texts = option_texts or keys
    ids, mp = encode(tok, state, instructions, texts)
    hidden = encoder(input_ids=torch.tensor([ids], device=device),
                     attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device)
                     ).last_hidden_state
    logits = head(hidden, torch.tensor([mp], device=device)).float()[0]
    probs = F.softmax(logits / TEMP, dim=0).tolist()
    return keys, probs


def answer_one(state, q):
    """Answer one typed question dict; returns the decisions entry."""
    t = str(q.get("type", "")).lower()
    if t in ("bool", "boolean", "noul"):
        # trained as a 2-marker choice with the same 是/否 option texts
        options, probs = score_choice(state, q.get("instructions", ""), {YES: 1, NO: 2})
        p_yes = probs[0]
        return {"type": "bool", "value": p_yes > 0.5, "p_true": round(p_yes, 4),
                "probabilities": {"true": round(p_yes, 4), "false": round(1 - p_yes, 4)}}
    if t == "score":
        criteria = q.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score questions need criteria as an ordered list of 2-10 levels")
        options, probs = score_choice(state, q.get("instructions", ""),
                                      {c: i for i, c in enumerate(criteria)})
        expected = sum(i * p for i, p in enumerate(probs)) + 1
        best = max(range(len(options)), key=lambda i: probs[i])
        return {"type": "score", "value": options[best], "expected": round(expected, 2),
                "probabilities": {o: round(p, 4) for o, p in zip(options, probs)},
                "confidence": round(probs[best], 4)}
    if t != "choice":
        raise ValueError(f"unknown question type: {q.get('type')!r} (use choice/bool/score)")
    criteria = q.get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise ValueError("choice questions need criteria with >= 2 options")
    options, probs = score_choice(state, q.get("instructions", ""), criteria,
                                  option_texts=[f"{criteria[k]} ({k})" for k in criteria]
                                  if all(str(k).startswith("idx") for k in criteria) else None)
    best = max(range(len(options)), key=lambda i: probs[i])
    return {"type": "choice", "value": options[best],
            "probabilities": {o: round(p, 4) for o, p in zip(options, probs)},
            "confidence": round(probs[best], 4)}


def server_class():
    class H(BaseHTTPRequestHandler):
        def send_json(self, code, data):
            content = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if urlparse(self.path).path == "/api/health":
                self.send_json(200, {"ready": True, "backend": "jev-distill-student",
                                     "temperature": TEMP})
                return
            self.send_json(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if urlparse(self.path).path != "/api/v1/decide":
                self.send_json(404, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    payload = json.loads(raw.decode("gbk"))
                if not isinstance(payload, dict):
                    raise ValueError('body must be a JSON object')
                if "states" not in payload and ("state" not in payload or "questions" not in payload):
                    raise ValueError('body must contain "state"+"questions" or "states"')
                if "states" in payload:  # batched NanoJev-style contract
                    batch = payload["states"]
                    if len(batch) > 128:
                        raise ValueError("max 128 states per request")
                    out_states = []
                    for s in batch:
                        if "state" not in s or "questions" not in s:
                            raise ValueError('each state needs "state" and "questions"')
                        answers = {qid: answer_one(s["state"], q) for qid, q in s["questions"].items()}
                        out_states.append({"id": s.get("id"), "answers": answers})
                    self.send_json(200, {"object": "decision", "model": "jev-distill-student",
                                         "states": out_states})
                    return
                decisions = {}
                for qid, q in payload["questions"].items():
                    decisions[str(qid)] = answer_one(payload["state"], q)
                self.send_json(200, {"object": "decision", "model": "jev-distill-student",
                                     "decisions": decisions})
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                self.send_json(500, {"error": "student inference failed; inspect the server process."})
                raise

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

    return H


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    encoder = AutoModel.from_pretrained(args.model_dir).to(device).eval()
    _hs = torch.load(os.path.join(args.model_dir, "head.pt"), map_location=device, weights_only=True)
    head.load_state_dict(_hs["head"])
    head.eval()
    TEMP = float(_hs.get("temperature", 1.0))
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}", "ready": True,
                      "backend": "jev-distill-student", "temperature": TEMP}), flush=True)
    HTTPServer(("127.0.0.1", args.port), server_class()).serve_forever()
