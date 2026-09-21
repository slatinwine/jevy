"""Minimal jevy intent-recognition client (stdlib only)."""
import json
import urllib.request


def decide(state, intents, endpoint="http://127.0.0.1:8767", timeout=60):
    """Return (chosen_intent, probability_distribution)."""
    payload = {"state": state,
               "questions": {"intent": {"type": "choice",
                                        "instructions": "判断用户消息最符合哪个意图",
                                        "criteria": intents}}}
    req = urllib.request.Request(endpoint + "/api/v1/decide",
                                 data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())["decisions"]["intent"]
    return d["value"], d["probabilities"]


if __name__ == "__main__":
    intents = {"hardware": "硬件故障", "billing": "账单问题",
               "shipping": "物流问题", "other": "其他"}
    value, probs = decide("我的手机充不进电，买了才两周", intents)
    print("intent:", value)
    print(json.dumps(probs, ensure_ascii=False, indent=1))
