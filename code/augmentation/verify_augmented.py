import json, re, sys
sys.path.insert(0, '.')
import build_augmented_sft as B
from transformers import AutoTokenizer
from sft_samples import build_samples
data = json.load(open(B.OUT)); orig = json.load(open(B.SRC))
bad = {"orders_not_all_opened": [], "unconfirmed_write": [], "unpaired_call": [], "bad_json": [], "adjacent_same_role": []}
for n, conv in enumerate(data):
    msgs = conv["messages"]; uid = B.user_id_of(msgs)
    # 1. every order opened before the first write
    fw = next((i for i, m in enumerate(msgs) if any(c["function"]["name"] in B.WRITE for c in (m.get("tool_calls") or []))), len(msgs))
    opened = {json.loads(c["function"]["arguments"]).get("order_id") for m in msgs[:fw] for c in (m.get("tool_calls") or []) if c["function"]["name"] == "get_order_details"}
    if not set(B.DB["users"][uid]["orders"]) <= opened: bad["orders_not_all_opened"].append(n)
    # 2. every write confirmed
    for i, m in enumerate(msgs):
        if any(c["function"]["name"] in B.WRITE for c in (m.get("tool_calls") or [])) and not B.confirmed(msgs, i): bad["unconfirmed_write"].append((n, i))
    # 3. each tool call answered by a tool message with the same id right after
    for i, m in enumerate(msgs):
        for c in (m.get("tool_calls") or []):
            nxt = [x for x in msgs[i + 1:i + 3] if x["role"] == "tool" and x["tool_call_id"] == c["id"]]
            if not nxt: bad["unpaired_call"].append((n, i))
            try: json.loads(c["function"]["arguments"])
            except Exception: bad["bad_json"].append((n, i))
        if i and m["role"] == "user" and msgs[i - 1]["role"] == "user": bad["adjacent_same_role"].append((n, i))
print({k: len(v) for k, v in bad.items()})
tok = AutoTokenizer.from_pretrained("unsloth/Qwen3-14B-unsloth-bnb-4bit")
def tot(convs):
    s_all, skipped, toks = 0, {"template_mismatch": 0, "too_long": 0}, 0
    for c in convs:
        s, sk = build_samples(c["messages"], c["tools"], tok)
        s_all += len(s); toks += sum(len(x["prompt_ids"]) + len(x["completion_ids"]) for x in s)
        for k in sk: skipped[k] += sk[k]
    return s_all, skipped, toks
a = tot(orig); b = tot(data)
print("original :", "samples", a[0], "skipped", a[1], "tokens (prompt+completion summed over samples)", a[2])
print("augmented:", "samples", b[0], "skipped", b[1], "tokens", b[2])
