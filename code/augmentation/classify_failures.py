"""Rule-based classification of the failed rollouts of the tau-bench retail 4-trial runs into the same causes used for tau3.
Uses the successful database changes of each rollout vs the target actions of the task, plus a few conversation facts."""
import json, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
DB = json.load(open(os.path.join(HERE, "..", "tau3", "tau2-bench", "data", "tau2", "domains", "retail", "db.json")))
TASKS = json.load(open(os.path.join(HERE, "data", "retail_test_tasks.json")))
WRITE = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items",
         "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"}
LOCK = ("non-pending order cannot be", "non-delivered order cannot be")
CAUSES = {1: "Wrong or incomplete order lookup (missed, mixed up or never opened the right order)",
          2: "Changed the database without an explicit yes, or before the facts were in (locked order, own choice of payment or reason)",
          3: "Wrong or invented values, or said something different from what it did",
          4: "Cancelled a whole order for a one-item request", 5: "Account lookup dead end: gave up, or invented details",
          6: "Simulated customer or scenario problem", 7: "Did not state the required information (database was right)",
          8: "Transferred to a human when the task was doable", 9: "Other / needs a human look"}
def canon(name, kw):
    kw = dict(kw)
    if "item_ids" in kw and "new_item_ids" in kw:
        pairs = sorted(zip(kw.pop("item_ids"), kw.pop("new_item_ids"))); kw["pairs"] = pairs
    elif "item_ids" in kw: kw["item_ids"] = sorted(kw["item_ids"])
    return (name, json.dumps(kw, sort_keys=True))
def parse(conv):
    calls, out, succ, errs, opened, transferred, authed = {}, [], [], [], [], False, False
    first_write_opened = None
    global _UID
    _UID = None
    for m in conv["messages"]:
        for c in (m.get("tool_calls") or []):
            f = c["function"]
            try: a = json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]
            except Exception: a = {}
            calls[c["id"]] = (f["name"], a)
            if f["name"] == "get_order_details": opened.append(a.get("order_id"))
            if f["name"] == "transfer_to_human_agents": transferred = True
            if f["name"] in WRITE and first_write_opened is None: first_write_opened = set(opened)
        if m["role"] == "tool":
            nm, a = calls.get(m.get("tool_call_id"), (m.get("name"), {}))
            content = (m.get("content") or "")
            if nm in ("get_user_details",) and content.startswith("{"): authed = True; _UID = a.get("user_id")
            if (nm or "").startswith("find_user_id") and content and not content.startswith("Error"): authed = True
            if nm in WRITE:
                if content.startswith("Error"): errs.append(content)
                else: succ.append((nm, a))
            elif content.startswith("Error") and any(l in content.lower() for l in LOCK): errs.append(content)
    return succ, errs, set(opened), transferred, authed
_UID = None
def classify(conv, task):
    succ, errs, opened, transferred, authed = parse(conv)
    tgt = [(a["name"], a["kwargs"]) for a in task["actions"] if a["name"] in WRITE]
    S, T = Counter(canon(*w) for w in succ), Counter(canon(*w) for w in tgt)
    missing, extra = T - S, S - T
    info = {"missing": [m[0] for m in missing.elements()], "extra": [e[0] for e in extra.elements()], "transferred": transferred, "locked": any(any(l in e.lower() for l in LOCK) for e in errs)}
    if not authed: return 5, info
    if transferred and any("payment method should be" in e.lower() for e in errs): return 6, info   # customer insists on a refund method the policy forbids
    if not missing and not extra: return 7, info
    tgt_orders = {a.get("order_id") for _, a in tgt if a.get("order_id")}
    extra_w = [w for w in succ if canon(*w) in extra]
    miss_w = [w for w in tgt if canon(*w) in missing]
    for n, a in extra_w:   # whole-order cancel of a multi-item order the task did not cancel
        if n == "cancel_pending_order" and a.get("order_id") not in {x.get("order_id") for nm, x in tgt if nm == "cancel_pending_order"}:
            if len(DB["orders"].get(a["order_id"], {}).get("items", [])) > 1: return 4, info
    if info["locked"]: return 2, info
    if any(a.get("order_id") and a["order_id"] not in tgt_orders for n, a in extra_w) or any(a.get("order_id") and a["order_id"] not in opened for n, a in miss_w): return 1, info
    # wrong address: the target address belongs to an order of this customer that was never opened -> lookup problem
    uid = _UID
    for n, a in miss_w:
        if n in ("modify_user_address", "modify_pending_order_address") and uid in DB["users"]:
            src = [o for o in DB["users"][uid]["orders"] if DB["orders"][o]["address"].get("address1") == a.get("address1") and DB["orders"][o]["address"].get("zip") == a.get("zip")]
            if src and not (set(src) & opened): return 1, info
    same = [(m, e) for m in miss_w for e in extra_w if m[0] == e[0] and m[1].get("order_id") == e[1].get("order_id")]
    if same:
        diff = set()
        for m, e in same:
            diff |= {k for k in set(m[1]) | set(e[1]) if m[1].get(k) != e[1].get(k)}
        return (2 if diff <= {"payment_method_id", "reason"} else 3), info
    if transferred and not extra_w: return 8, info
    if not extra_w and miss_w: return 3, info
    return 9, info
def run(results_path, transcripts_path):
    rows = {(json.loads(l)["task_index"], json.loads(l)["trial"]): json.loads(l) for l in open(results_path)}
    out = []
    for l in open(transcripts_path):
        c = json.loads(l)
        if c["reward"] >= 1: continue
        cause, info = classify(c, TASKS[c["task_index"]])
        out.append({"task": c["task_index"], "trial": c["trial"], "cause": cause, **info})
    return out
if __name__ == "__main__":
    D = os.path.join(HERE, "data")
    res = {}
    for name, f in (("control", "retail_starting_adapter_rerun_4trials"), ("distilled", "retail_new_model_4trials"), ("augmented", "retail_aug_4trials")):
        res[name] = run(f"{D}/{f}_results.jsonl", f"{D}/{f}_transcripts.jsonl")
    json.dump({"causes": CAUSES, "failures": res}, open(os.path.join(D, "failure_causes_auto.json"), "w"))
    print("cause | " + " | ".join(f"{n} ({len(v)} failed)" for n, v in res.items()))
    for c in CAUSES: print(c, "|", " | ".join(str(sum(1 for r in v if r["cause"] == c)) for v in res.values()), "|", CAUSES[c][:70])
