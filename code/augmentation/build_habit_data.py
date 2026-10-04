"""Habit-only SFT data: the ONLY supervised assistant turns are
   (a) lookup steps   : loading the account and opening EVERY order of the customer, one tool call per turn
   (b) confirmation steps : the agent message that lists exactly what it is about to change and asks yes/no, and the tool call
                            that follows the customer's yes (it must execute exactly what was confirmed)
Everything else in a conversation stays as unsupervised context. Output: habit_dataset.json = list of
   {"messages": [...], "tools": [...], "supervise": [message indices], "kind": "real" | "synthetic_lookup" | "synthetic_confirm"}
Real conversations come from the 178 augmented teacher conversations (train tasks). Synthetic ones are built from the retail
database using only customers that no test task uses.
"""
import copy, json, os, random, re
import build_augmented_sft as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "habit_dataset.json")
rnd = random.Random(7)
DB = B.DB
AUG = json.load(open(B.OUT))
TOOLS = AUG[0]["tools"]
SYSTEM = AUG[0]["messages"][0]["content"]
TESTT = json.load(open(os.path.join(HERE, "data", "retail_test_tasks.json")))

# customers that any test task uses (never used for synthetic examples)
test_users = set()
for t in TESTT:
    for a in t["actions"]:
        k = a["kwargs"]
        if a["name"].startswith("find_user_id"):
            for uid, u in DB["users"].items():
                if ("email" in k and u["email"] == k["email"]) or ("first_name" in k and u["name"]["first_name"] == k["first_name"] and u["name"]["last_name"] == k["last_name"] and u["address"]["zip"] == k["zip"]):
                    test_users.add(uid)
FREE = [u for u in DB["users"] if u not in test_users]

# --------------------------------------------------------------------------- reasoning texts (natural style, with the real ids)
def r_user(uid):
    return rnd.choice([f"\nOkay, I have the customer id {uid}. Next I'll load the account to see their payment methods and every order on it.\n",
                       f"\nGot the user id: {uid}. Now I should pull up the account details, which list all their orders.\n"])
def r_first(ids, first):
    n = len(ids)
    if n == 1:
        return f"\nOkay, I have the account details. The customer has one order, {ids[0]}. I'll open it to see exactly what is in it before I do or say anything else.\n"
    return rnd.choice([f"\nOkay, I have the account details. The customer has {n} orders: {', '.join(ids)}. I don't know yet which one matters, so before I answer or change anything I'll open every order, starting with {first}.\n",
                       f"\nThe account lists {n} orders: {', '.join(ids)}. Rather than guess which one the customer means, I'll read all of them first. Opening {first} now.\n"])
def r_next(seen, nxt, left):
    return rnd.choice([f"\nOrders opened so far: {', '.join(seen)}. Next I'll open {nxt}; {left} more after that.\n",
                       f"\nI've read {', '.join(seen)}. Still to open: {nxt} and {left} more. Opening {nxt} now.\n"])
def r_last(seen, nxt, n):
    return rnd.choice([f"\nOrders opened so far: {', '.join(seen)}. {nxt} is the last one; after it I'll have read all {n} orders.\n",
                       f"\nOnly {nxt} is left. Once I open it I will have seen every one of the {n} orders.\n"])
WHAT = {"cancel_pending_order": "cancel the whole order", "return_delivered_order_items": "return these items", "exchange_delivered_order_items": "exchange these items",
        "modify_pending_order_items": "change these items", "modify_pending_order_address": "change the shipping address", "modify_user_address": "change the default address",
        "modify_pending_order_payment": "change the payment method"}
def r_confirm(name):
    return rnd.choice([f"\nOkay, I know what to do now: {WHAT[name]}. This changes the database for good, so before calling the tool I have to list exactly what will happen and get an explicit yes. I'll write the summary and ask.\n",
                       f"\nI have everything I need to {WHAT[name]}. The policy says to list the action details and wait for an explicit yes before any change, so I'll summarise it for the customer and ask.\n"])
R_WRITE = ["\nThe customer said yes to exactly what I listed, so I can make the change now.\n", "\nThey confirmed the details, so I'll go ahead and run it exactly as listed.\n"]

# --------------------------------------------------------------------------- real conversations
LOOK_MARK = ("I have the customer id", "I have the customer's account. They have", "Orders opened so far")
def is_inserted_lookup(m):
    r = m.get("reasoning_content") or ""
    return m["role"] == "assistant" and any(k in r for k in LOOK_MARK)
def real_samples():
    out, cnt = [], {"user_details": 0, "orders": 0, "confirm": 0, "write": 0}
    for conv in AUG:
        msgs = copy.deepcopy(conv["messages"]); uid = B.user_id_of(msgs); orders = DB["users"][uid]["orders"]
        sup = set()
        opened = set()
        for i, m in enumerate(msgs):
            for c in (m.get("tool_calls") or []):
                nm = c["function"]["name"]; a = json.loads(c["function"]["arguments"])
                if nm == "get_order_details" and not is_inserted_lookup(m): opened.add(a.get("order_id"))
        # (a) lookup turns: get_user_details (original or inserted) and the inserted order turns, with natural reasoning for inserted ones
        seen = []
        block_total = [i for i, m in enumerate(msgs) if is_inserted_lookup(m) and m["tool_calls"][0]["function"]["name"] == "get_order_details"]
        first_in_block = True
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or not m.get("tool_calls"): continue
            nm = m["tool_calls"][0]["function"]["name"]; a = json.loads(m["tool_calls"][0]["function"]["arguments"])
            if nm == "get_user_details":
                if is_inserted_lookup(m): m["reasoning_content"] = r_user(uid)
                sup.add(i); cnt["user_details"] += 1
            if nm == "get_order_details" and is_inserted_lookup(m):
                todo = [o for o in orders if o not in seen]
                already = [o for o in orders if o in seen]
                if first_in_block: m["reasoning_content"] = r_first(orders, a["order_id"]); first_in_block = False
                elif len(block_total) and i == block_total[-1]: m["reasoning_content"] = r_last(seen, a["order_id"], len(orders))
                else:
                    left = len([x for x in block_total if x > i])
                    m["reasoning_content"] = r_next(seen, a["order_id"], left)
                sup.add(i); cnt["orders"] += 1
            if nm == "get_order_details": seen.append(a.get("order_id"))
        # (b) confirmation turns and the write call after the yes
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and any(c["function"]["name"] in B.WRITE for c in (m.get("tool_calls") or [])):
                sup.add(i); cnt["write"] += 1
                u = next((j for j in range(i - 1, -1, -1) if msgs[j]["role"] == "user"), None)
                a_ = next((j for j in range(u - 1, -1, -1) if msgs[j]["role"] == "assistant" and (msgs[j].get("content") or "").strip()), None) if u is not None else None
                if a_ is not None:
                    if (msgs[a_].get("content") or "").startswith("Before I"):
                        nm = msgs[i]["tool_calls"][0]["function"]["name"]; msgs[a_]["reasoning_content"] = r_confirm(nm)
                    sup.add(a_); cnt["confirm"] += 1
        out.append({"messages": msgs, "tools": conv["tools"], "supervise": sorted(sup), "kind": "real"})
    return out, cnt

# --------------------------------------------------------------------------- synthetic conversations
OPEN_POOL = [x["messages"][1]["content"] for x in AUG]
_ASK_ALL = [x["messages"][2]["content"] for x in AUG if x["messages"][2]["role"] == "assistant" and not x["messages"][2].get("tool_calls")]
ASK_POOL = [t for t in _ASK_ALL if not re.search(r"cancel|exchang|return|modif|chang|updat|refund", t, re.I)]   # neutral: no specific action mentioned
assert len(ASK_POOL) >= 5, len(ASK_POOL)
def cid(*p): return "chatcmpl-tool-" + B.h("hab", *p)[:32]
def auth_block(uid, tag, supervise_lookups, msgs, sup):
    u = DB["users"][uid]; first, last, zp = u["name"]["first_name"], u["name"]["last_name"], u["address"]["zip"]
    by_email = rnd.random() < 0.35
    msgs.append({"role": "assistant", "content": rnd.choice(ASK_POOL), "reasoning_content": "\nOkay, I need to verify the customer's identity first, by email or by name and zip code.\n"})
    if by_email:
        msgs.append({"role": "user", "content": rnd.choice([f"Sure, my email is {u['email']}.", f"It's {u['email']}.", f"My email address is {u['email']}."])})
        call, args = "find_user_id_by_email", {"email": u["email"]}
    else:
        msgs.append({"role": "user", "content": rnd.choice([f"Sure, my name is {first} {last} and my zip code is {zp}.", f"I'm {first} {last}, zip code {zp}.", f"{first} {last}, {zp}."])})
        call, args = "find_user_id_by_name_zip", {"first_name": first, "last_name": last, "zip": zp}
    i1 = cid(tag, "find"); msgs.append(B.call_msg(i1, call, args, "\nI can look the customer up with what they gave me.\n")); msgs.append(B.tool_msg(i1, call, uid))
    i2 = cid(tag, "user"); msgs.append(B.call_msg(i2, "get_user_details", {"user_id": uid}, r_user(uid)))
    if supervise_lookups: sup.add(len(msgs) - 1)
    msgs.append(B.tool_msg(i2, "get_user_details", json.dumps({k: v for k, v in u.items() if k != "user_id"})))
    orders = u["orders"]; seen = []
    for k, oid in enumerate(orders):
        left = len(orders) - k - 1
        r = r_first(orders, oid) if k == 0 else (r_last(seen, oid, len(orders)) if left == 0 else r_next(seen, oid, left))
        ic = cid(tag, "order", oid); msgs.append(B.call_msg(ic, "get_order_details", {"order_id": oid}, r))
        if supervise_lookups: sup.add(len(msgs) - 1)
        msgs.append(B.tool_msg(ic, "get_order_details", json.dumps(DB["orders"][oid]))); seen.append(oid)

def synthetic_lookup(uid, n):
    msgs, sup = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rnd.choice(OPEN_POOL)}], set()
    auth_block(uid, f"L{n}", True, msgs, sup)
    return {"messages": msgs, "tools": TOOLS, "supervise": sorted(sup), "kind": "synthetic_lookup"}

def synthetic_confirm(uid, n):
    u = DB["users"][uid]; pend = [o for o in u["orders"] if DB["orders"][o]["status"] == "pending"]; deliv = [o for o in u["orders"] if DB["orders"][o]["status"] == "delivered"]
    kinds = (["cancel"] if pend else []) + (["return"] if deliv else [])
    if not kinds: return None
    kind = rnd.choice(kinds)
    msgs, sup = [{"role": "system", "content": SYSTEM}], set()
    if kind == "cancel":
        oid = rnd.choice(pend); reason = rnd.choice(["no longer needed", "ordered by mistake"])
        why = "I don't need it anymore" if reason == "no longer needed" else "I ordered it by mistake"
        msgs.append({"role": "user", "content": rnd.choice([f"Hi, I'd like to cancel my order {oid}. {why}.", f"Hello! Please cancel order {oid}, {why}."])})
        auth_block(uid, f"C{n}", False, msgs, sup)
        args = {"order_id": oid, "reason": reason}; name = "cancel_pending_order"
    else:
        oid = rnd.choice(deliv); items = DB["orders"][oid]["items"]; pick = items if rnd.random() < 0.5 or len(items) == 1 else rnd.sample(items, rnd.randint(1, len(items) - 1))
        names = ", ".join(sorted({B.item_info(i["item_id"])[0].lower() for i in pick}))
        msgs.append({"role": "user", "content": rnd.choice([f"Hi, I need to return the {names} from order {oid}.", f"Hello, I'd like to return the {names} I received in order {oid}, refund to the original payment please."])})
        auth_block(uid, f"C{n}", False, msgs, sup)
        pay = [p for p in DB["orders"][oid].get("payment_history", []) if p.get("transaction_type") == "payment"][0]["payment_method_id"]
        args = {"order_id": oid, "item_ids": [i["item_id"] for i in pick], "payment_method_id": pay}; name = "return_delivered_order_items"
    msgs.append({"role": "assistant", "content": B.confirm_text(name, args, uid), "reasoning_content": r_confirm(name)}); sup.add(len(msgs) - 1)
    msgs.append({"role": "user", "content": rnd.choice(B.YES)})
    msgs.append(B.call_msg(cid(f"C{n}", "write"), name, args, rnd.choice(R_WRITE))); sup.add(len(msgs) - 1)
    return {"messages": msgs, "tools": TOOLS, "supervise": sorted(sup), "kind": "synthetic_confirm"}

def main():
    real, cnt = real_samples()
    multi = [u for u in FREE if len(DB["users"][u]["orders"]) >= 2]; one = [u for u in FREE if len(DB["users"][u]["orders"]) == 1]
    syn_l = [synthetic_lookup(u, n) for n, u in enumerate(multi)] + [synthetic_lookup(u, 10000 + n) for n, u in enumerate(rnd.sample(one, min(60, len(one))))]
    cand = [u for u in FREE if DB["users"][u]["orders"]]; rnd.shuffle(cand)
    syn_c = [x for x in (synthetic_confirm(u, n) for n, u in enumerate(cand[:420])) if x]
    data = real + syn_l + syn_c
    for d in data:
        for i in d["supervise"]: assert d["messages"][i]["role"] == "assistant"
    json.dump(data, open(OUT, "w"))
    stats = {"conversations": len(data), "real": len(real), "synthetic_lookup": len(syn_l), "synthetic_confirm": len(syn_c), "free_customers_used": len(FREE), "test_customers_excluded": len(test_users),
             "supervised_turns": sum(len(d["supervise"]) for d in data), "real_counts": cnt,
             "supervised_by_kind": {k: sum(len(d["supervise"]) for d in data if d["kind"] == k) for k in ("real", "synthetic_lookup", "synthetic_confirm")}}
    json.dump(stats, open(OUT.replace(".json", "_stats.json"), "w"), indent=1); print(json.dumps(stats, indent=1))
if __name__ == "__main__": main()
