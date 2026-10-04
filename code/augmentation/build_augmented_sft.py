"""Builds an augmented copy of the 180 teacher SFT conversations:

  1. LOOKUPS      right after the customer's account is loaded, open EVERY order of the customer (get_order_details), with a
                  short <think> ledger ("orders: A, B, C; opened so far: ...; opening X next"). Tool results are the real
                  database rows (tau3 retail db.json, identical to what the conversations already show).
  2. CONFIRMATIONS before every change to the database (cancel / return / exchange / modify) that was not already preceded by
                  an agent message asking for confirmation and a customer "yes": insert an agent message that lists exactly what
                  will be done (items, prices, refund / payment method, one-time-per-order warning) and ends with a yes/no
                  question, then a short customer "Yes". The agent's own <think> before it explains why.

Everything inserted is built by rules from the database, so it is correct by construction and verifiable.
Usage: python3 build_augmented_sft.py   ->  teacher32b_passed_sft_lookups_confirm.json (+ _stats.json)
"""
import copy, hashlib, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
DB = json.load(open(os.path.join(HERE, "..", "tau3", "tau2-bench", "data", "tau2", "domains", "retail", "db.json")))
SRC = os.path.join(HERE, "teacher32b_passed_sft_reasoning.json")
OUT = os.path.join(HERE, "teacher32b_passed_sft_lookups_confirm.json")

WRITE = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items",
         "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"}
AFFIRM = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|go ahead|please (go|proceed|do)|proceed|confirm|confirmed|correct|that'?s (right|correct)|absolutely|definitely|alright|let'?s do)\b", re.I)
CONFIRM_Q = re.compile(r"(confirm|proceed|go ahead|\(yes|yes/no|yes or no|shall i|would you like me to|do you want me to|is that (correct|ok|okay)|sound good)", re.I)
YES = ["Yes, please go ahead.", "Yes, that's right. Please proceed.", "Yes, go ahead.", "Yes, confirmed.", "Yes, please do that.", "Yes, that's correct, go ahead."]


def h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def money(x):
    return f"${x:,.2f}"


def item_info(item_id):
    for p in DB["products"].values():
        v = p["variants"].get(item_id)
        if v:
            opts = ", ".join(f"{k}: {val}" for k, val in v["options"].items())
            return p["name"], opts, v["price"]
    return "item " + item_id, "", 0.0


def item_text(item_id):
    n, o, p = item_info(item_id)
    return f"{n} ({o}, {money(p)})" if o else f"{n} ({money(p)})"


def pm_text(user_id, pm_id):
    pm = DB["users"][user_id]["payment_methods"].get(pm_id)
    if not pm:
        return pm_id
    if pm["source"] == "credit_card":
        return f"your {pm.get('brand', 'credit')} card ending in {pm.get('last_four', '')}".replace("your credit card", "your credit card")
    if pm["source"] == "gift_card":
        return f"your gift card (balance {money(pm.get('balance', 0))})"
    return "your PayPal account" if pm["source"] == "paypal" else pm_id


def order_items(order_id, item_ids):
    """Items of an order by item id (an id can appear several times)."""
    prices = {}
    for it in DB["orders"][order_id]["items"]:
        prices.setdefault(it["item_id"], it["price"])
    return [prices.get(i, item_info(i)[2]) for i in item_ids]


def tool_msg(call_id, name, content):
    return {"role": "tool", "content": content, "tool_call_id": call_id, "name": name}


def call_msg(call_id, name, args, reasoning):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": json.dumps(args), "name": name}, "id": call_id, "type": "function"}],
            "reasoning_content": reasoning}


# ---------------------------------------------------------------- confirmations
def confirm_text(name, a, user_id):
    if name == "cancel_pending_order":
        o = DB["orders"][a["order_id"]]
        total = sum(it["price"] for it in o["items"])
        pay = [p for p in o.get("payment_history", []) if p.get("transaction_type") == "payment"]
        how = pm_text(user_id, pay[0]["payment_method_id"]) if pay else "the original payment method"
        names = "; ".join(item_text(it["item_id"]) for it in o["items"])
        return (f"Before I cancel anything, please confirm the details:\n\n- Order: {a['order_id']} (the whole order will be cancelled, I cannot cancel single items)\n"
                f"- Items in it: {names}\n- Reason: \"{a['reason']}\"\n- Refund: {money(total)} to {how}\n\nDo you want me to go ahead and cancel this order? (yes/no)")
    if name == "return_delivered_order_items":
        its = "\n".join(f"  - {item_text(i)}" for i in a["item_ids"])
        total = sum(order_items(a["order_id"], a["item_ids"]))
        return (f"Before I start the return, please confirm the details:\n\n- Order: {a['order_id']}\n- Items to return:\n{its}\n"
                f"- Refund: {money(total)} to {pm_text(user_id, a['payment_method_id'])}\n- The order will change to \"return requested\" and you will get an email with return instructions.\n\n"
                "Do you want me to go ahead with this return? (yes/no)")
    if name in ("exchange_delivered_order_items", "modify_pending_order_items"):
        verb = "exchange" if name.startswith("exchange") else "modify"
        pairs = "\n".join(f"  - {item_text(o)}  ->  {item_text(n)}" for o, n in zip(a["item_ids"], a["new_item_ids"]))
        diff = sum(item_info(n)[2] for n in a["new_item_ids"]) - sum(order_items(a["order_id"], a["item_ids"]))
        pm = pm_text(user_id, a["payment_method_id"])
        money_line = (f"you pay the difference of {money(diff)} with {pm}" if diff > 0 else
                      f"the difference of {money(-diff)} is refunded to {pm}" if diff < 0 else "there is no price difference")
        return (f"Before I {verb} the items, please confirm the details:\n\n- Order: {a['order_id']}\n- Changes:\n{pairs}\n- Price: {money_line}\n"
                f"- This can only be done once per order, so please make sure this is the complete list of items you want to change.\n\n"
                f"Do you want me to go ahead with this {verb}? (yes/no)")
    if name == "modify_pending_order_address":
        return (f"Before I change the address, please confirm: order {a['order_id']} will be shipped to\n\n{a['address1']}, {a['address2']}\n{a['city']}, {a['state']} {a['zip']}, {a['country']}\n\n"
                "Do you want me to go ahead? (yes/no)")
    if name == "modify_user_address":
        return (f"Before I update your profile, please confirm: your default address will become\n\n{a['address1']}, {a['address2']}\n{a['city']}, {a['state']} {a['zip']}, {a['country']}\n\n"
                "Do you want me to go ahead? (yes/no)")
    if name == "modify_pending_order_payment":
        return (f"Before I change the payment, please confirm: order {a['order_id']} will be paid with {pm_text(user_id, a['payment_method_id'])}.\n\n"
                "Do you want me to go ahead? (yes/no)")
    raise ValueError(name)


def confirm_reasoning(name, a):
    what = {"cancel_pending_order": "cancel the whole order", "return_delivered_order_items": "return these items",
            "exchange_delivered_order_items": "exchange these items", "modify_pending_order_items": "change these items",
            "modify_pending_order_address": "change the shipping address", "modify_user_address": "change the default address",
            "modify_pending_order_payment": "change the payment method"}[name]
    return (f"\nI am about to {what}. This changes the database and cannot be undone, and the customer has not yet explicitly agreed to these exact details. "
            "Policy says I must list the action, the items or address and any amounts, and get an explicit \"yes\" before I call the tool. "
            "I will write the summary from what I have just read and ask for confirmation, and only act after the customer says yes.\n")


# ---------------------------------------------------------------- the two passes
def user_id_of(msgs):
    for m in msgs:
        for c in (m.get("tool_calls") or []):
            args = json.loads(c["function"]["arguments"])
            if "user_id" in args:
                return args["user_id"]
    for m in msgs:
        if m["role"] == "tool" and m.get("name", "").startswith("find_user_id") and m["content"] in DB["users"]:
            return m["content"]
    for m in msgs:   # fallback: the order rows carry the owner
        if m["role"] == "tool" and m.get("name") == "get_order_details" and m["content"].startswith("{"):
            try:
                return json.loads(m["content"])["user_id"]
            except Exception:
                pass
    return None


def add_lookups(msgs, tag):
    uid = user_id_of(msgs)
    if uid is None:
        return msgs, 0, "no user id"
    orders = DB["users"][uid]["orders"]
    first_write = next((i for i, m in enumerate(msgs) if any(c["function"]["name"] in WRITE for c in (m.get("tool_calls") or []))), len(msgs))
    out = list(msgs)
    # position: right after the first get_user_details result, or right after the user id lookup (then add get_user_details ourselves)
    pos = next((i for i, m in enumerate(out) if m["role"] == "tool" and m.get("name") == "get_user_details"), None)
    if pos is not None and pos > first_write:   # the teacher changed something before it ever loaded the account: load it earlier
        pos = None
    new = []
    if pos is None:
        pos = next(i for i, m in enumerate(out) if m["role"] == "tool" and m.get("name", "").startswith("find_user_id"))
        cid = "chatcmpl-tool-" + h(tag, "user")[:32]
        new.append(call_msg(cid, "get_user_details", {"user_id": uid},
                            f"\nI have the customer id {uid}. Before doing anything else I should load their account to see their payment methods and the list of all their orders.\n"))
        content = json.dumps({k: v for k, v in DB["users"][uid].items() if k != "user_id"})
        new.append(tool_msg(cid, "get_user_details", content))
    # orders already opened BEFORE the insertion point (the ledger must be true at that point in the conversation)
    opened = {json.loads(c["function"]["arguments"]).get("order_id") for m in out[:pos + 1] for c in (m.get("tool_calls") or []) if c["function"]["name"] == "get_order_details"}
    todo = [o for o in orders if o not in opened]
    if not todo:
        if new:
            out[pos + 1:pos + 1] = new
        return out, len(new) // 2, "all orders already opened"
    seen = list(opened & set(orders))
    for k, oid in enumerate(todo):
        cid = "chatcmpl-tool-" + h(tag, "order", oid)[:32]
        left = [x for x in todo[k + 1:]]
        if k == 0:
            r = (f"\nI have the customer's account. They have {len(orders)} orders: {', '.join(orders)}. I should not rely on whichever order I open first or on what the customer "
                 f"remembers, so I will open every order before deciding anything. That way I know every item, status and address on the account"
                 f"{' (already opened: ' + ', '.join(seen) + ')' if seen else ''}. Opening {oid} first.\n")
        elif left:
            r = f"\nOrders opened so far: {', '.join(seen)}. Still to open: {', '.join([oid] + left)}. Opening {oid} next.\n"
        else:
            r = f"\nOrders opened so far: {', '.join(seen)}. {oid} is the last one. After it I will have seen all {len(orders)} orders.\n"
        new.append(call_msg(cid, "get_order_details", {"order_id": oid}, r))
        new.append(tool_msg(cid, "get_order_details", json.dumps(DB["orders"][oid])))
        seen.append(oid)
    out[pos + 1:pos + 1] = new
    return out, len(todo) + (1 if len(new) // 2 > len(todo) else 0), "ok"


def confirmed(msgs, i):
    """Was the write in assistant message i preceded by an agent confirmation question and a customer yes?"""
    u = next((j for j in range(i - 1, -1, -1) if msgs[j]["role"] == "user"), None)
    if u is None or not AFFIRM.search((msgs[u].get("content") or "").strip()):
        return False
    a = next((j for j in range(u - 1, -1, -1) if msgs[j]["role"] == "assistant" and (msgs[j].get("content") or "").strip()), None)
    return a is not None and bool(CONFIRM_Q.search(msgs[a]["content"]))


def add_confirmations(msgs, tag):
    uid = user_id_of(msgs)
    out, inserted, writes = [], 0, 0
    work = list(msgs)
    for i, m in enumerate(work):
        calls = [c for c in (m.get("tool_calls") or []) if c["function"]["name"] in WRITE]
        if m["role"] == "assistant" and calls:
            writes += 1
            probe = out + [m]
            if not confirmed(probe, len(probe) - 1):
                c = calls[0]
                a = json.loads(c["function"]["arguments"])
                out.append({"role": "assistant", "content": confirm_text(c["function"]["name"], a, uid), "reasoning_content": confirm_reasoning(c["function"]["name"], a)})
                out.append({"role": "user", "content": YES[int(h(tag, i)[:4], 16) % len(YES)]})
                inserted += 1
        out.append(m)
    return out, inserted, writes


def main():
    data = json.load(open(SRC))
    res, stats = [], {"conversations_in": len(data), "lookups_inserted": 0, "user_details_inserted": 0, "writes": 0, "writes_already_confirmed": 0,
                      "confirmations_inserted": 0, "skipped": []}
    for n, conv in enumerate(data):
        msgs = copy.deepcopy(conv["messages"])
        tag = f"conv{n}"
        auth = next((i for i, m in enumerate(msgs) if m["role"] == "tool" and (m.get("name") == "get_user_details" or m.get("name", "").startswith("find_user_id"))), None)
        first_w = next((i for i, m in enumerate(msgs) if any(c["function"]["name"] in WRITE for c in (m.get("tool_calls") or []))), len(msgs))
        if auth is None or first_w < auth:
            stats.setdefault("dropped_changed_before_authenticating", []).append(n)   # the teacher never authenticated the customer: a policy violation
            continue
        before = len(msgs)
        msgs, n_look, why = add_lookups(msgs, tag)
        count = lambda ms, name: sum(1 for m in ms for c in (m.get("tool_calls") or []) if c["function"]["name"] == name)
        stats["lookups_inserted"] += count(msgs, "get_order_details") - count(conv["messages"], "get_order_details")
        stats["user_details_inserted"] += count(msgs, "get_user_details") - count(conv["messages"], "get_user_details")
        if why == "no user id":
            stats["skipped"].append(n)
        msgs, n_conf, writes = add_confirmations(msgs, tag)
        stats["writes"] += writes
        stats["confirmations_inserted"] += n_conf
        stats["writes_already_confirmed"] += writes - n_conf
        res.append({"messages": msgs, "tools": conv["tools"]})
    stats["conversations_out"] = len(res)
    stats["avg_messages_before"] = sum(len(c["messages"]) for c in data) / len(data)
    stats["avg_messages_after"] = sum(len(c["messages"]) for c in res) / len(res)
    json.dump(res, open(OUT, "w"))
    json.dump(stats, open(OUT.replace(".json", "_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
