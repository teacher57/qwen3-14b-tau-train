"""Renders habit_dataset.json: supervised turns highlighted, context dimmed.   python3 render_habits.py -> renders/habit_dataset.html"""
import html, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "renders", "habit_dataset.html")
data = json.load(open(os.path.join(HERE, "habit_dataset.json")))
stats = json.load(open(os.path.join(HERE, "habit_dataset_stats.json")))
esc = lambda s: html.escape(str(s if s is not None else ""))
WRITE = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"}
KIND = {"real": "real conversation", "synthetic_lookup": "synthetic lookup", "synthetic_confirm": "synthetic confirmation"}

def tool_summary(name, content):
    try:
        v = json.loads(content)
    except Exception:
        return esc(content[:200])
    if name == "get_order_details" and isinstance(v, dict):
        items = ", ".join(f"{i['name']} (${i['price']})" for i in v.get("items", []))
        return f"<b>order {esc(v.get('order_id'))}</b> &middot; {esc(v.get('status'))} &middot; {len(v.get('items', []))} item(s): {esc(items)}"
    if name == "get_user_details" and isinstance(v, dict):
        pm = ", ".join(f"{k} ({p.get('source')})" for k, p in (v.get("payment_methods") or {}).items())
        return f"<b>account</b> {esc(v['name']['first_name'])} {esc(v['name']['last_name'])} &middot; {len(v.get('orders', []))} order(s): {esc(', '.join(v.get('orders', [])))} &middot; payment: {esc(pm)}"
    return esc(json.dumps(v)[:300])

def render(conv):
    sup = set(conv["supervise"]); out = []
    for i, m in enumerate(conv["messages"]):
        if m["role"] == "system": continue
        s = " sup" if i in sup else ""
        tag = "<span class='tag'>trained</span>" if i in sup else ""
        if m["role"] == "tool":
            out.append(f"<div class='msg tool{s}'><div class='who'>tool result ({esc(m.get('name'))})</div>{tool_summary(m.get('name'), m.get('content') or '')}</div>"); continue
        think = (m.get("reasoning_content") or "").strip()
        th = f"<details class='think'{' open' if s else ''}><summary>reasoning</summary>{esc(think)}</details>" if think else ""
        c = (m.get("content") or "").strip()
        if c:
            out.append(f"<div class='msg {m['role']}{s}'><div class='who'>{'customer' if m['role'] == 'user' else 'agent'} {tag}</div>{esc(c)}{th}</div>"); th = ""
        for t in (m.get("tool_calls") or []):
            f = t["function"]; kind = "write" if f["name"] in WRITE else "call"
            out.append(f"<div class='msg call {kind}{s}'><div class='who'>{'database change' if kind == 'write' else 'tool call'} {tag}</div><code>{esc(f['name'])}</code> <span class='args'>{esc(f['arguments'])}</span>{th}</div>"); th = ""
    return "".join(out)

cards = []
for n, d in enumerate(data):
    kinds = {"lookup": sum(1 for i in d["supervise"] if d["messages"][i].get("tool_calls") and d["messages"][i]["tool_calls"][0]["function"]["name"] in ("get_user_details", "get_order_details")),
             "confirmation": sum(1 for i in d["supervise"] if (d["messages"][i].get("content") or "").strip() and not d["messages"][i].get("tool_calls")),
             "change": sum(1 for i in d["supervise"] if any(t["function"]["name"] in WRITE for t in (d["messages"][i].get("tool_calls") or [])))}
    cards.append(f"<section class='conv' data-id='{n}' data-kind='{d['kind']}'><header><h2>#{n}</h2><span class='pill'>{KIND[d['kind']]}</span>"
                 f"<span class='meta'>{len(d['supervise'])} trained turns: {kinds['lookup']} lookup, {kinds['confirmation']} confirmation, {kinds['change']} change &middot; {len(d['messages']) - 1} messages</span></header>"
                 f"<div class='dialog'>{render(d)}</div></section>")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Habit-only SFT data</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --mut:#667085; --line:#dfe3ea; --user:#eef2f8; --agent:#e8f4ee; --call:#fdf3e3; --tool:#f1f1f4; --sup:#fff3b0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171d; --card:#1b1f27; --ink:#e6e9ef; --mut:#9aa3b2; --line:#2c323d; --user:#222a38; --agent:#1c2e27; --call:#33291a; --tool:#252932; --sup:#4a4310; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; }}
#bar {{ position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
#bar h1 {{ font-size:16px; margin:0 10px 0 0; }} button, input {{ font:inherit; padding:5px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); cursor:pointer; }}
button.on {{ background:var(--ink); color:var(--card); }}
main {{ padding:16px; max-width:1100px; margin:0 auto; display:flex; flex-direction:column; gap:18px; }}
.conv {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.conv header {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }} .conv h2 {{ font-size:15px; margin:0; }}
.pill {{ font-size:12px; padding:2px 8px; border-radius:20px; background:var(--sup); }} .meta {{ color:var(--mut); font-size:12px; }}
.dialog {{ margin-top:8px; display:flex; flex-direction:column; gap:6px; max-height:75vh; overflow:auto; }}
.msg {{ border-radius:8px; padding:6px 9px; white-space:pre-wrap; word-break:break-word; opacity:.55; }} .msg.sup {{ opacity:1; outline:2px solid #d9b800; box-shadow:0 0 0 3px var(--sup); }}
.msg .who {{ font-size:11px; color:var(--mut); margin-bottom:2px; text-transform:uppercase; letter-spacing:.04em; }}
.msg.user {{ background:var(--user); margin-right:14%; }} .msg.assistant {{ background:var(--agent); margin-left:14%; }} .msg.call {{ background:var(--call); margin-left:14%; font-size:12px; }} .msg.write {{ border-left:4px solid #b3261e; }} .msg.tool {{ background:var(--tool); margin-left:14%; font-size:12px; }}
.tag {{ background:#d9b800; color:#222; border-radius:4px; padding:0 5px; margin-left:6px; text-transform:none; letter-spacing:0; }}
.think {{ margin-top:4px; color:var(--mut); }} .think summary {{ cursor:pointer; font-size:12px; }} code, .args {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; }} .args {{ color:var(--mut); word-break:break-all; }}
.hidden {{ display:none; }} body.only .msg:not(.sup) {{ display:none; }}
</style></head><body>
<div id="bar"><h1>Habit-only SFT data: {len(data)} conversations, {stats['supervised_turns']} trained turns</h1>
<button data-k="all" class="on">all</button><button data-k="real">real ({stats['real']})</button><button data-k="synthetic_lookup">synthetic lookups ({stats['synthetic_lookup']})</button><button data-k="synthetic_confirm">synthetic confirmations ({stats['synthetic_confirm']})</button>
<button id="only">show only trained turns</button><input id="q" placeholder="conversation #" size="14"><span class="meta" id="shown"></span>
<span class="meta">yellow outline = trained (loss), dimmed = context only</span></div>
<main>{''.join(cards)}</main>
<script>
const cs=[...document.querySelectorAll('.conv')]; let k='all';
function ap(){{const q=document.getElementById('q').value.trim();let n=0;cs.forEach(c=>{{const s=(k==='all'||c.dataset.kind===k)&&(!q||c.dataset.id===q);c.classList.toggle('hidden',!s);if(s)n++;}});document.getElementById('shown').textContent=n+' shown';}}
document.querySelectorAll('#bar button[data-k]').forEach(b=>b.onclick=()=>{{k=b.dataset.k;document.querySelectorAll('#bar button[data-k]').forEach(x=>x.classList.toggle('on',x===b));ap();}});
document.getElementById('only').onclick=e=>{{document.body.classList.toggle('only');e.target.classList.toggle('on');}};
document.getElementById('q').oninput=ap; ap();
</script></body></html>"""
os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write(page)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
