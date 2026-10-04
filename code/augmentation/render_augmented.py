"""Renders the augmented SFT conversations next to the originals; inserted messages are highlighted.
   python3 render_augmented.py  ->  sft_aug/renders/augmented_sft_dialogs.html
"""
import html, json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "renders", "augmented_sft_dialogs.html")
orig = json.load(open(os.path.join(HERE, "teacher32b_passed_sft_reasoning.json")))
aug = json.load(open(os.path.join(HERE, "teacher32b_passed_sft_lookups_confirm.json")))
stats = json.load(open(os.path.join(HERE, "teacher32b_passed_sft_lookups_confirm_stats.json")))
meta = json.load(open(os.path.join(HERE, "teacher32b_passed_meta.json")))
dropped = set(stats.get("dropped_changed_before_authenticating", []))
kept = [i for i in range(len(orig)) if i not in dropped]
assert len(kept) == len(aug)
esc = lambda s: html.escape(str(s if s is not None else ""))
WRITE = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items",
         "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"}


def key(m):
    return (m["role"], m.get("content") or "", json.dumps(m.get("tool_calls"), sort_keys=True), m.get("tool_call_id") or "")


def pretty(text):
    try:
        v = json.loads(text)
    except Exception:
        return text
    return json.dumps(v, indent=2, ensure_ascii=False) if isinstance(v, (dict, list)) else text


def clip(text, n=900):
    text = pretty(text)
    if len(text) <= n:
        return f"<div class='tool-out'>{esc(text)}</div>"
    return f"<details class='tool-out'><summary>{esc(text[:n])} <i>... (+{len(text) - n} chars)</i></summary>{esc(text)}</details>"


def render(msgs, inserted):
    out = []
    for i, m in enumerate(msgs):
        role, content = m["role"], (m.get("content") or "").strip()
        ins = " ins" if i in inserted else ""
        tag = "<span class='tag'>inserted</span>" if i in inserted else ""
        if role == "system":
            continue
        if role == "tool":
            out.append(f"<div class='msg tool{ins}'><div class='who'>tool result ({esc(m.get('name'))}) {tag}</div>{clip(content)}</div>")
            continue
        think = (m.get("reasoning_content") or "").strip()
        think_html = f"<details class='think'><summary>reasoning</summary>{esc(think)}</details>" if think else ""
        if content:
            who = "customer" if role == "user" else "agent"
            out.append(f"<div class='msg {role}{ins}'><div class='who'>{who} {tag}</div>{esc(content)}{think_html}</div>")
            think_html = ""
        for c in (m.get("tool_calls") or []):
            fn = c["function"]
            cls = "write" if fn["name"] in WRITE else "call"
            out.append(f"<div class='msg call {cls}{ins}'><div class='who'>{'database change' if cls == 'write' else 'tool call'} {tag}</div><code>{esc(fn['name'])}</code>"
                       f"<div class='args'>{esc(fn['arguments'])}</div>{think_html}</div>")
            think_html = ""
    return "".join(out) or "<div class='msg'>(empty)</div>"


cards, n_look_total, n_conf_total = [], 0, 0
for n, (oi, conv) in enumerate(zip(kept, aug)):
    o, a = orig[oi]["messages"], conv["messages"]
    pool = Counter(key(m) for m in o)
    ins = set()
    for j, m in enumerate(a):
        k = key(m)
        if pool[k] > 0:
            pool[k] -= 1
        else:
            ins.add(j)
    look = sum(1 for j in ins if a[j]["role"] == "assistant" and any(c["function"]["name"] in ("get_order_details", "get_user_details") for c in (a[j].get("tool_calls") or [])))
    conf = sum(1 for j in ins if a[j]["role"] == "assistant" and (a[j].get("content") or "").startswith("Before I"))
    n_look_total += look
    n_conf_total += conf
    first_user = next((m["content"] for m in a if m["role"] == "user"), "")
    kinds = Counter(c["function"]["name"] for m in a for c in (m.get("tool_calls") or []) if c["function"]["name"] in WRITE)
    cards.append(f"""
<section class="conv" data-id="{oi}" data-look="{1 if look else 0}" data-conf="{1 if conf else 0}" id="c{oi}">
  <header><h2>Conversation {oi}</h2><span class="pill">+{look} lookups</span><span class="pill">+{conf} confirmations</span>
    <span class="meta">{len(o) - 1} messages &rarr; {len(a) - 1} &middot; changes: {esc(', '.join(f'{k.split("_")[0]} x{v}' for k, v in kinds.items()) or 'none')}</span></header>
  <div class="cols">
    <div class="col"><h3>original</h3><div class="dialog">{render(o, set())}</div></div>
    <div class="col"><h3>augmented</h3><div class="dialog">{render(a, ins)}</div></div>
  </div>
</section>""")

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Augmented SFT conversations</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --mut:#667085; --line:#dfe3ea; --user:#eef2f8; --agent:#e8f4ee; --call:#fdf3e3; --tool:#f1f1f4; --ins:#fff3b0; --ok:#1e7a5c; --bad:#b3261e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171d; --card:#1b1f27; --ink:#e6e9ef; --mut:#9aa3b2; --line:#2c323d; --user:#222a38; --agent:#1c2e27; --call:#33291a; --tool:#252932; --ins:#4a4310; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; }}
#bar {{ position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
#bar h1 {{ font-size:16px; margin:0 10px 0 0; }}
button, input {{ font:inherit; padding:5px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); cursor:pointer; }}
button.on {{ background:var(--ink); color:var(--card); }}
main {{ padding:16px; max-width:1900px; margin:0 auto; display:flex; flex-direction:column; gap:22px; }}
.conv {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.conv header {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }} .conv h2 {{ font-size:15px; margin:0; }}
.pill {{ font-size:12px; padding:2px 8px; border-radius:20px; background:var(--ins); }} .meta {{ color:var(--mut); font-size:12px; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px; }} .col h3 {{ font-size:13px; margin:4px 0 6px; }}
.dialog {{ height:68vh; overflow:auto; border:1px solid var(--line); border-radius:8px; padding:8px; display:flex; flex-direction:column; gap:6px; }}
.msg {{ border-radius:8px; padding:6px 9px; white-space:pre-wrap; word-break:break-word; }}
.msg .who {{ font-size:11px; color:var(--mut); margin-bottom:2px; text-transform:uppercase; letter-spacing:.04em; }}
.msg.user {{ background:var(--user); margin-right:14%; }} .msg.assistant {{ background:var(--agent); margin-left:14%; }}
.msg.call {{ background:var(--call); margin-left:14%; font-size:12px; }} .msg.write {{ border-left:4px solid var(--bad); }} .msg.tool {{ background:var(--tool); margin-left:14%; font-size:12px; }}
.msg.ins {{ outline:2px solid #d9b800; box-shadow:0 0 0 3px var(--ins); }} .tag {{ background:#d9b800; color:#222; border-radius:4px; padding:0 5px; margin-left:6px; text-transform:none; letter-spacing:0; }}
.think {{ margin-top:4px; color:var(--mut); }} .think summary {{ cursor:pointer; font-size:12px; }}
code, .args {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; }} .args {{ color:var(--mut); display:block; word-break:break-all; }}
.tool-out {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; white-space:pre-wrap; }} details.tool-out summary {{ cursor:pointer; }}
.hidden {{ display:none; }}
@media (max-width:900px) {{ .cols {{ grid-template-columns:1fr; }} }}
</style></head><body>
<div id="bar">
  <h1>Augmented SFT data: {len(aug)} conversations</h1>
  <button data-f="all" class="on">all ({len(aug)})</button>
  <button data-f="conf">with inserted confirmations</button>
  <button data-f="look">with inserted lookups</button>
  <input id="q" placeholder="conversation number" size="18">
  <span class="meta" id="shown"></span>
  <span class="meta">{stats['lookups_inserted']} order lookups and {stats['confirmations_inserted']} confirmations inserted in total; yellow outline = inserted</span>
</div>
<main>{''.join(cards)}</main>
<script>
const convs = [...document.querySelectorAll('.conv')]; let filt = 'all';
function apply() {{
  const q = document.getElementById('q').value.trim(); let n = 0;
  convs.forEach(c => {{ const show = (filt === 'all' || c.dataset[filt] === '1') && (!q || c.dataset.id === q); c.classList.toggle('hidden', !show); if (show) n++; }});
  document.getElementById('shown').textContent = n + ' shown';
}}
document.querySelectorAll('#bar button').forEach(b => b.onclick = () => {{ filt = b.dataset.f; document.querySelectorAll('#bar button').forEach(x => x.classList.toggle('on', x === b)); apply(); }});
document.getElementById('q').oninput = apply; apply();
</script></body></html>"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(page)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB); inserted lookups {n_look_total} (assistant turns), confirmations {n_conf_total}")
