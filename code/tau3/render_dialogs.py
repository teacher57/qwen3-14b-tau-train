"""Renders all tau3 retail conversations of both models side by side into one self-contained HTML page.
   python3 render_dialogs.py   ->  tau3/renders/tau3_retail_dialogs.html
"""
import html, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "renders", "tau3_retail_dialogs.html")
esc = lambda s: html.escape(str(s if s is not None else ""))


def load(name):
    d = json.load(open(os.path.join(HERE, "data", f"tau3_{name}_results.json")))
    return d, {str(s["task_id"]): s for s in d["simulations"]}


old_d, OLD = load("old")
new_d, NEW = load("new")
TASKS = {str(t["id"]): t for t in old_d["tasks"]}


CAUSES = json.load(open(os.path.join(HERE, "new_model_failure_causes.json")))
OLD_CAUSES = json.load(open(os.path.join(HERE, "old_model_failure_causes.json")))


def passed(s):
    return ((s.get("reward_info") or {}).get("reward") or 0) >= 1


def pretty(text):
    """Tool results that are JSON are dumped with indent=2; anything else is shown unchanged."""
    text = str(text if text is not None else "")
    try:
        val = json.loads(text)
    except Exception:
        return text
    return json.dumps(val, indent=2, ensure_ascii=False) if isinstance(val, (dict, list)) else text


def clip(text, n=900):
    text = pretty(text)
    if len(text) <= n:
        return f"<div class='tool-out'>{esc(text)}</div>"
    return f"<details class='tool-out'><summary>{esc(text[:n])} <i>... (+{len(text) - n} chars)</i></summary>{esc(text)}</details>"


def render_msgs(msgs):
    out = []
    for m in msgs:
        role, content = m.get("role"), (m.get("content") or "")
        calls = m.get("tool_calls") or []
        if role == "tool":
            err = " err" if m.get("error") else ""
            out.append(f"<div class='msg tool{err}'><div class='who'>tool result{' (error)' if err else ''}</div>{clip(content)}</div>")
            continue
        text = content.strip()
        if text:
            out.append(f"<div class='msg {role}'><div class='who'>{'customer (GPT-4.1)' if role == 'user' else 'agent'}</div>{esc(text)}</div>")
        for c in calls:
            args = json.dumps(c.get("arguments"), ensure_ascii=False)
            out.append(f"<div class='msg call'><div class='who'>tool call</div><code>{esc(c.get('name'))}</code>"
                       f"<div class='args'>{esc(args)}</div></div>")
    return "".join(out) or "<div class='msg'>(no messages)</div>"


def badge(s):
    ri = s.get("reward_info") or {}
    ok = passed(s)
    bd = ri.get("reward_breakdown") or {}
    parts = " &middot; ".join(f"{esc(k)} {v:g}" for k, v in bd.items())
    return (f"<span class='pill {'ok' if ok else 'bad'}'>{'PASS' if ok else 'FAIL'}</span> "
            f"<span class='meta'>{esc(s.get('termination_reason'))} &middot; {s.get('duration', 0) / 60:.1f} min &middot; "
            f"{len(s.get('messages') or [])} msgs &middot; {parts}</span>")


def order_key(t):
    return int(t) if t.isdigit() else t


cards, counts = [], {"both": 0, "new": 0, "old": 0, "neither": 0}
for tid in sorted(OLD, key=order_key):
    o, n = OLD[tid], NEW.get(tid)
    if n is None:
        continue
    po, pn = passed(o), passed(n)
    cat = "both" if po and pn else "new" if pn else "old" if po else "neither"
    counts[cat] += 1
    t = TASKS.get(tid, {})
    ins = ((t.get("user_scenario") or {}).get("instructions")) or {}
    ins_html = "".join(f"<p><b>{esc(k.replace('_', ' '))}:</b> {esc(v)}</p>" for k, v in ins.items() if v)
    acts = ((t.get("evaluation_criteria") or {}).get("actions")) or []
    acts_html = "".join(f"<li><code>{esc(a.get('name'))}</code> <span class='args'>{esc(json.dumps(a.get('arguments'), ensure_ascii=False))}</span></li>" for a in acts)
    label = {"both": "both pass", "new": "only new passes", "old": "only old passes", "neither": "neither passes"}[cat]
    cause = CAUSES["tasks"].get(tid)
    ocause = OLD_CAUSES["tasks"].get(tid)
    cause_html = ""
    why = ""
    if ocause:
        cause_html += f"<span class='cause old'>old: cause {ocause[0]}: {esc(CAUSES['clusters'][str(ocause[0])])}</span>"
        why += f"<p class='why'><b>Why the old model failed:</b> {esc(ocause[1])}</p>"
    if cause:
        cause_html += f"<span class='cause'>new: cause {cause[0]}: {esc(CAUSES['clusters'][str(cause[0])])}</span>"
        why += f"<p class='why'><b>Why the new model failed:</b> {esc(cause[1])}</p>"
    ins_html = why + ins_html
    cards.append(f"""
<section class="task" data-cat="{cat}" data-cause="{cause[0] if cause else 0}" data-ocause="{ocause[0] if ocause else 0}" data-id="{esc(tid)}" id="t{esc(tid)}">
  <header><h2>Task {esc(tid)}</h2><span class="cat {cat}">{label}</span>{cause_html}</header>
  <details class="brief"{' open' if (cause or ocause) else ''}><summary>What the customer wants and what the agent must do</summary>{ins_html}
    <p><b>Target actions the agent has to perform ({len(acts)}):</b></p><ul>{acts_html or '<li>none (information only)</li>'}</ul></details>
  <div class="cols">
    <div class="col"><h3>old (starting adapter) {badge(o)}</h3><div class="dialog">{render_msgs(o.get('messages') or [])}</div></div>
    <div class="col"><h3>new (fine-tuned) {badge(n)}</h3><div class="dialog">{render_msgs(n.get('messages') or [])}</div></div>
  </div>
</section>""")

n_all = sum(counts.values())
from collections import Counter
cause_counts = Counter(v[0] for v in CAUSES["tasks"].values())
ocause_counts = Counter(v[0] for v in OLD_CAUSES["tasks"].values())
ocause_options = "".join(f"<option value='{k}'>{k}. {esc(CAUSES['clusters'][k])} ({ocause_counts.get(int(k), 0)})</option>" for k in sorted(CAUSES["clusters"], key=int))
cause_options = "".join(f"<option value='{k}'>{k}. {esc(CAUSES['clusters'][k])} ({cause_counts.get(int(k), 0)})</option>" for k in sorted(CAUSES["clusters"], key=int))
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>τ³ retail dialogs, old vs new</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --mut:#667085; --line:#dfe3ea; --user:#eef2f8; --agent:#e8f4ee; --call:#fdf3e3; --tool:#f1f1f4;
        --ok:#1e7a5c; --bad:#b3261e; --old:#2f5d8a; --new:#1e7a5c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171d; --card:#1b1f27; --ink:#e6e9ef; --mut:#9aa3b2; --line:#2c323d; --user:#222a38; --agent:#1c2e27; --call:#33291a; --tool:#252932; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; }}
#bar {{ position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
#bar h1 {{ font-size:16px; margin:0 10px 0 0; }}
button, input, select {{ font:inherit; padding:5px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); cursor:pointer; }}
button.on {{ background:var(--ink); color:var(--card); }}
main {{ padding:16px; max-width:1900px; margin:0 auto; display:flex; flex-direction:column; gap:22px; }}
.task {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.task header {{ display:flex; align-items:center; gap:12px; }}
.task h2 {{ font-size:15px; margin:0; }}
.cat {{ font-size:12px; padding:2px 8px; border-radius:20px; border:1px solid var(--line); color:var(--mut); }}
.cat.new {{ color:var(--new); border-color:var(--new); }} .cat.old {{ color:var(--old); border-color:var(--old); }}
.brief {{ margin:8px 0; color:var(--mut); }} .brief summary {{ cursor:pointer; }} .brief p, .brief li {{ margin:4px 0; color:var(--ink); }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.col h3 {{ font-size:13px; margin:4px 0 6px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
.dialog {{ height:68vh; overflow:auto; border:1px solid var(--line); border-radius:8px; padding:8px; display:flex; flex-direction:column; gap:6px; }}
.msg {{ border-radius:8px; padding:6px 9px; white-space:pre-wrap; word-break:break-word; }}
.msg .who {{ font-size:11px; color:var(--mut); margin-bottom:2px; text-transform:uppercase; letter-spacing:.04em; }}
.msg.user {{ background:var(--user); margin-right:14%; }} .msg.assistant {{ background:var(--agent); margin-left:14%; }}
.msg.call {{ background:var(--call); margin-left:14%; font-size:12px; }} .msg.tool {{ background:var(--tool); margin-left:14%; font-size:12px; }}
.msg.tool.err {{ border:1px solid var(--bad); }}
code, .args {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; }} .args {{ color:var(--mut); display:block; word-break:break-all; }}
.tool-out {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; white-space:pre-wrap; }} details.tool-out summary {{ cursor:pointer; }}
.pill {{ font-size:11px; font-weight:700; padding:1px 8px; border-radius:20px; color:#fff; }} .pill.ok {{ background:var(--ok); }} .pill.bad {{ background:var(--bad); }}
.meta {{ font-weight:400; color:var(--mut); font-size:12px; }}
.cause {{ font-size:12px; padding:2px 8px; border-radius:6px; background:var(--call); }} .cause.old {{ background:var(--tool); }} .why {{ background:var(--call); padding:6px 9px; border-radius:6px; }}
.hidden {{ display:none; }}
@media (max-width:900px) {{ .cols {{ grid-template-columns:1fr; }} }}
</style></head><body>
<div id="bar">
  <h1>τ³ retail: old vs new, {n_all} tasks</h1>
  <button data-f="all" class="on">all ({n_all})</button>
  <button data-f="both">both pass ({counts['both']})</button>
  <button data-f="new">only new passes ({counts['new']})</button>
  <button data-f="old">only old passes ({counts['old']})</button>
  <button data-f="neither">neither ({counts['neither']})</button>
  <select id="cs"><option value="">new model: any cause</option>{cause_options}</select>
  <select id="os"><option value="">old model: any cause</option>{ocause_options}</select>
  <input id="q" placeholder="task id, e.g. 42" size="12">
  <span class="meta" id="shown"></span>
</div>
<main>{''.join(cards)}</main>
<script>
const tasks = [...document.querySelectorAll('.task')];
let filt = 'all';
function apply() {{
  const q = document.getElementById('q').value.trim();
  let n = 0;
  tasks.forEach(t => {{
    const cz = document.getElementById('cs').value;
    const oz = document.getElementById('os').value;
    const show = (filt === 'all' || t.dataset.cat === filt) && (!q || t.dataset.id === q) && (!cz || t.dataset.cause === cz) && (!oz || t.dataset.ocause === oz);
    t.classList.toggle('hidden', !show);
    if (show) n++;
  }});
  document.getElementById('shown').textContent = n + ' shown';
}}
document.querySelectorAll('#bar button').forEach(b => b.onclick = () => {{
  filt = b.dataset.f;
  document.querySelectorAll('#bar button').forEach(x => x.classList.toggle('on', x === b));
  apply();
}});
document.getElementById('q').oninput = apply;
document.getElementById('cs').onchange = apply;
document.getElementById('os').onchange = apply;
apply();
</script></body></html>"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(page)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB), tasks: {n_all}, {counts}")
