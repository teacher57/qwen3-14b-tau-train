import sys, nbformat as nbf
OUT = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Fine-tuning Qwen3-14B on lookup-and-confirm data

**Experiment.** Train a fresh LoRA on the 14B base model with the 180 teacher conversations **augmented** so that the agent (1) opens **every** order of the customer before any change and (2) asks for an explicit **yes** before every database change. Same recipe as the earlier "new" model (r=16, lr 5e-5, batch 8, 1 epoch), so the only difference is the data. Then run the τ-bench retail test (GPT-4o user, 115 tasks, **4 trials per task**, temperature 0.7, the same setup as the earlier runs) and compare with the starting adapter (41.7% pass^1) and the distilled model (45.7%) measured the same way.

This notebook only **reads** `full_run_results/aug_sft_run.json`, rewritten every 30 s by `sft_aug/update_aug_json.py` (start it with `sft_aug/aug_json_updater.sh start`). Re-run a cell to refresh.

---
# Part 1: process info
""")
code("""
import json, datetime
from pathlib import Path
from IPython.display import HTML, display
ROOT = Path("..") if Path("../full_run_results").exists() else Path(".")
PATH = ROOT / "full_run_results" / "aug_sft_run.json"
load = lambda: json.load(open(PATH))
def fmt_min(m):
    if m is None: return "-"
    m = int(round(m)); return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"
def table(rows, head=None):
    td = "style='padding:3px 12px;border-bottom:1px solid #ddd;text-align:left'"
    h = "".join(f"<th {td}>{c}</th>" for c in head) if head else ""
    b = "".join("<tr>" + "".join(f"<td {td}>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table style='border-collapse:collapse;margin:6px 0 14px'>{('<tr>' + h + '</tr>') if head else ''}{b}</table>"
def info_html(r):
    p, t, c = r.get("pod", {}), r["training"], r["config"]
    o = [f"<h4>JSON written {r['updated']} &middot; stage: <span style='color:#1E7A5C'>{r['stage']}</span></h4>"]
    if not r.get("pod_reachable", True): o.append("<p style='color:#B3261E'><b>Pod unreachable</b>, showing the last known state.</p>")
    o.append("<b>Machine</b>" + table([["address", p.get("host")], ["GPU", p.get("gpu_name")], ["GPU load / memory / temperature", f"{p.get('gpu_util_pct')} / {p.get('gpu_mem_used')} of {p.get('gpu_mem_total')} / {p.get('gpu_temp_c')} C"],
        ["disk (size, used, free)", p.get("disk")]]))
    o.append("<b>Experiment</b>" + table([[k.replace("_", " "), v] for k, v in c.items()]))
    pct = f"{t['steps_done'] / t['steps_total']:.0%}" if t["steps_total"] else "-"
    ref = 193.3 / 384
    o.append("<b>Progress</b>" + table([["steps", f"{t['steps_done']} / {t['steps_total']} ({pct})"], ["elapsed", fmt_min(t["elapsed_min"])],
        ["seconds per step (reference run: 29.4)", f"{t['sec_per_step']:.1f}" if t["sec_per_step"] else "-"], ["time left", fmt_min(t["eta_min"])], ["training done at about (local)", t.get("eta_at") or "-"],
        ["checkpoints saved", ", ".join(t["checkpoints"]) or "-"]]))
    pl = r.get("plan")
    if pl:
        o.append("<b>Remaining time (training and evaluation)</b>" + table([[x["step"], fmt_min(x["minutes"]) if x["minutes"] else "done", x["basis"], x["done_at"] or "-"] for x in pl["steps"]], ["phase", "time left", "basis", "done at (local)"])
                 + f"<p>everything done, results on the Mac, at about <b>{pl['done_at']}</b> ({fmt_min(pl['total_min'])} from now)</p>")
    ev_ = r.get("eval", {})
    if ev_.get("rollouts_done"):
        o.append("<b>Evaluation progress</b>" + table([["rollouts", f"{ev_['rollouts_done']} / {ev_['rollouts_total']}"], ["pass^1 so far", f"{ev_['pass1']:.1%}"],
            ["rollouts per minute", f"{ev_['rollouts_per_min']:.2f}" if ev_.get("rollouts_per_min") else "-"], ["done at about", ev_.get("eta_at") or "-"]]))
    o.append("<b>Last lines of the training log</b><pre style='font-size:12px'>" + "\\n".join(l.replace("<", "&lt;") for l in (p.get("train_log_tail") or p.get("env_tail") or [])) + "</pre>")
    o.append("<b>Recent events</b>" + table([[e["time"], e["event"].replace("<", "&lt;")] for e in r.get("events", [])[-10:]]))
    return "".join(o)
RUN = load(); display(HTML(info_html(RUN)))
""")
md("### Live view (optional): refreshes the block above every 30 s until you stop the cell")
code("""
import time
from IPython.display import clear_output
try:
    while True:
        RUN = load(); clear_output(wait=True); display(HTML(info_html(RUN)))
        if str(RUN["stage"]).startswith(("training finished", "failed")): break
        time.sleep(30)
except KeyboardInterrupt:
    pass
""")
cells[-1].metadata["tags"] = ["skip-execution"]

md("---\n# Part 2: graphs (plotly)")
code("""
import numpy as np, plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook_connected"; pio.templates.default = "plotly_white"
RUN = load(); T = RUN["training"]
C_AUG, C_REF = "#1E7A5C", "#2F5D8A"
REF = json.load(open(ROOT / "full_run_results" / "distill_run.json"))["training"]   # the earlier 'new' model's SFT
def smooth(x, k=20):
    x = np.asarray(x, float); return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else x
def placeholder(title, text="no data yet"):
    f = go.Figure(); f.add_annotation(text=text, showarrow=False, font=dict(size=16, color="#888")); f.update_layout(title=title, height=260, xaxis=dict(visible=False), yaxis=dict(visible=False)); return f
print(f"stage: {RUN['stage']} | steps {T['steps_done']}/{T['steps_total']}")
""")
md("### 1. Loss: this run vs the earlier run (different data, so compare the shape, not the level)")
code("""
if T["step"]:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss, raw and 20-step average", "Gradient norm and learning rate"), specs=[[{}, {"secondary_y": True}]])
    fig.add_scatter(x=T["step"], y=T["loss"], mode="lines", line=dict(color="#C8D8D0", width=1), name="this run (raw)", row=1, col=1)
    s = smooth(T["loss"]); fig.add_scatter(x=T["step"][len(T["step"]) - len(s):], y=s, mode="lines", line=dict(color=C_AUG, width=3), name="this run (avg)", row=1, col=1)
    rs = smooth(REF["loss"]); fig.add_scatter(x=REF["step_index"][len(REF["step_index"]) - len(rs):], y=rs, mode="lines", line=dict(color=C_REF, width=2, dash="dot"), name="earlier 'new' run (avg)", row=1, col=1)
    fig.add_scatter(x=T["step"], y=T["grad_norm"], mode="lines", line=dict(color="#E0883B"), name="grad norm", row=1, col=2)
    fig.add_scatter(x=T["step"], y=T["lr"], mode="lines", line=dict(color="#7A4E9C"), name="learning rate", row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="optimizer step"); fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2)); fig.show()
else:
    placeholder("Loss", "training has not logged a step yet").show()
""")
md("### Loss function (training loss on the assistant turns, per optimizer step)")
code("""
if len(T["step"]) > 1:
    st = np.array(T["step"]); ls = np.array(T["loss"], float)
    fig = go.Figure()
    fig.add_scatter(x=st, y=ls, mode="lines+markers", marker=dict(size=4), line=dict(color="#C8D8D0", width=1), name="loss per step")
    for k, col in ((10, "#E0883B"), (30, C_AUG), (60, "#2F5D8A")):
        if len(ls) >= k:
            sm = smooth(ls, k); fig.add_scatter(x=st[len(st) - len(sm):], y=sm, mode="lines", line=dict(color=col, width=3 if k == 30 else 2), name=f"{k}-step average")
    rs = smooth(REF["loss"], 30); fig.add_scatter(x=REF["step_index"][len(REF["step_index"]) - len(rs):], y=rs, mode="lines", line=dict(color="#7A4E9C", width=2, dash="dot"), name="earlier run, 30-step average (other data)")
    for ck in (96, 192, 288, 384, 480):
        if ck <= st[-1]: fig.add_vline(x=ck, line_dash="dot", line_color="#999", annotation_text=f"step-{ck}", annotation_position="top")
    fig.update_layout(title="Training loss", height=520, xaxis=dict(title="optimizer step", rangeslider=dict(visible=True)), yaxis=dict(title="loss (cross-entropy per completion token)"), hovermode="x unified", legend=dict(orientation="h", y=-0.35))
    fig.show()
    n = min(10, len(ls)); print(f"first {n} steps: {ls[:n].mean():.3f} | latest {n} steps: {ls[-n:].mean():.3f} | lowest single step: {ls.min():.3f} at step {int(st[ls.argmin()])} | steps logged: {len(ls)}")
else:
    placeholder("Loss function", "training has not logged a step yet").show()
""")
md("### 2. Speed and time left")
code("""
if len(T["step"]) > 1:
    el = np.array(T["elapsed"]); st = np.array(T["step"]); dt = np.diff(el, prepend=0) * 60; tok = np.array(T["tokens"])
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Elapsed time vs steps (dotted: projection)", "Seconds per step", "Tokens per step and per second"), specs=[[{}, {}, {"secondary_y": True}]])
    fig.add_scatter(x=st, y=el, mode="lines", line=dict(color=C_AUG, width=3), name="elapsed (min)", row=1, col=1)
    rate = el[-1] / st[-1]; fig.add_scatter(x=[st[-1], T["steps_total"]], y=[el[-1], rate * T["steps_total"]], mode="lines", line=dict(color=C_AUG, dash="dot"), name="projection", row=1, col=1)
    fig.add_scatter(x=[0, T["steps_total"]], y=[0, 193.3 / 384 * T["steps_total"]], mode="lines", line=dict(color=C_REF, dash="dash"), name="earlier run's pace", row=1, col=1)
    fig.add_scatter(x=st, y=dt, mode="markers+lines", line=dict(color="#E0883B", width=1), name="s / step", row=1, col=2)
    fig.add_bar(x=st, y=tok, marker_color="#C8D8D0", name="tokens / step", row=1, col=3)
    fig.add_scatter(x=st, y=tok / np.maximum(dt, 1e-9), mode="lines", line=dict(color="#7A4E9C"), name="tokens / s", row=1, col=3, secondary_y=True)
    fig.update_xaxes(title_text="optimizer step"); fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2)); fig.show()
    print(f"projected total: {rate * T['steps_total'] / 60:.1f} h (earlier run: 3.2 h for 384 steps)")
else:
    placeholder("Speed", "needs at least 2 logged steps").show()
""")
md("### 3. GPU right now")
code("""
P = RUN["pod"]
def num(s): 
    try: return float(str(s).split()[0].replace("%", ""))
    except Exception: return None
used, tot = num(P.get("gpu_mem_used")), num(P.get("gpu_mem_total"))
if used is not None and tot:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("GPU load (%)", "GPU memory (GiB)"))
    fig.add_bar(x=[P.get("gpu_name")], y=[num(P.get("gpu_util_pct")) or 0], marker_color=C_AUG, row=1, col=1); fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.add_bar(x=["used", "total"], y=[used / 1024, tot / 1024], marker_color=[C_AUG, "#B8B8B8"], row=1, col=2)
    fig.update_layout(height=320, showlegend=False); fig.show()
else:
    placeholder("GPU", "pod not reporting").show()
""")
md("### 4. What is in the training data: original vs augmented")
code("""
SA = ROOT / "sft_aug"
stats = json.load(open(SA / "teacher32b_passed_sft_lookups_confirm_stats.json"))
orig = json.load(open(SA / "teacher32b_passed_sft_reasoning.json")); aug = json.load(open(SA / "teacher32b_passed_sft_lookups_confirm.json"))
WRITE = {"cancel_pending_order","exchange_delivered_order_items","modify_pending_order_address","modify_pending_order_items","modify_pending_order_payment","modify_user_address","return_delivered_order_items"}
count = lambda convs, names: sum(1 for c in convs for m in c["messages"] for t in (m.get("tool_calls") or []) if t["function"]["name"] in names)
fig = make_subplots(rows=1, cols=3, subplot_titles=("Tool calls per 100 conversations", "Messages per conversation", "Changes that were preceded by a confirmation"))
for name, convs, color in (("original (180)", orig, C_REF), ("augmented (178)", aug, C_AUG)):
    n = len(convs)
    fig.add_bar(x=["order lookups", "user lookups", "database changes"], y=[100 * count(convs, {"get_order_details"}) / n, 100 * count(convs, {"get_user_details"}) / n, 100 * count(convs, WRITE) / n], name=name, marker_color=color, row=1, col=1)
    fig.add_histogram(x=[len(c["messages"]) - 1 for c in convs], name=name, marker_color=color, opacity=0.6, nbinsx=30, showlegend=False, row=1, col=2)
fig.add_bar(x=["already confirmed", "confirmation inserted"], y=[stats["writes_already_confirmed"], stats["confirmations_inserted"]], marker_color=[C_REF, C_AUG], showlegend=False, text=[stats["writes_already_confirmed"], stats["confirmations_inserted"]], textposition="outside", row=1, col=3)
fig.update_layout(height=400, barmode="group", legend=dict(orientation="h", y=-0.2)); fig.show()
print({k: v for k, v in stats.items() if not isinstance(v, list)}); print("dropped conversations:", stats.get("dropped_changed_before_authenticating"))
""")
md("""
---
# Part 3: evaluation (fills in as the evaluation runs, then finishes after training)
Same test as the earlier runs: τ-bench retail test split, 115 tasks, **4 trials per task**, temperature 0.7, GPT-4o as the customer, 25 steps. Baselines are the two completed 4-trial runs: the **starting adapter (control, same pod)** and the **distilled model**. The new model's results come from `/root/passk_aug.jsonl` on the pod while it runs (via the JSON), and from the saved files after I copy them.
""")
code("""
import math, re
DATA = ROOT / "sft_aug" / "data"
def load_results(path):
    by = {}
    for l in open(path):
        r = json.loads(l); by.setdefault(r["task_index"], {})[r["trial"]] = 1 if r["reward"] >= 1 else 0
    return {t: [v[k] for k in sorted(v)] for t, v in by.items()}
def load_rows(path): return sorted((json.loads(l) for l in open(path)), key=lambda r: r["finished_at"])
RES = {"starting adapter (control)": DATA / "retail_starting_adapter_rerun_4trials_results.jsonl", "distilled (earlier)": DATA / "retail_new_model_4trials_results.jsonl"}
BY = {k: load_results(v) for k, v in RES.items()}
ROWS = {k: load_rows(v) for k, v in RES.items()}
final_aug = DATA / "retail_aug_4trials_results.jsonl"
E = RUN.get("eval", {})
if final_aug.exists():
    BY["augmented (this run)"] = load_results(final_aug); ROWS["augmented (this run)"] = load_rows(final_aug)
elif E.get("rollouts_done"):
    BY["augmented (this run)"] = {int(t): v for t, v in E["by_task"].items()}
def passk(v, k): return math.comb(sum(v), k) / math.comb(len(v), k) if len(v) >= k else None
def mean_se(xs):
    xs = list(xs); n = len(xs)
    if not n: return None, None
    m = sum(xs) / n; return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else None)
SUMM = {k: {kk: mean_se(passk(v, kk) for v in by.values() if passk(v, kk) is not None) for kk in (1, 2, 3, 4)} for k, by in BY.items()}
COL = {"starting adapter (control)": "#E0883B", "distilled (earlier)": "#2F5D8A", "augmented (this run)": "#1E7A5C"}
prog = f"{E.get('rollouts_done', 0)}/{E.get('rollouts_total', 460)} rollouts" + (f", about {fmt_min(E.get('eta_min'))} left, done at about {E.get('eta_at')}" if E.get("eta_min") else "")
print("augmented model evaluation:", "finished (saved file)" if final_aug.exists() else prog)
""")
code("""
fig = make_subplots(rows=1, cols=3, column_widths=[0.38, 0.32, 0.30], subplot_titles=("pass^k on the 115 retail test tasks (±1 s.e.)", "Running pass^1 in completion order", "pass^1 per trial"))
for name, S in SUMM.items():
    ks = [k for k in (1, 2, 3, 4) if S[k][0] is not None]
    fig.add_bar(x=[f"pass^{k}" for k in ks], y=[S[k][0] for k in ks], error_y=dict(type="data", array=[S[k][1] or 0 for k in ks]), name=name, marker_color=COL[name],
                text=[f"{S[k][0]:.0%}" for k in ks], textposition="outside", row=1, col=1)
for name, rows in ROWS.items():
    ok = np.array([1 if r["reward"] >= 1 else 0 for r in rows]); fig.add_scatter(y=np.cumsum(ok) / np.arange(1, len(ok) + 1), mode="lines", line=dict(color=COL[name], width=2.5), name=name, showlegend=False, row=1, col=2)
if not final_aug.exists() and E.get("running_pass1"):
    fig.add_scatter(y=E["running_pass1"], mode="lines", line=dict(color=COL["augmented (this run)"], width=2.5), name="augmented (running)", showlegend=False, row=1, col=2)
for name, rows in ROWS.items():
    tr = {}
    for r in rows: tr.setdefault(r["trial"], []).append(1 if r["reward"] >= 1 else 0)
    fig.add_scatter(x=[k + 1 for k in sorted(tr)], y=[sum(tr[k]) / len(tr[k]) for k in sorted(tr)], mode="lines+markers", line=dict(color=COL[name]), showlegend=False, row=1, col=3)
if not final_aug.exists() and E.get("per_trial"):
    fig.add_scatter(x=[int(k) + 1 for k in E["per_trial"]], y=[v["passed"] / v["done"] for v in E["per_trial"].values()], mode="lines+markers", line=dict(color=COL["augmented (this run)"]), showlegend=False, row=1, col=3)
fig.update_yaxes(range=[0, 0.8]); fig.update_xaxes(title_text="rollouts finished", row=1, col=2); fig.update_xaxes(title_text="trial", dtick=1, row=1, col=3)
fig.update_layout(height=450, barmode="group", legend=dict(orientation="h", y=-0.2)); fig.show()
""")
code("""
# paired comparison on tasks that have all 4 trials (needs the augmented run to be complete)
rng = np.random.default_rng(0)
def paired(a, b):
    ts = [t for t in a if t in b and len(a[t]) == 4 and len(b[t]) == 4]
    xs = np.array([np.mean(a[t]) - np.mean(b[t]) for t in ts])
    if len(xs) < 20: return None
    boots = np.array([rng.choice(xs, len(xs)).mean() for _ in range(10000)]); flips = np.array([(xs * rng.choice([-1, 1], len(xs))).mean() for _ in range(10000)])
    return dict(n=len(xs), mean=xs.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5), p=float((np.abs(flips) >= abs(xs.mean())).mean()))
A = "augmented (this run)"
if A in BY and final_aug.exists():
    rows_ = []
    for other in ("starting adapter (control)", "distilled (earlier)"):
        r = paired(BY[A], BY[other]); rows_.append([f"augmented vs {other}", r["n"], f"{r['mean']:+.3f}", f"{r['lo']:+.3f} to {r['hi']:+.3f}", f"{r['p']:.3f}"])
    display(HTML("<b>Paired by task, pass^1 difference</b>" + table(rows_, ["comparison", "tasks", "difference", "95% bootstrap interval", "permutation p"])))
else:
    display(HTML("<p><i>Paired tests appear once the augmented evaluation is complete and its files are saved.</i></p>"))
""")
md("### Behavior measures: did the model learn the habits? (from the saved conversations)")
code("""
WRITE = {"cancel_pending_order","exchange_delivered_order_items","modify_pending_order_address","modify_pending_order_items","modify_pending_order_payment","modify_user_address","return_delivered_order_items"}
AFF = re.compile(r"^\\W*(yes|yeah|yep|sure|ok|okay|go ahead|please (go|proceed|do)|proceed|confirm|correct|that'?s (right|correct)|absolutely|definitely)\\b", re.I)
def behaviour(path):
    multi = all_open = writes = yes = 0
    for l in open(path):
        conv = json.loads(l); orders, opened, fw, last_user, calls = None, set(), None, "", {}
        for m in conv["messages"]:
            if m["role"] == "user": last_user = (m.get("content") or "").strip()
            for c in (m.get("tool_calls") or []):
                calls[c["id"]] = c["function"]; name = c["function"]["name"]
                if name == "get_order_details":
                    try: opened.add(json.loads(c["function"]["arguments"]).get("order_id"))
                    except Exception: pass
                if name in WRITE:
                    writes += 1; yes += bool(AFF.search(last_user))
                    if fw is None: fw = set(opened)
            if m["role"] == "tool" and m.get("name") == "get_user_details" and (m.get("content") or "").startswith("{"):
                try: orders = json.loads(m["content"]).get("orders")
                except Exception: pass
        if orders and len(orders) >= 2:
            multi += 1; all_open += (fw if fw is not None else opened) >= set(orders)
    return {"all orders opened first": all_open / max(multi, 1), "changes right after a yes": yes / max(writes, 1), "conversations": multi}
TR = {"starting adapter (control)": DATA / "retail_starting_adapter_rerun_4trials_transcripts.jsonl", "distilled (earlier)": DATA / "retail_new_model_4trials_transcripts.jsonl", A: DATA / "retail_aug_4trials_transcripts.jsonl"}
BEH = {k: (behaviour(v) if v.exists() else None) for k, v in TR.items()}
fig = make_subplots(rows=1, cols=2, subplot_titles=("Tasks (2+ orders) where all orders were opened before the first change", "Changes that came right after an explicit yes"))
for name, r in BEH.items():
    if r: 
        fig.add_bar(x=[name], y=[r["all orders opened first"]], marker_color=COL[name], showlegend=False, text=[f"{r['all orders opened first']:.0%}"], textposition="outside", row=1, col=1)
        fig.add_bar(x=[name], y=[r["changes right after a yes"]], marker_color=COL[name], showlegend=False, text=[f"{r['changes right after a yes']:.0%}"], textposition="outside", row=1, col=2)
fig.update_yaxes(range=[0, 1]); fig.update_layout(height=400)
if BEH[A] is None: fig.add_annotation(text="augmented conversations are added after the evaluation is copied from the pod", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.02, font=dict(color="#888"))
fig.show()
display(HTML(table([[k] + ([f"{v['all orders opened first']:.0%}", f"{v['changes right after a yes']:.0%}"] if v else ["-", "-"]) for k, v in BEH.items()], ["model", "all orders opened first", "changes after a yes"])))
""")

md("""
---
# Part 4: tau3 retail with the default settings (baseline vs augmented)
After the τ-bench evaluation above, the same two models are tested on **τ³ (tau2-bench) retail** under its **default settings**: agent temperature **0**, up to **200 steps**, **GPT-4.1** as the simulated customer, all **114 tasks**, **2 trials** per task, the baseline (starting adapter) first and the augmented model second, one after the other. Earlier τ³ runs in this project used temperature 0.7 and 1 trial (starting adapter 53/114 = 46.5%, distilled model 55/114 = 48.2%), so they are shown as a reference only, not as a like-for-like comparison.

Data: `full_run_results/tau3_std_run.json`, rewritten every 30 s by `sft_aug/update_tau3std_json.py` (`sft_aug/std_json_updater.sh start`).
""")
code("""
P4 = ROOT / "full_run_results" / "tau3_std_run.json"
def std_info_html(r):
    b, a, pod, cfg, pl = r["old"], r["new"], r.get("pod", {}), r["config"], r["plan"]
    o = [f"<h4>JSON written {r['updated']} &middot; stage: <span style='color:#1E7A5C'>{r['stage']}</span></h4>"]
    if not r.get("pod_reachable", True): o.append("<p style='color:#B3261E'><b>Pod unreachable</b>, showing the last known state.</p>")
    o.append("<b>Machine</b>" + table([["address", pod.get("host")], ["GPU (name, load, memory used, total)", pod.get("gpu") or "-"], ["disk", pod.get("disk") or "-"],
        ["vLLM answering", "yes" if pod.get("vllm_up") else "no"], ["processes alive", ", ".join(k for k, v in (pod.get("procs") or {}).items() if v) or "none"],
        ["last setup log line", (pod.get("setup_tail") or "-").replace("<", "&lt;")[-150:]]]))
    o.append("<b>Settings</b>" + table([[k.replace("_", " "), v] for k, v in cfg.items()]))
    rows = [[n, f"{e['rollouts_done']} / {e['rollouts_total']}", e.get("passed", "-"), f"{e['pass1']:.1%} &plusmn; {e['pass1_se']:.1%}" if e.get("pass1") is not None and e.get("pass1_se") is not None else "-",
             f"{e['rollouts_per_min']:.2f}" if e.get("rollouts_per_min") else "-", f"${e.get('user_cost', 0):.2f}" if e.get("rollouts_done") else "-"] for n, e in (("baseline (starting adapter)", b), ("augmented", a))]
    o.append("<b>Progress</b>" + table(rows, ["run", "rollouts done", "passed", "pass^1 (&plusmn; s.e.)", "rollouts / min", "GPT-4.1 cost"]))
    o.append(f"<b>Remaining</b> (pace used {pl['pace_used']:.2f} rollouts/min)" + table([[x["step"], x["remaining"], fmt_min(x["minutes"]) if x["remaining"] else "done", x["done_at"] or "-"] for x in pl["steps"]], ["step", "rollouts left", "time", "done at (local)"])
             + f"<p>both runs done at about <b>{pl.get('done_at') or '-'}</b> local (setup and vLLM startup add about 25 minutes before the first rollout)</p>")
    p = r.get("paired")
    if p: o.append("<b>Baseline vs augmented, task by task (all trials pooled per task is not used here: trial 1 of each)</b>" + table([["both pass", len(p["both_pass"])], ["only augmented passes", len(p["only_new"])], ["only baseline passes", len(p["only_old"])], ["neither", len(p["neither"])]]))
    o.append("<b>Recent events</b>" + table([[e["time"], e["event"].replace("<", "&lt;")] for e in r.get("events", [])[-10:]]))
    return "".join(o)
if P4.exists():
    R4 = json.load(open(P4)); display(HTML(std_info_html(R4)))
else:
    R4 = None; display(HTML("<p><i>The τ³ default-settings run has not started yet (no tau3_std_run.json).</i></p>"))
""")
code("""
import time as _t
from IPython.display import clear_output as _clr
try:
    while P4.exists():
        R4 = json.load(open(P4)); _clr(wait=True); display(HTML(std_info_html(R4)))
        if str(R4["stage"]).startswith(("finished", "failed")): break
        _t.sleep(30)
except KeyboardInterrupt:
    pass
""")
cells[-1].metadata["tags"] = ["skip-execution"]
md("### τ³ results: pass rate, how rollouts ended, and the earlier runs for reference")
code("""
if R4 and (R4["old"]["rollouts_done"] or R4["new"]["rollouts_done"]):
    B4, A4 = R4["old"], R4["new"]
    fig = make_subplots(rows=1, cols=3, column_widths=[0.38, 0.32, 0.30], subplot_titles=("pass^1 (±1 s.e.); grey = previous results (other benchmark or settings, not like-for-like)", "Running pass rate", "How the rollouts ended"))
    names, vals, errs, cols = [], [], [], []
    for n, e, c in (("baseline", B4, "#2F5D8A"), ("augmented", A4, "#1E7A5C")):
        if e["rollouts_done"]: names.append(n); vals.append(e["pass1"]); errs.append(e.get("pass1_se") or 0); cols.append(c)
    try:
        old_ref = json.load(open(ROOT / "full_run_results" / "tau3_run.json"))
        for n, k in (("prev baseline (τ³)", "old"), ("prev distilled (τ³)", "new")):
            names.append(n); vals.append(old_ref[k]["pass1"]); errs.append(old_ref[k].get("pass1_se") or 0); cols.append("#B8B8B8")
    except Exception: pass
    try:   # the augmented model's previous result: tau-bench retail, 4 trials (it was never run on tau3 before)
        _by = {}
        for _l in open(ROOT / "sft_aug" / "data" / "retail_aug_4trials_results.jsonl"):
            _r = json.loads(_l); _by.setdefault(_r["task_index"], []).append(1 if _r["reward"] >= 1 else 0)
        _x = [sum(v) / len(v) for v in _by.values()]; _m = sum(_x) / len(_x); _se = (sum((a_ - _m) ** 2 for a_ in _x) / (len(_x) - 1) / len(_x)) ** 0.5
        names.append("prev augmented (τ-bench)"); vals.append(_m); errs.append(_se); cols.append("#B8B8B8")
    except Exception: pass
    fig.add_bar(x=names, y=vals, error_y=dict(type="data", array=errs), marker_color=cols, text=[f"{v:.1%}" for v in vals], textposition="outside", showlegend=False, row=1, col=1)
    for n, e, c in (("baseline", B4, "#2F5D8A"), ("augmented", A4, "#1E7A5C")):
        if e["rollouts_done"]: fig.add_scatter(y=e["running_pass1"], mode="lines", line=dict(color=c, width=2.5), name=n, row=1, col=2)
    reasons = sorted({k for e in (B4, A4) for k in e.get("terminations", {})}); pal = ["#1E7A5C", "#E0883B", "#B3261E", "#7A4E9C", "#888888", "#2F5D8A"]
    for i, k in enumerate(reasons):
        fig.add_bar(x=[n for n, e in (("baseline", B4), ("augmented", A4)) if e["rollouts_done"]], y=[e["terminations"].get(k, 0) for e in (B4, A4) if e["rollouts_done"]], name=k, marker_color=pal[i % len(pal)], row=1, col=3)
    fig.update_yaxes(range=[0, 0.8], row=1, col=1); fig.update_xaxes(title_text="rollouts finished", row=1, col=2)
    fig.update_layout(height=450, barmode="stack", legend=dict(orientation="h", y=-0.25)); fig.show()
else:
    placeholder("τ³ default-settings results", "no rollouts yet").show()
""")
md("### τ³: baseline vs augmented, task by task, and per trial")
code("""
if R4 and R4["old"]["rollouts_done"] and R4["new"]["rollouts_done"]:
    rows_b = {}; rows_a = {}
    for r_ in R4["old"]["rows"]: rows_b.setdefault(r_["task_id"], []).append(1 if (r_.get("reward") or 0) >= 1 else 0)
    for r_ in R4["new"]["rows"]: rows_a.setdefault(r_["task_id"], []).append(1 if (r_.get("reward") or 0) >= 1 else 0)
    tasks = sorted(set(rows_b) & set(rows_a), key=lambda x: int(x) if x.isdigit() else x)
    pb = [np.mean(rows_b[t]) for t in tasks]; pa = [np.mean(rows_a[t]) for t in tasks]
    d = np.array(pa) - np.array(pb); rng = np.random.default_rng(0)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)]); flips = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(10000)])
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Per-task pass rate: augmented (y) vs baseline (x)", "Tasks gained and lost (augmented - baseline)", "Bootstrap of the mean difference"))
    rs = np.random.default_rng(1); fig.add_scatter(x=np.array(pb) + rs.uniform(-.03, .03, len(pb)), y=np.array(pa) + rs.uniform(-.03, .03, len(pa)), mode="markers", marker=dict(size=7, color="#1E7A5C", opacity=.7),
        text=[f"task {t}: baseline {x:.2f}, augmented {y:.2f}" for t, x, y in zip(tasks, pb, pa)], hoverinfo="text", showlegend=False, row=1, col=1)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999", dash="dot"), showlegend=False, row=1, col=1)
    from collections import Counter as _C
    cn = _C(round(x * 2) / 2 for x in d); ks = sorted(cn)
    fig.add_bar(x=ks, y=[cn[k] for k in ks], marker_color=["#1E7A5C" if k > 0 else "#E0883B" if k < 0 else "#B8B8B8" for k in ks], showlegend=False, row=1, col=2)
    fig.add_histogram(x=boots, nbinsx=50, marker_color="#1E7A5C", showlegend=False, row=1, col=3); fig.add_vline(x=0, line_color="black", row=1, col=3)
    fig.update_layout(height=420); fig.show()
    print(f"mean difference (augmented - baseline): {d.mean():+.3f}, 95% bootstrap interval {np.percentile(boots, 2.5):+.3f} to {np.percentile(boots, 97.5):+.3f}, permutation p = {float((np.abs(flips) >= abs(d.mean())).mean()):.3f} over {len(d)} tasks")
else:
    placeholder("τ³ baseline vs augmented", "needs results from both runs").show()
""")
md("### τ³: how long and how chatty the rollouts are, and the cost")
code("""
if R4 and (R4["old"]["rollouts_done"] or R4["new"]["rollouts_done"]):
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Rollout duration (s)", "Messages per rollout", "Tool calls per rollout", "GPT-4.1 cost so far and projected (USD)"), specs=[[{}, {}, {}, {}]])
    for n, e, c in (("baseline", R4["old"], "#2F5D8A"), ("augmented", R4["new"], "#1E7A5C")):
        if e["rollouts_done"]:
            for col, key in ((1, "duration"), (2, "n_messages"), (3, "n_tool_calls")): fig.add_box(y=[r_.get(key) for r_ in e["rows"]], name=n, marker_color=c, boxmean=True, showlegend=(col == 1), row=1, col=col)
            per = e["user_cost"] / e["rollouts_done"]; fig.add_bar(x=[n + " so far", n + " projected"], y=[e["user_cost"], per * e["rollouts_total"]], marker_color=c, opacity=.9, showlegend=False, row=1, col=4)
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2)); fig.show()
else:
    placeholder("τ³ durations and cost", "no rollouts yet").show()
""")
nbf.write(nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}), OUT)
print("wrote", OUT, len(cells), "cells")
