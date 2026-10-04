"""Builds tau3_retail_eval.ipynb (next to this folder's parent). The notebook only READS full_run_results/tau3_run.json."""
import sys, nbformat as nbf

OUT = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# τ³-bench retail: starting adapter vs fine-tuned model

Both models are Qwen3-14B (4-bit) + a LoRA adapter, served by vLLM and run **one after the other** on the **τ³ (tau2-bench) retail** domain:

* **old** = the starting adapter (`round-0`), **new** = the adapter fine-tuned on 180 conversations of a Qwen3-32B teacher (`distill-epoch-1`).
* all 114 retail tasks (split `base`), **1 trial per task**, agent temperature 0.7, **GPT-4.1** as the simulated user, tau2 default of 200 steps.
* This notebook only **reads** `full_run_results/tau3_run.json`, which `tau3/update_tau3_json.py` rewrites every 30 s (start it with `tau3/tau3_json_updater.sh start`). Re-run a cell to refresh it.

> With one trial per task and 114 tasks, differences smaller than about 8 points cannot be told apart from noise.

---
# Part 1: process info
""")

code("""
import json, datetime
from pathlib import Path
from IPython.display import HTML, display

ROOT = Path("..") if Path("../full_run_results").exists() else Path(".")
PATH = ROOT / "full_run_results" / "tau3_run.json"

def load():
    return json.load(open(PATH))

def pct(x, se=None):
    if x is None:
        return "-"
    return f"{x:.1%}" + (f" &plusmn; {se:.1%}" if se is not None else "")

def fmt_min(m):
    if m is None:
        return "-"
    m = int(round(m))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"

def table(rows, head=None):
    td = "style='padding:3px 12px;border-bottom:1px solid #ddd;text-align:left'"
    h = "".join(f"<th {td}>{c}</th>" for c in head) if head else ""
    body = "".join("<tr>" + "".join(f"<td {td}>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table style='border-collapse:collapse;margin:6px 0 14px'>{('<tr>' + h + '</tr>') if head else ''}{body}</table>"

def info_html(run):
    old, new, pod, cfg, plan = run["old"], run["new"], run.get("pod", {}), run["config"], run["plan"]
    out = [f"<h4>JSON written {run['updated']} &middot; stage: <span style='color:#1E7A5C'>{run['stage']}</span></h4>"]
    if not run.get("pod_reachable", True):
        out.append("<p style='color:#B3261E'><b>Pod unreachable</b>, showing the last known state.</p>")
    out.append("<b>Setup of this run</b>" + table([[k.replace("_", " "), v] for k, v in cfg.items()]))
    gpu = pod.get("gpu") or "-"
    out.append("<b>Pod</b>" + table([["address", pod.get("host")], ["GPU (name, util, memory used, total)", gpu], ["disk (used / free)", (pod.get("disk") or "-")],
        ["vLLM answering", "yes" if pod.get("vllm_up") else "no"], ["processes alive", ", ".join(k for k, v in (pod.get("procs") or {}).items() if v) or "none"],
        ["last setup log line", (pod.get("setup_tail") or "-").replace("<", "&lt;")[-160:]]]))
    rows = []
    for name, ev in (("old (starting adapter)", old), ("new (fine-tuned)", new)):
        rows.append([name, f"{ev['rollouts_done']} / {ev['rollouts_total']}", ev.get("passed", "-"), pct(ev.get("pass1"), ev.get("pass1_se")),
                     f"{ev['rollouts_per_min']:.2f}" if ev.get("rollouts_per_min") else "-", f"${ev.get('user_cost', 0):.2f}" if ev.get("rollouts_done") else "-",
                     fmt_min((ev.get('avg_duration_s') or 0) / 60) if ev.get("rollouts_done") else "-"])
    out.append("<b>Progress</b>" + table(rows, ["run", "rollouts done", "passed", "pass^1 (&plusmn; s.e.)", "rollouts / min", "user (GPT-4.1) cost", "avg rollout time"]))
    steps = [[s["step"], s["remaining"], fmt_min(s["minutes"]) if s["remaining"] else "done", s["done_at"] or "-"] for s in plan["steps"]]
    out.append(f"<b>Remaining</b> (pace used: {plan['pace_used']:.2f} rollouts/min)" + table(steps, ["step", "rollouts left", "time", "done at (local)"])
               + f"<p>everything done at about <b>{plan.get('done_at') or '-'}</b> local</p>")
    p = run.get("paired")
    if p:
        out.append("<b>Old vs new, task by task</b>" + table([["both pass", len(p["both_pass"])], ["only the new model passes", len(p["only_new"])],
            ["only the old model passes", len(p["only_old"])], ["neither passes", len(p["neither"])],
            ["difference (new - old)", f"{p['diff']:+.1%} of {p['n_common']} common tasks"], ["sign test p", f"{p['sign_test_p']:.2f}" if p["sign_test_p"] is not None else "-"]]))
    ev = run.get("events", [])[-12:]
    out.append("<b>Recent events</b>" + table([[e["time"], e["event"].replace("<", "&lt;")] for e in ev]))
    return "".join(out)

RUN = load()
display(HTML(info_html(RUN)))
""")

md("""
### Live view (optional)
Run the next cell to refresh the block above every 30 s until you stop the cell (the square button). It only re-reads the JSON.
""")
code("""
import time
from IPython.display import clear_output
try:
    while True:
        RUN = load()
        clear_output(wait=True)
        display(HTML(info_html(RUN)))
        if str(RUN["stage"]).startswith(("finished", "failed")):
            break
        time.sleep(30)
except KeyboardInterrupt:
    pass
""")
cells[-1].metadata["tags"] = ["skip-execution"]   # the loop never ends on its own, so nbconvert must skip it

md("---\n# Part 2: graphs (plotly)")

code("""
import math
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "notebook_connected"
pio.templates.default = "plotly_white"
RUN = load()
OLD, NEW = RUN["old"], RUN["new"]
C_OLD, C_NEW = "#2F5D8A", "#1E7A5C"
HAVE_OLD, HAVE_NEW = OLD["rollouts_done"] > 0, NEW["rollouts_done"] > 0

def placeholder(title, text="no data yet"):
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(size=16, color="#888"))
    fig.update_layout(title=title, height=260, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def passed(r):
    return (r.get("reward") or 0) >= 1

print(f"JSON written {RUN['updated']} | stage: {RUN['stage']} | old {OLD['rollouts_done']}/{OLD['rollouts_total']}, new {NEW['rollouts_done']}/{NEW['rollouts_total']}")
""")

md("### 1. Progress and running pass rate")
code("""
if HAVE_OLD or HAVE_NEW:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Running pass^1 (in completion order)", "Rollouts finished over time"))
    for name, color, ev in (("old (starting adapter)", C_OLD, OLD), ("new (fine-tuned)", C_NEW, NEW)):
        if ev["rollouts_done"]:
            fig.add_scatter(y=ev["running_pass1"], mode="lines", line=dict(color=color, width=2.5), name=name, row=1, col=1)
            fig.add_scatter(x=ev["finish_times"], y=list(range(1, ev["rollouts_done"] + 1)), mode="lines", line=dict(color=color, width=2.5), name=name, showlegend=False, row=1, col=2)
    fig.update_yaxes(range=[0, 1], title_text="running pass^1", row=1, col=1); fig.update_xaxes(title_text="rollouts finished", row=1, col=1)
    fig.update_yaxes(title_text="rollouts", row=1, col=2)
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
    fig.show()
else:
    placeholder("Progress and running pass rate").show()
""")

md("### 2. Result: pass^1 per model, and how the rollouts ended")
code("""
if HAVE_OLD or HAVE_NEW:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("pass^1 (±1 standard error)", "How the rollouts ended"),
                        specs=[[{"type": "xy"}, {"type": "xy"}]])
    for name, color, ev in (("old (starting adapter)", C_OLD, OLD), ("new (fine-tuned)", C_NEW, NEW)):
        if ev["rollouts_done"]:
            fig.add_bar(x=[name], y=[ev["pass1"]], error_y=dict(type="data", array=[ev.get("pass1_se") or 0]), marker_color=color, showlegend=False,
                        text=[f"{ev['pass1']:.1%}<br>({ev['passed']}/{ev['rollouts_done']})"], textposition="outside", row=1, col=1)
    reasons = sorted({k for ev in (OLD, NEW) for k in ev.get("terminations", {})})
    palette = ["#1E7A5C", "#E0883B", "#B3261E", "#7A4E9C", "#888888", "#2F5D8A", "#C9A227"]
    for i, k in enumerate(reasons):
        names = [n for n, ev in (("old", OLD), ("new", NEW)) if ev["rollouts_done"]]
        vals = [ev["terminations"].get(k, 0) for ev in (OLD, NEW) if ev["rollouts_done"]]
        fig.add_bar(x=names, y=vals, name=k, marker_color=palette[i % len(palette)], row=1, col=2)
    fig.update_yaxes(range=[0, 1], row=1, col=1); fig.update_yaxes(title_text="rollouts", row=1, col=2)
    fig.update_layout(height=450, barmode="stack", legend=dict(orientation="h", y=-0.2))
    fig.show()
else:
    placeholder("Result").show()
""")

md("### 3. Old vs new, task by task")
code("""
P = RUN.get("paired")
if P:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("Tasks by outcome", "Pass / fail per task (1 = pass)"), specs=[[{"type": "xy"}, {"type": "xy"}]])
    cats = [("both pass", len(P["both_pass"]), "#1E7A5C"), ("only new", len(P["only_new"]), "#2FA37C"), ("only old", len(P["only_old"]), "#2F5D8A"), ("neither", len(P["neither"]), "#B8B8B8")]
    fig.add_bar(x=[c[0] for c in cats], y=[c[1] for c in cats], marker_color=[c[2] for c in cats], text=[c[1] for c in cats], textposition="outside", showlegend=False, row=1, col=1)
    ids = sorted(set(P["both_pass"] + P["only_new"] + P["only_old"] + P["neither"]), key=lambda x: int(x) if x.isdigit() else x)
    old_ok = {r["task_id"]: 1 if passed(r) else 0 for r in OLD["rows"]}; new_ok = {r["task_id"]: 1 if passed(r) else 0 for r in NEW["rows"]}
    z = [[old_ok.get(t, None) for t in ids], [new_ok.get(t, None) for t in ids]]
    fig.add_heatmap(z=z, x=ids, y=["old", "new"], colorscale=[[0, "#E6B8B5"], [1, "#1E7A5C"]], showscale=False, xgap=1, ygap=2, row=1, col=2)
    fig.update_xaxes(title_text="task id", showticklabels=False, row=1, col=2)
    fig.update_layout(height=380)
    fig.show()
    print(f"difference (new - old): {P['diff']:+.1%} of {P['n_common']} common tasks; sign test p = {P['sign_test_p']:.2f}" if P["sign_test_p"] is not None else "no differing tasks yet")
    print("only new passes:", ", ".join(P["only_new"]) or "-")
    print("only old passes:", ", ".join(P["only_old"]) or "-")
else:
    placeholder("Old vs new, task by task", "needs results from both runs").show()
""")

md("### 4. What the reward is made of, and how hard the tasks are")
code("""
if HAVE_OLD or HAVE_NEW:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("Share of rollouts passing each reward component", "pass^1 by number of target actions in the task"))
    keys = sorted({k for ev in (OLD, NEW) for k in (ev.get("breakdown_pass_rate") or {})})
    for name, color, ev in (("old", C_OLD, OLD), ("new", C_NEW, NEW)):
        if ev["rollouts_done"] and keys:
            fig.add_bar(x=keys, y=[(ev["breakdown_pass_rate"] or {}).get(k) for k in keys], name=name, marker_color=color, row=1, col=1)
    for name, color, ev in (("old", C_OLD, OLD), ("new", C_NEW, NEW)):
        if ev["rollouts_done"]:
            g = {}
            for r in ev["rows"]:
                g.setdefault(min(r.get("n_target_actions") or 0, 6), []).append(1 if passed(r) else 0)
            ks = sorted(g)
            fig.add_bar(x=[("6+" if k == 6 else str(k)) for k in ks], y=[sum(g[k]) / len(g[k]) for k in ks], name=name, marker_color=color, showlegend=False,
                        text=[f"n={len(g[k])}" for k in ks], textposition="outside", row=1, col=2)
    fig.update_yaxes(range=[0, 1.05]); fig.update_xaxes(title_text="target actions the task needs", row=1, col=2)
    fig.update_layout(height=420, barmode="group", legend=dict(orientation="h", y=-0.2))
    fig.show()
else:
    placeholder("Reward components").show()
""")

md("### 5. How long and how chatty the rollouts are")
code("""
if HAVE_OLD or HAVE_NEW:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Rollout duration (s)", "Messages per rollout", "Tool calls per rollout"))
    for name, color, ev in (("old", C_OLD, OLD), ("new", C_NEW, NEW)):
        if ev["rollouts_done"]:
            for col, key in ((1, "duration"), (2, "n_messages"), (3, "n_tool_calls")):
                fig.add_box(y=[r.get(key) for r in ev["rows"]], name=name, marker_color=color, boxmean=True, showlegend=(col == 1), row=1, col=col)
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.15))
    fig.show()
else:
    placeholder("Durations").show()
""")

md("### 6. Cost so far (GPT-4.1 as the user; the local agent is free)")
code("""
rows = [(n, ev) for n, ev in (("old", OLD), ("new", NEW)) if ev["rollouts_done"]]
if rows:
    fig = go.Figure()
    for (n, ev), color in zip(rows, (C_OLD, C_NEW)):
        per = ev["user_cost"] / ev["rollouts_done"]
        fig.add_bar(x=[n + " (so far)", n + " (projected for all 114)"], y=[ev["user_cost"], per * ev["rollouts_total"]], marker_color=[color, color], opacity=0.9,
                    text=[f"${ev['user_cost']:.2f}", f"${per * ev['rollouts_total']:.2f}"], textposition="outside", showlegend=False)
    fig.update_layout(title="User-simulator cost (USD)", height=380, yaxis_title="USD")
    fig.show()
else:
    placeholder("Cost").show()
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
