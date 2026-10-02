"""Insert (or replace) notebook Section 12: distilling the 32B teacher into the 14B (SFT) and the tau-bench tests of the result.
Layout: a header cell; a TRAINING part (status cell + plots cell); a TAU-BENCH EVALUATION part (status cell + plots cell, retail and airline).
The code cells only READ full_run_results/distill_run.json and draw it when run; nothing rewrites the notebook afterwards. Idempotent.
Usage: python3 add_distill_section.py
"""
import json, os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen3_14b_training_and_rollouts.ipynb")

HEADER = """## Section 12: fine-tuning the 14B on the 32B teacher's passed conversations, then tau-bench tests (live)

Stage 2 and 3 of the distillation. The 32B teacher (Section 11) solved 86 of the 114 combo tasks the 14B does not reliably solve and
produced 180 passing conversations (60 from tasks the 14B never solves, 120 from ones it solves sometimes), about 17 agent turns each.
Each conversation becomes one training example per agent turn, laid out exactly as the model sees it at inference, with the 32B's
thinking kept (3,065 examples, one epoch, 4-bit LoRA rank 16, learning rate 5e-5, batch 8, loss on the agent's reply only).

This section has two parts, each with its own status cell and plots cell:
1. **Training**: the SFT run itself (loss, gradient norm, learning rate, checkpoints, timing).
2. **tau-bench evaluation**: the tests of the result, retail and airline (below).

**Retail test.** The new adapter takes the full 115-task retail test at temperature 0.7 (the setup of Section 10): first two trials per task,
then two more trials for both the new model and the starting adapter to cut the noise (so up to four per task). With several trials, **pass^1** is the
average pass rate and **pass^2** is the chance that two random trials of a task both pass (C(c,2) / C(n,2) for c passes in n trials, averaged over tasks;
with two trials it is "both pass").
Two baselines are shown next to it: the **earlier test** of the starting adapter from Section 10 (different pod and GPU), and a **re-run of
the starting adapter on the same pod, server and settings** as the new model, which removes the pod/GPU difference.

**Airline test.** After retail trials 3 and 4, the same two adapters (the new model, then the starting adapter) are run on the **airline** domain:
50 tasks (the only split airline has), two trials each at temperature 0.7, tool-calling agent with GPT-4o as the customer, 25 steps, so 100 rollouts
per model. The fine-tuning never saw airline conversations, so this checks whether the retail gain carries over to a different domain or the
model got worse elsewhere. The runs go one after the other on the same server (never in parallel), and at the retail pace (about 4-5 rollouts per minute) each model takes
roughly 20-25 minutes. The airline test was started once and paused after a few rollouts so the retail trials 3 and 4 could run first; it resumes
from the saved rollouts. The evaluation status cell shows the remaining steps with their estimated finish times.

Each pass rate over 115 retail tasks has a standard error of about 4 points (about 6 over the 50 airline tasks), so gaps under roughly
8 points should be read with care. The cells below only read `full_run_results/distill_run.json` (written with timestamps by
`teacher/update_distill_json.py`); re-run them to redraw. Set `LIVE = True` in a status cell to have it redraw itself every 30 seconds.
"""

TRAIN_HEADING = """### Part 1: training (SFT of the 14B)"""
EVAL_HEADING = """### Part 2: tau-bench evaluation (retail and airline)"""

TRAIN_STATUS = """import datetime, html, json, time
from IPython.display import HTML, clear_output, display

RUN = "full_run_results/distill_run.json"
LIVE = False  # True: re-read the JSON and redraw every 30 s until training is finished (interrupt to stop)


def fmt_min(x):
    if x is None:
        return "estimating..."
    h, m = divmod(int(round(x)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def render_html(run):
    td = "border:1px solid #8886;padding:3px 10px;text-align:left"
    def table(head, body):
        h = "".join(f"<th style='{td}'>{c}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td style='{td}'>{c}</td>" for c in row) + "</tr>" for row in body)
        return f"<table style='border-collapse:collapse;margin:6px 0 14px'><tr>{h}</tr>{b}</table>"
    t, ev, d = run["training"], run["events_local"], run["data"]
    finished = "training_finished" in ev
    state = "finished" if finished else ("running" if run["stage"] == "training" else html.escape(run["stage"]))
    out = [f"<h4>Training (JSON written {html.escape(run['updated'])} = {html.escape(run['updated_utc'])})</h4>"]
    rows = [["state", state + ("" if run["pod_reachable"] else " (pod unreachable, showing last known state)")],
            ["data", f"{d['conversations']} passed conversations from the 32B teacher, {d['samples']} per-turn examples, {d['epochs']} epoch"],
            ["setup", "4-bit LoRA rank 16, learning rate 5e-5 (linear decay), batch 8 examples, loss on the agent's reply only"]]
    if t.get("steps_total"):
        rows.append(["progress", f"step {t['steps_done']} / {t['steps_total']} ({t['steps_done'] / t['steps_total']:.0%})"
                     + (f", {t['sec_per_step']:.0f} s per step" if t.get("sec_per_step") else "")])
        if t.get("loss_last10") is not None:
            rows.append(["loss", f"{t['loss_first10']:.3f} (first 10 steps) to {t['loss_last10']:.3f} (last 10); gradient norm max {max(t['grad_norm']):.2f}"])
        rows.append(["time", ("done in " + fmt_min(t.get("elapsed_min"))) if finished else f"{fmt_min(t.get('elapsed_min'))} so far, about {fmt_min(t.get('eta_min'))} left"])
        rows.append(["checkpoints on the pod", ", ".join(c for c in t["checkpoints"] if c.startswith(("step", "epoch"))) or "-"])
    out.append(table(["", ""], rows))
    keys = ("teacher_finished", "training_started", "training_finished")
    out.append("<b>Timestamps (local)</b>" + table(["event", "time"], [[k.replace("_", " "), html.escape(ev[k])] for k in keys if k in ev] or [["(none yet)", ""]]))
    return "".join(out)


# --- run ---
if LIVE:
    while True:
        run = json.load(open(RUN))
        clear_output(wait=True)
        display(HTML(render_html(run)))
        if "training_finished" in run["events_local"] or run["stage"].startswith("failed"):
            break
        time.sleep(30)
else:
    display(HTML(render_html(json.load(open(RUN)))))
"""

TRAIN_CHART = """import json
import matplotlib.pyplot as plt

run = json.load(open("full_run_results/distill_run.json"))
t = run["training"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.3))

ax = axes[0]
if t["loss"]:
    ax.plot(t["step_index"], t["loss"], color="#8888", lw=0.8, label="per step")
    w = 15
    if len(t["loss"]) >= w:
        ma = [sum(t["loss"][i - w:i]) / w for i in range(w, len(t["loss"]) + 1)]
        ax.plot(t["step_index"][w - 1:], ma, color="#1E7A5C", lw=2, label=f"{w}-step average")
    for c in t["checkpoints"]:
        if c.startswith("step-"):
            ax.axvline(int(c.split("-")[1]), color="#2F5D8A", ls=":", lw=1)
    ax.legend(fontsize=8)
ax.set_xlabel("optimizer step (8 examples each; dotted lines: saved checkpoints)")
ax.set_ylabel("loss on the agent's reply")
ax.set_title("Training loss" + (f" ({t['steps_done']}/{t['steps_total']} steps)" if t["steps_total"] else ""))

ax = axes[1]
if t["grad_norm"]:
    ax.plot(t["step_index"], t["grad_norm"], color="#A33B3B", lw=1)
    ax.set_ylabel("gradient norm", color="#A33B3B")
    ax2 = ax.twinx()
    ax2.plot(t["step_index"], t["lr"], color="#2F5D8A", lw=1)
    ax2.set_ylabel("learning rate", color="#2F5D8A")
ax.set_xlabel("optimizer step")
ax.set_title("Gradient norm and learning rate")

ax = axes[2]
if t["tokens"]:
    ax.bar(t["step_index"], t["tokens"], color="#8888", width=1.0)
ax.set_xlabel("optimizer step")
ax.set_ylabel("completion tokens in the step")
ax.set_title("Size of each training step")

plt.tight_layout()
plt.show()
"""

EVAL_STATUS = """import datetime, html, json, time
from IPython.display import HTML, clear_output, display

RUN = "full_run_results/distill_run.json"
LIVE = False  # True: re-read the JSON and redraw every 30 s until all tests are done (interrupt to stop)


def fmt_min(x):
    if x is None:
        return "estimating..."
    h, m = divmod(int(round(x)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def pct(x, se=None):
    return "-" if x is None else f"{x:.1%}" + (f" +/- {se:.1%}" if se is not None else "")


def render_html(run):
    td = "border:1px solid #8886;padding:3px 10px;text-align:left"
    def table(head, body):
        h = "".join(f"<th style='{td}'>{c}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td style='{td}'>{c}</td>" for c in row) + "</tr>" for row in body)
        return f"<table style='border-collapse:collapse;margin:6px 0 14px'><tr>{h}</tr>{b}</table>"
    _plan = {st["proc"]: st for st in (run.get("plan") or {}).get("steps", [])}
    def plan_note(proc, d):
        # time left comes from the plan (pace of the process running now), so it matches the "now / next" table
        st = _plan.get(proc)
        if not st or st["remaining"] <= 0 or run["stage"] == "finished":
            return ""
        return (f", {st['remaining']} left, about {fmt_min(st['done_in_min'])}, done at about {st['done_at']}" if st["running"]
                else f", waiting, {st['remaining']} to do, done at about {st['done_at']}")
    e, r2, b, ev = run["eval"], run.get("rerun") or {}, run["baseline"], run["events_local"]
    air = run.get("airline") or {}
    out = [f"<h4>tau-bench evaluation (JSON written {html.escape(run['updated'])} = {html.escape(run['updated_utc'])})</h4>"]
    rows = [["stage", html.escape(run["stage"]) + ("" if run["pod_reachable"] else " (pod unreachable, showing last known state)")]]
    if e["rollouts_done"]:
        rows.append(["retail test, new model", f"{e['rollouts_done']} / {e['rollouts_total']} rollouts, {e['errors']} errors" + plan_note("distilled", e)])
    if r2.get("rollouts_done") is not None:
        rows.append(["retail test, starting adapter re-run (same pod)", f"{r2.get('rollouts_done', 0)} / {r2.get('rollouts_total', 460)} rollouts, {r2.get('errors', 0)} errors" + plan_note("baseline2", r2)])
    for key, label in (("new", "airline test, new model"), ("base", "airline test, starting adapter")):
        a_ = air.get(key) or {}
        if a_.get("rollouts_done") or run.get("airline_eta"):
            rows.append([label, f"{a_.get('rollouts_done', 0)} / {a_.get('rollouts_total', 100)} rollouts, {a_.get('errors', 0)} errors"])
    plan = run.get("plan") or {}
    if plan.get("steps"):
        for st in plan["steps"]:
            rows.append([("now: " if st["running"] else "next: ") + st["step"],
                         f"{st['remaining']} rollouts left, about {fmt_min(st['minutes'])}, done at about {st['done_at']}"])
        rows.append(["everything done at about", f"{plan['done_at']} local (in about {fmt_min(plan['total_min'])}; pace {plan['pace_used']:.1f} rollouts per minute, "
                     + ("measured now" if plan.get("pace_measured") else "from the earlier retail runs") + ")"])
    out.append(table(["", ""], rows))
    if e["rollouts_done"]:
        bs = lambda k: b.get(k) if b else None
        cons = lambda x: f"{x:.2f}" if x else "-"
        out.append("<b>Retail test: new model vs the starting adapter</b>" + table(
            ["", "new model (SFT on the teacher)", "starting adapter, re-run on the same pod", "starting adapter, earlier test (other pod)"],
            [["pass^1", pct(e["pass1"], e.get("pass1_se")), pct(r2.get("pass1"), r2.get("pass1_se")), pct(bs("pass1"), bs("pass1_se"))],
             ["pass^2 (two random trials both pass)", pct(e["pass2"], e.get("pass2_se")), pct(r2.get("pass2"), r2.get("pass2_se")), pct(bs("pass2"), bs("pass2_se"))],
             ["pass^3 (three random trials all pass)", pct(e.get("pass3"), e.get("pass3_se")), pct(r2.get("pass3"), r2.get("pass3_se")), pct(bs("pass3"), bs("pass3_se"))],
             ["pass^4 (all four trials pass)", pct(e.get("pass4"), e.get("pass4_se")), pct(r2.get("pass4"), r2.get("pass4_se")), pct(bs("pass4"), bs("pass4_se"))],
             ["tasks with 3+ / 4 trials done", f"{e.get('tasks_for_pass3', 0)} / {e.get('tasks_for_pass4', 0)}", f"{r2.get('tasks_for_pass3', 0)} / {r2.get('tasks_for_pass4', 0)}", "-"],
             ["consistency (pass^2 / pass^1)", cons(e.get("consistency")), cons(r2.get("consistency")), cons(bs("consistency"))]]))
        tr_ = lambda src: (src or {}).get("by_trial") or {}
        ks = sorted({int(k) for src in (e, r2, b) for k in tr_(src)})
        if ks:
            def tcell(src, k):
                x = tr_(src).get(str(k))
                return "-" if not x or not x["done"] else f"{x['pass_rate']:.1%} ({x['passed']}/{x['done']})"
            rows_t = [[f"trial {k + 1}" + (" (second half)" if k >= 2 else ""), tcell(e, k), tcell(r2, k), tcell(b, k)] for k in ks]
            rows_t.append(["all trials", f"{e['pass1']:.1%} ({e['passed']}/{e['rollouts_done']})" if e.get("pass1") is not None else "-",
                           f"{r2['pass1']:.1%} ({r2['passed']}/{r2['rollouts_done']})" if r2.get("pass1") is not None else "-",
                           f"{b['pass1']:.1%}" if b and b.get("pass1") is not None else "-"])
            out.append("<b>Retail pass rate by trial (each trial = one attempt at all 115 tasks; passed / rollouts done)</b>" + table(
                ["", "new model", "starting adapter, re-run (same pod)", "starting adapter, earlier test"], rows_t))
        g = run.get("groups") or {}
        if g.get("new"):
            def cell(src, kind, k):
                x = (g.get(src) or {}).get(kind)
                return "-" if not x or x.get(k) is None else f"{x[k]:.1%}"
            out.append("<b>Retail by task type (64 combo tasks, 51 others)</b>" + table(
                ["", "new", "re-run (same pod)", "earlier test"],
                [[f"{kind} tasks, {k.replace('pass', 'pass^')}", cell("new", kind, k), cell("rerun", kind, k), cell("earlier", kind, k)]
                 for kind in ("combo", "other") for k in ("pass1", "pass2", "pass3", "pass4")]))
        c = run.get("comparisons") or {}
        lines = []
        for key, label in (("new_vs_rerun", "new model vs re-run on the same pod"), ("new_vs_earlier", "new model vs the earlier test")):
            if c.get(key):
                x = c[key]
                lines.append(f"{label}: pass^2 is higher for the new model on {x['new_only']} tasks and higher for the baseline on {x['other_only']} "
                             f"({x.get('ties', 0)} tied; sign test on {x['n_common']} tasks, p = {x['p_value']:.3f})")
        if lines:
            out.append("<p><b>Retail pass^2 comparisons per task</b><br>" + "<br>".join(lines) + "</p>")
    an, ab = air.get("new") or {}, air.get("base") or {}
    if an.get("pass1") is not None or ab.get("pass1") is not None:
        cons_ = lambda x: f"{x:.2f}" if x else "-"
        out.append("<b>Airline test (50 tasks, never trained on): new model vs the starting adapter</b>" + table(
            ["", "new model", "starting adapter"],
            [["pass^1", pct(an.get("pass1"), an.get("pass1_se")), pct(ab.get("pass1"), ab.get("pass1_se"))],
             ["pass^2 (two random trials both pass)", pct(an.get("pass2"), an.get("pass2_se")), pct(ab.get("pass2"), ab.get("pass2_se"))],
             ["consistency (pass^2 / pass^1)", cons_(an.get("consistency")), cons_(ab.get("consistency"))]]))
        at_ = lambda src, k: ((src or {}).get("by_trial") or {}).get(str(k))
        akeys = sorted({int(k) for src in (an, ab) for k in ((src or {}).get("by_trial") or {})})
        if akeys:
            fmt_ = lambda x: "-" if not x or not x["done"] else f"{x['pass_rate']:.1%} ({x['passed']}/{x['done']})"
            out.append("<b>Airline pass rate by trial (passed / rollouts done)</b>" + table(
                ["", "new model", "starting adapter"], [[f"trial {k + 1}", fmt_(at_(an, k)), fmt_(at_(ab, k))] for k in akeys]))
        ac = air.get("comparison")
        if ac:
            out.append(f"<p>Airline pass^2 per task: higher for the new model on {ac['new_only']} tasks, higher for the starting adapter on {ac['other_only']} "
                       f"({ac.get('ties', 0)} tied; sign test on {ac['n_common']} tasks, p = {ac['p_value']:.3f})</p>")
    keys = ("eval_started", "rerun_finished", "extra_trials_started", "extra_new_finished", "extra_base_finished", "airline_new_finished", "airline_finished", "pipeline_done")
    out.append("<b>Timestamps (local)</b>" + table(["event", "time"], [[k.replace("_", " "), html.escape(ev[k])] for k in keys if k in ev] or [["(none yet)", ""]]))
    return "".join(out)


# --- run ---
if LIVE:
    while True:
        run = json.load(open(RUN))
        clear_output(wait=True)
        display(HTML(render_html(run)))
        if run["stage"] == "finished" or run["stage"].startswith("failed"):
            break
        time.sleep(30)
else:
    display(HTML(render_html(json.load(open(RUN)))))
"""

EVAL_CHART = """import json
import matplotlib.pyplot as plt

run = json.load(open("full_run_results/distill_run.json"))
e, r2, b = run["eval"], run.get("rerun") or {}, run["baseline"]
g = run.get("groups") or {}
air = run.get("airline") or {}
an, ab = air.get("new") or {}, air.get("base") or {}
C_NEW, C_RERUN, C_OLD = "#1E7A5C", "#E0883B", "#2F5D8A"
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# retail: running pass^1, all runs on the SAME axes
ax = axes[0][0]
if e["running_pass1"]:
    ax.plot(range(1, len(e["running_pass1"]) + 1), e["running_pass1"], color=C_NEW, label=f"new model ({e['rollouts_done']}/{e['rollouts_total']})")
if r2.get("running_pass1"):
    ax.plot(range(1, len(r2["running_pass1"]) + 1), r2["running_pass1"], color=C_RERUN, label=f"starting adapter, re-run ({r2['rollouts_done']}/{r2['rollouts_total']})")
if b and b.get("running_pass1"):
    ax.plot(range(1, len(b["running_pass1"]) + 1), b["running_pass1"], color=C_OLD, label=f"starting adapter, earlier test ({len(b['running_pass1'])}/230)")
elif b and b.get("pass1") is not None:
    ax.axhline(b["pass1"], color=C_OLD, ls="--", lw=1, label="starting adapter, earlier test (final)")
ax.set_ylim(0, 1)
ax.set_xlabel("rollouts finished (completion order; quick tasks finish first)")
ax.set_ylabel("running pass^1")
ax.set_title("Retail test: running pass rate")
ax.legend(fontsize=8)

# retail: overall bars
ax = axes[0][1]
sources = [("earlier test", C_OLD, b or {}), ("re-run, same pod", C_RERUN, r2), ("new model", C_NEW, e)]
shown = [(n, c, d) for n, c, d in sources if d.get("pass1") is not None]
if shown:
    w = 0.8 / len(shown)
    for j, (n, c, d) in enumerate(shown):
        for i, k in enumerate(("pass1", "pass2", "pass3", "pass4")):
            x = i - 0.4 + w * (j + 0.5)
            if d.get(k) is not None:
                ax.bar(x, d[k], w, yerr=d.get(k + "_se"), capsize=3, color=c, label=n if i == 0 else None)
                ax.text(x, d[k] + 0.05, f"{d[k]:.0%}", ha="center", fontsize=8)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["pass^1", "pass^2", "pass^3", "pass^4"])
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "no retail test results yet", ha="center", va="center"); ax.set_axis_off()
ax.set_ylim(0, 1)
ax.set_title("Retail overall (bars: +/-1 standard error)")

# airline: bars
ax = axes[0][2]
if an.get("pass1") is not None or ab.get("pass1") is not None:
    for j, (n, c, d) in enumerate((("starting adapter", C_OLD, ab), ("new model", C_NEW, an))):
        for i, k in enumerate(("pass1", "pass2")):
            x = i - 0.4 + 0.4 * (j + 0.5)
            if d.get(k) is not None:
                ax.bar(x, d[k], 0.4, yerr=d.get(k + "_se"), capsize=3, color=c, label=n if i == 0 else None)
                ax.text(x, d[k] + 0.05, f"{d[k]:.0%}", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pass^1", "pass^2"])
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "airline test not started yet", ha="center", va="center"); ax.set_axis_off()
ax.set_ylim(0, 1)
ax.set_title("Airline, 50 tasks (bars: +/-1 standard error)")

# retail by task type
for ax, k, title in ((axes[1][0], "pass1", "Retail pass^1 by task type"), (axes[1][1], "pass2", "Retail pass^2 by task type"),
                     (axes[2][0], "pass3", "Retail pass^3 by task type (only tasks with 3+ trials done)"), (axes[2][1], "pass4", "Retail pass^4 by task type (only tasks with 4 trials done)")):
    srcs = [(n, c, g.get(key)) for n, c, key in (("earlier test", C_OLD, "earlier"), ("re-run, same pod", C_RERUN, "rerun"), ("new model", C_NEW, "new")) if g.get(key)]
    if srcs:
        w = 0.8 / len(srcs)
        for j, (n, c, d) in enumerate(srcs):
            for i, kind in enumerate(("combo", "other")):
                v = d[kind][k]
                if v is not None:
                    x = i - 0.4 + w * (j + 0.5)
                    ax.bar(x, v, w, color=c, label=n if i == 0 else None)
                    ax.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["combo tasks (64)", "other tasks (51)"])
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no results yet", ha="center", va="center"); ax.set_axis_off()
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)

# airline running pass^1
ax = axes[1][2]
for n, c, d in (("new model", C_NEW, an), ("starting adapter", C_OLD, ab)):
    if d.get("running_pass1"):
        ax.plot(range(1, len(d["running_pass1"]) + 1), d["running_pass1"], color=c, label=f"{n} ({d['rollouts_done']}/100)")
ax.set_ylim(0, 1)
ax.set_xlabel("rollouts finished")
ax.set_ylabel("running pass^1")
ax.set_title("Airline test: running pass rate")
if ax.get_legend_handles_labels()[0]:
    ax.legend(fontsize=8)

axes[2][2].set_axis_off()
plt.tight_layout()
plt.show()
"""


def md(src):
    return {"cell_type": "markdown", "metadata": {"section": "distill"}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {"section": "distill"}, "outputs": [], "source": src.splitlines(True)}


nb = json.load(open(NB))
cells = [c for c in nb["cells"] if c.get("metadata", {}).get("section") != "distill"]
at = len(cells) - 1 if cells and not "".join(cells[-1]["source"]).strip() else len(cells)
cells[at:at] = [md(HEADER), md(TRAIN_HEADING), code(TRAIN_STATUS), code(TRAIN_CHART), md(EVAL_HEADING), code(EVAL_STATUS), code(EVAL_CHART)]
nb["cells"] = cells
tmp = NB + ".tmp"
json.dump(nb, open(tmp, "w"), indent=1, ensure_ascii=False)
os.replace(tmp, NB)
print("Section 12 inserted;", len(cells), "cells in the notebook")
