"""Insert (or replace) notebook Section 11, the 32B teacher run. The cells only READ full_run_results/teacher_run.json
and display it when run; nothing rewrites the notebook afterwards. Idempotent.
Usage: python3 add_notebook_section.py
"""
import json, os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen3_14b_training_and_rollouts.ipynb")

HEADER = """## Section 11: a 32B teacher on the hard combo tasks (stage 1 of distillation)

Plan: let Qwen3-32B-AWQ play every combo train task that the 14B does not reliably solve (up to 4 samples each, temperature 0.7;
a task stops being sampled once it has 2 passes), keep the passing conversations, and fine-tune the 14B on them. "Does not reliably
solve" comes from our earlier diagnostic of the 153 combo train tasks: 58 never solved and 56 solved sometimes (114 tasks, all
included); the 39 the 14B always solves are skipped. They run one after the other, never in parallel: first the 65 from the original
unsolved list (46 never + 19 sometimes), then 49 more (12 never + 37 sometimes). Stage 1 is the go/no-go check on the never-solved
tasks: if the 32B solves under about 30% of the 58 there is too little hard data to train on.

(The original list over-counted because 126 of the 422 first-run rollouts had no recorded task id.)

The cells below only read `full_run_results/teacher_run.json` (written with timestamps by `teacher/update_teacher_json.py`);
re-run them to redraw. Set `LIVE = True` in the status cell to have it redraw itself every 30 seconds.
"""

STATUS = """import datetime, html, json, time
from IPython.display import HTML, clear_output, display

RUN = "full_run_results/teacher_run.json"
LIVE = False  # True: re-read the JSON and redraw every 30 s until the run finishes (interrupt to stop)


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
    s, ts = run["summary"], run["timestamps"]
    out = [f"<h4>32B teacher run (JSON written {html.escape(run['updated'])} = {html.escape(run['updated_utc'])})</h4>"]
    verdict = "-"
    if s["tasks_with_all_trials"] >= 0.5 * s["n_tasks"] or run["stage"] == "finished":
        verdict = "GO" if s["solved_fraction"] >= s["go_no_go_threshold"] else "NO-GO (too little data)"
    out.append(table(["", ""], [
        ["stage", html.escape(run["stage"]) + ("" if run["pod_reachable"] else " (pod unreachable, showing last known state)")],
        ["rollouts", f"{s['rollouts_done']} run, up to {s['total_rollouts']} ({s['rollouts_passed']} passed, {s['errors']} errors, "
                     f"{s.get('rollouts_skipped', 0)} skipped because the task already had {s.get('stop_after_passes', 2)} passes)"],
        ["tasks solved at least once", f"{s['tasks_solved']} / {s['n_tasks']}; go/no-go on {s.get('go_no_go_basis', 'all tasks')}: {s['solved_fraction']:.0%} (threshold {s['go_no_go_threshold']:.0%})"],
        ["verdict (shown once half the hard tasks are complete)", verdict],
        ["pace", f"{s['rollouts_per_min']:.2f} rollouts/min" if s.get("rollouts_per_min") else "not measured yet (needs 10 finished rollouts)"],
        ["time left (upper bound: skipped trials shorten it)", "done" if run["stage"] == "finished" else fmt_min(s.get("eta_min"))],
    ]))
    out.append("<b>Timestamps</b>" + table(["event", "time (local)"], [
        [k.replace("_", " "), html.escape(ts.get(k, "-"))] for k in ("setup_seen", "setup_done", "vllm_ready", "rollouts_started", "teacher_done")]
        + [["last rollout finished", html.escape(run["rollout_finish_times"][-1]) if run["rollout_finish_times"] else "-"]]))
    g = run["groups_14b"]
    out.append("<b>By how the 14B did on these tasks in the earlier diagnostic</b>" + table(
        ["14B group", "tasks", "solved by the 32B", "rollouts passed / done"],
        [[k, g[k]["n_tasks"], f"{g[k]['solved']} ({g[k]['solved'] / g[k]['n_tasks']:.0%})" if g[k]["n_tasks"] else "-",
          f"{g[k]['rollouts_passed']} / {g[k]['rollouts_done']}"] for k in ("never", "mixed", "always")]))
    return "".join(out)


# --- run ---
if LIVE:
    while True:
        run = json.load(open(RUN))
        clear_output(wait=True)
        display(HTML(render_html(run)))
        if run["stage"] == "finished":
            break
        time.sleep(30)
else:
    display(HTML(render_html(json.load(open(RUN)))))
"""

CHART = """import json
import matplotlib.pyplot as plt

run = json.load(open("full_run_results/teacher_run.json"))
s, g = run["summary"], run["groups_14b"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

axes[0].barh(["rollouts"], [s["rollouts_done"]], color="#1E7A5C")
axes[0].barh(["rollouts"], [s["rollouts_passed"]], color="#2F5D8A")
axes[0].set_xlim(0, s["total_rollouts"])
axes[0].set_xlabel(f"of {s['total_rollouts']} (blue: passed)")
axes[0].set_title("Progress")

names = ["never", "mixed", "always"]
solved = [g[k]["solved"] for k in names]
unsolved = [g[k]["tasks_with_any_rollout"] - g[k]["solved"] for k in names]
pending = [g[k]["n_tasks"] - g[k]["tasks_with_any_rollout"] for k in names]
axes[1].bar(names, solved, color="#2F5D8A", label="solved by the 32B")
axes[1].bar(names, unsolved, bottom=solved, color="#A33B3B", label="not solved yet")
axes[1].bar(names, pending, bottom=[a + b for a, b in zip(solved, unsolved)], color="#8888", label="not started")
axes[1].set_ylabel("tasks")
axes[1].set_title("Tasks by 14B group (earlier diagnostic)")
axes[1].legend(fontsize=8)

counts = [sum(1 for p in run["per_task"] if p["n_done"] >= 1 and p["passes"] == k) for k in range(s["trials"] + 1)]
axes[2].bar([str(k) for k in range(s["trials"] + 1)], counts, color="#1E7A5C")
axes[2].set_xlabel("passing rollouts per task (of 4)")
axes[2].set_ylabel("tasks")
axes[2].set_title("How often each task is solved")

plt.tight_layout()
plt.show()
"""


def md(src):
    return {"cell_type": "markdown", "metadata": {"section": "teacher"}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {"section": "teacher"}, "outputs": [], "source": src.splitlines(True)}


nb = json.load(open(NB))
cells = [c for c in nb["cells"] if c.get("metadata", {}).get("section") != "teacher"]
at = len(cells) - 1 if cells and not "".join(cells[-1]["source"]).strip() else len(cells)
cells[at:at] = [md(HEADER), code(STATUS), code(CHART)]
nb["cells"] = cells
tmp = NB + ".tmp"
json.dump(nb, open(tmp, "w"), indent=1, ensure_ascii=False)
os.replace(tmp, NB)
print("Section 11 inserted;", len(cells), "cells in the notebook")
