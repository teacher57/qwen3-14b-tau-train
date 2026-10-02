"""Keep Section 10 (pass^2 test of gentle round 5 vs the starting adapter) live in the notebook.

Every INTERVAL seconds: pull the pod's per-rollout result files, rebuild full_run_results/passk_run.json
(with an ETA), and rewrite the three auto cells of Section 10 (header / chart with embedded image / status).
Exits after both runs are finished.

Usage: ./.venv-nb/bin/python live_passk_section.py [--once] [--interval 30]
Single writer: do not run another notebook updater at the same time.
"""
import base64, datetime, io, json, math, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import update_notebook as un

HERE, NB, HOST, PORT, KEY = un.HERE, un.NB, un.HOST, un.PORT, un.KEY
FR = os.path.join(HERE, "full_run_results")
RUN_JSON = os.path.join(FR, "passk_run.json")
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 30
N_TASKS, TRIALS, TEMP = 115, 2, 0.7
TOTAL = N_TASKS * TRIALS
MODELS = [("gentle5", "gentle round 5 (combo-pool GRPO, lr 1e-6)"), ("baseline", "starting adapter (control)")]
PRIOR_MIN = (95, 145)  # minutes per model before any pace is measured (230 rollouts at 1.6-2.4 per minute)
REF_PASS1_T0 = 0.452   # earlier full eval of the un-tuned Qwen3-14B-AWQ at temperature 0, one trial


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", PORT, "-i", KEY, HOST, cmd],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.stdout if r.returncode == 0 else None


def pull(model):
    out = ssh(f"cat /root/passk_{model}.jsonl 2>/dev/null; true")
    return None if out is None else [json.loads(l) for l in out.splitlines() if l.strip()]


def per_task(rows):
    by = {}
    for r in rows:
        by.setdefault(r["task_index"], []).append(r["reward"] >= 1)
    return by


def model_state(model, label, rows, running, finished, prev):
    rows = rows if rows is not None else prev.get("rows", [])
    done = len(rows)
    by = per_task(rows)
    two = {t: v for t, v in by.items() if len(v) >= TRIALS}
    passed = sum(1 for r in rows if r["reward"] >= 1)
    st = {"model": model, "label": label, "done": done, "passed": passed, "total": TOTAL,
          "rows": [{"task_index": r["task_index"], "trial": r["trial"], "reward": r["reward"], "seconds": r["seconds"],
                    "finished_at": r["finished_at"], "error": r.get("error")} for r in rows],
          "errors": sum(1 for r in rows if r.get("error")), "running": model in running, "finished": bool(finished)}
    st["pass1"] = passed / done if done else None
    st["pass1_se"] = math.sqrt(st["pass1"] * (1 - st["pass1"]) / done) if done else None
    st["tasks_both_done"] = len(two)
    st["both_pass"] = sum(1 for v in two.values() if all(v))
    st["pass2"] = st["both_pass"] / len(two) if two else None
    st["pass2_se"] = math.sqrt(st["pass2"] * (1 - st["pass2"]) / len(two)) if two else None
    # pass^1 on the same tasks, so the ratio compares like with like
    p1_same = sum(sum(v) for v in two.values()) / (TRIALS * len(two)) if two else None
    st["pass1_same_tasks"] = p1_same
    st["consistency"] = (st["pass2"] / p1_same) if (two and p1_same) else None
    if done:
        start = rows[0]["started_run_at"]
        last = max(r["finished_at"] for r in rows)
        st["elapsed_min"] = (last - start) / 60
        if done >= 10 and st["elapsed_min"] > 0:
            st["rollouts_per_min"] = done / st["elapsed_min"]
            st["eta_min"] = 0 if finished else (TOTAL - done) / st["rollouts_per_min"]
    return st


def paired_pass2(a, b):
    ta = {t: all(v) for t, v in per_task([r for r in a["_raw"]]).items() if len(per_task(a["_raw"])[t]) >= TRIALS}
    tb = {t: all(v) for t, v in per_task([r for r in b["_raw"]]).items() if len(per_task(b["_raw"])[t]) >= TRIALS}
    common = sorted(set(ta) & set(tb))
    up = sum(1 for t in common if ta[t] and not tb[t])
    down = sum(1 for t in common if tb[t] and not ta[t])
    n = up + down
    p = 1.0
    if n:
        k = min(up, down)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n_common": len(common), "gentle5_only": up, "baseline_only": down, "p_value": p}


def build(prev):
    now = datetime.datetime.now().astimezone()
    procs = ssh("ps -eo args | grep '[f]ull_passk.py' | awk '{print $3}'; echo ---; "
                "for m in gentle5 baseline; do grep -q PASSK_DONE /root/passk_$m.log 2>/dev/null && echo done_$m; done; true")
    prev_models = {m["model"]: m for m in prev.get("models", [])}
    reachable = procs is not None
    if reachable:
        head, _, tail = procs.partition("---")
        running = {x.strip() for x in head.split() if x.strip()}
        marks = {x.strip()[5:] for x in tail.split() if x.strip().startswith("done_")}
    else:
        running, marks = set(prev.get("_running", [])), set(prev.get("_done", []))
    models, raws = [], {}
    for m, label in MODELS:
        rows = pull(m) if reachable else None
        st = model_state(m, label, rows, running, m in marks, prev_models.get(m, {}))
        raws[m] = rows if rows is not None else []
        models.append(st)
    g, b = models
    if g.get("eta_min") is not None:
        g_total = g["elapsed_min"] + g["eta_min"]
        b_eta = b["eta_min"] if b.get("eta_min") is not None else (0 if b["finished"] else g_total)
        eta = {"gentle5_min": g["eta_min"], "both_min": g["eta_min"] + b_eta}
    else:
        eta = {"gentle5_min": None, "both_min": None, "prior_per_model_min": list(PRIOR_MIN)}
    comp = None
    if g["tasks_both_done"] and b["tasks_both_done"]:
        comp = paired_pass2({"_raw": raws["gentle5"]}, {"_raw": raws["baseline"]})
    return {"updated": now.isoformat(timespec="seconds"), "last_update": now.isoformat(timespec="seconds"),
            "last_update_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_tasks": N_TASKS, "trials": TRIALS, "temperature": TEMP, "total_rollouts": TOTAL,
            "reference_pass1_t0": REF_PASS1_T0, "models": models, "eta": eta, "comparison": comp,
            "all_done": all(m["finished"] for m in models), "pod_unreachable": not reachable,
            "_running": sorted(running), "_done": sorted(marks)}


CHART_SRC = """import json
import matplotlib.pyplot as plt

run = json.load(open("full_run_results/passk_run.json"))
TOTAL = run["total_rollouts"]
colors = {"gentle5": "#1E7A5C", "baseline": "#2F5D8A"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

for m in run["models"]:
    axes[0].barh([m["model"]], [m["done"]], color=colors[m["model"]])
    axes[0].text(m["done"] + 3, m["model"], f"{m['done']}/{TOTAL}", va="center")
axes[0].set_xlim(0, TOTAL * 1.18)
axes[0].set_xlabel("rollouts finished (115 tasks x 2 trials)")
axes[0].set_title("Progress")

for m in run["models"]:
    rows = sorted(m["rows"], key=lambda r: r["finished_at"])
    if rows:
        cum, ok = [], 0
        for i, r in enumerate(rows, 1):
            ok += r["reward"] >= 1
            cum.append(ok / i)
        axes[1].plot(range(1, len(rows) + 1), cum, color=colors[m["model"]], label=m["model"])
axes[1].axhline(run["reference_pass1_t0"], color="#888", ls="--", lw=1, label="earlier AWQ baseline, temp 0 (45.2%)")
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("rollouts finished (completion order)")
axes[1].set_ylabel("running pass^1")
axes[1].set_title("Running pass^1 (quick tasks finish first)")
axes[1].legend(fontsize=8)

xs, w = 0, 0.38
labels = []
for i, m in enumerate(run["models"]):
    if m["pass1"] is None:
        continue
    axes[2].bar(i - w / 2, m["pass1"], w, yerr=m["pass1_se"], capsize=3, color=colors[m["model"]], alpha=0.55)
    axes[2].text(i - w / 2, m["pass1"] + 0.07, f"{m['pass1']:.0%}", ha="center", fontsize=9)
    if m["pass2"] is not None:
        axes[2].bar(i + w / 2, m["pass2"], w, yerr=m["pass2_se"], capsize=3, color=colors[m["model"]])
        axes[2].text(i + w / 2, m["pass2"] + 0.07, f"{m['pass2']:.0%}", ha="center", fontsize=9)
axes[2].set_xticks(range(len(run["models"])))
axes[2].set_xticklabels([m["model"] + "\\n(light: pass^1, solid: pass^2)" for m in run["models"]], fontsize=8)
axes[2].set_ylim(0, 1)
axes[2].set_ylabel("rate (bars: +/-1 standard error)")
axes[2].set_title("pass^1 vs pass^2 so far")

plt.tight_layout()
plt.show()
"""

STATUS_SRC = """import datetime, html, json, time
from IPython.display import HTML, clear_output, display

RUN = "full_run_results/passk_run.json"
LIVE = False  # True: keep redrawing this cell every 30 s until both runs are done (interrupt to stop)


def fmt_min(x):
    if x is None:
        return "estimating..."
    h, m = divmod(int(round(x)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def pct(x, se=None):
    if x is None:
        return "-"
    return f"{x:.1%}" + (f" +/- {se:.1%}" if se is not None else "")


def render_html(run):
    td = "border:1px solid #8886;padding:3px 10px;text-align:left"
    def table(head, body):
        h = "".join(f"<th style='{td}'>{c}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td style='{td}'>{c}</td>" for c in row) + "</tr>" for row in body)
        return f"<table style='border-collapse:collapse;margin:6px 0 14px'><tr>{h}</tr>{b}</table>"
    rows = []
    for m in run["models"]:
        state = "finished" if m["finished"] else ("running" if m["running"] else "waiting")
        rows.append([html.escape(m["label"]), state, f"{m['done']} / {run['total_rollouts']}",
                     f"{m['tasks_both_done']} / {run['n_tasks']}", pct(m["pass1"], m["pass1_se"]),
                     pct(m["pass2"], m["pass2_se"]),
                     f"{m['consistency']:.2f}" if m["consistency"] is not None else "-",
                     f"{m['rollouts_per_min']:.2f}" if m.get("rollouts_per_min") else "-",
                     "done" if m["finished"] else fmt_min(m.get("eta_min")), m["errors"]])
    out = [f"<h4>pass^2 test: live status (data written {html.escape(run['last_update'])} = {html.escape(run['last_update_utc'])})</h4>"]
    out.append(table(["model", "state", "rollouts", "tasks with both trials", "pass^1", "pass^2", "consistency (pass^2 / pass^1)",
                      "rollouts/min", "time left", "errors"], rows))
    eta = run["eta"]
    if eta.get("both_min") is not None:
        out.append(f"<p><b>Estimated time to finish:</b> round 5 in about {fmt_min(eta['gentle5_min'])}; "
                   f"both runs in about {fmt_min(eta['both_min'])} (the control is assumed to take as long as round 5).</p>")
    else:
        lo, hi = eta["prior_per_model_min"]
        out.append(f"<p><b>Estimated time to finish:</b> the pace is not measured yet (needs about 10 finished rollouts); "
                   f"expect {lo}-{hi} min per model, so {lo * 2}-{hi * 2} min for both. Early finishers are the quick tasks, "
                   f"so the first estimates run optimistic.</p>")
    c = run.get("comparison")
    if c:
        out.append(f"<p><b>pass^2, round 5 vs control, on the {c['n_common']} tasks both have finished:</b> "
                   f"{c['gentle5_only']} tasks only round 5 passes twice, {c['baseline_only']} only the control does "
                   f"(paired test p = {c['p_value']:.2f}).</p>")
    out.append(f"<p style='opacity:.7'>Each task is run {run['trials']} times at temperature {run['temperature']}. pass^2 = share of tasks "
               f"where both trials pass; consistency = pass^2 / pass^1 on the same tasks (1.0 would mean fully repeatable). "
               f"Reference: the earlier un-tuned Qwen3-14B-AWQ full eval scored {run['reference_pass1_t0']:.1%} at temperature 0, one trial "
               f"(different setup, a rough guide only).</p>")
    return "".join(out)


# --- run ---
if LIVE:
    while True:
        run = json.load(open(RUN))
        clear_output(wait=True)
        display(HTML(render_html(run)))
        if run["all_done"]:
            break
        time.sleep(30)
else:
    display(HTML(render_html(json.load(open(RUN)))))
"""

HEADER_MD = """## Section 10: pass^2 test of gentle round 5 (live)

The 20-task probe was too noisy to say whether the gentle run's round-5 checkpoint (0.60) is really better than the starting
adapter (0.40; the same adapter scored 0.55 and 0.50 in earlier setups), and it measures one attempt per task. This section
measures consistency instead: the full retail test split (all 115 tasks), each task run twice at temperature 0.7, for round 5
and then for the starting adapter as a control with identical settings (tool-calling agent, GPT-4o as the simulated user,
25 steps).

- **pass^1** is the average pass rate over all trials.
- **pass^2** is the share of tasks where both trials pass (the tau-bench definition, C(c,2)/C(n,2) with n = 2).
- **consistency** = pass^2 / pass^1 on the same tasks. Temperature above 0 is needed for this to mean anything: at 0 the two
  trials differ only through the user simulator and GPU arithmetic.

A pass rate over 115 tasks has a standard error of about 4-5 points, so gaps under roughly 10 points should not be read as a
difference. The data file is rewritten every 30 seconds by `live_passk_section.py`; re-run the chart and status cells to redraw
them (set `LIVE = True` in the status cell to have it redraw itself).
"""


def render_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        plt.close("all")
        exec(CHART_SRC.replace("plt.show()", ""), {})
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", dpi=110)
        plt.close("all")
    finally:
        os.chdir(cwd)
    return base64.b64encode(buf.getvalue()).decode()


def status_html(run):
    ns = {}
    exec(STATUS_SRC.split("# --- run ---")[0], ns)
    return ns["render_html"](run)


def update_notebook(run):
    nb = json.load(open(NB))
    cells = nb["cells"]
    want = {
        "passk_header": {"cell_type": "markdown", "metadata": {"auto": "passk_header"}, "source": HEADER_MD.splitlines(True)},
        "passk_chart": {"cell_type": "code", "execution_count": None, "metadata": {"auto": "passk_chart"},
                        "source": CHART_SRC.splitlines(True),
                        "outputs": [{"output_type": "display_data", "metadata": {},
                                     "data": {"image/png": render_png(), "text/plain": ["<Figure>"]}}]},
        "passk_status": {"cell_type": "code", "execution_count": None, "metadata": {"auto": "passk_status"},
                         "source": STATUS_SRC.splitlines(True),
                         "outputs": [{"output_type": "display_data", "metadata": {},
                                      "data": {"text/html": status_html(run).splitlines(True),
                                               "text/plain": ["<pass^2 live status>"]}}]},
    }
    have = {c.get("metadata", {}).get("auto"): i for i, c in enumerate(cells)}
    for key in ("passk_header", "passk_chart", "passk_status"):
        if key in have:
            cells[have[key]] = want[key]
        else:
            at = len(cells) - 1 if not "".join(cells[-1]["source"]).strip() else len(cells)
            cells.insert(at, want[key])
            have = {c.get("metadata", {}).get("auto"): i for i, c in enumerate(cells)}
    tmp = NB + ".tmp"
    json.dump(nb, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, NB)


def once():
    prev = json.load(open(RUN_JSON)) if os.path.exists(RUN_JSON) else {}
    run = build(prev)
    json.dump(run, open(RUN_JSON, "w"), indent=1)
    update_notebook(run)
    g, b = run["models"]
    eta = run["eta"]["both_min"]
    return run, (f"{run['updated']} gentle5={g['done']}/{TOTAL} baseline={b['done']}/{TOTAL} "
                 f"eta_both={'-' if eta is None else round(eta)}min")


if __name__ == "__main__":
    if "--once" in sys.argv:
        print(once()[1])
        sys.exit(0)
    while True:
        try:
            run, msg = once()
            print(msg, flush=True)
            if run["all_done"]:
                print("both runs finished; final update written, exiting", flush=True)
                break
        except Exception as e:
            print("update failed:", type(e).__name__, str(e)[:150], flush=True)
        time.sleep(INTERVAL)
