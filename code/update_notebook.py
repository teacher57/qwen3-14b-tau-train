"""Refresh the combo pass-rate diagnostic in the notebook.

1. scp /root/combo_passrate.jsonl from the pod and save it as
   full_run_results/combo_passrate.json: {updated, summary, per_task[], rollouts[]}
   (per_task carries instruction, target tool calls, rewards, pass rate, category).
2. Recompute stats (rollouts, coverage, mean reward, never/mixed/always-pass tasks).
3. Rewrite the auto-generated status cell in Section 7 of the notebook
   (cell metadata {"auto": "combo_status"}; created right after the Section 7 chart if missing).

Usage: python3 update_notebook.py [--no-pull]
Env overrides: POD_HOST, POD_PORT, POD_KEY.
"""
import base64, collections, datetime, io, json, os, subprocess, sys

VENV_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv-nb", "bin", "python")
try:
    import matplotlib  # noqa: F401
except ImportError:
    if os.path.exists(VENV_PY) and sys.executable != VENV_PY:
        os.execv(VENV_PY, [VENV_PY] + sys.argv)
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "qwen3_14b_training_and_rollouts.ipynb")
OUT = os.path.join(HERE, "full_run_results", "combo_passrate.json")
RAW = OUT + ".pull.jsonl"
TASKS = os.path.join(HERE, "full_run_results", "combo_tasks.json")
HOST = os.environ.get("POD_HOST", "root@194.68.245.68")
PORT = os.environ.get("POD_PORT", "22050")
KEY = os.path.expanduser(os.environ.get("POD_KEY", "~/.ssh/id_ed25519"))
TARGET_PER_TASK = 4
N_TASKS = 153


CHART_SRC = """import json, collections
import matplotlib.pyplot as plt

sources = ["test split", "train split", "passing rollouts", "GRPO task slots"]
combo = [64, 153, 73, 16]
total = [115, 500, 296, 52]
shares = [100 * c / t for c, t in zip(combo, total)]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].bar(sources, shares, color=["#A33B3B", "#2F5D8A", "#6B4FA0", "#1E7A5C"])
for i, (sh, c, t) in enumerate(zip(shares, combo, total)):
    axes[0].text(i, sh + 1, f"{sh:.1f}%\\n({c}/{t})", ha="center", fontsize=9)
axes[0].set_ylim(0, 70)
axes[0].set_ylabel("share of tasks that are combos (%)")
axes[0].set_title("Combo share by source")
axes[0].tick_params(axis="x", labelrotation=15)

try:
    data = json.load(open("full_run_results/combo_passrate.json"))
    recs = data["rollouts"] if isinstance(data, dict) else data
    per_task = collections.defaultdict(list)
    for r in recs:
        per_task[r["task_index"]].append(r["reward"])
    counted = sum(min(len(v), 4) for v in per_task.values())
    axes[1].barh(["rollouts counted\\n(target 612)", "tasks fully done\\n(target 153)"],
                 [counted / 612 * 100, sum(len(v) >= 4 for v in per_task.values()) / 153 * 100],
                 color="#1E7A5C")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("% of the 4-per-task diagnostic")
    axes[1].set_title(f"Diagnostic progress: {counted}/612 rollouts")
    rates = [sum(v) / len(v) for v in per_task.values() if len(v) >= 4]
    axes[2].hist(rates, bins=[i / 4 - 0.125 for i in range(6)], color="#1E7A5C", edgecolor="white")
    axes[2].set_xlabel("per-task pass rate (round-0 adapter, temp=1.0)")
    axes[2].set_ylabel("number of combo train tasks")
    mixed = sum(0 < r < 1 for r in rates)
    axes[2].set_title(f"Finished tasks: {mixed}/{len(rates)} mixed pass/fail")
except FileNotFoundError:
    for ax in axes[1:]:
        ax.text(0.5, 0.5, "pass-rate diagnostic not saved yet\\n(full_run_results/combo_passrate.json)",
                ha="center", va="center")
        ax.set_axis_off()

plt.tight_layout()
plt.show()
"""

def pull():
    r = subprocess.run(
        ["scp", "-q", "-o", "ConnectTimeout=15", "-P", PORT, "-i", KEY,
         f"{HOST}:/root/combo_passrate.jsonl", RAW],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("pull failed, using the last saved copy:", r.stderr.strip()[:150])
        return
    recs = [json.loads(l) for l in open(RAW) if l.strip()]
    os.remove(RAW)
    json.dump(build_data(recs), open(OUT, "w"), indent=1)
    print(f"saved {len(recs)} rollouts to {OUT}")


PROBE_LOG = os.path.join(HERE, "full_run_results", "grpo_eval_probe_log.json")
PROBE_RUN = "combo_grpo"


def sync_probe_log():
    """Merge the pod's probe results (/root/grpo_eval_probe.log.jsonl) into
    grpo_eval_probe_log.json, tagged run=PROBE_RUN. Other runs' entries are left untouched."""
    raw = PROBE_LOG + ".pull.jsonl"
    r = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=15", "-P", PORT, "-i", KEY,
                        f"{HOST}:/root/grpo_eval_probe.log.jsonl", raw], capture_output=True, text=True)
    if r.returncode != 0:
        return  # no probes on the pod yet (or pod unreachable)
    new = [dict(json.loads(l), run=PROBE_RUN) for l in open(raw) if l.strip()]
    os.remove(raw)
    old = json.load(open(PROBE_LOG))
    kept = [e for e in old if e.get("run") != PROBE_RUN]
    json.dump(kept + new, open(PROBE_LOG, "w"), indent=1)
    print(f"probe log: {len(new)} {PROBE_RUN} entries merged")


def categorize(rewards):
    n, k = len(rewards), sum(rewards)
    if 0 < k < n:
        return "mixed"
    base = "never" if k == 0 else "always"
    return base if n >= TARGET_PER_TASK else base + f" (so far, n={n})"


def build_data(recs):
    tasks = {t["idx"]: t for t in json.load(open(TASKS))}
    by = collections.defaultdict(list)
    for r in sorted(recs, key=lambda r: r["rep"]):
        by[r["task_index"]].append(r)
    per_task = []
    for tid in sorted(by):
        rs = by[tid]
        rewards = [r["reward"] for r in rs]
        t = tasks.get(tid, {})
        per_task.append({
            "task_index": tid,
            "n_rollouts": len(rewards),
            "rewards": rewards,
            "pass_rate": sum(rewards) / len(rewards),
            "category": categorize(rewards),
            "errors": [r["error"] for r in rs if "error" in r],
            "instruction": t.get("instruction"),
            "target_actions": t.get("actions"),
        })
    full = [p for p in per_task if p["n_rollouts"] >= TARGET_PER_TASK]
    summary = {
        "rollouts": len(recs),
        "counted_toward_target": sum(min(p["n_rollouts"], TARGET_PER_TASK) for p in per_task),
        "target": N_TASKS * TARGET_PER_TASK,
        "tasks_started": len(per_task),
        "tasks_with_all_4": len(full),
        "errors": sum("error" in r for r in recs),
        "mean_reward": sum(r["reward"] for r in recs) / len(recs) if recs else 0.0,
        "tasks_mixed_so_far": sum(p["category"] == "mixed" for p in per_task),
        "tasks_always_fail_so_far": sum(p["category"].startswith("never") for p in per_task),
        "tasks_always_pass_so_far": sum(p["category"].startswith("always") for p in per_task),
    }
    return {"updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "summary": summary, "per_task": per_task, "rollouts": recs}


def load_rollouts():
    data = json.load(open(OUT))
    return data["rollouts"] if isinstance(data, dict) else data


def task_table_markdown():
    data = json.load(open(OUT))
    if not isinstance(data, dict):
        return []
    order = {"mixed": 0}
    rows = sorted(data["per_task"], key=lambda p: (order.get(p["category"], 1), p["category"], -p["pass_rate"], p["task_index"]))
    out = ["\n<details><summary>All tasks measured so far (%d), mixed first</summary>\n\n" % len(rows),
           "| task | rewards | pass rate | category | target tool calls | instruction |\n|---|---|---|---|---|---|\n"]
    for p in rows:
        acts = ", ".join(a["name"] for a in (p["target_actions"] or [])) or "-"
        instr = (p["instruction"] or "").replace("|", "/").replace("\n", " ")
        instr = (instr[:110] + "...") if len(instr) > 110 else instr
        rw = "".join("1" if r else "0" for r in p["rewards"])
        out.append(f"| {p['task_index']} | {rw} | {p['pass_rate']:.2f} | {p['category']} | {acts} | {instr} |\n")
    out.append("\n</details>\n")
    return out


def stats():
    recs = load_rollouts()
    per = collections.defaultdict(list)
    for r in recs:
        per[r["task_index"]].append(r["reward"])
    counted = sum(min(len(v), TARGET_PER_TASK) for v in per.values())
    full = {t: v for t, v in per.items() if len(v) >= TARGET_PER_TASK}
    rates = {t: sum(v) / len(v) for t, v in full.items()}
    return {
        "rollouts": len(recs),
        "counted": counted,
        "target": N_TASKS * TARGET_PER_TASK,
        "tasks_started": len(per),
        "tasks_full": len(full),
        "errors": sum("error" in r for r in recs),
        "mean": sum(r["reward"] for r in recs) / len(recs) if recs else 0.0,
        "never": sum(r == 0 for r in rates.values()),
        "mixed": sum(0 < r < 1 for r in rates.values()),
        "always": sum(r == 1 for r in rates.values()),
    }


def status_markdown(s):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pct = 100 * s["counted"] / s["target"]
    lines = [
        f"### Combo pass-rate diagnostic: live status (auto-updated {now})\n",
        "\n",
        "| | |\n|---|---|\n",
        f"| rollouts saved | {s['rollouts']} ({s['errors']} errors) |\n",
        f"| counted toward {TARGET_PER_TASK}/task target | {s['counted']} / {s['target']} ({pct:.0f}%) |\n",
        f"| tasks with all {TARGET_PER_TASK} rollouts | {s['tasks_full']} / {N_TASKS} |\n",
        f"| mean reward so far | {s['mean']:.3f} |\n",
    ]
    if s["tasks_full"]:
        lines += [
            f"| of finished tasks: never pass / mixed / always pass | "
            f"{s['never']} / {s['mixed']} / {s['always']} |\n",
        ]
    lines += task_table_markdown()
    lines.append("\nGenerated by `update_notebook.py`; the chart above reads `full_run_results/combo_passrate.json` (per-task summaries + raw rollouts).\n")
    return lines



def render_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        plt.close("all")
        ns = {}
        exec(CHART_SRC.replace("plt.show()", ""), ns)
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", dpi=110)
        plt.close("all")
    finally:
        os.chdir(cwd)
    return base64.b64encode(buf.getvalue()).decode()

def update_notebook(s):
    nb = json.load(open(NB))
    cells = nb["cells"]
    new = {"cell_type": "markdown", "metadata": {"auto": "combo_status"},
           "source": status_markdown(s)}
    for i, c in enumerate(cells):
        if c.get("metadata", {}).get("auto") == "combo_status":
            cells[i] = new
            break
    else:
        # insert right after the Section 7 chart cell (the code cell following the Section 7 markdown)
        idx = next((i for i, c in enumerate(cells)
                    if c["cell_type"] == "markdown" and "## Section 7" in "".join(c["source"])), None)
        if idx is None:
            sys.exit("Section 7 not found in notebook")
        cells.insert(idx + 2, new)
    # the code cell right after the Section 7 markdown is the chart: rewrite its source and embed the render
    m = next(i for i, c in enumerate(cells)
             if c["cell_type"] == "markdown" and "## Section 7" in "".join(c["source"]))
    chart = cells[m + 1]
    assert chart["cell_type"] == "code", "expected the Section 7 chart code cell"
    chart["source"] = CHART_SRC.splitlines(True)
    chart["outputs"] = [{"output_type": "display_data", "metadata": {},
                         "data": {"image/png": render_chart(), "text/plain": ["<Figure size 1760x462 with 3 Axes>"]}}]
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    if "--no-pull" not in sys.argv:
        pull()
        sync_probe_log()
    s = stats()
    update_notebook(s)
    print(f"notebook updated: {s['counted']}/{s['target']} counted, mean {s['mean']:.3f}, "
          f"never/mixed/always = {s['never']}/{s['mixed']}/{s['always']}")
