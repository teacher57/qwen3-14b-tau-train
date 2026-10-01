"""Keep Section 8 (combo GRPO run) of the notebook live.

Every INTERVAL seconds: pull the pod's training log, stdout and probe log, rebuild
full_run_results/combo_grpo_run.json, and rewrite the three auto cells of Section 8
(header / chart with embedded image / status tables). Exits after the run has stopped.

Usage: /Users/serjtankian/.claude/jobs/2f730982/tmp/nb_venv/bin/python live_grpo_section.py [--once] [--interval 30]
Single writer: do not run update_notebook.py at the same time (both rewrite the notebook file).
"""
import base64, datetime, io, json, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import update_notebook as un  # reuse pod settings + sync_probe_log

HERE, NB, HOST, PORT, KEY = un.HERE, un.NB, un.HOST, un.PORT, un.KEY
FR = os.path.join(HERE, "full_run_results")
RUN_JSON = os.path.join(FR, "combo_grpo_run.json")
TRAIN_LOG = os.path.join(FR, "combo_grpo_train_log.jsonl")
STDOUT_LOG = os.path.join(FR, "combo_grpo_stdout.log")
PROBE_LOG = un.PROBE_LOG
MAX_ROUNDS = 30
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 30


def scp(remote, local):
    r = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=15", "-P", PORT, "-i", KEY,
                        f"{HOST}:{remote}", local], capture_output=True, text=True)
    return r.returncode == 0


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", PORT, "-i", KEY, HOST, cmd],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.stdout.strip() if r.returncode == 0 else None


def parse_stdout(path):
    rounds, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.match(r"=== round (\d+) tasks=\[(.*?)\] ===", line)
        if m:
            cur = int(m.group(1))
            rounds[cur] = {"round": cur, "tasks": {}, "errors": 0,
                           "task_ids": [int(x) for x in m.group(2).split(",")]}
            continue
        if cur is None:
            continue
        m = re.match(r"\s+task (\d+): rewards=\[(.*?)\] mean=", line)
        if m:
            rounds[cur]["tasks"][int(m.group(1))] = [float(x) for x in m.group(2).split(",")]
            continue
        m = re.match(r"\s+round (\d+): avg_reward=([\d.]+) avg_loss=([-\d.eE]+) n_updated=(\d+)/(\d+)", line)
        if m:
            r = rounds.get(int(m.group(1)))
            if r is not None:
                r.update(avg_reward=float(m.group(2)), avg_loss=float(m.group(3)),
                         n_updated=int(m.group(4)), n_rollouts=int(m.group(5)))
            continue
        if "[rollout error]" in line:
            rounds[cur]["errors"] += 1
    return rounds


def build_run(prev):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    seen = {r["round"]: r.get("seen_at") for r in prev.get("rounds", [])}
    done_at = {r["round"]: r.get("finished_at") for r in prev.get("rounds", [])}
    rounds = parse_stdout(STDOUT_LOG) if os.path.exists(STDOUT_LOG) else {}
    out = []
    for n in sorted(rounds):
        r = rounds[n]
        r["seen_at"] = seen.get(n) or now
        if "avg_reward" in r:
            r["finished_at"] = done_at.get(n) or now
        out.append(r)
    probes = []
    if os.path.exists(PROBE_LOG):
        probes = [e for e in json.load(open(PROBE_LOG)) if e.get("run") == un.PROBE_RUN]
    alive = ssh("pgrep -f '^python3 /root/grpo_combo_train.py' | wc -l; cat /root/STOPPED_ON_DECLINE 2>/dev/null; true")
    if alive is None:  # pod unreachable: keep the last known state rather than call it stopped
        running = prev.get("running", True)
        return {"updated": now, "running": running, "stopped_on_decline": prev.get("stopped_on_decline"),
                "max_rounds": MAX_ROUNDS, "rounds": out, "probes": probes, "pod_unreachable": True}
    running = alive.splitlines()[0].strip() not in ("0", "")
    stopped = (alive.splitlines()[1] if alive and len(alive.splitlines()) > 1 else None)
    return {"updated": now, "running": running, "stopped_on_decline": stopped,
            "max_rounds": MAX_ROUNDS, "rounds": out, "probes": probes}


def durations(run):
    fin = [datetime.datetime.fromisoformat(r["finished_at"]) for r in run["rounds"] if r.get("finished_at")]
    return [(b - a).total_seconds() for a, b in zip(fin, fin[1:])]


CHART_SRC = """import json
import matplotlib.pyplot as plt

run = json.load(open("full_run_results/combo_grpo_run.json"))
rounds = [r for r in run["rounds"] if "avg_reward" in r]
probes = run["probes"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
if rounds:
    xs = [r["round"] for r in rounds]
    axes[0].bar(xs, [r["avg_reward"] for r in rounds], color="#1E7A5C")
    axes[0].set_ylim(0, 1)
axes[0].set_xlabel("round")
axes[0].set_ylabel("avg training reward (16 rollouts)")
axes[0].set_title("Training reward per round (noisy, not the signal)")

if rounds:
    axes[1].bar([r["round"] for r in rounds], [r.get("n_updated", 0) for r in rounds], color="#2F5D8A")
axes[1].set_xlabel("round")
axes[1].set_ylabel("rollouts with a gradient (of 16)")
axes[1].set_title("Learning signal per round")

if probes:
    px = [p["round"] for p in probes]
    py = [p["probe_avg_reward"] for p in probes]
    axes[2].plot(px, py, marker="o", color="#A33B3B")
    for x, y in zip(px, py):
        axes[2].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    axes[2].set_ylim(0, 1)
    axes[2].set_xticks(px)
    axes[2].set_xticklabels(["base\\n(r0 adapter)" if x == -1 else f"r{x}" for x in px])
else:
    axes[2].text(0.5, 0.5, "no probe scores yet", ha="center", va="center")
    axes[2].set_axis_off()
axes[2].set_title("Eval probe: 20 held-out test tasks, temp=0")
axes[2].set_ylabel("probe avg reward")

plt.tight_layout()
plt.show()
"""

STATUS_SRC = """import datetime, html, json, time
from IPython.display import HTML, clear_output, display

RUN = "full_run_results/combo_grpo_run.json"
LIVE = False  # True: keep re-reading and redrawing this cell every 30 s until the trainer stops (interrupt to stop)


def _mark(v):
    return "".join("1" if x else "0" for x in v)


def render_html(run):
    rounds = run["rounds"]
    done = [r for r in rounds if "avg_reward" in r]
    fin = [datetime.datetime.fromisoformat(r["finished_at"]) for r in done if r.get("finished_at")]
    gaps = [(b - a).total_seconds() for a, b in zip(fin, fin[1:])]
    state = ("running" if run["running"] else
             ("stopped on decline: " + run["stopped_on_decline"]) if run.get("stopped_on_decline") else "not running")
    lu, uu = run.get("last_update", ""), run.get("last_update_utc", "")
    best = max(run["probes"], key=lambda p: p["probe_avg_reward"]) if run["probes"] else None
    t = "border-collapse:collapse;margin:6px 0 14px"
    td = "border:1px solid #8886;padding:3px 10px;text-align:left"
    def table(head, body):
        h = "".join(f"<th style='{td}'>{c}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td style='{td}'>{c}</td>" for c in row) + "</tr>" for row in body)
        return f"<table style='{t}'><tr>{h}</tr>{b}</table>"
    out = [f"<h4>Combo GRPO run: live status (data written {html.escape(lu)} = {html.escape(uu)})</h4>"]
    out.append(table(["", ""], [
        ["trainer", state],
        ["rounds finished", f"{len(done)} / {run['max_rounds']} max"],
        ["minutes per round (observed)", f"{sum(gaps) / len(gaps) / 60:.1f}" if gaps else "-"],
        ["rollout errors", sum(r["errors"] for r in rounds)],
        ["best probe so far", f"{best['probe_avg_reward']:.2f} (round {best['round']})" if best else "-"],
    ]))
    out.append("<b>Probe scores</b>" + table(["checkpoint", "probe avg reward"], [
        ["baseline (original round-0 adapter)" if p["round"] == -1 else f"round {p['round']}", f"{p['probe_avg_reward']:.2f}"]
        for p in run["probes"]] or [["none yet", ""]]))
    out.append("<b>Rounds</b>" + table(["round", "tasks (reward per rollout)", "avg reward", "gradient rollouts", "errors"], [
        [r["round"], html.escape("; ".join(f"{k}: {_mark(v)}" for k, v in r["tasks"].items()) or "running..."),
         f"{r['avg_reward']:.2f}" if "avg_reward" in r else "-",
         f"{r['n_updated']}/{r['n_rollouts']}" if "n_updated" in r else "-", r["errors"]] for r in rounds]))
    return "".join(out)


# --- run ---
if LIVE:
    while True:
        run = json.load(open(RUN))
        clear_output(wait=True)
        display(HTML(render_html(run)))
        if not run["running"]:
            break
        time.sleep(30)
else:
    display(HTML(render_html(json.load(open(RUN)))))
"""

HEADER_MD = """## Section 8: GRPO on the combo pool (live)

Resumes from the round-0 LoRA checkpoint and trains GRPO only on the 56 *mixed pass/fail* combo
train tasks found in Section 7 (4 tasks per round, 4 rollouts each, lr 1e-5). The signal is the
held-out probe (the same 20 real test tasks at temperature 0), not the training reward: it is scored
for the original round-0 adapter first (the baseline, labelled `-1`) and then for every saved
checkpoint. The pod stops training itself if a probe drops more than 0.10 below the best so far.

Notes from setting this up: with two LoRA adapters live (the training policy and the probe), vLLM
needed `--max-loras 3`; with the default of 1 the adapters took turns and rollouts timed out. This
section's data file is rewritten every 30 seconds by `live_grpo_section.py`; the chart and status cells below
are code cells, so re-run them to redraw (set `LIVE = True` in the status cell to have it redraw itself).
"""


def status_md(run):
    now = run["updated"].replace("T", " ")
    rounds = run["rounds"]
    done = [r for r in rounds if "avg_reward" in r]
    d = durations(run)
    lu, uu = run.get("last_update"), run.get("last_update_utc")
    stamp = f"{lu[:10]} {lu[11:19]} local (UTC{lu[19:]}) = {uu[11:19]} UTC" if lu and uu else now
    lines = [f"### Live status (auto-updated {stamp})\n\n"]
    state = "running" if run["running"] else ("stopped on decline: " + run["stopped_on_decline"]
                                              if run["stopped_on_decline"] else "not running")
    lines.append("| | |\n|---|---|\n")
    lines.append(f"| trainer | {state} |\n")
    lines.append(f"| rounds finished | {len(done)} / {run['max_rounds']} max |\n")
    if d:
        avg = sum(d) / len(d)
        lines.append(f"| minutes per round (observed) | {avg / 60:.1f} |\n")
        lines.append(f"| full pass over the 56-task pool (14 rounds) | about {14 * avg / 3600:.1f} h |\n")
    errs = sum(r["errors"] for r in rounds)
    lines.append(f"| rollout errors so far | {errs} |\n")
    if run["probes"]:
        best = max(run["probes"], key=lambda p: p["probe_avg_reward"])
        lines.append(f"| best probe so far | {best['probe_avg_reward']:.2f} (round {best['round']}) |\n")
    lines.append("\n**Probe scores**\n\n| checkpoint | probe avg reward |\n|---|---|\n")
    for p in run["probes"]:
        name = "baseline (original round-0 adapter)" if p["round"] == -1 else f"round {p['round']}"
        lines.append(f"| {name} | {p['probe_avg_reward']:.2f} |\n")
    if not run["probes"]:
        lines.append("| (none yet) | |\n")
    lines.append("\n<details><summary>Rounds (%d)</summary>\n\n" % len(rounds))
    lines.append("| round | tasks (rewards per rollout) | avg reward | gradient rollouts | errors |\n|---|---|---|---|---|\n")
    for r in rounds:
        tasks = "; ".join(f"{t}: {''.join('1' if x else '0' for x in v)}" for t, v in r["tasks"].items()) or "running..."
        avg = f"{r['avg_reward']:.2f}" if "avg_reward" in r else "-"
        nu = f"{r['n_updated']}/{r['n_rollouts']}" if "n_updated" in r else "-"
        lines.append(f"| {r['round']} | {tasks} | {avg} | {nu} | {r['errors']} |\n")
    lines.append("\n</details>\n")
    return lines


def render_chart():
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


LIVE_HTML = os.path.join(HERE, "combo_grpo_live.html")


def write_live_html(run):
    import html as H
    lu, uu = run.get("last_update", ""), run.get("last_update_utc", "")
    done = [r for r in run["rounds"] if "avg_reward" in r]
    d = durations(run)
    state = "running" if run["running"] else (("stopped on decline: " + run["stopped_on_decline"]) if run["stopped_on_decline"] else "not running")
    rows = ""
    for r in run["rounds"]:
        tasks = "; ".join(f"{t}: {''.join('1' if x else '0' for x in v)}" for t, v in r["tasks"].items()) or "running..."
        rows += (f"<tr><td>{r['round']}</td><td>{H.escape(tasks)}</td><td>{r.get('avg_reward', '-')}</td>"
                 f"<td>{str(r['n_updated']) + '/' + str(r['n_rollouts']) if 'n_updated' in r else '-'}</td><td>{r['errors']}</td></tr>")
    prow = "".join(f"<tr><td>{'baseline (original round-0 adapter)' if p['round'] == -1 else 'round ' + str(p['round'])}</td>"
                   f"<td>{p['probe_avg_reward']:.2f}</td></tr>" for p in run["probes"]) or "<tr><td colspan=2>none yet</td></tr>"
    per = f"{sum(d) / len(d) / 60:.1f} min" if d else "-"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>Combo GRPO run (live)</title>
<style>:root{{color-scheme:light dark}}body{{font:15px system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;margin:8px 0 20px}}td,th{{border:1px solid #8884;padding:4px 10px;text-align:left}}
img{{max-width:100%}}.t{{opacity:.7}}</style></head><body>
<h2>Combo GRPO run (live)</h2>
<p class="t">Last update: {lu} = {uu}. This page reloads itself every 30 seconds.</p>
<table><tr><td>trainer</td><td>{state}</td></tr>
<tr><td>rounds finished</td><td>{len(done)} / {run['max_rounds']} max</td></tr>
<tr><td>minutes per round (observed)</td><td>{per}</td></tr>
<tr><td>rollout errors</td><td>{sum(r['errors'] for r in run['rounds'])}</td></tr></table>
<img alt="charts" src="data:image/png;base64,{run.get('_png', '')}">
<h3>Probe scores (20 held-out test tasks, temp 0)</h3><table><tr><th>checkpoint</th><th>score</th></tr>{prow}</table>
<h3>Rounds</h3><table><tr><th>round</th><th>tasks (reward per rollout)</th><th>avg</th><th>gradient rollouts</th><th>errors</th></tr>{rows}</table>
</body></html>"""
    tmp = LIVE_HTML + ".tmp"
    open(tmp, "w").write(page)
    os.replace(tmp, LIVE_HTML)


def status_html(run):
    ns = {}
    exec(STATUS_SRC.split("# --- run ---")[0], ns)
    return ns["render_html"](run)


def update_notebook(run):
    nb = json.load(open(NB))
    cells = nb["cells"]
    png = render_chart()
    run["_png"] = png
    write_live_html(run)
    run.pop("_png", None)
    want = {
        "grpo_header": {"cell_type": "markdown", "metadata": {"auto": "grpo_header"},
                        "source": HEADER_MD.splitlines(True)},
        "grpo_chart": {"cell_type": "code", "execution_count": None, "metadata": {"auto": "grpo_chart"},
                       "source": CHART_SRC.splitlines(True),
                       "outputs": [{"output_type": "display_data", "metadata": {},
                                    "data": {"image/png": png, "text/plain": ["<Figure>"]}}]},
        "grpo_status": {"cell_type": "code", "execution_count": None, "metadata": {"auto": "grpo_status"},
                        "source": STATUS_SRC.splitlines(True),
                        "outputs": [{"output_type": "display_data", "metadata": {},
                                     "data": {"text/html": status_html(run).splitlines(True),
                                              "text/plain": ["<combo GRPO live status>"]}}]},
    }
    have = {c.get("metadata", {}).get("auto"): i for i, c in enumerate(cells)}
    for key in ("grpo_header", "grpo_chart", "grpo_status"):
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
    scp("/root/grpo_train.log.jsonl", TRAIN_LOG)
    scp("/root/grpo_train.stdout.log", STDOUT_LOG)
    un.sync_probe_log()
    prev = json.load(open(RUN_JSON)) if os.path.exists(RUN_JSON) else {}
    run = build_run(prev)
    run["last_update"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    run["last_update_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    json.dump(run, open(RUN_JSON, "w"), indent=1)
    update_notebook(run)
    done = sum("avg_reward" in r for r in run["rounds"])
    return run, f"{run['updated']} running={run['running']} rounds={done} probes={len(run['probes'])}"


if __name__ == "__main__":
    if "--once" in sys.argv:
        print(once()[1])
        sys.exit(0)
    stopped_checks = 0
    while True:
        try:
            run, msg = once()
            print(msg, flush=True)
            stopped_checks = 0 if run["running"] else stopped_checks + 1
            if stopped_checks >= 3:
                print("trainer not running for 3 checks; final update written, exiting", flush=True)
                break
        except Exception as e:
            print("update failed:", type(e).__name__, str(e)[:150], flush=True)
        time.sleep(INTERVAL)
