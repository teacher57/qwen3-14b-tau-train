"""Write full_run_results/teacher_run.json (status + timestamps) for the 32B teacher run. JSON only; the notebook reads it.

Usage: python3 update_teacher_json.py [--once] [--interval 30]
Env: TEACHER_HOST (default root@154.54.102.23), TEACHER_PORT (default 14422), TEACHER_KEY.
"""
import datetime, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "full_run_results", "teacher_run.json")
GROUPS = os.path.join(HERE, "..", "full_run_results", "combo_passrate.json")
TASKS = os.path.join(HERE, "hard_tasks.json")
HOST = os.environ.get("TEACHER_HOST", "root@154.54.102.23")
PORT = os.environ.get("TEACHER_PORT", "14422")
KEY = os.path.expanduser(os.environ.get("TEACHER_KEY", "~/.ssh/id_ed25519"))
TRIALS, THRESHOLD, STOP_AT = 4, 0.30, 2
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 30

REMOTE = r"""
for k in "setup_running:pgrep -f [s]etup_pod.sh" "vllm_running:pgrep -f [v]llm.serve" "rollouts_running:pgrep -f [t]eacher_rollouts.py"; do
  n=${k%%:*}; c=${k#*:}; if eval "$c" >/dev/null 2>&1; then echo "$n=1"; else echo "$n=0"; fi
done
grep -q SETUP_COMPLETE /root/setup.log 2>/dev/null && echo setup_done=1 || echo setup_done=0
grep -q 'Application startup complete' /root/vllm.log 2>/dev/null && echo vllm_ready=1 || echo vllm_ready=0
grep -q TEACHER_ALL_DONE /root/teacher.log 2>/dev/null && echo teacher_done=1 || echo teacher_done=0
echo ROWS
python3 - <<'PY'
import json, os
for p in ("/root/teacher_rollouts.jsonl", "/root/teacher_rollouts_b.jsonl"):
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                print(json.dumps({k: r.get(k) for k in ("task_index", "trial", "reward", "seconds", "finished_at", "started_run_at", "error", "skipped")}))
PY
"""


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", PORT, "-i", KEY, HOST, cmd],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.stdout if r.returncode == 0 else None


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def build(prev):
    out = ssh(REMOTE)
    flags, rows, reachable = {}, [], out is not None
    if reachable:
        head, _, tail = out.partition("ROWS\n")
        flags = {l.split("=")[0]: l.split("=")[1] == "1" for l in head.splitlines() if "=" in l}
        rows = [json.loads(l) for l in tail.splitlines() if l.strip()]
    else:
        flags, rows = prev.get("flags", {}), prev.get("_rows", [])
    tasks = json.load(open(TASKS))
    cat = {p["task_index"]: p["category"].split(" ")[0] for p in json.load(open(GROUPS))["per_task"]}
    ts = dict(prev.get("timestamps", {}))
    for name, cond in [("setup_seen", True), ("setup_done", flags.get("setup_done")), ("vllm_ready", flags.get("vllm_ready")),
                       ("rollouts_started", bool(rows)), ("teacher_done", flags.get("teacher_done"))]:
        if cond and name not in ts:
            ts[name] = now_iso()
    skipped_rows = [r for r in rows if r.get("skipped")]
    rows = [r for r in rows if not r.get("skipped")]
    skipped_by = {}
    for r in skipped_rows:
        skipped_by[r["task_index"]] = skipped_by.get(r["task_index"], 0) + 1
    by = {}
    for r in rows:
        by.setdefault(r["task_index"], []).append(r)
    per_task = []
    for t in tasks:
        rs = by.get(t, [])
        passes = sum(1 for r in rs if (r["reward"] or 0) >= 1)
        per_task.append({"task_index": t, "group_14b": cat.get(t, "?"), "n_done": len(rs), "n_skipped": skipped_by.get(t, 0), "passes": passes,
                         "solved": passes > 0, "rewards": [r["reward"] for r in sorted(rs, key=lambda r: r["trial"])],
                         "avg_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 1) if rs else None,
                         "last_finished_at": datetime.datetime.fromtimestamp(max(r["finished_at"] for r in rs)).astimezone().isoformat(timespec="seconds") if rs else None})
    groups = {}
    for g in ("never", "mixed", "always"):
        pts = [p for p in per_task if p["group_14b"] == g]
        groups[g] = {"n_tasks": len(pts), "tasks_with_any_rollout": sum(1 for p in pts if p["n_done"]),
                     "solved": sum(1 for p in pts if p["solved"]),
                     "rollouts_done": sum(p["n_done"] for p in pts), "rollouts_passed": sum(p["passes"] for p in pts)}
    done, passed = len(rows), sum(1 for r in rows if (r["reward"] or 0) >= 1)
    total = len(tasks) * TRIALS
    st = {"model": "Qwen3-32B-AWQ", "temperature": 0.7, "trials": TRIALS, "n_tasks": len(tasks), "total_rollouts": total,
          "rollouts_done": done, "rollouts_passed": passed, "errors": sum(1 for r in rows if r.get("error")),
          "tasks_solved": sum(1 for p in per_task if p["solved"]),
          "rollouts_skipped": len(skipped_rows), "stop_after_passes": STOP_AT,
          "tasks_with_all_trials": sum(1 for p in per_task if p["passes"] >= STOP_AT or p["n_done"] + p["n_skipped"] >= TRIALS),
          "go_no_go_threshold": THRESHOLD}
    hard = groups["never"]
    st["solved_fraction"] = hard["solved"] / hard["n_tasks"] if hard["n_tasks"] else 0.0  # go/no-go basis: the tasks the 14B never solves
    st["go_no_go_basis"] = "tasks the 14B never solves"
    if rows:
        # the batches run one after the other, so the pace is that of the most recent batch (the latest process start)
        procs = {}
        for r in rows + skipped_rows:
            procs.setdefault(r["started_run_at"], []).append(r)
        latest = max(r["finished_at"] for r in rows)
        rate = 0.0
        for st0 in sorted(procs, reverse=True):
            real_ = [r for r in procs[st0] if not r.get("skipped")]
            span = (max((r["finished_at"] for r in real_), default=st0) - st0) / 60
            if len(real_) >= 5 and span > 0:
                rate = len(real_) / span
                break
        st["elapsed_min"] = (latest - min(procs)) / 60
        if rate > 0:
            st["rollouts_per_min"] = rate
            remaining = max(total - done - len(skipped_rows), 0)  # upper bound: unreached trials of solved tasks will be skipped
            st["eta_min"] = 0 if flags.get("teacher_done") else remaining / rate
    if flags.get("teacher_done"):
        stage = "finished"
    elif flags.get("rollouts_running"):
        stage = "rollouts running"
    elif flags.get("vllm_ready"):
        stage = "32B server ready"
    elif flags.get("vllm_running"):
        stage = "loading the 32B model"
    elif flags.get("setup_running"):
        stage = "installing vLLM and tau-bench"
    elif flags.get("setup_done"):
        stage = "setup done, server not started"
    else:
        stage = "waiting"
    return {"updated": now_iso(), "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pod_reachable": reachable, "stage": stage, "flags": flags, "timestamps": ts, "summary": st, "groups_14b": groups,
            "per_task": per_task, "rollout_finish_times": sorted(datetime.datetime.fromtimestamp(r["finished_at"]).astimezone().isoformat(timespec="seconds") for r in rows),
            "_rows": rows}


def once():
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data = build(prev)
    tmp = OUT + ".tmp"
    json.dump(data, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    s = data["summary"]
    return data, f"{data['updated']} {data['stage']} rollouts={s['rollouts_done']}/{s['total_rollouts']} solved_tasks={s['tasks_solved']}"


if __name__ == "__main__":
    if "--once" in sys.argv:
        print(once()[1])
        sys.exit(0)
    while True:
        try:
            data, msg = once()
            print(msg, flush=True)
            if data["stage"] == "finished":
                print("teacher run finished; final JSON written, exiting", flush=True)
                break
        except Exception as e:
            print("update failed:", type(e).__name__, str(e)[:150], flush=True)
        time.sleep(INTERVAL)
