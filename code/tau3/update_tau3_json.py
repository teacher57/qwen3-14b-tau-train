"""Keeps full_run_results/tau3_run.json current: pulls a compact summary from the pod (summarize_remote.py) and computes stats.
The notebook tau3_retail_eval.ipynb only READS that JSON.

  python3 update_tau3_json.py            # loop every 30 s until the run is finished
  python3 update_tau3_json.py --once
Env: TAU3_HOST (default root@202.181.159.229), TAU3_PORT (default 14606), TAU3_KEY.
"""
import datetime, json, math, os, subprocess, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "full_run_results", "tau3_run.json")
HOST = os.environ.get("TAU3_HOST", "root@202.181.159.229")
PORT = os.environ.get("TAU3_PORT", "14606")
KEY = os.path.expanduser(os.environ.get("TAU3_KEY", "~/.ssh/id_ed25519"))
N_TASKS, TRIALS = 114, 1
DEFAULT_PACE = 1.0   # rollouts per minute until measured


def ssh(cmd, stdin=None, timeout=90):
    # stdin=DEVNULL: a detached ssh must never wait on a terminal; keep-alives + hard timeout so one bad call cannot freeze the loop
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
                        "-o", "ServerAliveCountMax=3", "-p", PORT, "-i", KEY, HOST, cmd],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.stdout


def local_iso(ts):
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def parse_time(s):
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def sd_se(xs):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def summarize(run):
    rows = [r for r in (run or {}).get("rows", []) if r.get("reward") is not None or r.get("term")]
    if not rows:
        return {"rollouts_done": 0, "rollouts_total": N_TASKS * TRIALS}
    for r in rows:
        r["t_end"] = parse_time(r.get("end_time") or "")
    rows.sort(key=lambda r: (r["t_end"] or 0))
    passed = [1 if (r["reward"] or 0) >= 1 else 0 for r in rows]
    n = len(rows)
    done_times = [r["t_end"] for r in rows if r["t_end"]]
    starts = [parse_time(r.get("start_time") or "") for r in rows]
    starts = [s for s in starts if s]
    rate = None
    if len(done_times) >= 6 and starts:
        span = (max(done_times) - min(starts)) / 60
        rate = n / span if span > 0 else None
    breakdown = {}
    for r in rows:
        for k, v in (r.get("breakdown") or {}).items():
            breakdown.setdefault(k, []).append(v)
    return {
        "rollouts_done": n, "rollouts_total": N_TASKS * TRIALS, "passed": sum(passed),
        "pass1": sum(passed) / n, "pass1_se": sd_se(passed),
        "running_pass1": [sum(passed[:i + 1]) / (i + 1) for i in range(n)],
        "finish_times": [local_iso(t) for t in done_times],
        "terminations": dict(Counter(str(r["term"]) for r in rows)),
        "breakdown_pass_rate": {k: sum(1 for x in v if (x or 0) >= 1) / len(v) for k, v in breakdown.items()},
        "agent_cost": sum(r.get("agent_cost") or 0 for r in rows), "user_cost": sum(r.get("user_cost") or 0 for r in rows),
        "avg_duration_s": sum(r.get("duration") or 0 for r in rows) / n,
        "rollouts_per_min": rate, "agent_llm": run.get("agent_llm"), "user_llm": run.get("user_llm"),
        "rows": [{k: r.get(k) for k in ("task_id", "trial", "reward", "term", "duration", "user_cost", "n_messages", "n_tool_calls",
                                         "n_target_actions", "target_actions", "db_match", "breakdown")} for r in rows],
    }


def paired(old, new):
    a = {r["task_id"]: 1 if (r["reward"] or 0) >= 1 else 0 for r in old.get("rows", [])}
    b = {r["task_id"]: 1 if (r["reward"] or 0) >= 1 else 0 for r in new.get("rows", [])}
    common = sorted(set(a) & set(b), key=lambda x: int(x) if x.isdigit() else x)
    if not common:
        return None
    both = [t for t in common if a[t] and b[t]]
    only_new = [t for t in common if b[t] and not a[t]]
    only_old = [t for t in common if a[t] and not b[t]]
    neither = [t for t in common if not a[t] and not b[t]]
    p = None
    if only_new or only_old:
        k, m = len(only_new), len(only_new) + len(only_old)
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m)
    return {"n_common": len(common), "both_pass": both, "only_new": only_new, "only_old": only_old, "neither": neither,
            "diff": (len(only_new) - len(only_old)) / len(common), "sign_test_p": p}


def build(prev):
    now = time.time()
    try:
        raw = ssh("python3 /root/summarize_remote.py")
        remote = json.loads(raw) if raw.strip().startswith("{") else None
    except Exception:
        remote = None
    if remote is None:
        out = dict(prev) if prev else {}
        out.update({"updated": local_iso(now), "pod_reachable": False})
        return out
    runs = remote.get("runs", {})
    old, new = summarize(runs.get("tau3_old")), summarize(runs.get("tau3_new"))
    log = remote.get("pipeline_log", [])
    events = []
    for line in log:
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0].endswith("Z"):
            try:
                events.append({"time": local_iso(datetime.datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()), "event": parts[1][:200]})
            except Exception:
                pass
    text = "\n".join(log)
    procs = remote.get("procs", {})
    if "TAU3_SETUP_FAILED" in text:
        stage = "failed: pod setup"
    elif "TAU3_FAILED" in text:
        stage = "failed: vLLM start"
    elif "TAU3_DONE" in text:
        stage = "finished"
    elif old["rollouts_done"] >= N_TASKS * TRIALS and new["rollouts_done"] == 0 and procs.get("run"):
        stage = "switching to the fine-tuned model"
    elif new["rollouts_done"] > 0 or "run tau3_new started" in text:
        stage = "running the fine-tuned model (new)"
    elif old["rollouts_done"] > 0 or "run tau3_old started" in text:
        stage = "running the starting adapter (old)"
    elif procs.get("run") or "tau3: starting vllm" in text:
        stage = "starting vLLM and loading adapters"
    elif procs.get("setup") or "TAU3_SETUP_DONE" not in text:
        stage = "setting up the pod"
    else:
        stage = "setup done, waiting to start the runs"
    # plan: remaining rollouts and time
    pace = (new.get("rollouts_per_min") if stage.startswith("running the fine") else old.get("rollouts_per_min")) or \
        old.get("rollouts_per_min") or DEFAULT_PACE
    steps, cum = [], 0.0
    for label, ev in (("starting adapter (old)", old), ("fine-tuned model (new)", new)):
        rem = max(N_TASKS * TRIALS - ev["rollouts_done"], 0)
        mins = rem / pace if pace else None
        cum += mins or 0
        steps.append({"step": label, "remaining": rem, "minutes": mins, "done_in_min": cum,
                      "done_at": datetime.datetime.fromtimestamp(now + cum * 60).strftime("%H:%M") if rem else None})
    out = {
        "updated": local_iso(now), "pod_reachable": True, "stage": stage, "pod": {
            "host": HOST + ":" + PORT, "gpu": remote.get("gpu"), "disk": remote.get("disk"), "vllm_up": remote.get("vllm_up"),
            "procs": procs, "setup_tail": remote.get("setup_tail"), "vllm_log_tail": remote.get("vllm_log_tail")},
        "config": {"domain": "retail", "task_split": "base (all 114 tasks)", "trials": TRIALS, "agent_temperature": 0.7,
                   "user_llm": "gpt-4.1", "max_steps": 200, "concurrency": 24, "agent_base": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
                   "old_adapter": "round-0 (starting adapter)", "new_adapter": "distill-epoch-1 (SFT on the 32B teacher)"},
        "old": old, "new": new, "paired": paired(old, new),
        "plan": {"steps": steps, "pace_used": pace, "total_min": cum, "done_at": steps[-1]["done_at"] or steps[0]["done_at"]},
        "events": events,
    }
    return out


def write(out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, OUT)


if __name__ == "__main__":
    prev = json.load(open(OUT)) if os.path.exists(OUT) else None
    while True:
        out = build(prev)
        write(out)
        prev = out
        print(out.get("updated"), out.get("stage"), flush=True)
        if "--once" in sys.argv or str(out.get("stage", "")).startswith(("finished", "failed")):
            break
        time.sleep(30)
