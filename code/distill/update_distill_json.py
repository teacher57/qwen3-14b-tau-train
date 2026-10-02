"""Write full_run_results/distill_run.json (status + timestamps + training curve + eval results) for the distillation run:
SFT of the 14B on the 32B teacher's passed conversations, then the tau-bench tests of the result (retail with up to 4 trials per task,
the starting adapter re-run on the same pod, and the airline test). JSON only; the notebook reads it.

pass^1 = average pass rate over all trials. pass^2 for a task with c passes out of n trials = C(c,2) / C(n,2) (the chance that two
random trials both pass; with n = 2 it is "both trials pass"), averaged over tasks. Standard errors are computed over tasks.

Usage: python3 update_distill_json.py [--once] [--interval 30]
Env: TEACHER_HOST (default root@154.54.102.23), TEACHER_PORT (default 14422), TEACHER_KEY.
"""
import datetime, json, math, os, re, subprocess, sys, time
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "full_run_results", "distill_run.json")
BASE = os.path.join(HERE, "..", "full_run_results", "passk_run.json")   # earlier pass^2 test (starting adapter = baseline)
COMBO = os.path.join(HERE, "..", "github_repo", "dataset", "combo_tasks_test_64.json")
HOST = os.environ.get("TEACHER_HOST", "root@154.54.102.23")
PORT = os.environ.get("TEACHER_PORT", "14422")
KEY = os.path.expanduser(os.environ.get("TEACHER_KEY", "~/.ssh/id_ed25519"))
N_RETAIL, RETAIL_TRIALS = 115, 4
RETAIL_TOTAL = N_RETAIL * RETAIL_TRIALS          # 460 rollouts per model once the extra trials are done
N_AIRLINE, AIRLINE_TRIALS = 50, 2
AIR_TOTAL = N_AIRLINE * AIRLINE_TRIALS           # 100 rollouts per model
DEFAULT_PACE = 4.5                                # rollouts per minute measured on this pod (retail)
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 30

REMOTE = r"""
echo ===PIPE; cat /root/pipeline.log 2>/dev/null
echo ===TRAIN; cat /root/sft_out/train_log.jsonl 2>/dev/null
echo ===CKPT; ls /root/sft_out 2>/dev/null
echo ===FLAGS
pgrep -f '[t]rain_sft.py' >/dev/null && echo trainer_running=1 || echo trainer_running=0
grep -q AIRLINE_DONE /root/pipeline.log 2>/dev/null && echo airline_done=1 || echo airline_done=0
pgrep -f '[r]un_more_chain' >/dev/null && echo more_chain_alive=1 || echo more_chain_alive=0
for n in $(ps -eo args | grep '[f]ull_passk.py' | awk '{print $3}'); do echo proc_$n=1; done
python3 - <<'PY'
import json, os
for tag, p in (("EVAL", "/root/passk_distilled.jsonl"), ("EVAL2", "/root/passk_baseline2.jsonl"),
               ("EVAL3", "/root/passk_airline_new.jsonl"), ("EVAL4", "/root/passk_airline_base.jsonl")):
    print("===" + tag)
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                print(json.dumps({k: r.get(k) for k in ("task_index", "trial", "reward", "seconds", "finished_at", "started_run_at", "error")}))
PY
"""


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", PORT, "-i", KEY, HOST, cmd],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.stdout if r.returncode == 0 else None


def now_local():
    return datetime.datetime.now().astimezone()


def utc_to_local(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).astimezone()


def split(out):
    parts, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^===(\w+)$", line)
        if m:
            cur = m.group(1)
            parts[cur] = []
        elif cur:
            parts[cur].append(line)
    return parts


def pk(v, k=2):
    n = len(v)
    return comb(sum(v), k) / comb(n, k) if n >= k else None


def sd_se(xs):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def latest_rate(rows):
    """Pace of the most recent process (rows share a started_run_at): rollouts per minute since it started."""
    groups = {}
    for r in rows:
        groups.setdefault(r.get("started_run_at") or 0, []).append(r)
    for st0 in sorted(groups, reverse=True):
        g = groups[st0]
        span = (max(r["finished_at"] for r in g) - st0) / 60
        if len(g) >= 5 and span > 0:
            return len(g) / span
    return None


def summarize(rows, total):
    by = {}
    for r in rows:
        by.setdefault(r["task_index"], []).append(1 if r["reward"] >= 1 else 0)
    multi = {t: v for t, v in by.items() if len(v) >= 2}
    passed = sum(1 for r in rows if r["reward"] >= 1)
    p1_tasks = [sum(v) / len(v) for v in by.values()]
    p2_tasks = [pk(v) for v in multi.values()]
    ev = {"rollouts_done": len(rows), "rollouts_total": total, "passed": passed, "errors": sum(1 for r in rows if r.get("error")),
          "tasks_with_both_trials": len(multi), "trials_per_task_min": min((len(v) for v in by.values()), default=0),
          "pass1": passed / len(rows) if rows else None, "pass1_se": sd_se(p1_tasks),
          "pass2": sum(p2_tasks) / len(p2_tasks) if p2_tasks else None, "pass2_se": sd_se(p2_tasks),
          "finish_times": sorted(datetime.datetime.fromtimestamp(r["finished_at"]).astimezone().isoformat(timespec="seconds") for r in rows),
          "running_pass1": [], "by_task": {str(t): v for t, v in by.items()}}
    for kk in (3, 4):
        tk = [pk(v, kk) for v in by.values() if len(v) >= kk]
        ev[f"pass{kk}"] = sum(tk) / len(tk) if tk else None
        ev[f"pass{kk}_se"] = sd_se(tk)
        ev[f"tasks_for_pass{kk}"] = len(tk)
    bt = {}
    for r in rows:
        k = r.get("trial")
        if k is None:
            continue
        x = bt.setdefault(str(k), {"done": 0, "passed": 0})
        x["done"] += 1
        x["passed"] += 1 if r["reward"] >= 1 else 0
    for x in bt.values():
        x["pass_rate"] = x["passed"] / x["done"]
    ev["by_trial"] = dict(sorted(bt.items(), key=lambda kv: int(kv[0])))
    if rows:
        ok = 0
        for i, r in enumerate(sorted(rows, key=lambda r: r["finished_at"]), 1):
            ok += r["reward"] >= 1
            ev["running_pass1"].append(round(ok / i, 4))
        p1_same = [sum(v) / len(v) for v in multi.values()]
        m1 = sum(p1_same) / len(p1_same) if p1_same else None
        ev["consistency"] = ev["pass2"] / m1 if (ev["pass2"] is not None and m1) else None
        rate = latest_rate(rows)
        if rate:
            ev["rollouts_per_min"] = rate
            ev["eta_min"] = max(total - len(rows), 0) / rate
    return ev


def group_rates(ev, combo):
    out = {}
    for name, keep in (("combo", lambda t: t in combo), ("other", lambda t: t not in combo)):
        v = [x for t, x in ev["by_task"].items() if keep(int(t)) and len(x) >= 2]
        n = len(v)
        out[name] = {"tasks": n, "pass1": sum(sum(x) / len(x) for x in v) / n if n else None,
                     "pass2": sum(pk(x) for x in v) / n if n else None,
                     **{f"pass{kk}": (lambda w: sum(w) / len(w) if w else None)([pk(x, kk) for x in v if len(x) >= kk]) for kk in (3, 4)}}
    return out


def paired(a, b):
    """Per task, whose pass^2 is higher (a or b), over tasks both have >= 2 trials for, with an exact two-sided sign test on the non-ties."""
    ta = {t: pk(v) for t, v in a["by_task"].items() if len(v) >= 2}
    tb = {t: pk(v) for t, v in b["by_task"].items() if len(v) >= 2}
    common = sorted(set(ta) & set(tb))
    up = sum(1 for t in common if ta[t] > tb[t])
    down = sum(1 for t in common if ta[t] < tb[t])
    n = up + down
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(up, down) + 1)) / 2 ** n) if n else 1.0
    return {"n_common": len(common), "new_only": up, "other_only": down, "ties": len(common) - n, "p_value": p}


def build(prev):
    out = ssh(REMOTE)
    if out is None:
        d = dict(prev)
        d.update(updated=now_local().isoformat(timespec="seconds"), pod_reachable=False)
        return d
    p = split(out)
    pipe = [l for l in p.get("PIPE", []) if l.strip()]
    events = {}
    for l in pipe:
        m = re.match(r"^(\S+Z) (.*)$", l)
        if not m:
            continue
        t, msg = m.groups()
        if msg.startswith("teacher run finished"):
            events["teacher_finished"] = t
        elif msg.startswith("restarting training") or (msg.startswith("training the 14B") and "training_started" not in events):
            events["training_started"] = t
        elif msg.startswith("training finished"):
            events["training_finished"] = t
        elif msg.startswith("14B server ready"):
            events["eval_started"] = t
        elif msg.startswith("retail test finished"):
            events["eval_finished"] = t
        elif msg.startswith("retail re-run finished"):
            events["rerun_finished"] = t
        elif msg.startswith("retail trials 3 and 4: starting"):
            events["extra_trials_started"] = t
        elif msg.startswith("retail trials 3 and 4, new model finished"):
            events["extra_new_finished"] = t
        elif msg.startswith("retail trials 3 and 4, starting adapter finished"):
            events["extra_base_finished"] = t
        elif msg.startswith("airline test, new model finished"):
            events["airline_new_finished"] = t
        elif msg.startswith("airline test finished"):
            events["airline_finished"] = t
        elif msg.startswith("FAILED") or msg.startswith("STOPPED"):
            events["failed"] = msg
        elif msg.startswith("PIPELINE_DONE"):
            events["pipeline_done"] = t
    if "PIPELINE_DONE" in " ".join(pipe):
        events.setdefault("pipeline_done", "yes")
    flags = {}
    for l in p.get("FLAGS", []):
        if "=" in l:
            k, v = l.split("=")
            flags[k] = v == "1"
    # training
    steps = [json.loads(l) for l in p.get("TRAIN", []) if l.strip().startswith("{")]
    tr = {"steps_done": len(steps), "steps_total": steps[-1]["of"] if steps else None,
          "loss": [s["loss"] for s in steps], "grad_norm": [s["grad_norm"] for s in steps], "lr": [s["lr"] for s in steps],
          "tokens": [s["tokens"] for s in steps], "step_index": [s["step"] for s in steps],
          "checkpoints": [c for c in p.get("CKPT", []) if c.strip()]}
    if steps:
        recent = steps[-20:]
        tr["sec_per_step"] = (recent[-1]["elapsed_min"] - recent[0]["elapsed_min"]) * 60 / max(len(recent) - 1, 1) if len(recent) > 1 else None
        tr["elapsed_min"] = steps[-1]["elapsed_min"]
        if tr["sec_per_step"]:
            tr["eta_min"] = (tr["steps_total"] - tr["steps_done"]) * tr["sec_per_step"] / 60
        tr["loss_first10"] = sum(tr["loss"][:10]) / len(tr["loss"][:10])
        tr["loss_last10"] = sum(tr["loss"][-10:]) / len(tr["loss"][-10:])
    # evals
    def rows_of(tag):
        return [json.loads(l) for l in p.get(tag, []) if l.strip().startswith("{")]
    rows, rows2, rows3, rows4 = rows_of("EVAL"), rows_of("EVAL2"), rows_of("EVAL3"), rows_of("EVAL4")
    ev, ev2 = summarize(rows, RETAIL_TOTAL), summarize(rows2, RETAIL_TOTAL)
    ev3, ev4 = summarize(rows3, AIR_TOTAL), summarize(rows4, AIR_TOTAL)
    airline = {"n_tasks": N_AIRLINE, "trials": AIRLINE_TRIALS, "new": ev3, "base": ev4,
               "comparison": paired(ev3, ev4) if (rows3 and rows4) else None}
    # the earlier test of the starting adapter (Section 10), summarized the same way
    base = earlier = None
    if os.path.exists(BASE):
        for m in json.load(open(BASE))["models"]:
            if m["model"] == "baseline":
                brow = [{"task_index": r["task_index"], "trial": r.get("trial"), "reward": r["reward"], "finished_at": r["finished_at"]} for r in m["rows"]]
                earlier = summarize(brow, len(brow))
                base = {k: earlier.get(k) for k in ("pass1", "pass1_se", "pass2", "pass2_se", "consistency", "running_pass1", "by_trial", "pass3", "pass3_se", "pass4", "pass4_se", "tasks_for_pass3", "tasks_for_pass4")}
                base["done"] = len(brow)
    combo = {t["task_index"] for t in json.load(open(COMBO))} if os.path.exists(COMBO) else set()
    groups = {"new": group_rates(ev, combo) if rows else None, "rerun": group_rates(ev2, combo) if rows2 else None,
              "earlier": group_rates(earlier, combo) if earlier else None}
    comparisons = {}
    if rows and rows2:
        comparisons["new_vs_rerun"] = paired(ev, ev2)
    if rows and earlier:
        comparisons["new_vs_earlier"] = paired(ev, earlier)
    if rows2 and earlier:
        comparisons["rerun_vs_earlier"] = paired(ev2, earlier)
    # what is running, and what is left (steps run strictly one after the other)
    running = {k[5:] for k, v in flags.items() if k.startswith("proc_") and v}
    pace_cur = next((e_.get("rollouts_per_min") for name, e_ in (("distilled", ev), ("baseline2", ev2), ("airline_new", ev3), ("airline_base", ev4)) if name in running), None)
    steps_left = []
    for name, label, e_, total in (("distilled", "retail trials 3 and 4, new model", ev, RETAIL_TOTAL),
                                   ("baseline2", "retail trials 3 and 4, starting adapter", ev2, RETAIL_TOTAL),
                                   ("airline_new", "airline test, new model", ev3, AIR_TOTAL),
                                   ("airline_base", "airline test, starting adapter", ev4, AIR_TOTAL)):
        rem = max(total - e_["rollouts_done"], 0)
        if rem and not flags.get("airline_done") and (flags.get("more_chain_alive") or name in running):
            pace = (e_.get("rollouts_per_min") if name in running else None) or pace_cur or DEFAULT_PACE
            steps_left.append({"step": label, "proc": name, "remaining": rem, "minutes": rem / pace, "running": name in running})
    n = now_local()
    acc = 0.0
    for s_ in steps_left:
        acc += s_["minutes"]
        s_["done_in_min"] = acc
        s_["done_at"] = (n + datetime.timedelta(minutes=acc)).strftime("%H:%M")
    plan = {"steps": steps_left, "total_min": acc, "done_at": (n + datetime.timedelta(minutes=acc)).strftime("%H:%M") if steps_left else None,
            "pace_used": pace_cur or DEFAULT_PACE, "pace_measured": pace_cur is not None}
    air_steps = [s_ for s_ in steps_left if s_["proc"].startswith("airline")]
    airline_eta = None
    if air_steps:
        before = sum(s_["minutes"] for s_ in steps_left if not s_["proc"].startswith("airline"))
        run_min = sum(s_["minutes"] for s_ in air_steps)
        started = any(s_["running"] for s_ in air_steps)
        airline_eta = {"started": started, "starts_in_min": before, "starts_at": (n + datetime.timedelta(minutes=before)).strftime("%H:%M"),
                       "duration_min": run_min, "done_in_min": before + run_min, "done_at": (n + datetime.timedelta(minutes=before + run_min)).strftime("%H:%M"),
                       "basis": "measured pace" if pace_cur else "retail pace measured earlier on this pod", "rate_assumed_per_min": pace_cur or DEFAULT_PACE}
    # stage
    if "failed" in events:
        stage = "failed: " + events["failed"][:80]
    elif events.get("pipeline_done") and ("distilled" in running or ("baseline2" in running and rows2 and len(rows2) > 230)):
        stage = "retail trials 3 and 4: new model" if "distilled" in running else "retail trials 3 and 4: starting adapter"
    elif events.get("pipeline_done") and "baseline2" in running:
        stage = "retail trials 3 and 4: starting adapter" if events.get("extra_trials_started") else "baseline re-run running"
    elif events.get("pipeline_done") and {"airline_new", "airline_base"} & running:
        stage = "airline test running"
    elif events.get("pipeline_done") and flags.get("more_chain_alive"):
        stage = "switching between tests"
    elif events.get("pipeline_done") and not flags.get("airline_done") and (rows3 or rows4):
        stage = "paused"
    elif events.get("pipeline_done"):
        stage = "finished"
    elif flags.get("eval_running") or events.get("eval_started"):
        stage = "retail test running"
    elif events.get("training_finished"):
        stage = "starting the 14B server"
    elif flags.get("trainer_running") or steps:
        stage = "training"
    elif events.get("teacher_finished"):
        stage = "preparing training"
    else:
        stage = "waiting for the teacher run"
    if stage == "finished" and not flags.get("airline_done") and (rows3 or rows4):
        stage = "paused"
    expected = None
    if stage == "training" and tr.get("eta_min") is not None:
        expected = {"training_done_in_min": tr["eta_min"], "result_in_min_low": tr["eta_min"] + 66, "result_in_min_high": tr["eta_min"] + 126}
    return {"updated": n.isoformat(timespec="seconds"),
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pod_reachable": True, "stage": stage, "flags": flags,
            "events_local": {k: utc_to_local(v).isoformat(timespec="seconds") for k, v in events.items() if v.endswith("Z")},
            "training": tr, "eval": ev, "rerun": ev2, "airline": airline, "airline_eta": airline_eta, "plan": plan,
            "baseline": base, "groups": groups, "comparisons": comparisons, "expected": expected,
            "retail_trials_target": RETAIL_TRIALS, "data": {"conversations": 180, "samples": 3065, "epochs": 1}}


def once():
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d = build(prev)
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    t, e, e2 = d.get("training", {}), d.get("eval", {}), d.get("rerun", {})
    return d, (f"{d['updated']} {d['stage']} | retail new {e.get('rollouts_done', 0)}/{RETAIL_TOTAL}, "
               f"starting adapter {e2.get('rollouts_done', 0)}/{RETAIL_TOTAL} | left {round((d.get('plan') or {}).get('total_min', 0))} min")


if __name__ == "__main__":
    if "--once" in sys.argv:
        print(once()[1])
        sys.exit(0)
    while True:
        try:
            d, msg = once()
            print(msg, flush=True)
            if d["stage"] == "finished" or d["stage"].startswith("failed"):
                print("pipeline over; final JSON written, exiting", flush=True)
                break
        except Exception as e:
            print("update failed:", type(e).__name__, str(e)[:150], flush=True)
        time.sleep(INTERVAL)
