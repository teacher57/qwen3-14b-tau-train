"""Keeps full_run_results/aug_sft_run.json current (pod state + training progress). The notebook aug_sft_experiment.ipynb only READS it.
  python3 update_aug_json.py [--once]      env: AUG_HOST (default root@195.26.233.55), AUG_PORT (default 43757)
"""
import datetime, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "full_run_results", "aug_sft_run.json")
HOST, PORT = os.environ.get("AUG_HOST", "root@195.26.233.55"), os.environ.get("AUG_PORT", "43757")
KEY = os.path.expanduser("~/.ssh/id_ed25519")
REMOTE = r'''
echo ===GPU; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null
echo ===DISK; df -h / | tail -1
echo ===PIPE; tail -n 30 /root/pipeline.log 2>/dev/null
echo ===ENV; tail -n 3 /root/train_env_install.log 2>/dev/null | cut -c1-200
echo ===PROCS; pgrep -f "[t]rain_sft.py" | head -1; pgrep -f "[s]etup_train_env" | head -1
echo ===TRAINLOG; tail -n 3 /root/train_sft.log 2>/dev/null | cut -c1-250
echo ===STEPS; cat /root/sft_out_aug/train_log.jsonl 2>/dev/null
echo ===VLLM; (curl -s -m 3 localhost:8000/v1/models >/dev/null && echo up) 2>/dev/null; pgrep -f '[f]ull_passk.py' | head -1; tail -n 1 /root/vllm011_install.log 2>/dev/null
echo ===EVAL; cat /root/passk_aug.jsonl 2>/dev/null
echo ===CKPT; ls -d /root/sft_out_aug/* 2>/dev/null
'''
def local(ts): return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
def ssh(cmd):
    r = subprocess.run(["ssh","-n","-o","BatchMode=yes","-o","ConnectTimeout=15","-o","ServerAliveInterval=10","-o","ServerAliveCountMax=3","-p",PORT,"-i",KEY,HOST,cmd],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
    return r.stdout
def sections(raw):
    out, cur = {}, None
    for l in raw.splitlines():
        if l.startswith("==="): cur = l[3:]; out[cur] = []
        elif cur: out[cur].append(l)
    return out
def eval_summary(rows, now):
    total = 460
    if not rows: return {"rollouts_done": 0, "rollouts_total": total}
    rows = sorted(rows, key=lambda r: r["finished_at"]); passed = [1 if r["reward"] >= 1 else 0 for r in rows]
    by_task, per_trial = {}, {}
    for r, ok in zip(rows, passed):
        by_task.setdefault(r["task_index"], []).append(ok); x = per_trial.setdefault(str(r["trial"]), [0, 0]); x[0] += 1; x[1] += ok
    st = min(r.get("started_run_at", r["finished_at"]) for r in rows); span = (rows[-1]["finished_at"] - st) / 60
    rate = len(rows) / span if span > 0 and len(rows) >= 6 else None
    eta = (total - len(rows)) / rate if rate else None
    return {"rollouts_done": len(rows), "rollouts_total": total, "passed": sum(passed), "pass1": sum(passed) / len(rows),
            "running_pass1": [sum(passed[:i + 1]) / (i + 1) for i in range(len(rows))], "by_task": {str(k): v for k, v in by_task.items()},
            "per_trial": {k: {"done": v[0], "passed": v[1]} for k, v in sorted(per_trial.items())}, "rollouts_per_min": rate, "eta_min": eta,
            "eta_at": datetime.datetime.fromtimestamp(now + eta * 60).strftime("%H:%M") if eta else None}
PREP_MIN, EVAL_PACE_ASSUMED, COPY_MIN = 12, 3.5, 8
def make_plan(stage, train_eta, ev, now):
    """Remaining time per phase. Evaluation pace is assumed (3.5 rollouts/min, the earlier 4-trial runs managed 3-4.5) until its rollouts really run."""
    train_left = 0 if stage.startswith(("training finished", "starting vLLM", "evaluating", "finished")) else (train_eta if train_eta is not None else 5.0 * 60)
    prep_left = 0 if stage.startswith(("evaluating", "finished")) else PREP_MIN
    done = ev.get("rollouts_done", 0); total = ev.get("rollouts_total", 460)
    pace = ev.get("rollouts_per_min") or EVAL_PACE_ASSUMED
    eval_left = 0 if stage.startswith("finished") else (total - done) / pace
    steps, cum = [], 0.0
    for label, mins, note in (("training", train_left, "measured pace" if train_eta is not None else "assumed 5 h"), ("vLLM startup and model loading", prep_left, "assumed"),
                              ("evaluation: 460 rollouts (115 tasks x 4 trials)", eval_left, "measured pace" if ev.get("rollouts_per_min") else f"assumed {EVAL_PACE_ASSUMED} rollouts/min (likely range 1.5-2.5 h)"),
                              ("copy results to the Mac", 0 if stage.startswith("finished") else COPY_MIN, "assumed")):
        cum += mins
        steps.append({"step": label, "minutes": mins, "basis": note, "done_in_min": cum, "done_at": datetime.datetime.fromtimestamp(now + cum * 60).strftime("%H:%M") if mins else None})
    return {"steps": steps, "total_min": cum, "done_at": datetime.datetime.fromtimestamp(now + cum * 60).strftime("%a %H:%M")}
def build(prev):
    now = time.time()
    try: raw = ssh(REMOTE); S = sections(raw) if "===GPU" in raw else None
    except Exception: S = None
    if S is None:
        o = dict(prev or {}); o.update({"updated": local(now), "pod_reachable": False}); return o
    steps = [json.loads(l) for l in S.get("STEPS", []) if l.startswith("{")]
    pipe = "\n".join(S.get("PIPE", []))
    total = steps[-1]["of"] if steps else 511
    done = steps[-1]["step"] if steps else 0
    sec = (steps[-1]["elapsed_min"] * 60 / done) if done else None
    eta = (total - done) * sec / 60 if sec else None
    procs = [x for x in S.get("PROCS", []) if x.strip()]
    ev_rows = [json.loads(l) for l in S.get("EVAL", []) if l.startswith("{")]
    if "EVAL_DONE" in pipe: stage = "finished (training and evaluation)"
    elif "FAILED" in pipe or any("TRAIN_ENV_FAILED" in l for l in S.get("ENV", [])): stage = "failed (see events)"
    elif ev_rows or "eval starts" in pipe: stage = "evaluating on the retail test (4 trials)"
    elif "tau-bench installed" in pipe: stage = "starting vLLM for the evaluation"
    elif "TRAINING_DONE" in pipe: stage = "training finished, preparing the evaluation"
    elif steps and procs: stage = "training"
    elif "training starts" in pipe: stage = "loading the model and building samples"
    else: stage = "installing the training environment"
    ev = []
    for l in S.get("PIPE", []):
        p = l.split(" ", 1)
        if len(p) == 2 and p[0].endswith("Z"):
            try: ev.append({"time": local(datetime.datetime.fromisoformat(p[0].replace("Z", "+00:00")).timestamp()), "event": p[1][:200]})
            except Exception: pass
    gpu = (S.get("GPU") or [""])[0].split(", ")
    return {"updated": local(now), "pod_reachable": True, "stage": stage,
            "pod": {"host": f"{HOST}:{PORT}", "gpu_name": gpu[0] if gpu else None, "gpu_util_pct": gpu[1] if len(gpu) > 1 else None,
                    "gpu_mem_used": gpu[2] if len(gpu) > 2 else None, "gpu_mem_total": gpu[3] if len(gpu) > 3 else None, "gpu_temp_c": gpu[4] if len(gpu) > 4 else None,
                    "disk": (S.get("DISK") or [""])[0], "env_tail": S.get("ENV", []), "train_log_tail": S.get("TRAINLOG", [])},
            "config": {"base": "unsloth/Qwen3-14B-unsloth-bnb-4bit (fresh LoRA r=16)", "data": "teacher32b_passed_sft_lookups_confirm.json (178 conversations, 4086 samples)",
                       "lr": 5e-5, "batch": 8, "epochs": 1, "max_tokens": 20000, "reference_run": "distill-epoch-1: 384 steps, 193 min on an A100"},
            "training": {"steps_done": done, "steps_total": total, "sec_per_step": sec, "elapsed_min": steps[-1]["elapsed_min"] if steps else 0, "eta_min": eta,
                         "eta_at": datetime.datetime.fromtimestamp(now + eta * 60).strftime("%H:%M") if eta else None,
                         "step": [s["step"] for s in steps], "loss": [s["loss"] for s in steps], "grad_norm": [s["grad_norm"] for s in steps], "lr": [s["lr"] for s in steps],
                         "tokens": [s["tokens"] for s in steps], "samples": [s["samples"] for s in steps], "elapsed": [s["elapsed_min"] for s in steps],
                         "checkpoints": [os.path.basename(x) for x in S.get("CKPT", [])]},
            "eval": eval_summary(ev_rows, now), "plan": make_plan(stage, eta, eval_summary(ev_rows, now), now), "events": ev}
def write(o):
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)
if __name__ == "__main__":
    prev = json.load(open(OUT)) if os.path.exists(OUT) else None
    while True:
        o = build(prev); write(o); prev = o; print(o["updated"], o.get("stage"), flush=True)
        if "--once" in sys.argv or str(o.get("stage", "")).startswith(("finished", "failed")): break
        time.sleep(30)
