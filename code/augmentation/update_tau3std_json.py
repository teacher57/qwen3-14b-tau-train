"""Keeps full_run_results/tau3_std_run.json current for the tau3 standard-settings run (baseline vs augmented) on the pod.
The notebook aug_sft_experiment.ipynb (Part 4) only READS that JSON.   python3 update_tau3std_json.py [--once]
Env: STD_HOST (default root@154.54.102.23), STD_PORT (default 16842)."""
import datetime, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tau3"))
import update_tau3_json as U
HERE = os.path.dirname(os.path.abspath(__file__))
U.OUT = os.path.join(HERE, "..", "full_run_results", "tau3_std_run.json")
U.HOST, U.PORT = os.environ.get("STD_HOST", "root@154.54.102.23"), os.environ.get("STD_PORT", "16842")
U.N_TASKS, U.TRIALS = 114, 2
N, T = U.N_TASKS, U.TRIALS
def build(prev):
    now = time.time()
    try:
        raw = U.ssh("python3 /root/summarize_std_remote.py"); remote = json.loads(raw) if raw.strip().startswith("{") else None
    except Exception: remote = None
    if remote is None:
        o = dict(prev or {}); o.update({"updated": U.local_iso(now), "pod_reachable": False}); return o
    runs = remote.get("runs", {})
    base, aug = U.summarize(runs.get("tau3std_base")), U.summarize(runs.get("tau3std_aug"))
    log = remote.get("pipeline_log", []); text = "\n".join(log); procs = remote.get("procs", {})
    ev = []
    for l in log:
        p = l.split(" ", 1)
        if len(p) == 2 and p[0].endswith("Z"):
            try: ev.append({"time": U.local_iso(datetime.datetime.fromisoformat(p[0].replace("Z", "+00:00")).timestamp()), "event": p[1][:200]})
            except Exception: pass
    if "FAILED" in text: stage = "failed (see events)"
    elif "TAU3_STD_DONE" in text: stage = "finished"
    elif aug["rollouts_done"] > 0 or "tau3 run tau3std_aug started" in text: stage = "running the augmented model"
    elif base["rollouts_done"] >= N * T and procs.get("run"): stage = "switching to the augmented model"
    elif base["rollouts_done"] > 0 or "tau3 run tau3std_base started" in text: stage = "running the baseline (starting adapter)"
    elif "vllm ready" in text: stage = "vLLM ready, starting the baseline"
    elif "setup done" in text: stage = "starting vLLM"
    else: stage = "setting up the pod"
    pace = (aug if stage.startswith("running the aug") else base).get("rollouts_per_min") or base.get("rollouts_per_min") or 3.0
    steps, cum = [], 0.0
    for label, e in (("baseline (starting adapter)", base), ("augmented model", aug)):
        rem = max(N * T - e["rollouts_done"], 0); mins = rem / pace; cum += mins
        steps.append({"step": label, "remaining": rem, "minutes": mins, "done_in_min": cum, "done_at": datetime.datetime.fromtimestamp(now + cum * 60).strftime("%H:%M") if rem else None})
    pod = {"host": f"{U.HOST}:{U.PORT}", "gpu": remote.get("gpu"), "disk": remote.get("disk"), "vllm_up": remote.get("vllm_up"), "procs": procs,
           "setup_tail": remote.get("setup_tail"), "vllm_log_tail": remote.get("vllm_log_tail")}
    return {"updated": U.local_iso(now), "pod_reachable": True, "stage": stage, "pod": pod,
            "config": {"benchmark": "tau3 (tau2-bench) retail, 114 tasks (split base)", "agent_temperature": 0.0, "max_steps": 200, "user": "gpt-4.1 (default temperature)", "trials": T,
                       "concurrency": 32, "baseline": "starting adapter (round-0)", "augmented": "aug-epoch-1 (lookup + confirmation augmented SFT)"},
            "old": base, "new": aug, "paired": U.paired(base, aug), "plan": {"steps": steps, "pace_used": pace, "total_min": cum, "done_at": steps[-1]["done_at"] or steps[0]["done_at"]}, "events": ev}
if __name__ == "__main__":
    prev = json.load(open(U.OUT)) if os.path.exists(U.OUT) else None
    while True:
        o = build(prev); U.write(o); prev = o; print(o.get("updated"), o.get("stage"), flush=True)
        if "--once" in sys.argv or str(o.get("stage", "")).startswith(("finished", "failed")): break
        time.sleep(30)
