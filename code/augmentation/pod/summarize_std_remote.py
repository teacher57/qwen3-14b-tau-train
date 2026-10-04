"""Runs ON THE POD: prints one compact JSON with pod state and a per-simulation summary of the tau2 runs (no conversations)."""
import glob, json, os, subprocess, time

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

out = {"now": time.time()}
out["pipeline_log"] = open("/root/pipeline.log").read().splitlines()[-80:] if os.path.exists("/root/pipeline.log") else []
out["setup_tail"] = sh("tail -n 2 /root/setup.log 2>/dev/null | cut -c1-200")
out["gpu"] = sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null")
out["disk"] = sh("df -h / | tail -1")
out["vllm_up"] = bool(sh("curl -s -m 3 localhost:8000/v1/models"))
out["procs"] = {"setup": bool(sh("pgrep -f '[s]etup_pod.sh'")), "run": bool(sh("pgrep -f '[r]un_tau3.sh'")),
                "tau2": bool(sh("pgrep -f '[t]au2 run'")), "vllm": bool(sh("pgrep -f '[v]llm serve'"))}
out["vllm_log_tail"] = sh("tail -n 2 /root/vllm14b.log 2>/dev/null | cut -c1-200")
runs = {}
for name in ("tau3std_base", "tau3std_aug"):
    cands = [p for p in glob.glob(f"/root/tau2-bench/data/simulations/{name}*") ]
    files = []
    for p in cands:
        files += [p] if p.endswith(".json") else glob.glob(p + "/results.json")
    if not files:
        continue
    path = sorted(files, key=os.path.getmtime)[-1]
    try:
        d = json.load(open(path))
    except Exception as e:  # file being rewritten
        runs[name] = {"error": f"unreadable right now: {e}"}
        continue
    tasks = {str(t["id"]): t for t in d.get("tasks", [])}
    rows = []
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        msgs = s.get("messages") or []
        t = tasks.get(str(s["task_id"]), {})
        acts = ((t.get("evaluation_criteria") or {}).get("actions")) or []
        rows.append({
            "task_id": str(s["task_id"]), "trial": s.get("trial"), "reward": ri.get("reward"), "term": s.get("termination_reason"),
            "duration": s.get("duration"), "agent_cost": s.get("agent_cost"), "user_cost": s.get("user_cost"),
            "end_time": s.get("end_time"), "start_time": s.get("start_time"), "n_messages": len(msgs),
            "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs),
            "breakdown": ri.get("reward_breakdown"), "db_match": (ri.get("db_check") or {}).get("db_match"),
            "n_target_actions": len(acts), "target_actions": [a.get("name") for a in acts],
        })
    info = d.get("info") or {}
    runs[name] = {"path": path, "n_tasks_in_file": len(tasks), "num_trials": info.get("num_trials"), "rows": rows,
                  "agent_llm": (info.get("agent_info") or {}).get("llm"), "user_llm": (info.get("user_info") or {}).get("llm")}
out["runs"] = runs
print(json.dumps(out))
