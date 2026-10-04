"""pass^k eval of one LoRA adapter on the full tau-bench retail TEST split (115 tasks), run on the pod.

Each task is run `trials` times at `temperature` (tool-calling agent, GPT-4o as the simulated user, 25 steps).
pass^k for a task = C(c,k)/C(n,k) where c of n trials passed; with n = k = 2 that is 1 only if both trials pass.
Resumable: one JSON line per finished (task, trial) in /root/passk_<name>.jsonl, full conversations in
/root/passk_<name>_transcripts.jsonl.

Usage: python3 full_passk.py <name> <adapter_dir> [trials=2] [temperature=0.7] [workers=48] [env=retail] [n_tasks=115]
(env "airline" has 50 test tasks.)
"""
import json, os, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent

litellm.request_timeout = 3600
VLLM = "http://localhost:8000"
os.environ["HOSTED_VLLM_API_BASE"] = VLLM + "/v1"
os.environ["HOSTED_VLLM_API_KEY"] = "dummy"

name, ckpt = sys.argv[1], sys.argv[2]
trials = int(sys.argv[3]) if len(sys.argv) > 3 else 2
temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 48
env_name = sys.argv[6] if len(sys.argv) > 6 else "retail"
n_tasks = int(sys.argv[7]) if len(sys.argv) > 7 else 115
adapter = f"passk-{name}"
OUT = f"/root/passk_{name}.jsonl"
TR = f"/root/passk_{name}_transcripts.jsonl"
N_TASKS, MAX_STEPS, TASK_DEADLINE = n_tasks, int(os.environ.get("MAX_STEPS", "25")), 45 * 60   # MAX_STEPS env: 30 is the tau-bench default

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            r = json.loads(l)
            done.add((r["task_index"], r["trial"]))

r = requests.post(f"{VLLM}/v1/load_lora_adapter", json={"lora_name": adapter, "lora_path": ckpt}, timeout=180)
print("load adapter ->", r.status_code, r.text[:120], flush=True)
lock = threading.Lock()
start = time.time()


def one(job):
    tid, k = job
    t0 = time.time()
    env = get_env(env_name, user_strategy="llm", user_model="gpt-4o", user_provider="openai",
                  task_split="test", task_index=tid)
    agent = ToolCallingAgent(tools_info=env.tools_info, wiki=env.wiki,
                             model=f"hosted_vllm/{adapter}", provider="hosted_vllm", temperature=temperature)
    try:
        res = agent.solve(env=env, task_index=tid, max_num_steps=MAX_STEPS)
        return {"task_index": tid, "trial": k, "reward": res.reward, "messages": res.messages, "t0": t0}
    except Exception as e:
        return {"task_index": tid, "trial": k, "reward": 0.0, "messages": [], "error": str(e)[:300], "t0": t0}


def record(rec):
    now = time.time()
    line = {"task_index": rec["task_index"], "trial": rec["trial"], "reward": rec["reward"],
            "n_messages": len(rec["messages"]), "seconds": round(now - rec["t0"], 1), "finished_at": now,
            "started_run_at": start, "temperature": temperature}
    if "error" in rec:
        line["error"] = rec["error"]
    with lock:
        with open(OUT, "a") as f:
            f.write(json.dumps(line) + "\n")
        with open(TR, "a") as f:
            f.write(json.dumps({"task_index": rec["task_index"], "trial": rec["trial"], "reward": rec["reward"],
                                "messages": rec["messages"]}) + "\n")


jobs = [(t, k) for t in range(N_TASKS) for k in range(trials) if (t, k) not in done]
print(f"{len(jobs)} rollouts to run ({len(done)} already done), {workers} workers, temp {temperature}", flush=True)
ex = ThreadPoolExecutor(workers)
futs = {ex.submit(one, j): j for j in jobs}
finished = set()
try:
    for f in as_completed(futs, timeout=TASK_DEADLINE * (len(jobs) / max(workers, 1) + 1)):
        record(f.result())
        finished.add(futs[f])
except Exception as e:
    print("stopped waiting:", type(e).__name__, flush=True)
for j in jobs:
    if j not in finished:
        record({"task_index": j[0], "trial": j[1], "reward": 0.0, "messages": [],
                "error": "did not finish before the deadline", "t0": start})
rows = [json.loads(l) for l in open(OUT) if l.strip()]
p1 = sum(r["reward"] >= 1 for r in rows) / len(rows)
print(f"PASSK_DONE name={name} pass1={p1:.4f} n={len(rows)}", flush=True)
ex.shutdown(wait=False, cancel_futures=True)
os._exit(0)
