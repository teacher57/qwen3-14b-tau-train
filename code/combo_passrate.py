"""Measure per-task pass rate of the round-0 adapter on the 153 combo TRAIN tasks.
N rollouts per task at temp=1.0 (same as GRPO exploration). Appends one JSON line per
rollout to /root/combo_passrate.jsonl so a crash loses nothing; resumable."""
import sys, json, os, requests, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent

VLLM_BASE = "http://localhost:8000/v1"
os.environ["HOSTED_VLLM_API_BASE"] = VLLM_BASE
os.environ["HOSTED_VLLM_API_KEY"] = "dummy"

ADAPTER = "round0"
CKPT = "/root/round0_ckpt"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = "/root/combo_passrate.jsonl"

tasks = [t["idx"] for t in json.load(open("/root/train_combos_full.json"))]

done = {}
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        done[(r["task_index"], r["rep"])] = True

r = requests.post("http://localhost:8000/v1/load_lora_adapter",
                  json={"lora_name": ADAPTER, "lora_path": CKPT}, timeout=180)
print("load adapter ->", r.status_code, r.text[:150], flush=True)

lock = threading.Lock()

def one(tid, rep):
    env = get_env("retail", user_strategy="llm", user_model="gpt-4o", user_provider="openai",
                  task_split="train", task_index=tid)
    agent = ToolCallingAgent(tools_info=env.tools_info, wiki=env.wiki,
                             model=f"hosted_vllm/{ADAPTER}", provider="hosted_vllm", temperature=1.0)
    try:
        res = agent.solve(env=env, task_index=tid, max_num_steps=20)
        rec = {"task_index": tid, "rep": rep, "reward": res.reward}
    except Exception as e:
        rec = {"task_index": tid, "rep": rep, "reward": 0.0, "error": str(e)[:200]}
    with lock:
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec

jobs = [(t, k) for t in tasks for k in range(N) if (t, k) not in done]
print(f"{len(jobs)} rollouts to run ({len(done)} already done)", flush=True)
n = 0
with ThreadPoolExecutor(WORKERS) as ex:
    futs = [ex.submit(one, t, k) for t, k in jobs]
    for f in as_completed(futs):
        n += 1
        if n % 20 == 0:
            print(f"progress {n}/{len(jobs)}", flush=True)
print("PASSRATE_DONE", flush=True)
