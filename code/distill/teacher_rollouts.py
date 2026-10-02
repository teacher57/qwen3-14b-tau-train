"""Teacher rollouts: run a served model on a list of tau-bench retail TRAIN tasks, several samples each, and save the
full conversation of every rollout (reward + messages) so passing ones can become SFT data.

Usage: python3 teacher_rollouts.py <tasks.json> [trials=4] [temperature=0.7] [workers=24] [served_name=teacher32b] [stop_after_passes=2]
A task is no longer sampled once it has `stop_after_passes` passing rollouts (its remaining trials are logged as skipped).
Jobs run in rounds (every task's trial 0, then trial 1, ...) so that skipping actually saves work.
Resumable: one JSON line per finished (task, trial) in /root/teacher_rollouts.jsonl (messages included).
"""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent

litellm.request_timeout = 3600
os.environ["HOSTED_VLLM_API_BASE"] = "http://localhost:8000/v1"
os.environ["HOSTED_VLLM_API_KEY"] = "dummy"

tasks = json.load(open(sys.argv[1]))
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 4
temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 24
served = sys.argv[5] if len(sys.argv) > 5 else "teacher32b"
stop_at = int(sys.argv[6]) if len(sys.argv) > 6 else 2
OUT, MAX_STEPS = os.environ.get("TEACHER_OUT", "/root/teacher_rollouts.jsonl"), 30

done = set()
passes = {}
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            r = json.loads(l)
            done.add((r["task_index"], r["trial"]))
            if (r.get("reward") or 0) >= 1:
                passes[r["task_index"]] = passes.get(r["task_index"], 0) + 1
lock = threading.Lock()
start = time.time()


def one(job):
    tid, k = job
    with lock:
        enough = passes.get(tid, 0) >= stop_at
    if enough:
        rec = {"task_index": tid, "trial": k, "reward": None, "skipped": True, "messages": [], "seconds": 0,
               "finished_at": time.time(), "started_run_at": start, "temperature": temperature, "model": served}
        with lock:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec
    t0 = time.time()
    env = get_env("retail", user_strategy="llm", user_model="gpt-4o", user_provider="openai",
                  task_split="train", task_index=tid)
    agent = ToolCallingAgent(tools_info=env.tools_info, wiki=env.wiki, model=f"hosted_vllm/{served}",
                             provider="hosted_vllm", temperature=temperature)
    try:
        res = agent.solve(env=env, task_index=tid, max_num_steps=MAX_STEPS)
        rec = {"task_index": tid, "trial": k, "reward": res.reward, "messages": res.messages}
    except Exception as e:
        rec = {"task_index": tid, "trial": k, "reward": 0.0, "messages": [], "error": str(e)[:300]}
    rec.update(seconds=round(time.time() - t0, 1), finished_at=time.time(), started_run_at=start,
               temperature=temperature, model=served)
    with lock:
        if (rec["reward"] or 0) >= 1:
            passes[tid] = passes.get(tid, 0) + 1
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


jobs = [(t, k) for k in range(trials) for t in tasks if (t, k) not in done]
print(f"{len(jobs)} rollouts to run ({len(done)} already done), {workers} workers, temp {temperature}", flush=True)
n_ok = 0
with ThreadPoolExecutor(workers) as ex:
    for i, f in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
        n_ok += (f.result()["reward"] or 0) >= 1
        if i % 10 == 0:
            print(f"progress {i}/{len(jobs)} passed so far {n_ok}", flush=True)
print(f"TEACHER_DONE passed={n_ok} of {len(jobs)}", flush=True)
