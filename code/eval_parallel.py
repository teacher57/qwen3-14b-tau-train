"""
Lightweight eval probe for a GRPO checkpoint, run WITHOUT interrupting training.

Loads the given checkpoint into the already-running vLLM server under its own
adapter name (distinct from "policy", the live training adapter, so this never
touches what training is actively updating/reading). Runs a fixed subset of
real tau-bench retail TEST-split tasks (never seen during training, which only
samples from the train split) at temperature=0 for a deterministic read, using
the same reward tau-bench itself reports. Unloads the probe adapter when done
so it doesn't accumulate GPU memory across many checkpoints.

Usage: python3 eval_checkpoint.py <checkpoint_dir> <round_number>
"""
import sys, json, os, requests
import litellm
litellm.request_timeout = 3600
from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent

VLLM_BASE = "http://localhost:8000/v1"
os.environ["HOSTED_VLLM_API_BASE"] = VLLM_BASE
os.environ["HOSTED_VLLM_API_KEY"] = "dummy"

# fixed subset of real retail TEST tasks for a fast, consistent probe across
# checkpoints (task 0 is a known hang in this harness, skipped)
PROBE_TASK_IDS = list(range(1, 21))

def main():
    ckpt_dir = sys.argv[1]
    round_num = sys.argv[2]
    adapter_name = f"eval-round-{round_num}"

    r = requests.post(f"{VLLM_BASE.replace('/v1','')}/v1/load_lora_adapter",
                       json={"lora_name": adapter_name, "lora_path": ckpt_dir}, timeout=180)
    print(f"load adapter -> {r.status_code} {r.text[:200]}", flush=True)

    transcript_path = f"/root/grpo_eval_probe_transcripts_round_{round_num}.json"

    from concurrent.futures import ThreadPoolExecutor
    def one(tid):
        env = get_env("retail", user_strategy="llm", user_model="gpt-4o", user_provider="openai",
                       task_split="test", task_index=tid)
        agent = ToolCallingAgent(tools_info=env.tools_info, wiki=env.wiki,
                                  model=f"hosted_vllm/{adapter_name}", provider="hosted_vllm", temperature=0.0)
        try:
            res = agent.solve(env=env, task_index=tid, max_num_steps=20)
            return {"task_index": tid, "reward": res.reward, "messages": res.messages}
        except Exception as e:
            return {"task_index": tid, "reward": 0.0, "messages": [], "error": str(e)}
    with ThreadPoolExecutor(5) as ex:
        transcripts = list(ex.map(one, PROBE_TASK_IDS))
    rewards = [t["reward"] for t in transcripts]
    for t in transcripts:
        print(f"  task {t['task_index']}: reward={t['reward']}", flush=True)
    with open(transcript_path, "w") as f:
        json.dump(transcripts, f, indent=1)

    avg = sum(rewards) / len(rewards)
    print(f"PROBE_RESULT round={round_num} avg_reward={avg:.4f} n_tasks={len(rewards)}", flush=True)

    with open("/root/grpo_eval_probe.log.jsonl", "a") as f:
        f.write(json.dumps({"round": int(round_num), "probe_avg_reward": avg, "n_tasks": len(rewards)}) + "\n")

    try:
        requests.post(f"{VLLM_BASE.replace('/v1','')}/v1/unload_lora_adapter",
                       json={"lora_name": adapter_name}, timeout=15)
    except Exception:
        pass

if __name__ == "__main__":
    main()
