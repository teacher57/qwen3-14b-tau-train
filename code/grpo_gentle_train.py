"""
GRPO training for Qwen3-14B on tau-bench retail, reward = tau-bench's own
calculate_reward() (0/1 per episode, from final-DB-state hash match against
ground truth -- see tau_bench/envs/base.py).

Design:
  - vLLM serves the base model + a LoRA adapter ("policy") on localhost:8000,
    reachable via litellm's "hosted_vllm" provider (isolated from the
    OPENAI_API_BASE used for the GPT-4o user simulator, so both can run
    concurrently without clobbering each other's endpoint).
  - Rollout: tau_bench's own ToolCallingAgent.solve() is reused UNMODIFIED
    against a real tau_bench retail Env + GPT-4o user, for G rollouts per
    task (temperature=1.0 for exploration). This gives us the exact reward
    tau-bench itself would report.
  - Advantage: standard GRPO group-relative normalization -- per task,
    A_i = (r_i - mean(r_group)) / (std(r_group) + eps). Tasks whose G
    rollouts all got the same reward carry no signal and are skipped.
  - Update: vLLM did the generation, so we recompute logprobs of the exact
    assistant tokens it produced via a teacher-forced forward pass on a
    separate unsloth-loaded copy of the model (LoRA weights kept in sync by
    saving to disk after each round and hot-reloading into vLLM). Loss per
    rollout = -advantage * mean(logprob(assistant tokens)), matching the
    train_on_responses_only masking used for SFT (only assistant spans
    contribute; user/tool/system tokens are masked out).
"""
import json, re, random, time, os, requests, sys
from concurrent.futures import ThreadPoolExecutor
import torch
import litellm
litellm.request_timeout = 3600
from unsloth import FastLanguageModel
from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent

random.seed(0)

VLLM_BASE = "http://localhost:8000/v1"
LORA_NAME = "policy"
LORA_DIR = "/root/grpo_lora"
LOG_PATH = "/root/grpo_train.log.jsonl"

os.environ["HOSTED_VLLM_API_BASE"] = VLLM_BASE
os.environ["HOSTED_VLLM_API_KEY"] = "dummy"

N_ROUNDS = 15
TASKS_PER_ROUND = 4
GROUP_SIZE = 4          # G rollouts per task
MAX_NUM_STEPS = 20
TEMPERATURE = 1.0
LR = 1e-6
GRAD_CLIP = 1.0
POOL = json.load(open("/root/combo_pool.json"))  # the 56 mixed pass/fail combo train tasks

print("loading trainer-side model (unsloth) for gradient updates...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-14B-unsloth-bnb-4bit",
    max_seq_length=16384, load_in_4bit=True, dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
_missing = set_peft_model_state_dict(model, load_file("/root/round0_ckpt/adapter_model.safetensors"))
print("loaded round-0 adapter weights; unexpected keys:", len(getattr(_missing, "unexpected_keys", [])), flush=True)
FastLanguageModel.for_training(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def save_and_reload_lora():
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    try:
        requests.post(f"{VLLM_BASE.replace('/v1','')}/v1/unload_lora_adapter",
                       json={"lora_name": LORA_NAME}, timeout=10)
    except Exception:
        pass
    try:
        r = requests.post(f"{VLLM_BASE.replace('/v1','')}/v1/load_lora_adapter",
                           json={"lora_name": LORA_NAME, "lora_path": LORA_DIR}, timeout=180)
        print(f"  lora reload -> {r.status_code} {r.text[:200]}", flush=True)
    except requests.exceptions.ReadTimeout:
        # the server can legitimately take >60s to swap in a 14B-scale LoRA;
        # a client-side timeout here doesn't mean the load failed -- verify
        # via /v1/models instead of crashing the whole training run on it
        models = requests.get(f"{VLLM_BASE}/models", timeout=30).json()
        loaded = any(m["id"] == LORA_NAME for m in models.get("data", []))
        print(f"  lora reload -> client timed out, server-side loaded={loaded}", flush=True)

def run_one_rollout(task_index):
    env = get_env(
        "retail", user_strategy="llm", user_model="gpt-4o", user_provider="openai",
        task_split="train", task_index=task_index,
    )
    agent = ToolCallingAgent(
        tools_info=env.tools_info, wiki=env.wiki,
        model=f"hosted_vllm/{LORA_NAME}", provider="hosted_vllm", temperature=TEMPERATURE,
    )
    try:
        res = agent.solve(env=env, task_index=task_index, max_num_steps=MAX_NUM_STEPS)
        return {"task_index": task_index, "reward": res.reward, "messages": res.messages}
    except Exception as e:
        print(f"  [rollout error] task {task_index}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return {"task_index": task_index, "reward": 0.0, "messages": [], "error": str(e)}

def sanitize_messages(messages):
    # litellm's raw message dump has content=None for assistant turns that made
    # a tool call (only pure-text replies have a string) -- Qwen3's chat
    # template chokes on None content ('NoneType' object is not subscriptable).
    # Strip down to only the fields the template actually needs, same as the
    # SFT script's sanitize().
    clean = []
    for m in messages:
        nm = {"role": m["role"], "content": m.get("content") if m.get("content") is not None else ""}
        if m.get("tool_calls"):
            nm["tool_calls"] = m["tool_calls"]
        if m.get("role") == "tool":
            nm["tool_call_id"] = m.get("tool_call_id")
            nm["name"] = m.get("name")
        clean.append(nm)
    return clean

def find_assistant_spans(messages):
    """Tokenize the full conversation with the chat template, return
    (input_ids, list of (start, end) token spans belonging to assistant turns)."""
    messages = sanitize_messages(messages)
    ids = tokenizer.apply_chat_template(messages, tools=None, tokenize=True, add_generation_prompt=False)
    # locate each assistant turn's token span by re-templating incrementally
    spans = []
    running = []
    for i, m in enumerate(messages):
        prefix_ids = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False) if i > 0 else []
        upto_ids = tokenizer.apply_chat_template(messages[:i+1], tokenize=True, add_generation_prompt=False)
        if m.get("role") == "assistant":
            spans.append((len(prefix_ids), len(upto_ids)))
    return ids, spans

def compute_policy_loss(messages, advantage):
    if abs(advantage) < 1e-6:
        return None
    try:
        ids, spans = find_assistant_spans(messages)
    except Exception as e:
        import traceback
        print(f"  [skip] template/span error: {e}", flush=True)
        traceback.print_exc()
        return None
    if not spans:
        return None
    input_ids = torch.tensor([ids], device=model.device)
    out = model(input_ids=input_ids)
    # only materialise float32 log-probs at the assistant-token positions (the full
    # seq x vocab float32 copy is what ran the 44GB card out of memory)
    pos = [t - 1 for start, end in spans for t in range(max(start, 1), end)]
    n_tokens = len(pos)
    if n_tokens == 0:
        return None
    pos_t = torch.tensor(pos, device=model.device)
    tgt = input_ids[0][pos_t + 1]
    sel = out.logits[0][pos_t].float()
    total_logprob = -torch.nn.functional.cross_entropy(sel, tgt, reduction="sum")
    del out, sel
    mean_logprob = total_logprob / n_tokens
    loss = -advantage * mean_logprob
    return loss

def task_order_generator():
    # hand tasks out from a full random shuffle of all NUM_TRAIN_TASKS, epoch
    # style, instead of resampling independently each round (so every task
    # gets visited an even number of times, not just whichever happens to be
    # drawn)
    while True:
        order = list(POOL)
        random.shuffle(order)
        for tid in order:
            yield tid

def main():
    print("registering initial LoRA adapter with vLLM before any rollouts...", flush=True)
    save_and_reload_lora()  # so "hosted_vllm/policy" resolves from round 0 onward

    task_stream = task_order_generator()
    with open(LOG_PATH, "a") as logf:
        for round_i in range(N_ROUNDS):
            task_ids = [next(task_stream) for _ in range(TASKS_PER_ROUND)]
            print(f"\n=== round {round_i} tasks={task_ids} ===", flush=True)
            all_rollouts = []
            with ThreadPoolExecutor(8) as ex:
                futs = {tid: [ex.submit(run_one_rollout, tid) for _ in range(GROUP_SIZE)] for tid in task_ids}
            for tid in task_ids:
                group = [f.result() for f in futs[tid]]
                rewards = [g["reward"] for g in group]
                mean_r = sum(rewards) / len(rewards)
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
                for g in group:
                    g["advantage"] = 0.0 if std_r < 1e-6 else (g["reward"] - mean_r) / (std_r + 1e-6)
                print(f"  task {tid}: rewards={rewards} mean={mean_r:.2f}", flush=True)
                all_rollouts.extend(group)

            optimizer.zero_grad()
            losses = []
            for r in all_rollouts:
                if not r["messages"]:
                    continue
                loss = compute_policy_loss(r["messages"], r["advantage"])
                if loss is not None:
                    loss.backward()
                    torch.cuda.empty_cache()
                    losses.append(loss.item())
            if losses:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            avg_reward = sum(r["reward"] for r in all_rollouts) / len(all_rollouts)
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            print(f"  round {round_i}: avg_reward={avg_reward:.3f} avg_loss={avg_loss:.4f} n_updated={len(losses)}/{len(all_rollouts)}", flush=True)
            logf.write(json.dumps({"round": round_i, "avg_reward": avg_reward, "avg_loss": avg_loss, "n_updated": len(losses)}) + "\n")
            logf.flush()

            save_and_reload_lora()
            if round_i in (1, 3) or (round_i % 5 == 0 and round_i > 0):
                model.save_pretrained(f"/root/grpo_checkpoints/round-{round_i}")

if __name__ == "__main__":
    main()
