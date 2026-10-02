"""SFT of Qwen3-14B (4-bit QLoRA, unsloth) on per-turn samples built from the 32B teacher's passed conversations.

Run inside the training venv on the pod: /root/train_env/bin/python train_sft.py [--epochs 2] [--lr 5e-5] [--batch 8] ...
Loss is taken on the assistant completion only (thinking + text + tool call), token-weighted over each optimizer step.
Saves the adapter after every epoch to <out>/epoch-N and logs every step to <out>/train_log.jsonl; writes <out>/SFT_DONE at the end.
"""
import argparse, json, math, os, random, time

import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel

from sft_samples import build_samples

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/teacher32b_passed_sft_reasoning.json")
ap.add_argument("--out", default="/root/sft_out")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=8, help="samples per optimizer step")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max_tokens", type=int, default=20000)
ap.add_argument("--save_every", type=int, default=96, help="also save the adapter every N steps (step-N)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
random.seed(0)
torch.manual_seed(0)

print("loading the 4-bit base model...", flush=True)
model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3-14B-unsloth-bnb-4bit",
                                               max_seq_length=args.max_tokens, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=args.rank, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=args.rank, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
FastLanguageModel.for_training(model)

data = json.load(open(args.data))
samples, skipped = [], {"template_mismatch": 0, "too_long": 0}
for d in data:
    s, sk = build_samples(d["messages"], d["tools"], tok, args.max_tokens)
    samples += s
    for k in skipped:
        skipped[k] += sk[k]
n_comp = sum(len(s["completion_ids"]) for s in samples)
print(f"{len(data)} conversations -> {len(samples)} per-turn samples ({n_comp} completion tokens), skipped {skipped}", flush=True)
if not samples:
    raise SystemExit("no samples")

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(samples) / args.batch)
total_steps = args.epochs * steps_per_epoch
warmup = max(1, min(5, total_steps // 10))


def lr_at(step):  # linear warmup, then linear decay to 10%
    if step < warmup:
        return args.lr * (step + 1) / warmup
    return args.lr * (1 - 0.9 * (step - warmup) / max(total_steps - warmup, 1))


log = open(os.path.join(args.out, "train_log.jsonl"), "a")
step, t0 = 0, time.time()
for epoch in range(args.epochs):
    random.shuffle(samples)
    for b in range(0, len(samples), args.batch):
        chunk = samples[b:b + args.batch]
        n_tok = sum(len(s["completion_ids"]) for s in chunk)
        opt.zero_grad()
        loss_val = 0.0
        for s in chunk:
            p_len = len(s["prompt_ids"])
            ids = torch.tensor([s["prompt_ids"] + s["completion_ids"]], device=model.device)
            out = model(input_ids=ids)
            logits = out.logits[0, p_len - 1:-1]          # positions that predict the completion tokens
            loss = F.cross_entropy(logits.float(), ids[0, p_len:], reduction="sum") / n_tok
            loss.backward()
            loss_val += loss.item()
            del out, logits, loss
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        step += 1
        rec = {"step": step, "of": total_steps, "epoch": epoch + 1, "loss": round(loss_val, 4), "grad_norm": round(float(gn), 3),
               "lr": lr_at(step - 1), "samples": len(chunk), "tokens": n_tok, "elapsed_min": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if args.save_every and step % args.save_every == 0 and step < total_steps:
            path = os.path.join(args.out, f"step-{step}")
            model.save_pretrained(path)
            print("saved", path, flush=True)
    path = os.path.join(args.out, f"epoch-{epoch + 1}")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    print("saved", path, flush=True)
open(os.path.join(args.out, "SFT_DONE"), "w").write("done\n")
print("SFT_DONE", flush=True)
