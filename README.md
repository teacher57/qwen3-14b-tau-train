# Distilling Qwen3-32B into Qwen3-14B on τ-bench retail

This branch contains **one experiment only**: a Qwen3-32B-AWQ teacher plays hard retail *train* tasks, its passing
conversations are used to fine-tune Qwen3-14B (QLoRA), and the 14B is tested on the retail test split against a
same-pod control. Earlier experiments (SFT, GRPO) are on `main`.

**Start here: `notebooks/distillation_experiment.ipynb`** (all charts are matplotlib; a standalone copy is
`renders/distillation_experiment.html`). It recomputes every number from the files in this branch.

## Recipe

1. **Teacher rollouts.** Qwen3-32B-AWQ (vLLM, hermes tool parser, qwen3 reasoning parser; GPT-4o as the simulated customer,
   25 steps) plays 114 hard train tasks (tasks the 14B solved rarely or never, plus mixed ones), up to 4 samples each,
   sampling of a task stopped after 2 passes. It solved **86 of 114 tasks** (199 passing rollouts); 180 conversations
   went into SFT.
2. **SFT.** One example per assistant turn (prompt = chat template of the earlier messages, completion = `<think>` reasoning,
   tool call or reply, `<|im_end|>`), loss on the completion only. QLoRA r=16 on `unsloth/Qwen3-14B-unsloth-bnb-4bit`,
   lr 5e-5, batch 8, 1 epoch = 384 steps (3.2 h on one A100), loss 0.53 → 0.35.
3. **Test.** Full retail test split (115 tasks), **4 trials per task**, temperature 0.7, GPT-4o user, for the new adapter and
   for the starting adapter as a control **re-run on the same pod with identical settings** (vLLM 0.11.0, LoRA adapters with
   `--enable-lora --max-loras 3`).

## Result

| retail test, 115 tasks | new model | starting adapter, same-pod re-run | starting adapter, earlier test (2 trials, other pod) |
|---|---|---|---|
| pass^1 | **45.7% ± 3.4** | 41.7% ± 3.4 | 37.8% ± 3.7 |
| pass^2 | 29.9% | 27.1% | 21.7% |
| pass^3 | 22.0% | 20.2% | n/a |
| pass^4 | 16.5% | 16.5% | n/a |

Paired by task:

| comparison | difference in pass^1 | 95% interval | permutation p |
|---|---|---|---|
| new vs same-pod control, 4 trials | **+3.9** | -1.1 to +8.9 | 0.15 |
| new vs earlier test, first 2 trials (matched) | +8.7 | +1.3 to +16.1 | 0.03 |
| control vs earlier test, first 2 trials (matched) | +2.2 | -5.2 to +9.6 | 0.65 |

Reading it honestly: against the same-pod control the gain is **+3.9 points and not statistically significant**
(sign test on pass^2: 31 tasks better, 23 worse, p = 0.34; pass^4 is identical). Against the earlier baseline measurement the
gain looks nominally significant, but the same starting adapter re-run on the new pod already scored +2.2 points above that
earlier test, so part of that gap is pod-to-pod drift. The new model led the control by +6.5 points on trials 1-2 and by
+1.3 on trials 3-4, so the early lead partly regressed with more sampling. The difference is concentrated in non-combo tasks
(+7.4 points vs +1.2 on the 64 combo tasks), even though training used only combo-task conversations; at this sample size that
is a hypothesis, not a finding. The airline test was started and stopped after 10 rollouts: there is no airline comparison.

## Files

```
notebooks/distillation_experiment.ipynb   matplotlib notebook (executed, outputs included)
renders/distillation_experiment.html      the same notebook as a standalone page
renders/retail_test_new_solves.html       tasks the new model solves that the control does not (earlier analysis)
results/distill/distill_run.json          4-trial retail test: per-trial, per-type, comparisons, training curves
results/distill/teacher_run.json          teacher rollouts per task and timing
results/distill/new_solves_analysis.json  per-task analysis (made when the new model had 2 trials; the notebook recomputes with 4)
results/passk_run.json                    the earlier baseline test (per-rollout results only)
dataset/distill/retail_eval_conversations/   ALL conversations of the 4-trial retail test (460 per model) + per-rollout results
dataset/distill/teacher_rollouts_raw.jsonl   all teacher conversations (passed or not)
dataset/distill/teacher32b_passed_sft*.json  the 180 passing conversations as SFT data (with and without reasoning)
dataset/distill/*.json                       task lists and task definitions used
code/distill/    teacher rollouts, SFT export/training, pod setup and eval chains, JSON writer
code/full_passk.py   pass^k eval of an adapter on the retail/airline test split (trials, temperature, env configurable)
```

The trained adapter (257 MB) is not in the repo. The conversations of the earliest baseline test (37.8%) were not kept, only its
rewards. Pod addresses in the scripts are defaults for pods that no longer exist; API keys come from the environment and are
never stored.
