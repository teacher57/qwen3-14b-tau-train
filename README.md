# Qwen3-14B on τ-bench: a fine-tuning attempt that quietly broke everything

**TL;DR:** Fine-tuning Qwen3-14B on 422 real, passing multi-turn τ-bench conversations
took a model that solved 45% of retail tasks and made it solve *2.6%* — while training
loss looked fine. The cause wasn't the usual suspects (bad data format, wrong learning
rate). The model learned a rigid **positional habit** — "call the authentication
function at message #4" — from a training set where 98% of examples had that exact
same conversational shape. It fires that habit regardless of whether the user has
actually provided the info yet, with fabricated placeholder values when it hasn't.

Fixing the data (section 6) and then pivoting to RL entirely (section 7, GRPO with
τ-bench's own reward — no custom scoring) were both reasonable next moves. **Neither
one produced a model better than the lightly-trained starting point** — the GRPO run's
own objective eval probe shows pass rate *declining* with more training (0.55 → 0.35 →
0.30 across checkpoints), confirmed on an independent rerun. This repo documents all
three attempts, including the two that didn't pan out, with full transcripts and the
root-cause diagnosis for the one that did.

**Follow-up (section 8):** test tasks are far more "combo" (several actions in one
conversation) than our training data, so we re-ran GRPO on only the combo tasks the
model solves some of the time. The held-out probe fell the same way (0.45 → 0.20 by
round 5), and training stopped itself. A gentler run (10x lower learning rate) did not
fall, but a full 115-task test with two trials per task could not tell its round-5
checkpoint from the starting adapter (pass^2 25.2% vs 21.7%, p = 0.57): a null result.

LoRA checkpoint weights (~250MB each, too large for plain git) are on
[Hugging Face](https://huggingface.co/teacher57/qwen3-14b-tau-grpo-checkpoints).
Everything else — eval results, training data, training curves, and full
conversation transcripts — is in this repo.

## The benchmark

[τ-bench](https://github.com/sierra-research/tau-bench) (Sierra Research) evaluates a
tool-calling agent against a simulated user (here: GPT-4o) across realistic retail/airline
customer-service tasks, with strict programmatic grading — every ground-truth tool call
has to match exactly (name, arguments) for a task to count as passed.

## What we actually tried

### 1. Baseline
Qwen3-14B-AWQ, no fine-tuning, served via vLLM (`--tool-call-parser hermes`,
`--reasoning-parser qwen3`). Full retail test split (115 tasks), GPT-4o as user-simulator.

**Result: 45.2% pass rate (52/115).**

### 2. First fine-tuning attempt — single-turn ground-truth distillation

Built an SFT dataset for free by taking each retail *train*-split task's ground-truth
action list and replaying it through the real tool functions (deterministic, no LLM
calls needed) to get authentic tool outputs. The user turn was the task's full
instruction dumped as one upfront message.

Trained QLoRA (r=16) for 3 epochs. Training loss converged cleanly to 0.47.

**Result: 2.6% pass rate (3/115).** Catastrophic regression.

**Diagnosis:** every training example had the *complete* user request available in
turn 1. The model learned "the moment you see a user message, you have everything you
need — act immediately." Real τ-bench conversations reveal information gradually
(GPT-4o plays a user who answers what's asked, not more) — so at eval time the model
fired tool calls with hallucinated argument values instead of asking for missing info.

### 3. Fix: real multi-turn rollouts via rejection sampling

Instead of synthetic single-turn dumps, generated genuine multi-turn training data:
ran the actual **base** (non-fine-tuned) model against the real τ-bench environment +
GPT-4o user-simulator on the 500-task retail *train* split, up to 2 attempts per task,
keeping only trajectories that scored reward=1.0. This is pure rejection sampling — no
new capability is taught, only successful behavior the model already has is reinforced.

**Result: 422/500 tasks passed (84.4%) within 2 tries.** Total GPT-4o cost: ~$1.41.

### 4. Second fine-tuning attempt — on the real multi-turn data

Same QLoRA config (r=16, lr=2e-4), trained on the 422 genuine rollouts, checkpointed
every 5 steps instead of once at the end, specifically to find an early stopping point
before memorization set in.

| Checkpoint | Pass rate |
|---|---|
| Baseline (untrained) | 45.2% (52/115) |
| Step 5 (~9% into epoch 1) | 44.3% (51/115) |
| **Step 10 (~19% into epoch 1)** | **5.2% (6/115)** |
| Full 3 epochs (159 steps) | 2.6% (3/115) |

There is essentially **no usable middle ground** — it goes from "indistinguishable
from baseline" to "nearly as broken as full training" in five steps.

### 5. Root cause: a positional shortcut, not literal memorization

Checked the training set's structure directly:

- **99.1%** of the 422 examples call an authentication function
  (`find_user_id_by_email` / `find_user_id_by_name_zip`) at *some* point.
- **98%** of those do it at the *exact same message index (4)*: system → user →
  assistant-clarifies → user-provides-info → assistant-calls-auth.

That's an extraordinarily rigid structural pattern, which makes sense given how the
data was generated — rejection-sampling from one fixed policy (same model, same
temperature, same prompt) on retail tasks whose personas commonly instruct the
simulated user to be "reactive" (won't volunteer info until asked) naturally converges
to one conversational shape.

A rank-16 LoRA adapter has more than enough capacity to pick up **"fire the auth call
around turn 4"** as a positional habit within a handful of gradient steps — faster than
it can learn the actual precondition ("only if the user has given you their info").
At eval time, on a differently-shaped conversation, the habit fires on schedule anyway,
with made-up placeholder values (`"John Doe"`, zip `"12345"`, `something@example.com`)
standing in for the real info the model never received.

This is a different failure mode from #2 above: not "the model over-acts on the first
turn because it always saw everything upfront," but "the model has learned *when* to
act by counting turns, not by checking preconditions."

## What we think would fix it

Not yet tried (see Open Questions), but follows directly from the diagnosis:

1. **Turn-position diversity in the training data** — the real fix. Rejection-sampling
   more data from the same setup would likely just reproduce the same shape again,
   since it's the *task personas'* behavior driving the homogeneity, not sampling luck.
   Needs either a different user-simulator strategy/persona style, or deliberately
   mixed task selection.
2. **Lower LoRA rank / learning rate / add dropout** — cheaper to try, might buy a
   wider window before the shortcut gets picked up, but doesn't address the root cause.
3. **Oversampling the few minority-shaped examples** (4 of 422 have auth at a different
   turn) — free, quick to test, but too little real diversity to expect much from.

### 6. Data-level fix: an augmented dataset grounded in the real DB

Instead of chasing the positional shortcut with hyperparameters, split the 422
rollouts into three parts and rebuilt the homogeneous ones:

1. **Unchanged** (140) — real passing rollouts, kept as-is.
2. **Zip/email withheld, negative examples** (139) — same real tasks, but the
   auth info is withheld at the point it would normally be given, so the
   *target* becomes "ask again" rather than "assume and proceed" — directly
   breaking the turn-4 positional habit by teaching a real precondition check
   instead.
3. **New synthetic policy-refusal / nonexistent-lookup / out-of-stock examples**
   (141) — built from *real* extracted specifics (names, zips, order IDs,
   payment methods, product variants — all pulled from τ-bench's own
   `orders.json`/`users.json`/`products.json`, never invented placeholders),
   covering requests that are genuinely impossible per the real retail policy
   (cancel an already-delivered order, exchange to a different product type,
   act on someone else's account, etc.) where the correct target is to refuse
   or first verify via a real tool call before refusing — not simply
   pattern-match a refusal from the request text.

420 examples total, rendered in full (instruction + target tool-calls-or-absence
+ connected data, no raw dialogue) in `renders/augmented_dataset_structure.html`.

### 7. Pivot to RL: GRPO with τ-bench's own reward

SFT — on any of the above data — still only teaches "reproduce this one
correct trajectory." Pivoted to RL, which only needs a pass/fail signal and
lets the model find its own correct trajectories, using **tau-bench's own
`calculate_reward()`** untouched (replay ground-truth actions on a fresh DB
copy, hash it, compare to the hash from the agent's actual trajectory — no
custom scoring). Setup:

- `unsloth/Qwen3-14B-unsloth-bnb-4bit`, QLoRA r=16, served from vLLM
  (`--quantization bitsandbytes --enable-lora`) **and** loaded separately via
  Unsloth for the gradient updates — one base checkpoint, two roles, to avoid
  juggling two quantization formats on a 30GB pod disk.
- Rollouts: τ-bench's own `ToolCallingAgent.solve()` reused unmodified against
  the live vLLM server (via litellm's `hosted_vllm` provider, isolated from
  `OPENAI_API_BASE` so the GPT-4o user-simulator keeps hitting the real OpenAI
  API on the same pod). G=4 rollouts/task, temperature=1.0.
- Advantage: standard GRPO group-relative normalization; groups that scored
  identically (all-0 or all-1) carry no gradient signal and are skipped.
- Update: vLLM did the generation, so logprobs of the exact assistant token
  spans are recomputed via a teacher-forced forward pass on the Unsloth-side
  model (same response-only masking as the SFT runs), loss =
  `-advantage * mean(logprob(assistant tokens))`. Adapter saved + hot-reloaded
  into vLLM every round.
- Objective signal, separate from the noisy small-batch training reward: every
  5 rounds, the saved checkpoint is loaded into vLLM under its own adapter
  name (training's live adapter untouched) and probed at temperature=0 against
  20 fixed real τ-bench **test**-split tasks.

**Three real bugs found and fixed along the way** (all in `code/grpo_train.py`):
the LoRA adapter wasn't registered with vLLM before the first round's rollouts
(every early rollout 404'd silently); vLLM's dynamic LoRA-reload endpoints need
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` or they don't exist; and the same
`content: null` chat-template crash from section 5 above, this time inside the
training loop's logprob-recompute step, not just at eval time.

**Result — it didn't help:**

| Checkpoint | Eval probe (20 fixed real test tasks, temp=0) |
|---|---|
| round 0 (1 round of training) | 0.55 (0.50 on rerun) |
| round 5 | 0.35 |
| round 10 | 0.30 (0.35 on rerun) |

A clean, monotonic-ish decline confirmed by an independent rerun (with full
dialogues saved this time, in `results/probe_transcripts/`) — not noise. Training
was stopped at round 12 once the trend was confirmed. Full dialogues for all
60 probed conversations (3 checkpoints × 20 tasks) are in
`renders/grpo_probe_dialogues.html`. Checkpoint weights (LoRA adapters, ~250MB
each) are on
[Hugging Face](https://huggingface.co/teacher57/qwen3-14b-tau-grpo-checkpoints).

Likely candidates, none confirmed: batch size too small (4 tasks/round, only
4-12 of 16 rollouts contributing gradient per round — high-variance updates on
a 14B model), `lr=1e-5` never tuned, or genuinely too few rounds (12) for noisy
RL to show net-positive movement. This specific run doesn't indict GRPO as a
method, just this configuration of it.

### 8. Combo tasks: where the test split differs, and a second GRPO run

A *combo* task asks for 2+ distinct actions (return / exchange / cancel / modify /
change payment) in one conversation. Combos are far more common in the test split than
in what we trained on:

| source | combo share |
|---|---|
| test split | 64 / 115 (55.7%) |
| train split (500 tasks) | 153 / 500 (30.6%) |
| the 422 passing rollouts (those with a task id) | 73 / 296 (24.7%) |
| GRPO task slots, rounds 0-12 of section 7 (rebuilt from the `random.seed(0)` draw) | 16 / 52 (30.8%) |

Rejection sampling keeps only passing rollouts, so harder multi-action tasks were
filtered out of the SFT data (80 of 153 combo train tasks never produced a pass).

**Which combo tasks can GRPO learn from?** GRPO learns from the difference between
winning and losing rollouts of the same task, so tasks the model always passes or
always fails give no signal, and the old rejection-sampling run never recorded
per-task pass rates. We measured them: each of the 153 combo train tasks got 4 or more
rollouts (640 in total, temperature 1.0) with the round-0 adapter against the real
env and GPT-4o as the user. Result (mean reward 0.42): **58 tasks never pass, 56 are
mixed, 39 always pass**. The 56 mixed tasks became the training pool. Per-task
results, with the instruction and ground-truth tool calls, are in
`results/combo_passrate.json`; the 153 task definitions are in `results/combo_tasks.json`.

**Combo-pool GRPO run.** Same start (round-0 adapter), the 56-task pool, lr 1e-5,
4 tasks x 4 rollouts per round, probe on the same 20 held-out test tasks at
temperature 0, with an automatic stop if a probe falls more than 0.10 below the best:

| checkpoint | probe (20 held-out test tasks) |
|---|---|
| baseline (the starting round-0 adapter) | 0.40 |
| round 0 (one update) | 0.45 |
| round 5 | **0.20** |

Training stopped itself during round 7. The training reward over rounds 0-6 stayed
between 0.31 and 0.69 with no trend, so it gave no warning; only the held-out probe
showed the drop. About a third of the tasks drawn (7 of the first 20) came out all-pass
or all-fail in training, so the "mixed" label from 4 rollouts is only partly reliable.
Dialogues for the three probes are in `results/probe_transcripts/combo_*.json`.

**Caveats.** The probe has 20 tasks and is noisy (about +/-0.1 per run). The same
starting adapter scored 0.55 and 0.50 in section 7's setup and 0.40 here (different
parallelism and vLLM settings), so absolute numbers should not be compared across runs,
and the 0.20 vs 0.40-0.45 gap is suggestive rather than conclusive on its own. Two GRPO
runs falling the same way is the stronger evidence. Checkpoints were saved only at
rounds 0 and 5, so the decline cannot be located between them.

**Infrastructure lessons** (each of these silently wasted a run at first):
- vLLM 0.30 dropped bitsandbytes quantization; vLLM 0.11.0 in its own virtualenv still
  serves the 4-bit model with LoRA adapters.
- With two LoRA adapters live (the training policy and a probe), vLLM needs
  `--max-loras 3`. At the default of 1 the adapters take turns, calls queue past
  litellm's 600 s timeout, and every rollout is silently scored 0.
- The update step ran out of GPU memory from a float32 log-softmax over the whole
  sequence; computing it only at assistant-token positions fixed that.
- 4-bit decoding is slow (about 300 tokens/s in total), so a round of 16 rollouts takes
  about 20 minutes and the 612-rollout diagnostic took about 4.5 hours.

**Gentle follow-up (lr 1e-6, same start and pool).** Everything as above except a 10x
lower learning rate; checkpoints saved and probed at rounds 1, 3, 5 and 10, with the stop
rule widened to a drop of more than 0.15 below the best (the probe is noisy). The same
starting adapter was reused as the baseline (0.40, not re-measured):

| checkpoint | probe (20 held-out test tasks) |
|---|---|
| baseline (the starting adapter) | 0.40 |
| gentle round 1 | 0.50 |
| gentle round 3 | 0.45 |
| gentle round 5 | 0.60 |
| gentle round 10 | 0.40 |

Training stopped itself during round 12 after the round-10 score fell 0.20 below the round-5
best. That stop was itself within the probe's noise. Training reward over rounds 0-11 stayed
between 0.31 and 0.75 with no trend; 8-16 of the 16 rollouts contributed a gradient per round
(all 16 in rounds 3, 7, 8 and 10). The gentle run never collapsed the way the lr 1e-5 run did, but its five probes
average about 0.49, the same as the starting adapter's usual score (0.40-0.55 across our
setups), so the 20-task probe cannot say whether it helped.

### 9. pass^2 test: is gentle round 5 better than the starting adapter?

To settle it we ran the full retail test split (all 115 tasks), each task twice at
temperature 0.7, for the round-5 checkpoint and for the starting adapter as a control with
identical settings (tool-calling agent, GPT-4o as the simulated user, 25 steps). pass^1 is the
average over all trials; pass^2 is the share of tasks where both trials pass (the τ-bench
definition, C(c,2)/C(n,2) with n = 2); consistency is pass^2 / pass^1. Temperature above 0 is
needed for pass^k to measure the model's own consistency: at 0 the trials differ only through
the user simulator (which sets no temperature, so it samples at the API default) and GPU
arithmetic.

| | gentle round 5 | starting adapter (control) |
|---|---|---|
| pass^1 (230 rollouts) | 40.4% ± 3.2 | 37.8% ± 3.2 |
| pass^2 (115 tasks) | 25.2% ± 4.0 | 21.7% ± 3.8 |
| consistency | 0.62 | 0.57 |

Round 5 passes both trials on 16 tasks where the control does not; the control wins 12 the
other way (paired test p = 0.57), with no errors in either run. The difference is well inside
the margin of error, so **the gentle run neither helped nor hurt measurably**, and the 0.60
probe score was most likely noise. Per-rollout results are in `results/passk_run.json` and
the live notebook section; the full conversations from this test and from the gentle-run
probes were only on the training pod and were lost when it was terminated, so they are not
included.

**Where this leaves us.** SFT broke the model, GRPO at lr 1e-5 degraded it, and GRPO at lr
1e-6 left it unchanged. About 12 rounds of 16 rollouts is too little signal to move a 14B model
by more than the measurement noise (about 4 points on 115 tasks). Untried: prompt and
scaffolding changes, distillation from a stronger teacher with mixed conversation shapes,
offline rejection-sampling fine-tuning on the combo pool, a larger base model.

### 10. Distillation from a stronger teacher: Qwen3-32B-AWQ plays the hard combo tasks, the 14B learns from the wins

Idea: the 14B cannot reliably solve many combo train tasks, so RL finds little to learn from. A 32B teacher
(Qwen3-32B-AWQ, vLLM, hermes tool parser, qwen3 reasoning parser, same agent and GPT-4o user) plays those tasks, and
only its **passing** conversations are used to fine-tune the 14B.

**Stage 1, teacher rollouts.** 114 hard train tasks (the ones the 14B solved rarely or never, plus mixed ones), up to 4
samples each, sampling of a task stopped after 2 passes. The teacher solved **86 of 114 tasks** and produced **180 passing
conversations** (all teacher conversations, passed or not: `dataset/distill/teacher_rollouts_raw.jsonl`).

**Stage 2, SFT.** One example per assistant turn (prompt = chat template of the messages so far; completion = the
`<think>` reasoning, the tool call or reply and `<|im_end|>`), loss on the completion only. QLoRA r=16 on
`unsloth/Qwen3-14B-unsloth-bnb-4bit`, lr 5e-5, batch 8, 1 epoch = 384 steps (3 h 13 min on one A100); loss 0.53 (first 10 steps)
to 0.35 (last 10). Data: `dataset/distill/teacher32b_passed_sft*.json`.

**Stage 3, test.** Full retail test split (115 tasks), **4 trials per task**, temperature 0.7, GPT-4o user, 25 steps, for
the new adapter and for the starting adapter as a control **re-run on the same pod with identical settings** (vLLM 0.11.0,
LoRA adapters via `--enable-lora --max-loras 3`). pass^k per task = C(c,k)/C(n,k); standard errors are over tasks.

| retail test, 115 tasks | new model | starting adapter, same-pod re-run | starting adapter, earlier test (2 trials, other pod) |
|---|---|---|---|
| pass^1 | **45.7% ± 3.4** | 41.7% ± 3.4 | 37.8% |
| pass^2 | 29.9% | 27.1% | 21.7% |
| pass^3 | 22.0% | 20.2% | n/a |
| pass^4 | 16.5% | 16.5% | n/a |
| consistency (pass^2 / pass^1) | 0.65 | 0.65 | 0.57 |

By task type, pass^1: combo tasks (64) 34.8% vs 33.6% (control); other tasks (51) 59.3% vs 52.0%. The gain, such as it is,
is on the non-combo tasks, although training used only combo conversations.

**Is it significant? No.** Paired by task over all 4 trials, the new model is **+3.9 points pass^1** over the same-pod control
(standard error 2.6, paired t = 1.53, 95% bootstrap interval -1.1 to +8.9, permutation p = 0.15). The sign test on pass^2
(new higher on 31 tasks, control higher on 23, 61 ties) gives p = 0.34. pass^4 is identical. The first 2-trial comparison looked
like +6.5 points (and +9 against the earlier test), but re-running the *same* starting adapter on the same pod scored 41.7%
instead of 37.8%: part of the early "gain" was run-to-run drift of the baseline (the control also beats the earlier test, p = 0.005).
Honest summary: a small positive effect (about +4 points) that is plausible but not established. Per-trial pass rates:
new 43.5 / 49.6 / 41.7 / 47.8%; control 40.9 / 39.1 / 41.7 / 45.2%.

The airline test (2 trials, 50 tasks) was started but stopped after 10 rollouts, so there is no airline comparison.

Files: all 460 conversations per model (`dataset/distill/retail_eval_conversations/`), per-rollout results,
`results/distill/distill_run.json` (everything the notebook's Section 12 plots), `results/distill/teacher_run.json`,
`results/distill/new_solves_analysis.json` (which tasks the new model solves that the control does not),
`renders/retail_test_new_solves.html`. The conversations of the *earliest* baseline test (37.8%) were not kept, only its
rewards (`results/passk_run.json`). The trained adapter (`distill-epoch-1`, 257 MB) is not in this repo.

## Repo contents

```
results/
  14b_retail_full_115.json          baseline eval, full 115-task transcripts + rewards
  14b_lora_retail_full_115.json     single-turn-SFT (attempt 1) eval, full transcripts
  14b_step5_retail_full_115.json    step-5 checkpoint eval, full transcripts
  14b_step10_retail_full_115.json   step-10 checkpoint eval, full transcripts
  32b_retail_full_115.json          Qwen3-32B-AWQ baseline, for reference (62% on
                                     mutating-action retail tasks)
  32b_airline_full_50.json          Qwen3-32B-AWQ airline baseline, for reference
  tau_eval_comparison.json          summary pass-rate table used for the bar chart
  grpo_train_log.jsonl              per-round GRPO training reward/loss
  grpo_eval_probe_log.json          per-checkpoint eval-probe pass rate (original + rerun)
  probe_transcripts/                full 60-conversation transcripts backing the probe
                                     (round_*.json: section 7; combo_*.json: section 8)
  combo_passrate.json               section 8 diagnostic: per-task pass rates (instruction,
                                     target tool calls, rewards) + all 640 rollouts
  combo_tasks.json                  the 153 combo train tasks (instruction + target actions)
  combo_grpo_run.json               combo-pool GRPO run: per-round rewards, probes, timings
  combo_grpo_train_log.jsonl, combo_grpo_stdout.log    raw logs of that run
  gentle_grpo_run.json, gentle_grpo_train_log.jsonl, gentle_grpo_stdout.log
                                    the gentle (lr 1e-6) run: per-round rewards, probes, logs
  distill/                          section 10: distill_run.json (4-trial retail test, per-trial and per-type
                                     stats, comparisons), teacher_run.json, new_solves_analysis.json
  passk_run.json                    section 9 pass^2 test: per-rollout results, pass^1/pass^2,
                                     paired comparison (conversations were lost with the pod)
dataset/
  qwen3_14b_retail_train_rollout_sft.json      the 422 genuine passing multi-turn
                                                rollouts used for fine-tuning attempt #2
  qwen3_14b_retail_train_augmented.json        the 420-example data-level fix (section 6)
  combo_pool.json                              the 56 task indices GRPO trained on in section 8
                                                (copy to /root/combo_pool.json for code/grpo_combo_train.py)
  combo_pool_56_mixed.json                     the same 56 tasks with instruction, target tool
                                                calls and their diagnostic rewards / pass rate
  combo_tasks_test_64.json                     the 64 combo tasks of the test split (instruction +
                                                target tool calls)
  combo_tasks_grpo_16.json                     the 16 combo tasks among the 52 task slots GRPO drew
                                                in section 7 (with the rounds they were drawn in)
  (the 153 combo train tasks are in results/combo_tasks.json, with pass rates in results/combo_passrate.json;
   these are task definitions, not training conversations: no combo-augmented SFT set has been generated)
  distill/
    teacher_rollouts_raw.jsonl, teacher32b_passed_sft(_reasoning).json, teacher32b_passed_meta.json
                                     section 10: all teacher conversations, the passing ones as SFT data
    retail_eval_conversations/       all 4-trial retail test conversations (new model and same-pod control),
                                     per-rollout results, GRPO probe conversations, early benchmark transcripts
    groups.json, tools.json, *_tasks lists, retail_test_tasks.json   task lists and task definitions used
notebooks/
  qwen3_14b_training_and_rollouts.ipynb     all training-loss curves, rollout-generation
                                             progress, eval comparison chart, and the
                                             GRPO training + eval-probe charts
code/
  grpo_train.py        the GRPO training loop (section 7)
  eval_checkpoint.py    loads a checkpoint into vLLM under its own adapter name and
                         probes it against real test-split tasks, without touching
                         the live training adapter
  launch_vllm.sh, remote_setup.sh, autowatch.sh    pod setup / vLLM launch / the
                         30-min watch loop that syncs logs and downloads+verifies
                         new checkpoints as they save
  combo_passrate.py     section 8 diagnostic: N rollouts per combo task, resumable
  grpo_combo_train.py   section 8 GRPO loop (combo pool, memory-lean loss, parallel rollouts)
  grpo_gentle_train.py  same with lr 1e-6 (the gentle follow-up)
  eval_parallel.py, probe_watcher*.sh, disk_guard*.sh    parallel probe, the watcher that
                         probes new checkpoints and stops training on a decline, disk guard
  full_passk.py, run_passk_chain.sh    section 9: pass^k eval of an adapter on the full 115-task
                         test (trials and temperature configurable), and the chain that ran it
  live_passk_section.py, section10_updater.sh    auto-updating notebook section for that test
                         and the detached launcher that keeps it current
  update_notebook.py, live_grpo_section.py, live_gentle_section.py, sync_checkpoints.py
                         laptop-side helpers that keep the notebook, a live HTML page and
                         local checkpoint copies current (pod address/paths are defaults)
  distill/               section 10: teacher rollouts (teacher_rollouts.py), SFT export and training (export_sft.py,
                         sft_samples.py, train_sft.py), pod setup and eval chains, and the JSON writer and notebook
                         Section 11/12 builders (update_distill_json.py, add_distill_section.py); full_passk.py above
                         now takes env and task count arguments
renders/
  retail_test_new_solves.html        tasks the distilled model solves that the control does not
  combo_task_comparison.html         the combo tasks of the test split, train split, passing
                                      rollouts and GRPO draws, with instruction + target tool calls
  augmented_dataset_structure.html   all 420 augmented-dataset examples, structure only
  grpo_probe_dialogues.html          all 60 full eval-probe conversations (3 checkpoints
                                      × 20 tasks), system message, instruction, every
                                      user/agent/tool turn, reasoning, tool calls+results
  retail_full_115.html, retail_train_rollouts_422.html, runpod_*.html, a100_*.html
                                      earlier full-eval and rollout-generation transcripts
```

## Setup notes (for anyone reproducing this)

- Qwen3-14B-AWQ served via vLLM 0.30.0, `--tool-call-parser hermes
  --reasoning-parser qwen3` — using the wrong parser (e.g. `qwen3_xml`) silently
  discards every tool call the model generates with no error, which looks exactly like
  a capability failure and cost real time to diagnose.
- QLoRA training via [Unsloth](https://github.com/unslothai/unsloth) on
  `unsloth/Qwen3-14B-unsloth-bnb-4bit`, r=16, target modules
  `q/k/v/o_proj, gate/up/down_proj`, `max_seq_length=16384` for the multi-turn data
  (8192 was enough for the single-turn version).
- Qwen3's chat template scans every message's `content` field for reasoning-tag
  stripping and crashes with `TypeError: 'NoneType' object is not subscriptable` if
  any message has `content: None` — which the OpenAI tool-call convention normally
  allows for a tool-call-only turn. Fixed by using `content: ""` instead.

## Open questions

- Does τ-bench's user-simulator expose an alternate strategy/persona style that would
  produce more varied conversational shapes without needing a different LLM?
- Would a much lower rank (r=4) or learning rate alone buy a meaningfully wider safe
  window, or is the positional shortcut just going to get learned regardless, faster
  than anything useful?
- Airline domain has no train split in this version of τ-bench — retail-only for now.
