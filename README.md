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
dataset/
  qwen3_14b_retail_train_rollout_sft.json      the 422 genuine passing multi-turn
                                                rollouts used for fine-tuning attempt #2
  qwen3_14b_retail_train_augmented.json        the 420-example data-level fix (section 6)
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
renders/
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
