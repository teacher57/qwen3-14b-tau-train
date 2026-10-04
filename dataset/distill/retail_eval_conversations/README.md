# Retail eval conversations (all that exist)
- retail_new_model_4trials_*: new SFT-on-teacher model, 115 test tasks x 4 trials, temp 0.7 (results = per-rollout reward; transcripts = full messages).
- retail_starting_adapter_rerun_4trials_*: starting adapter, same pod, same settings, 4 trials.
  (The earlier 2-trial files in teacher/data/ are subsets of these 4-trial files.)
- grpo_probes_20_tasks/round_{0,5,10}.json: probe conversations on the 20-task sample during the GRPO runs.
- benchmark_transcript_{base,finetuned}.jsonl: early benchmark runs (older SFT experiments).
- airline_partial_new_*: the 10 airline conversations of the distilled model that ran before the airline test was stopped.
- Augmentation experiment conversations: `dataset/augmentation/retail_aug_4trials_*` (τ-bench retail, 4 trials) and `dataset/tau3_eval/` (τ³ retail: `tau3std_*` = default settings, 2 trials, baseline vs augmented; `first_run_*` = first run, 1 trial, starting adapter vs distilled).
NOT available: conversations of the original baseline test (pass^1 37.8%, 2 trials, other pod) - only per-rollout rewards were kept in full_run_results/passk_run.json.
