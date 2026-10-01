#!/bin/bash
# Probe the baseline (original round-0 adapter) and every new GRPO checkpoint;
# stop training if the latest probe is >0.10 below the best (needs >=3 probes).
source /root/.taubench_env
cd /root/tau-bench
LOG=/root/grpo_eval_probe.log.jsonl
probe() { python3 /root/eval_parallel.py "$1" "$2" > /root/probe_round_$2.log 2>&1; }
check() {
  python3 - <<'PY'
import json, subprocess
recs = [json.loads(l) for l in open("/root/grpo_eval_probe.log.jsonl")]
if len(recs) >= 3:
    best = max(r["probe_avg_reward"] for r in recs)
    last = recs[-1]["probe_avg_reward"]
    if best - last > 0.15:
        print("DECLINE best=%.3f latest=%.3f -> stopping training" % (best, last), flush=True)
        subprocess.run(["pkill", "-f", "^python3 /root/grpo_gentle_train.py"])
        open("/root/STOPPED_ON_DECLINE", "w").write("best=%.3f latest=%.3f" % (best, last))
PY
}
[ -s $LOG ] || probe /root/round0_ckpt -1
declare -A seen
while true; do
  for d in /root/grpo_checkpoints/round-*; do
    [ -f "$d/adapter_model.safetensors" ] || continue
    n=${d##*round-}
    [ -n "${seen[$n]}" ] && continue
    sleep 20; seen[$n]=1
    probe "$d" "$n"; check
  done
  [ -f /root/STOPPED_ON_DECLINE ] && break
  pgrep -f '^python3 /root/grpo_gentle_train.py' >/dev/null || { [ -n "${seen[done]}" ] && break; seen[done]=1; }
  sleep 30
done
echo WATCHER_DONE
