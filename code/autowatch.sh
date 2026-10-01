#!/bin/bash
REMOTE="root@213.173.105.6"
PORT=30164
KEY=~/.ssh/id_ed25519
LOCAL_SAVES=/Users/serjtankian/claude_trusted/tau-train/saves
LOCAL_LOG=/Users/serjtankian/claude_trusted/tau-train/full_run_results/grpo_train_log.jsonl
LOCAL_PROBE_LOG_RAW=/Users/serjtankian/.claude/jobs/2f730982/tmp/grpo_eval_probe_raw.jsonl
LOCAL_PROBE_LOG=/Users/serjtankian/claude_trusted/tau-train/full_run_results/grpo_eval_probe_log.json

ssh_cmd() { ssh -o ConnectTimeout=10 -p $PORT -i $KEY $REMOTE "$1"; }

while true; do
  ALIVE=$(ssh_cmd "ps aux | grep grpo_train.py | grep -v grep | wc -l" 2>/dev/null)
  if [ -z "$ALIVE" ]; then
    echo "[$(date +%H:%M)] SSH UNREACHABLE (pod may be down)"
  elif [ "$ALIVE" = "0" ]; then
    echo "[$(date +%H:%M)] grpo_train.py not running (stopped by watchdog or crashed)"
    ssh_cmd "tail -20 /root/grpo_train.stdout.log"
  else
    scp -o ConnectTimeout=10 -P $PORT -i $KEY $REMOTE:/root/grpo_train.log.jsonl $LOCAL_LOG 2>/dev/null
    LATEST=$(tail -1 $LOCAL_LOG 2>/dev/null)
    echo "[$(date +%H:%M)] alive, latest: $LATEST"

    # check for new checkpoints not yet downloaded
    REMOTE_CKPTS=$(ssh_cmd "ls /root/grpo_checkpoints/" 2>/dev/null)
    for ck in $REMOTE_CKPTS; do
      if [ ! -d "$LOCAL_SAVES/$ck" ]; then
        echo "[$(date +%H:%M)] downloading new checkpoint: $ck"
        scp -o ConnectTimeout=15 -P $PORT -i $KEY -r $REMOTE:/root/grpo_checkpoints/$ck "$LOCAL_SAVES/$ck" 2>/dev/null
        LOCAL_SHA=$(shasum -a 256 "$LOCAL_SAVES/$ck/adapter_model.safetensors" 2>/dev/null | awk '{print $1}')
        REMOTE_SHA=$(ssh_cmd "sha256sum /root/grpo_checkpoints/$ck/adapter_model.safetensors" | awk '{print $1}')
        if [ "$LOCAL_SHA" = "$REMOTE_SHA" ] && [ -n "$LOCAL_SHA" ]; then
          echo "[$(date +%H:%M)] $ck downloaded and VERIFIED ($LOCAL_SHA)"
          ROUND_NUM=$(echo "$ck" | sed 's/round-//')
          echo "[$(date +%H:%M)] launching eval probe for $ck (round $ROUND_NUM) in background on pod"
          ssh_cmd "cd /root/tau-bench && source /root/.taubench_env && nohup python3 eval_checkpoint.py /root/grpo_checkpoints/$ck $ROUND_NUM > /root/probe_round_${ROUND_NUM}.log 2>&1 &"
        else
          echo "[$(date +%H:%M)] $ck CHECKSUM MISMATCH -- local=$LOCAL_SHA remote=$REMOTE_SHA"
          rm -rf "$LOCAL_SAVES/$ck"
        fi
      fi
    done

    # sync probe results (converting jsonl -> a proper JSON array file) and
    # check for peak-then-decline (overfitting signal)
    scp -o ConnectTimeout=10 -P $PORT -i $KEY $REMOTE:/root/grpo_eval_probe.log.jsonl $LOCAL_PROBE_LOG_RAW 2>/dev/null
    if [ -s "$LOCAL_PROBE_LOG_RAW" ]; then
      python3 -c "
import json
recs = [json.loads(l) for l in open('$LOCAL_PROBE_LOG_RAW')]
json.dump(recs, open('$LOCAL_PROBE_LOG', 'w'), indent=1)
"
      echo "[$(date +%H:%M)] probe results so far:"
      cat "$LOCAL_PROBE_LOG"
      python3 - "$LOCAL_PROBE_LOG" << 'PYEOF'
import json, sys
recs = json.load(open(sys.argv[1]))
if len(recs) >= 3:
    best = max(recs, key=lambda r: r["probe_avg_reward"])
    latest = recs[-1]
    if latest is not best and best["probe_avg_reward"] - latest["probe_avg_reward"] > 0.10:
        print(f"OVERFIT_SIGNAL best_round={best['round']} best={best['probe_avg_reward']:.3f} latest_round={latest['round']} latest={latest['probe_avg_reward']:.3f}")
PYEOF
    fi
  fi
  sleep 1800
done
