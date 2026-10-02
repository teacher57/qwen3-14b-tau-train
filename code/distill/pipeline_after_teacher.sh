#!/bin/bash
# Runs on the pod, detached. After the teacher run finishes it does, in order:
#   export passed conversations (reasoning kept) -> stop the 32B server and free its disk -> fine-tune the 14B
#   -> (installing vLLM 0.11 in the background while it trains) -> serve the 14B with the new adapter
#   -> full 115-task retail test, 2 trials per task at temperature 0.7 (same settings as the earlier baseline).
# Every step is logged with a timestamp to /root/pipeline.log. Epochs are read from /root/epochs.txt (default 2) when training starts.
cd /root
source /root/.taubench_env
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }

log "waiting for the teacher run to finish"
while ! grep -q TEACHER_ALL_DONE /root/teacher.log 2>/dev/null; do sleep 30; done
log "teacher run finished"

TEACHER_ON_POD=1 python3 /root/export_sft.py --keep-reasoning >> /root/pipeline.log 2>&1
N=$(python3 -c "import json;print(len(json.load(open('/root/teacher32b_passed_sft_reasoning.json'))))" 2>/dev/null || echo 0)
log "exported $N passed hard/medium conversations"
if [ "$N" -lt 10 ]; then log "STOPPED: fewer than 10 passed conversations, not worth training on"; exit 1; fi

pkill -f 'vllm serve'
sleep 20
rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3-32B-AWQ
log "32B server stopped and its weights removed; $(df -h / | tail -1 | awk '{print $4}') disk free"

while ! grep -q TRAIN_ENV_DONE /root/train_env_install.log 2>/dev/null; do sleep 20; done
(bash /root/setup_vllm011.sh > /root/vllm011_install.log 2>&1 &)
EPOCHS=$(cat /root/epochs.txt 2>/dev/null || echo 2)
log "training the 14B: $EPOCHS epochs"
/root/train_env/bin/python /root/train_sft.py --epochs "$EPOCHS" > /root/train_sft.log 2>&1
if [ ! -f /root/sft_out/SFT_DONE ]; then log "FAILED: training did not finish (see /root/train_sft.log)"; exit 1; fi
log "training finished; adapters in /root/sft_out"

while ! grep -q VLLM011_DONE /root/vllm011_install.log 2>/dev/null; do sleep 20; done
/root/launch_vllm_14b.sh >> /root/pipeline.log 2>&1
while ! grep -qE 'Application startup complete|Engine core initialization failed' /root/vllm14b.log 2>/dev/null; do sleep 15; done
if ! grep -q 'Application startup complete' /root/vllm14b.log; then log "FAILED: 14B server did not start (see /root/vllm14b.log)"; exit 1; fi
log "14B server ready; starting the full retail test (pass^2) on /root/sft_out/epoch-$EPOCHS"

cd /root/tau-bench
python3 /root/full_passk.py distilled "/root/sft_out/epoch-$EPOCHS" 2 0.7 48 > /root/passk_distilled.log 2>&1
log "retail test finished"
echo PIPELINE_DONE >> /root/pipeline.log
