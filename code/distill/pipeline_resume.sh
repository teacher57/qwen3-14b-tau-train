#!/bin/bash
# Resumes the pod pipeline at the training step with 1 epoch (the data is already exported, the 32B is gone, vLLM 0.11 is installed).
cd /root
source /root/.taubench_env
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
rm -rf /root/sft_out
log "restarting training with 1 epoch (2 epochs would take ~7 h at ~33 s per step); checkpoints every 96 steps"
/root/train_env/bin/python /root/train_sft.py --epochs 1 --save_every 96 > /root/train_sft.log 2>&1
if [ ! -f /root/sft_out/SFT_DONE ]; then log "FAILED: training did not finish (see /root/train_sft.log)"; exit 1; fi
log "training finished; final adapter /root/sft_out/epoch-1"
/root/launch_vllm_14b.sh >> /root/pipeline.log 2>&1
while ! grep -qE 'Application startup complete|Engine core initialization failed' /root/vllm14b.log 2>/dev/null; do sleep 15; done
if ! grep -q 'Application startup complete' /root/vllm14b.log; then log "FAILED: 14B server did not start (see /root/vllm14b.log)"; exit 1; fi
log "14B server ready; starting the full retail test (pass^2) on /root/sft_out/epoch-1"
cd /root/tau-bench
python3 /root/full_passk.py distilled /root/sft_out/epoch-1 2 0.7 48 > /root/passk_distilled.log 2>&1
log "retail test finished"
echo PIPELINE_DONE >> /root/pipeline.log
