#!/bin/bash
# Resumes the pod pipeline after training: (re)install vLLM 0.11 properly, serve the 14B with the new adapter, run the retail test.
cd /root
source /root/.taubench_env
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
log "the vLLM 0.11 install had failed earlier (the disk was full at the time); reinstalling it"
rm -rf /root/vllm011_env /root/vllm14b.log
bash /root/setup_vllm011.sh > /root/vllm011_install.log 2>&1
if ! grep -q VLLM011_DONE /root/vllm011_install.log; then log "FAILED: vLLM 0.11 install (see /root/vllm011_install.log)"; exit 1; fi
log "vLLM 0.11 installed"
/root/launch_vllm_14b.sh >> /root/pipeline.log 2>&1
while ! grep -qE 'Application startup complete|Engine core initialization failed|No such file' /root/vllm14b.log 2>/dev/null; do sleep 15; done
if ! grep -q 'Application startup complete' /root/vllm14b.log; then log "FAILED: 14B server did not start (see /root/vllm14b.log)"; exit 1; fi
log "14B server ready; starting the full retail test (pass^2) on /root/sft_out/epoch-1"
cd /root/tau-bench
python3 /root/full_passk.py distilled /root/sft_out/epoch-1 2 0.7 48 > /root/passk_distilled.log 2>&1
log "retail test finished"
echo PIPELINE_DONE >> /root/pipeline.log
