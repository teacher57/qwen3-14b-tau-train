#!/bin/bash
# after training: tau-bench + vLLM 0.11 serving the new adapter, then the full 115-task retail test, 4 trials, temperature 0.7 (same as the control and the earlier distilled model)
cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
while ! grep -qE "TRAINING_DONE|FAILED" /root/pipeline.log; do sleep 30; done
grep -q "FAILED" /root/pipeline.log && exit 1
while ! grep -qE "VLLM011_DONE|VLLM011_FAILED" /root/vllm011_install.log 2>/dev/null; do sleep 20; done
grep -q VLLM011_FAILED /root/vllm011_install.log && { log "FAILED: vLLM install"; exit 1; }
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
[ -d /root/tau-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau-bench.git /root/tau-bench
(cd /root/tau-bench && pip install -q -e . > /root/taubench_install.log 2>&1) || { log "FAILED: tau-bench install"; exit 1; }
log "tau-bench installed; starting vllm"
source /root/.taubench_env
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
nohup /root/vllm011_env/bin/vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit --quantization bitsandbytes --load-format bitsandbytes \
  --enable-lora --max-loras 3 --max-lora-rank 16 --tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.85 --max-model-len 16384 --port 8000 > /root/vllm14b.log 2>&1 &
for i in $(seq 1 120); do grep -q "Application startup complete" /root/vllm14b.log && break; grep -q "Engine core initialization failed" /root/vllm14b.log && break; sleep 10; done
grep -q "Application startup complete" /root/vllm14b.log || { log "FAILED: vllm did not start"; exit 1; }
log "vllm ready; eval starts (retail test, 4 trials, temp 0.7)"
cd /root/tau-bench
python3 /root/full_passk.py aug /root/sft_out_aug/epoch-1 4 0.7 48 > /root/passk_aug.log 2>&1
log "EVAL_DONE"
