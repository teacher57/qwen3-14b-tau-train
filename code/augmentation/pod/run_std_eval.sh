#!/bin/bash
# Standard tau-bench settings (temperature 0, 30 steps, GPT-4o user): the starting adapter (baseline) first, then the augmented model, strictly one after the other. 2 trials per task.
cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
python3 -c "import torch;print(torch.__version__)" > /root/torch_ver.txt 2>&1
echo "torch==$(python3 -c 'import torch;print(torch.__version__.split("+")[0])')" > /root/constraints.txt
log "setup: vllm env"
bash /root/setup_vllm_env.sh > /root/vllm011_install.log 2>&1
grep -q VLLM011_DONE /root/vllm011_install.log || { log "FAILED: vllm install"; exit 1; }
[ -d /root/tau-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau-bench.git /root/tau-bench
(cd /root/tau-bench && pip install -q -e . > /root/taubench_install.log 2>&1) || { log "FAILED: tau-bench install"; exit 1; }
/root/vllm011_env/bin/python - >> /root/pipeline.log 2>&1 <<'PY' || { log "FAILED: adapter download"; exit 1; }
from huggingface_hub import snapshot_download
snapshot_download("teacher57/qwen3-14b-tau-grpo-checkpoints", allow_patterns=["round-0/*"], local_dir="/root/adapters")
PY
ls /root/adapters/round-0/adapter_model.safetensors /root/adapters/aug-epoch-1/adapter_model.safetensors > /dev/null 2>&1 || { log "FAILED: adapters missing"; exit 1; }
log "setup done; starting vllm"
source /root/.taubench_env
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
nohup /root/vllm011_env/bin/vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit --quantization bitsandbytes --load-format bitsandbytes \
  --enable-lora --max-loras 3 --max-lora-rank 16 --tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.85 --max-model-len 16384 --port 8000 > /root/vllm14b.log 2>&1 &
for i in $(seq 1 180); do grep -q "Application startup complete" /root/vllm14b.log && break; grep -q "Engine core initialization failed" /root/vllm14b.log && break; sleep 10; done
grep -q "Application startup complete" /root/vllm14b.log || { log "FAILED: vllm did not start"; exit 1; }
log "vllm ready"
cd /root/tau-bench
export MAX_STEPS=30
log "run std_control started (temperature 0, 30 steps, 2 trials)"
python3 /root/full_passk_std.py std_control /root/adapters/round-0 2 0.0 48 > /root/passk_std_control.log 2>&1
log "run std_control finished"
log "run std_aug started (temperature 0, 30 steps, 2 trials)"
python3 /root/full_passk_std.py std_aug /root/adapters/aug-epoch-1 2 0.0 48 > /root/passk_std_aug.log 2>&1
log "STD_EVAL_DONE"
