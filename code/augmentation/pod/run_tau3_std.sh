#!/bin/bash
# tau3 (tau2-bench) retail, default settings (agent temperature 0, 200 steps, GPT-4.1 customer), 2 trials per task, 114 tasks:
# baseline (starting adapter round-0) first, then the augmented model, strictly one after the other.
cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
echo "torch==$(python3 -c 'import torch;print(torch.__version__.split("+")[0])')" > /root/constraints.txt
log "setup: vllm env"
bash /root/setup_vllm_env.sh > /root/vllm011_install.log 2>&1
grep -q VLLM011_DONE /root/vllm011_install.log || { log "FAILED: vllm install"; exit 1; }
log "setup: tau2-bench"
[ -d /root/tau2-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau2-bench /root/tau2-bench
(cd /root/tau2-bench && UV_NO_CACHE=1 uv sync > /root/tau2_install.log 2>&1 && UV_NO_CACHE=1 uv pip install -q websockets >> /root/tau2_install.log 2>&1) || { log "FAILED: tau2 install"; exit 1; }
[ -x /root/tau2-bench/.venv/bin/tau2 ] || { log "FAILED: tau2 binary"; exit 1; }
/root/vllm011_env/bin/python - >> /root/pipeline.log 2>&1 <<'PY' || { log "FAILED: adapter download"; exit 1; }
from huggingface_hub import snapshot_download
snapshot_download("teacher57/qwen3-14b-tau-grpo-checkpoints", allow_patterns=["round-0/*"], local_dir="/root/adapters")
PY
ls /root/adapters/round-0/adapter_model.safetensors /root/adapters/aug-epoch-1/adapter_model.safetensors > /dev/null 2>&1 || { log "FAILED: adapters missing"; exit 1; }
log "setup done; starting vllm"
source /root/.tau3_env
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
nohup /root/vllm011_env/bin/vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit --quantization bitsandbytes --load-format bitsandbytes \
  --enable-lora --max-loras 3 --max-lora-rank 16 --tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.85 --max-model-len 16384 --port 8000 > /root/vllm14b.log 2>&1 &
for i in $(seq 1 180); do grep -q "Application startup complete" /root/vllm14b.log && break; grep -q "Engine core initialization failed" /root/vllm14b.log && break; sleep 10; done
grep -q "Application startup complete" /root/vllm14b.log || { log "FAILED: vllm did not start"; exit 1; }
curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"base","lora_path":"/root/adapters/round-0"}' > /root/load_base.log
curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"aug","lora_path":"/root/adapters/aug-epoch-1"}' > /root/load_aug.log
log "vllm ready; adapters: $(head -c 60 /root/load_base.log) | $(head -c 60 /root/load_aug.log)"
cd /root/tau2-bench
run() {
  log "tau3 run $1 started"
  .venv/bin/tau2 run --domain retail --task-split-name base \
    --agent-llm "hosted_vllm/$2" --agent-llm-args '{"temperature": 0.0, "api_base": "http://localhost:8000/v1", "api_key": "dummy"}' \
    --user-llm gpt-4.1 --num-trials 2 --max-concurrency 32 --timeout 1200 --save-to "$1" --auto-resume > "/root/tau3_$1.log" 2>&1
  log "tau3 run $1 finished (exit $?)"
}
run tau3std_base base
run tau3std_aug aug
log "TAU3_STD_DONE"
