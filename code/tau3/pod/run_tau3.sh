#!/bin/bash
# tau3 (tau2-bench) retail eval, strictly one run after the other:
#   1) starting adapter ("old", round-0)   2) fine-tuned adapter ("new", distill-epoch-1)
# 114 tasks (split "base"), 1 trial, agent temperature 0.7, GPT-4.1 as the simulated user, tau2 default 200 steps.
source /root/.tau3_env
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
log "tau3: starting vllm"
nohup /root/vllm011_env/bin/vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --quantization bitsandbytes --load-format bitsandbytes \
  --enable-lora --max-loras 3 --max-lora-rank 16 \
  --tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --port 8000 > /root/vllm14b.log 2>&1 &
for i in $(seq 1 180); do curl -s localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
curl -s localhost:8000/v1/models > /dev/null 2>&1 || { log "TAU3_FAILED: vllm did not start"; exit 1; }
log "tau3: vllm ready"
curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"old","lora_path":"/root/adapters/round-0"}' > /root/load_old.log
curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"new","lora_path":"/root/adapters/distill-epoch-1"}' > /root/load_new.log
log "tau3: adapters loaded: $(cat /root/load_old.log | head -c 80) | $(cat /root/load_new.log | head -c 80)"
cd /root/tau2-bench
run() {  # name adapter
  log "tau3: run $1 started"
  .venv/bin/tau2 run --domain retail --task-split-name base \
    --agent-llm "hosted_vllm/$2" --agent-llm-args '{"temperature": 0.7, "api_base": "http://localhost:8000/v1", "api_key": "dummy"}' \
    --user-llm gpt-4.1 --num-trials 1 --max-concurrency 24 --timeout 1200 \
    --save-to "$1" --auto-resume > "/root/tau3_$1.log" 2>&1
  log "tau3: run $1 finished (exit $?)"
}
run tau3_old old
run tau3_new new
log "TAU3_DONE"
