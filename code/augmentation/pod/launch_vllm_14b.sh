#!/bin/bash
# Serve the 4-bit Qwen3-14B base with LoRA adapters that can be loaded at runtime (same flags as the earlier evals).
source /root/.taubench_env
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
cd /root/tau-bench
nohup /root/vllm011_env/bin/vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --quantization bitsandbytes --load-format bitsandbytes \
  --enable-lora --max-loras 3 --max-lora-rank 16 \
  --tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --port 8000 > /root/vllm14b.log 2>&1 &
disown
echo "VLLM14B_LAUNCHED pid=$!"
