#!/bin/bash
source /root/.taubench_env
export HF_HOME=/workspace/.cache/huggingface
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
cd /root/tau-bench
nohup /root/vllm_env/bin/vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --quantization bitsandbytes --load-format bitsandbytes \
  --enable-lora --max-loras 3 --max-lora-rank 16 \
  --tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.40 --max-model-len 16384 \
  --port 8000 > /root/vllm.log 2>&1 &
disown
echo "VLLM_LAUNCHED pid=$!"
