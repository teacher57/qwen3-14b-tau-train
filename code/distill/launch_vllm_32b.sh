#!/bin/bash
# Serve Qwen3-32B-AWQ as "teacher32b" on port 8000.
source /root/.taubench_env
cd /root/tau-bench
nohup vllm serve Qwen/Qwen3-32B-AWQ --served-model-name teacher32b \
  --tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --port 8000 > /root/vllm.log 2>&1 &
disown
echo "VLLM_LAUNCHED pid=$!"
