#!/bin/bash
set -e
cd /root
git clone --depth 1 https://github.com/sierra-research/tau-bench.git
cd tau-bench
pip install -e . -q
sed -i 's/max_num_steps: int = 30/max_num_steps: int = 25/' tau_bench/agents/tool_calling_agent.py
pip install -q unsloth trl vllm bitsandbytes accelerate peft
mkdir -p /workspace/.cache/huggingface
echo "SETUP_COMPLETE"
