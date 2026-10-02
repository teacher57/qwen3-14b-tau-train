#!/bin/bash
# One-time setup of a fresh pod for the 32B teacher run (50GB root disk: vLLM ~12GB + AWQ model ~19GB).
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
cd /root
pip install --no-cache-dir -q vllm openai litellm 2>&1 | tail -2
git clone --depth 1 https://github.com/sierra-research/tau-bench.git
cd tau-bench
pip install --no-cache-dir -q -e . 2>&1 | tail -2
echo SETUP_COMPLETE
