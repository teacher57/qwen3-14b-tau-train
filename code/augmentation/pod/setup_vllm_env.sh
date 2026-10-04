#!/bin/bash
# vLLM 0.11.0 in an isolated venv that reuses the system torch 2.8.0 (no second copy). Prints VLLM011_DONE only if the binary exists.
export PIP_NO_CACHE_DIR=1
python3 -m venv --system-site-packages /root/vllm011_env || { echo VLLM011_FAILED; exit 1; }
/root/vllm011_env/bin/pip install -q -c /root/constraints.txt vllm==0.11.0 || { echo VLLM011_FAILED; exit 1; }
/root/vllm011_env/bin/pip install -q -c /root/constraints.txt 'transformers>=4.56,<5' 'bitsandbytes>=0.46.1' || { echo VLLM011_FAILED; exit 1; }
[ -x /root/vllm011_env/bin/vllm ] || { echo VLLM011_FAILED; exit 1; }
echo VLLM011_DONE
