#!/bin/bash
# vLLM 0.11.0 in its own venv: the newest vLLM no longer serves bitsandbytes 4-bit models, and this is the setup the
# earlier baseline numbers (pass^1 37.8% / pass^2 21.7%) were measured with.
# Fails loudly: VLLM011_DONE is printed only if the vllm program really exists afterwards (an earlier run hid a full-disk failure).
set -o pipefail
python3 -m venv /root/vllm011_env
/root/vllm011_env/bin/pip install --no-cache-dir -q vllm==0.11.0 || { echo VLLM011_FAILED; exit 1; }
/root/vllm011_env/bin/pip install --no-cache-dir -q 'transformers>=4.56,<5' 'bitsandbytes>=0.46.1' || { echo VLLM011_FAILED; exit 1; }
[ -x /root/vllm011_env/bin/vllm ] || { echo VLLM011_FAILED; exit 1; }
echo VLLM011_DONE
