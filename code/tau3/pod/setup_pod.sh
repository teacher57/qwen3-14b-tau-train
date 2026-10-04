#!/bin/bash
# One-time pod setup for the tau3 retail eval: vLLM 0.11.0 venv (bitsandbytes + LoRA serving), tau2-bench venv, model + adapters.
# Disk is only 30 GB: no pip/uv caches. Prints TAU3_SETUP_DONE only if every step really worked.
LOG=/root/setup.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a /root/pipeline.log >> $LOG; }
fail() { log "TAU3_SETUP_FAILED: $*"; exit 1; }
log "setup: vllm venv"
python3 -m venv /root/vllm011_env || fail venv
/root/vllm011_env/bin/pip install --no-cache-dir -q vllm==0.11.0 >> $LOG 2>&1 || fail vllm
/root/vllm011_env/bin/pip install --no-cache-dir -q 'transformers>=4.56,<5' 'bitsandbytes>=0.46.1' >> $LOG 2>&1 || fail bnb
[ -x /root/vllm011_env/bin/vllm ] || fail vllm-binary
log "setup: tau2-bench"
[ -d /root/tau2-bench ] || git clone -q --depth 1 https://github.com/sierra-research/tau2-bench /root/tau2-bench >> $LOG 2>&1 || fail clone
cd /root/tau2-bench && UV_NO_CACHE=1 uv sync >> $LOG 2>&1 || fail uv-sync
UV_NO_CACHE=1 uv pip install -q websockets >> $LOG 2>&1 || fail websockets   # imported unconditionally by tau2 even in text mode
[ -x /root/tau2-bench/.venv/bin/tau2 ] || fail tau2-binary
log "setup: model and adapters"
/root/vllm011_env/bin/python - >> $LOG 2>&1 <<'PY' || fail download
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3-14B-unsloth-bnb-4bit")
snapshot_download("teacher57/qwen3-14b-tau-grpo-checkpoints", allow_patterns=["round-0/*"], local_dir="/root/adapters")
snapshot_download("teacher57/qwen3-14b-tau-distilled-from-32b", local_dir="/root/adapters/distill-epoch-1")
PY
ls /root/adapters/round-0/adapter_model.safetensors /root/adapters/distill-epoch-1/adapter_model.safetensors >> $LOG 2>&1 || fail adapters
log "TAU3_SETUP_DONE"
df -h / | tail -1 >> /root/pipeline.log
