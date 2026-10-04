#!/bin/bash
# Training env on a 30 GB-disk pod: reuse the system torch 2.8.0 (no second copy), no pip cache, then pre-download the 4-bit base model.
LOG=/root/train_env_install.log
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
python3 -m venv --system-site-packages /root/train_env >> $LOG 2>&1 || { echo TRAIN_ENV_FAILED venv >> $LOG; exit 1; }
echo "torch==2.8.0" > /root/constraints.txt
/root/train_env/bin/pip install -q -c /root/constraints.txt unsloth unsloth_zoo bitsandbytes accelerate peft trl >> $LOG 2>&1 || { echo TRAIN_ENV_FAILED pip >> $LOG; exit 1; }
/root/train_env/bin/python -c "import torch, unsloth; print('torch', torch.__version__, 'unsloth ok')" >> $LOG 2>&1 || { echo TRAIN_ENV_FAILED import >> $LOG; exit 1; }
/root/train_env/bin/python - >> $LOG 2>&1 <<'PY' || { echo TRAIN_ENV_FAILED download >> $LOG; exit 1; }
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3-14B-unsloth-bnb-4bit")
PY
echo TRAIN_ENV_DONE >> $LOG
df -h / | tail -1 >> $LOG
