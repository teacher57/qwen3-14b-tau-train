#!/bin/bash
# waits for the training env, then trains the 14B from the base on the augmented data (same recipe as the "new" model: r16, lr 5e-5, batch 8, 1 epoch)
cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
while ! grep -qE "TRAIN_ENV_DONE|TRAIN_ENV_FAILED" /root/train_env_install.log 2>/dev/null; do sleep 20; done
grep -q TRAIN_ENV_FAILED /root/train_env_install.log && { log "FAILED: training env install (see train_env_install.log)"; exit 1; }
log "training env ready; $(df -h / | tail -1 | awk '{print $4}') disk free; training starts"
/root/train_env/bin/python /root/train_sft.py --data /root/teacher32b_passed_sft_lookups_confirm.json --out /root/sft_out_aug --epochs 1 --save_every 96 > /root/train_sft.log 2>&1
[ -f /root/sft_out_aug/SFT_DONE ] && log "TRAINING_DONE" || log "FAILED: training did not finish (see train_sft.log)"
