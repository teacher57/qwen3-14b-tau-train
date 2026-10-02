#!/bin/bash
# 1) two MORE retail trials (trials 3 and 4; full_passk.py resumes from the saved ones) for the new model, then the starting adapter,
# 2) then the rest of the airline test (2 trials per task) for both. Strictly one after the other.
source /root/.taubench_env
cd /root/tau-bench
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
log "extra retail trials: starting with the new model (trials 3 and 4)"
python3 /root/full_passk.py distilled /root/sft_out/epoch-1 4 0.7 48 > /root/passk_distilled_more.log 2>&1
log "extra retail trials, new model finished; starting adapter next"
python3 /root/full_passk.py baseline2 /root/round0_ckpt 4 0.7 48 > /root/passk_baseline2_more.log 2>&1
log "extra retail trials, starting adapter finished; resuming the airline test"
python3 /root/full_passk.py airline_new /root/sft_out/epoch-1 2 0.7 48 airline 50 > /root/passk_airline_new_more.log 2>&1
log "airline test, new model finished; starting adapter next"
python3 /root/full_passk.py airline_base /root/round0_ckpt 2 0.7 48 airline 50 > /root/passk_airline_base.log 2>&1
log "airline test finished"
echo AIRLINE_DONE >> /root/pipeline.log
