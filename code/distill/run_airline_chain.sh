#!/bin/bash
# After the retail re-run finishes: the airline test (50 tasks x 2 trials, temp 0.7) for the new model, then for the starting adapter.
# Sequential, never in parallel with anything else.
source /root/.taubench_env
cd /root/tau-bench
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
while ! grep -q PASSK_DONE /root/passk_baseline2.log 2>/dev/null; do sleep 30; done
log "retail re-run finished; starting the airline test, new model"
python3 /root/full_passk.py airline_new /root/sft_out/epoch-1 2 0.7 48 airline 50 > /root/passk_airline_new.log 2>&1
log "airline test, new model finished; starting the starting adapter"
python3 /root/full_passk.py airline_base /root/round0_ckpt 2 0.7 48 airline 50 > /root/passk_airline_base.log 2>&1
log "airline test finished"
echo AIRLINE_DONE >> /root/pipeline.log
