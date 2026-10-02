#!/bin/bash
# pass^2 at temperature 0.7: round 5 first, then the starting adapter as a control with identical settings
source /root/.taubench_env
cd /root/tau-bench
python3 /root/full_passk.py gentle5 /root/gentle5_ckpt 2 0.7 48 > /root/passk_gentle5.log 2>&1
python3 /root/full_passk.py baseline /root/round0_ckpt 2 0.7 48 > /root/passk_baseline.log 2>&1
echo CHAIN_DONE > /root/passk_chain_done
