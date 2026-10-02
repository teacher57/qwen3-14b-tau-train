#!/bin/bash
# hard core first (tasks the 14B never solves), then the medium ones; the easy ones are skipped
source /root/.taubench_env
cd /root/tau-bench
python3 /root/teacher_rollouts.py /root/never46.json 4 0.7 32 teacher32b 2 >> /root/teacher.log 2>&1
python3 /root/teacher_rollouts.py /root/mixed19.json 4 0.7 32 teacher32b 2 >> /root/teacher.log 2>&1
echo TEACHER_CHAIN_DONE >> /root/teacher.log
