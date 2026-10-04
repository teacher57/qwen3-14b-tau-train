#!/bin/bash
# usage: send_std_files.sh PORT   (pod IP is read from the first argument too: send_std_files.sh IP PORT)  -- copies scripts + the augmented adapter, pipes the OpenAI key without printing it, starts the run
IP=$1; PORT=$2; K=~/.ssh/id_ed25519; D="$(cd "$(dirname "$0")" && pwd)"
source ~/.zshenv
scp -q -P $PORT -i $K "$D/full_passk_std.py" "$D/setup_vllm_env.sh" "$D/run_std_eval.sh" root@$IP:/root/ || exit 1
ssh -p $PORT -i $K root@$IP 'mkdir -p /root/adapters/aug-epoch-1'
scp -q -P $PORT -i $K "$D/../saves/aug-epoch-1/adapter_config.json" "$D/../saves/aug-epoch-1/adapter_model.safetensors" root@$IP:/root/adapters/aug-epoch-1/ || exit 1
ssh -p $PORT -i $K root@$IP 'umask 077; cat > /root/.taubench_env' <<ENV
export OPENAI_API_KEY="$OPENAI_API_KEY"
ENV
ssh -p $PORT -i $K root@$IP 'chmod 600 /root/.taubench_env; nohup bash /root/run_std_eval.sh > /root/run_std_stdout.log 2>&1 & disown; echo started'
