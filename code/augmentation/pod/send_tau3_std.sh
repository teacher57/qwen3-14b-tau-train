#!/bin/bash
# usage: send_tau3_std.sh IP PORT  -- copies scripts and the augmented adapter, pipes the OpenAI key (never printed), starts the run
IP=$1; PORT=$2; K=~/.ssh/id_ed25519; D="$(cd "$(dirname "$0")" && pwd)"
source ~/.zshenv
scp -q -P $PORT -i $K "$D/setup_vllm_env.sh" "$D/run_tau3_std.sh" "$D/summarize_std_remote.py" root@$IP:/root/ || exit 1
ssh -p $PORT -i $K root@$IP 'mkdir -p /root/adapters/aug-epoch-1'
scp -q -P $PORT -i $K "$D/../saves/aug-epoch-1/adapter_config.json" "$D/../saves/aug-epoch-1/adapter_model.safetensors" root@$IP:/root/adapters/aug-epoch-1/ || exit 1
ssh -p $PORT -i $K root@$IP 'umask 077; cat > /root/.tau3_env' <<ENV
export OPENAI_API_KEY="$OPENAI_API_KEY"
ENV
ssh -p $PORT -i $K root@$IP 'chmod 600 /root/.tau3_env; nohup bash /root/run_tau3_std.sh > /root/run_tau3_stdout.log 2>&1 & disown; echo started'
