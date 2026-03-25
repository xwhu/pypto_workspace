#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source ~/pypto_workspace/.venv/bin/activate
cd ~/pypto_workspace/rust_llm_server

rust-gdb -batch -ex "run --backend ascend --device-id 2 --weights ../../Qwen3-0.6B/" -ex "bt" --args target/debug/rust_llm_server --backend ascend --device-id 2 > gdb_log.txt 2>&1 &
PID=$!
sleep 15
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3","messages":[{"role":"user","content":"Hi"}]}' > /dev/null
sleep 10
kill -9 $PID 2>/dev/null
cat gdb_log.txt | tail -n 150
