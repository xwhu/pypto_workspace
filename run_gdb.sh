#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source ~/pypto_workspace/.venv/bin/activate
cd ~/pypto_workspace/rust_llm_server

cat << 'GDB_EOF' > gdb_script.txt
set logging on
set logging file gdb_full.log
run --backend ascend --device-id 2
bt full
quit
GDB_EOF

echo "Starting GDB server on device 2..."
rm -f gdb_full.log
gdb -batch -x gdb_script.txt target/debug/rust_llm_server > gdb.log 2>&1 &
GDB_PID=$!

echo "Waiting for server to start..."
sleep 15

echo "Sending curl request..."
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hi"}]}' > curl.log 2>&1

echo "Waiting for crash..."
sleep 5
kill -9 $GDB_PID 2>/dev/null || true

echo "GDB Output extracted:"
tail -n 200 gdb_full.log
