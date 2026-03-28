#!/bin/bash
# Launch HCCL smoke test with 2 ranks on 2 NPUs.
# Usage: bash examples/run_hccl_smoke_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building HCCL smoke test ==="
cargo build --example hccl_smoke_test --features "ascend,hccl"

echo ""
echo "=== Launching rank 0 (device 0) and rank 1 (device 1) ==="

ASCEND_DEVICE_ID=0 ./target/debug/examples/hccl_smoke_test --rank 0 --nranks 2 &
PID0=$!

sleep 0.5  # Give rank 0 a moment to write root info

ASCEND_DEVICE_ID=1 ./target/debug/examples/hccl_smoke_test --rank 1 --nranks 2 &
PID1=$!

echo "Waiting for both ranks to complete..."
wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

echo ""
if [ $STATUS0 -eq 0 ] && [ $STATUS1 -eq 0 ]; then
    echo "✅ HCCL smoke test PASSED"
else
    echo "❌ HCCL smoke test FAILED (rank0=$STATUS0, rank1=$STATUS1)"
fi
