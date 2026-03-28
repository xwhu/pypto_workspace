#!/bin/bash
# Compile and run HCCL C++ test locally
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Try to find CANN installation path if not specified
if [ -z "$ASCEND_HOME_PATH" ]; then
    if [ -n "$ASCEND_TOOLKIT_HOME" ]; then
        export ASCEND_HOME_PATH=$ASCEND_TOOLKIT_HOME
    elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
    elif [ -d "$HOME/workspace/cann" ]; then
        # Check if they have a local workspace build
        export ASCEND_HOME_PATH=$HOME/workspace/cann/runtime
        export HCCL_INC=$HOME/workspace/cann/hcomm/include
        export ACL_INC=$HOME/workspace/cann/runtime/include/external
    else
        echo "Error: ASCEND_HOME_PATH is not set. Please source the Ascend environment setup script (e.g. source /usr/local/Ascend/ascend-toolkit/set_env.sh)"
        exit 1
    fi
fi

if [ -z "$HCCL_INC" ]; then
    export HCCL_INC=$ASCEND_HOME_PATH/include
fi
if [ -z "$ACL_INC" ]; then
    export ACL_INC=$ASCEND_HOME_PATH/include
fi

echo "=== Compiling C++ HCCL test ==="
# We look for libraries in standard places or the CANN path
g++ -o hccl_cpp_test hccl_cpp_test.cpp \
    -I$ACL_INC \
    -I$HCCL_INC \
    -L$ASCEND_HOME_PATH/lib64 \
    -lascendcl -lhccl -lstdc++

echo "=== Running C++ HCCL test ==="

# VERY IMPORTANT: This flag bypasses the AICPU exception 507018 by forcing the
# HCCL operation onto the AI Vector Core (AIV)
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=512

ASCEND_DEVICE_ID=0 ./hccl_cpp_test --rank 0 --nranks 2 &
PID0=$!

sleep 0.5

ASCEND_DEVICE_ID=1 ./hccl_cpp_test --rank 1 --nranks 2 &
PID1=$!

wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

echo ""
if [ $STATUS0 -eq 0 ] && [ $STATUS1 -eq 0 ]; then
    echo "✅ C++ HCCL smoke test PASSED"
else
    echo "❌ C++ HCCL smoke test FAILED"
fi
