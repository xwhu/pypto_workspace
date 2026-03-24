#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# remote_test.sh — Deploy & run tests on remote NPU server
#
# Usage:
#   ./scripts/remote_test.sh gen-ref        # 生成 Python 参考数据
#   ./scripts/remote_test.sh deploy-rust    # 打包上传 Rust 代码
#   ./scripts/remote_test.sh cargo-test     # 远程 cargo test (需硬件)
#   ./scripts/remote_test.sh cargo-build    # 远程 cargo build --features ascend
#   ./scripts/remote_test.sh all            # 以上全部
#   ./scripts/remote_test.sh ssh            # 打开交互式 ssh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ──
REMOTE_USER="h00546948"
REMOTE_HOST="996server"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_DIR="pypto_workspace"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"  # project root

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${CYAN}[remote_test]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }

# ── Helper: run command on remote ──
remote_exec() {
    ssh "${REMOTE}" "$@"
}

# ── Helper: run command on remote with venv ──
remote_python() {
    ssh "${REMOTE}" "source ~/${REMOTE_DIR}/.venv/bin/activate && cd ~/${REMOTE_DIR} && $*"
}

# ── Helper: run command on remote with cargo env ──
remote_cargo() {
    ssh "${REMOTE}" "source ~/.cargo/env 2>/dev/null; cd ~/${REMOTE_DIR}/rust_llm_server && $*"
}

# ─────────────────────────────────────────────────────────────────
# Command: deploy-python
#   SCP Python test files to remote
# ─────────────────────────────────────────────────────────────────
cmd_deploy_python() {
    log "Deploying Python test files to ${REMOTE}:${REMOTE_DIR}/tests/"

    # Ensure remote tests dir exists
    remote_exec "mkdir -p ~/${REMOTE_DIR}/tests"

    # Copy all Python test files
    scp -r "${LOCAL_ROOT}/tests/"*.py "${REMOTE}:~/${REMOTE_DIR}/tests/" 2>/dev/null || {
        warn "No .py files in tests/ — nothing to deploy"
        return 0
    }

    ok "Python test files deployed"
}

# ─────────────────────────────────────────────────────────────────
# Command: gen-ref
#   Deploy Python + generate reference data on remote
# ─────────────────────────────────────────────────────────────────
cmd_gen_ref() {
    cmd_deploy_python

    log "Generating reference data on remote (PyTorch)..."
    remote_python "python tests/gen_reference.py"

    # Copy reference data back to local for Rust tests
    log "Downloading reference data to local..."
    mkdir -p "${LOCAL_ROOT}/rust_llm_server/tests/reference_data"
    scp -r "${REMOTE}:~/${REMOTE_DIR}/tests/reference_data/"* \
           "${LOCAL_ROOT}/rust_llm_server/tests/reference_data/"

    ok "Reference data generated and downloaded"
}

# ─────────────────────────────────────────────────────────────────
# Command: deploy-rust
#   Pack rust_llm_server + rustBindings, upload and unpack
# ─────────────────────────────────────────────────────────────────
cmd_deploy_rust() {
    log "Packing Rust code..."
    local tarball="/tmp/rust_deploy_$$.tar.gz"

    # Pack, excluding target dirs and .git
    tar czf "${tarball}" \
        -C "${LOCAL_ROOT}" \
        --exclude='*/target' \
        --exclude='*/.git' \
        --exclude='*/reference_data' \
        rust_llm_server \
        rustBindings

    local size=$(du -h "${tarball}" | cut -f1)
    log "Tarball: ${tarball} (${size})"

    log "Uploading to ${REMOTE}:~/${REMOTE_DIR}/ ..."
    scp "${tarball}" "${REMOTE}:/tmp/rust_deploy.tar.gz"

    log "Unpacking on remote..."
    remote_exec "cd ~/${REMOTE_DIR} && tar xzf /tmp/rust_deploy.tar.gz && rm /tmp/rust_deploy.tar.gz"

    # Cleanup local tarball
    rm -f "${tarball}"

    ok "Rust code deployed to ${REMOTE}:~/${REMOTE_DIR}/"
}

# ─────────────────────────────────────────────────────────────────
# Command: deploy-ref
#   Upload local reference data to remote for Rust tests
# ─────────────────────────────────────────────────────────────────
cmd_deploy_ref() {
    local ref_dir="${LOCAL_ROOT}/rust_llm_server/tests/reference_data"
    if [ ! -d "${ref_dir}" ] || [ -z "$(ls -A "${ref_dir}" 2>/dev/null)" ]; then
        warn "No reference data found at ${ref_dir}. Run 'gen-ref' first."
        return 1
    fi

    log "Uploading reference data to remote..."
    remote_exec "mkdir -p ~/${REMOTE_DIR}/rust_llm_server/tests/reference_data"
    scp -r "${ref_dir}/"* "${REMOTE}:~/${REMOTE_DIR}/rust_llm_server/tests/reference_data/"
    ok "Reference data uploaded"
}

# ─────────────────────────────────────────────────────────────────
# Command: cargo-build
#   Build Rust project on remote with ascend feature
# ─────────────────────────────────────────────────────────────────
cmd_cargo_build() {
    log "Building on remote (cargo build --features ascend)..."
    remote_cargo "cargo build --features ascend 2>&1"
    ok "Build succeeded"
}

# ─────────────────────────────────────────────────────────────────
# Command: cargo-test
#   Run Rust integration tests on remote NPU
# ─────────────────────────────────────────────────────────────────
cmd_cargo_test() {
    log "Running Rust tests on remote NPU..."
    remote_cargo "cargo test --features ascend -- --ignored 2>&1"
    ok "Tests completed"
}

# ─────────────────────────────────────────────────────────────────
# Command: cargo-test-unit
#   Run unit tests only (no NPU needed)
# ─────────────────────────────────────────────────────────────────
cmd_cargo_test_unit() {
    log "Running unit tests on remote..."
    remote_cargo "cargo test --features ascend 2>&1"
    ok "Unit tests completed"
}

# ─────────────────────────────────────────────────────────────────
# Command: ssh — interactive session
# ─────────────────────────────────────────────────────────────────
cmd_ssh() {
    log "Opening interactive SSH session..."
    ssh -t "${REMOTE}" "cd ~/${REMOTE_DIR} && exec bash -l"
}

# ─────────────────────────────────────────────────────────────────
# Command: all — full pipeline
# ─────────────────────────────────────────────────────────────────
cmd_all() {
    cmd_gen_ref
    echo ""
    cmd_deploy_rust
    echo ""
    cmd_deploy_ref
    echo ""
    cmd_cargo_build
    echo ""
    cmd_cargo_test
}

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
case "${1:-help}" in
    gen-ref)        cmd_gen_ref ;;
    deploy-python)  cmd_deploy_python ;;
    deploy-rust)    cmd_deploy_rust ;;
    deploy-ref)     cmd_deploy_ref ;;
    cargo-build)    cmd_cargo_build ;;
    cargo-test)     cmd_cargo_test ;;
    cargo-test-unit) cmd_cargo_test_unit ;;
    ssh)            cmd_ssh ;;
    all)            cmd_all ;;
    *)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  gen-ref         Generate Python reference data on remote"
        echo "  deploy-python   Upload Python test files"
        echo "  deploy-rust     Pack & upload Rust code (rust_llm_server + rustBindings)"
        echo "  deploy-ref      Upload reference data to remote for Rust tests"
        echo "  cargo-build     Build Rust on remote (--features ascend)"
        echo "  cargo-test      Run Rust NPU tests on remote (--ignored)"
        echo "  cargo-test-unit Run Rust unit tests on remote"
        echo "  ssh             Interactive SSH session"
        echo "  all             Full pipeline: gen-ref → deploy → build → test"
        exit 1
        ;;
esac
