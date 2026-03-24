#!/usr/bin/env bash
# remote-test.sh — 在远程服务器上执行构建和测试
# 使用方式 (本地调用):
#   ssh 996server 'bash /data/h00546948/pypto_workspace/scripts/remote-test.sh [OPTIONS]'
#
# 选项:
#   -b, --branch BRANCH   要测试的分支 (默认: 当前分支)
#   -c, --clean           构建前执行 cargo clean
#   -t, --test-only       仅运行测试，不构建
#   -p, --package PKG     仅测试指定的 package
#   --check               仅运行 cargo check + clippy (快速检查)
#   -h, --help            显示帮助

set -euo pipefail

# ============================================================
# 环境设置
# ============================================================
REPO_DIR="/data/h00546948/pypto_workspace"

# 确保 cargo 在 PATH 中
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# ============================================================
# 输出工具函数
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[PASS]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()    { echo -e "${RED}[FAIL]${NC} $*"; }

# JSON 格式的结果输出，供 Agent 解析
output_result() {
    local status="$1"
    local stage="$2"
    local message="$3"
    local duration="$4"
    echo ""
    echo "===RESULT_JSON_BEGIN==="
    cat <<EOF
{
  "status": "${status}",
  "stage": "${stage}",
  "message": "${message}",
  "duration_seconds": ${duration},
  "branch": "${BRANCH}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "===RESULT_JSON_END==="
}

# ============================================================
# 参数解析
# ============================================================
BRANCH=""
CLEAN=false
TEST_ONLY=false
CHECK_ONLY=false
PACKAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--branch) BRANCH="$2"; shift 2;;
        -c|--clean) CLEAN=true; shift;;
        -t|--test-only) TEST_ONLY=true; shift;;
        -p|--package) PACKAGE="$2"; shift 2;;
        --check) CHECK_ONLY=true; shift;;
        -h|--help)
            echo "Usage: remote-test.sh [OPTIONS]"
            echo "  -b, --branch BRANCH   Branch to test (default: current)"
            echo "  -c, --clean           Run cargo clean before build"
            echo "  -t, --test-only       Run tests only, skip build"
            echo "  -p, --package PKG     Test specific package"
            echo "  --check               Quick check (cargo check + clippy)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ============================================================
# 前置检查
# ============================================================
if ! command -v cargo &> /dev/null; then
    fail "cargo not found. Please install Rust first:"
    fail "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    output_result "error" "preflight" "cargo not found" 0
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    fail "Repository not found at $REPO_DIR"
    fail "Please clone it first:"
    fail "  git clone git@github.com:xwhu/pypto_workspace.git $REPO_DIR"
    output_result "error" "preflight" "repository not found at $REPO_DIR" 0
    exit 1
fi

# ============================================================
# 切换到仓库目录
# ============================================================
cd "$REPO_DIR"
TIMER_START=$(date +%s)

# ============================================================
# 分支切换
# ============================================================
if [ -n "$BRANCH" ]; then
    info "Fetching latest from origin..."
    git fetch origin --prune
    
    # 检查分支是否存在
    if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
        info "Checking out branch: $BRANCH"
        git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
        git reset --hard "origin/$BRANCH"
        info "Updating submodules..."
        git submodule update --init --recursive
    else
        fail "Branch '$BRANCH' not found on remote"
        output_result "error" "git" "branch not found: $BRANCH" 0
        exit 1
    fi
else
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    info "Using current branch: $BRANCH"
    info "Pulling latest changes..."
    git pull --ff-only origin "$BRANCH" 2>/dev/null || true
    info "Updating submodules..."
    git submodule update --init --recursive
fi

info "Current commit: $(git log --oneline -1)"
echo ""

# ============================================================
# Clean (可选)
# ============================================================
if [ "$CLEAN" = true ]; then
    info "Running cargo clean..."
    cargo clean
fi

# ============================================================
# 快速检查模式
# ============================================================
if [ "$CHECK_ONLY" = true ]; then
    info "=== Running cargo check ==="
    if cargo check --all-targets 2>&1; then
        success "cargo check passed"
    else
        fail "cargo check failed"
        TIMER_END=$(date +%s)
        output_result "fail" "check" "cargo check failed" $((TIMER_END - TIMER_START))
        exit 1
    fi
    
    echo ""
    info "=== Running clippy ==="
    if cargo clippy --all-targets -- -D warnings 2>&1; then
        success "clippy passed"
    else
        warn "clippy has warnings"
    fi
    
    TIMER_END=$(date +%s)
    output_result "pass" "check" "cargo check + clippy passed" $((TIMER_END - TIMER_START))
    exit 0
fi

# ============================================================
# 构建
# ============================================================
if [ "$TEST_ONLY" = false ]; then
    info "=== Building ==="
    CARGO_BUILD_ARGS=""
    if [ -n "$PACKAGE" ]; then
        CARGO_BUILD_ARGS="-p $PACKAGE"
    fi
    
    if cargo build $CARGO_BUILD_ARGS 2>&1; then
        success "Build succeeded"
    else
        fail "Build failed"
        TIMER_END=$(date +%s)
        output_result "fail" "build" "cargo build failed" $((TIMER_END - TIMER_START))
        exit 1
    fi
    echo ""
fi

# ============================================================
# 测试
# ============================================================
info "=== Running tests ==="
CARGO_TEST_ARGS=""
if [ -n "$PACKAGE" ]; then
    CARGO_TEST_ARGS="-p $PACKAGE"
fi

if cargo test $CARGO_TEST_ARGS 2>&1; then
    success "All tests passed"
    TIMER_END=$(date +%s)
    output_result "pass" "test" "all tests passed" $((TIMER_END - TIMER_START))
    exit 0
else
    fail "Some tests failed"
    TIMER_END=$(date +%s)
    output_result "fail" "test" "some tests failed" $((TIMER_END - TIMER_START))
    exit 1
fi
