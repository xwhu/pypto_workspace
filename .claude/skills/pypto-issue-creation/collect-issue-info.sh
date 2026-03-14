#!/bin/bash
#
# collect-issue-info.sh - Collect information needed for creating issues
# Usage: ./collect-issue-info.sh [submodule_name]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "PTO Superproject Issue Info Collector"
echo "========================================="
echo ""

# Get superproject info
echo -e "${GREEN}1. Superproject Commit ID:${NC}"
git rev-parse HEAD
echo ""

# Get all submodule commits
echo -e "${GREEN}2. Submodule Commit IDs:${NC}"
git submodule status --recursive | grep "modules/"
echo ""

# Get submodule info if specified
if [ -n "$1" ]; then
    SUBMODULE="modules/$1"
    
    if [ ! -d "$SUBMODULE" ]; then
        echo -e "${RED}Error: Submodule '$1' not found in modules/${NC}"
        echo "Available submodules:"
        ls -1 modules/
        exit 1
    fi
    
    echo -e "${GREEN}3. Recent Commits in $1:${NC}"
    cd "$SUBMODULE"
    git log --oneline -5
    cd - > /dev/null
    echo ""
    
    echo -e "${GREEN}4. Recent Changes in $1:${NC}"
    cd "$SUBMODULE"
    git diff --name-only HEAD~3 HEAD 2>/dev/null || echo "(No recent changes or shallow clone)"
    cd - > /dev/null
    echo ""
    
    echo -e "${GREEN}5. Submodule Details:${NC}"
    echo "Path: $SUBMODULE"
    cd "$SUBMODULE"
    echo "Remote URL: $(git remote get-url origin 2>/dev/null || echo 'N/A')"
    echo "Current Branch: $(git branch --show-current 2>/dev/null || echo 'detached HEAD')"
    cd - > /dev/null
    echo ""
fi

# Print issue creation template
echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}Issue Template (copy and fill in):${NC}"
echo -e "${YELLOW}=========================================${NC}"
echo ""
echo "## Environment"
echo ""
echo "**Superproject Commit**: \`$(git rev-parse HEAD)\`"
echo ""
echo "**Submodule Commits**:"
git submodule status --recursive | grep "modules/" | while read commit path rest; do
    name=$(basename "$path")
    echo "- $name: \`$commit\`"
done
echo ""
echo "## Reproduction Steps"
echo ""
echo '```bash'
echo '# TODO: Add your reproduction commands here'
echo '```'
echo ""
echo "## Expected Behavior"
echo ""
echo "TODO: Describe what should happen"
echo ""
echo "## Actual Behavior"
echo ""
echo "TODO: Describe what actually happens"
echo ""
echo "## Error Location"
echo ""
echo "- File: \`modules/TODO/path/to/file:LINENO\`"
echo "- Function: \`TODO\`"
echo "- Error message: \`TODO\`"
echo ""
echo "## Logs and Outputs"
echo ""
echo "**Build logs**: \`modules/TODO/build/logs/\`"
echo "**Test output**: \`modules/TODO/test/output/\`"
echo "**Relevant log excerpt**:"
echo '```'
echo 'TODO: Paste relevant log lines'
echo '```'
echo ""
echo "## Root Cause Analysis"
echo ""
echo "**Suspected cause**: TODO"
echo ""
echo "**Supporting evidence**: TODO"
echo ""
echo "## Repository to Create Issue"
echo ""
echo "TODO: Select one of:"
echo "- https://github.com/zhangstevenunity/PTOAS"
echo "- https://github.com/PTO-ISA/pto-isa"
echo "- https://github.com/hw-native-sys/pypto"
echo "- https://github.com/hengliao1972/pypto-lib"
echo "- https://github.com/hengliao1972/pypto-serving"
echo "- https://github.com/ChaoWao/simpler"
echo ""

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Collection complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
