# PTO Superproject Issue Creation

When creating an issue in the PTO superproject, you MUST gather and include the following information:

## Required Information Checklist

### 1. Current Commit IDs of All Submodules

Run this command to get all submodule commit IDs:

```bash
git submodule status --recursive
```

Output format:
```
4a20283f0d8d3e9e2f1a0b9c8d7e6f5a4b3c2d1e modules/PTOAS (v1.2.3-45-g4a20283)
7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6 modules/pto-isa (v2.0.0)
...
```

**MUST INCLUDE**: All 6 submodules in modules/ directory

### 2. Reproduction Scripts/Commands

Provide EXACT commands to reproduce the issue:

```bash
# Example format:
cd modules/PTOAS
./build.sh
./test/run_tests.sh
```

**Requirements**:
- Start from superproject root
- Include all navigation commands (cd, etc.)
- Include build commands if needed
- Include the exact command that fails
- Note any environment variables or dependencies

### 3. Output Locations and Logs

Document where to find outputs:

```
Build logs: modules/PTOAS/build/logs/
Test output: modules/PTOAS/test/output/
Error logs: modules/PTOAS/build/errors.log
```

**Requirements**:
- Full relative path from superproject root
- Specific log file names
- Any relevant output directories

### 4. Specific Failure Location

Identify the exact location using `file:linenumber` format:

```
modules/PTOAS/lib/Transforms/InsertSync.cpp:247
modules/pto-isa/demos/baseline/add/csrc/kernel/add_custom.cpp:89
```

**How to find**:
- Search error messages for file paths
- Use grep to find relevant code
- Look at stack traces
- Check build output for compilation errors

### 5. Root Cause Analysis

Provide your best guess at the root cause:

```markdown
**Suspected Root Cause**: 
The synchronization primitive is not properly initialized before use in the InsertSync pass.
This appears to be a race condition when processing nested loops.

**Supporting Evidence**:
- Error occurs only with nested loop structures
- Memory analyzer shows uninitialized sync state
- Similar issues found in related passes
```

## Issue Creation Decision Tree

### Step 1: Identify the Failing Submodule

Based on the error location, determine which submodule:

| If error is in | Create issue in |
|----------------|-----------------|
| modules/PTOAS/ | https://github.com/zhangstevenunity/PTOAS |
| modules/pto-isa/ | https://github.com/PTO-ISA/pto-isa |
| modules/pypto/ | https://github.com/hw-native-sys/pypto |
| modules/pypto-lib/ | https://github.com/hengliao1972/pypto-lib |
| modules/pypto-serving/ | https://github.com/hengliao1972/pypto-serving |
| modules/simpler/ | https://github.com/ChaoWao/simpler |

### Step 2: Check for Related Issues

Before creating:
1. Search existing issues in the target repository
2. Check if similar issue exists
3. Reference related issues if found

### Step 3: Create the Issue

Use this template:

```markdown
## Environment

**Superproject Commit**: `abc123def456`

**Submodule Commits**:
- PTOAS: `4a20283f0d8d...`
- pto-isa: `7b8c9d0e1f2a...`
- pypto: `9c0d1e2f3a4b...`
- pypto-lib: `1d2e3f4a5b6c...`
- pypto-serving: `3f4a5b6c7d8e...`
- simpler: `5b6c7d8e9f0a...`

## Reproduction Steps

```bash
cd modules/<submodule>
<exact commands to reproduce>
```

## Expected Behavior

<What should happen>

## Actual Behavior

<What actually happens>

## Error Location

- File: `modules/<submodule>/path/to/file.cpp:123`
- Function: `functionName()`
- Error message: `<exact error message>`

## Logs and Outputs

**Build logs**: `modules/<submodule>/build/logs/`
**Test output**: `modules/<submodule>/test/output/`
**Relevant log excerpt**:
```
<paste relevant log lines>
```

## Root Cause Analysis

**Suspected cause**: <your analysis>

**Supporting evidence**: <why you think this>

## Additional Context

<Any other relevant information>
```

## Example Issue Creation Flow

### Example 1: Build Failure in PTOAS

```bash
# 1. Get submodule commits
git submodule status --recursive

# 2. Identify failure
# Error: "undefined reference to `InsertSyncAnalysis::run()`"
# Location: modules/PTOAS/lib/Transforms/InsertSync.cpp

# 3. Create issue in: https://github.com/zhangstevenunity/PTOAS
```

### Example 2: Runtime Error in pypto

```bash
# 1. Get submodule commits
git submodule status --recursive

# 2. Identify failure
# Error: "AttributeError: 'Tensor' object has no attribute 'shape'"
# Location: modules/pypto/pypto/tensor.py:156

# 3. Create issue in: https://github.com/hw-native-sys/pypto
```

## Pre-Creation Checklist

- [ ] Collected all submodule commit IDs
- [ ] Documented exact reproduction steps
- [ ] Identified all output/log locations
- [ ] Located exact failure point (file:line)
- [ ] Provided root cause analysis
- [ ] Selected correct submodule repository
- [ ] Checked for existing similar issues

## Commands for Information Gathering

```bash
# Get all submodule commits
git submodule status --recursive

# Get superproject commit
git rev-parse HEAD

# Check recent changes in submodule
cd modules/<submodule> && git log --oneline -5

# Search for error patterns
grep -r "error_pattern" modules/<submodule>/

# Find recently modified files
git diff --name-only HEAD~5 HEAD modules/<submodule>/
```
