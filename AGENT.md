# AGENT.md

This file contains important information for AI agents working with this repository.

## Project Overview

This is a **super project** containing Git submodules for the PTO (Parallel Tensor Operation) ecosystem. It aggregates multiple related repositories into a unified workspace.

## Directory Structure

```
├── docs/                          # Documentation submodules
│   └── pypto_top_level_design_documents/  # Design docs
├── modules/                       # Code submodules
│   ├── PTOAS/                     # PTO Assembly/Dialect
│   ├── pto-isa/                   # PTO ISA specification
│   ├── pypto/                     # Core Python implementation
│   ├── pypto-lib/                 # Library utilities
│   ├── pypto-serving/             # Serving infrastructure
│   └── simpler/                   # Utilities
├── AGENT.md                       # This file
├── README.md                      # User documentation
└── .gitmodules                    # Submodule configuration
```

## Critical Rules for AI Agents

### 1. Submodule Management

- **NEVER** directly modify `.gitmodules` unless explicitly requested
- **NEVER** move submodules without updating `.gitmodules` and running `git submodule sync`
- **ALWAYS** use `git submodule add` to add new submodules, not manual edits
- **ALWAYS** verify paths in `.gitmodules` match actual directory structure

### 2. Making Changes

- Work in the appropriate submodule directory
- Commit changes in submodules FIRST
- Then update the super project with the new submodule commit
- Use descriptive commit messages mentioning which submodule changed

### 3. File Operations

- When editing files in submodules, treat them as independent repositories
- Each submodule has its own `.git` directory (inside super project's `.git/modules/`)
- Changes in submodules won't show in super project until committed in both places

### 4. Commands to Use

```bash
# Check submodule status
git submodule status

# Update all submodules to latest
git submodule update --remote

# Sync submodule configuration (after manual .gitmodules edits)
git submodule sync

# Execute command in all submodules
git submodule foreach '<command>'
```

### 5. Commands to AVOID

- `git add .` at super project root (may add unintended submodule changes)
- Moving submodule directories manually without proper git commands
- Deleting submodule directories without deinit

## Submodule Details

| Name | Type | Language | Purpose |
|------|------|----------|---------|
| PTOAS | Code | C++/MLIR | PTO Assembly compiler |
| pto-isa | Code/Docs | C++/Docs | ISA specification and reference |
| pypto | Code | Python | Core Python implementation |
| pypto-lib | Code | Python/C++ | Library functions |
| pypto-serving | Code | Python | Serving framework |
| simpler | Code | Various | Utility functions |
| pypto_top_level_design_documents | Docs | Markdown | Architecture docs |

## Common Tasks

### Update a Submodule

1. Change to submodule directory
2. Pull latest changes: `git pull origin main`
3. Return to super project
4. Stage the submodule: `git add <path>`
5. Commit: `git commit -m "Update <name> submodule"`

### Check for Updates

```bash
# See if submodules are behind
git submodule status

# Check latest in each submodule
git submodule foreach 'git fetch && git log HEAD..origin/main --oneline'
```

### Fix Submodule Issues

If a submodule shows as modified but you haven't changed anything:

```bash
# Reset submodule to committed state
git submodule update --init <path>

# Or if needed, force reset
cd <path>
git reset --hard HEAD
cd -
git submodule update --init <path>
```

## Testing

- Test changes in the relevant submodule before committing
- Some submodules may have CI/CD pipelines
- Check submodule's README for testing instructions

## Documentation

- Each submodule should have its own README.md
- High-level documentation is in `docs/pypto_top_level_design_documents/`
- This super project's README.md has usage instructions

## Contact

- Check individual submodule repositories for maintainers
- For super project issues, check with project owner

---

**Last Updated**: 2026-03-14
**Agent Instructions Version**: 1.0
