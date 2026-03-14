# PTO Super Project

This is a super project that aggregates all PTO (Parallel Tensor Operation) related repositories as Git submodules.

## Project Structure

```
.
├── docs/
│   └── pypto_top_level_design_documents/    # Top-level design documents
├── modules/
│   ├── PTOAS/                               # PTO AS - Assembly/Dialect
│   ├── pto-isa/                             # PTO ISA - Instruction Set Architecture
│   ├── pypto/                               # Core Python PTO implementation
│   ├── pypto-lib/                           # PTO library
│   ├── pypto-serving/                       # PTO serving infrastructure
│   └── simpler/                             # Simpler utilities
└── README.md
```

## Submodules

| Submodule | Path | Repository | Description |
|-----------|------|------------|-------------|
| PTOAS | `modules/PTOAS` | https://github.com/zhangstevenunity/PTOAS | PTO Assembly and dialect implementation |
| pto-isa | `modules/pto-isa` | https://github.com/PTO-ISA/pto-isa | PTO Virtual ISA specification and demos |
| pypto | `modules/pypto` | https://github.com/hw-native-sys/pypto | Core Python PTO implementation |
| pypto-lib | `modules/pypto-lib` | https://github.com/hengliao1972/pypto-lib | PTO library utilities |
| pypto-serving | `modules/pypto-serving` | https://github.com/hengliao1972/pypto-serving | PTO serving infrastructure |
| simpler | `modules/simpler` | https://github.com/ChaoWao/simpler | Simpler utilities |
| pypto_top_level_design_documents | `docs/pypto_top_level_design_documents` | https://github.com/hengliao1972/pypto_top_level_design_documents | Design documentation |

## Quick Start

### Clone the Super Project

```bash
git clone <super-project-url>
cd <super-project-directory>
```

### Initialize All Submodules

```bash
# Initialize and update all submodules
git submodule update --init --recursive
```

### Update All Submodules

```bash
# Update all submodules to latest commits
git submodule update --remote

# Update specific submodule
git submodule update --remote modules/pypto
```

## Working with Submodules

### Pull Latest Changes

```bash
# Pull super project changes
git pull

# Pull and update all submodules
git pull --recurse-submodules
```

### Make Changes in a Submodule

```bash
# Navigate to submodule
cd modules/pypto

# Make changes, commit, and push
git add .
git commit -m "Your changes"
git push

# Go back to super project
cd ../..

# Commit the submodule update in super project
git add modules/pypto
git commit -m "Update pypto submodule"
```

### Add a New Submodule

```bash
git submodule add <repository-url> <path>
git commit -m "Add new submodule: <name>"
```

### Remove a Submodule

```bash
# Remove from .gitmodules and config
git submodule deinit -f <path>
rm -rf .git/modules/<path>
git rm -f <path>
git commit -m "Remove submodule: <name>"
```

## Development Workflow

1. **Always work on feature branches** in both super project and submodules
2. **Commit submodule changes first**, then update and commit in super project
3. **Use `git status` frequently** to check submodule status
4. **Run tests** in relevant submodules before committing

## Troubleshooting

### Submodule not initialized

```bash
git submodule update --init
```

### Detached HEAD in submodule

```bash
cd <submodule-path>
git checkout <branch-name>
cd -
git add <submodule-path>
git commit -m "Switch submodule to branch"
```

### Clean all submodules

```bash
git submodule foreach --recursive 'git clean -xfd'
```

## License

See individual submodule repositories for their respective licenses.

## Contributing

Please refer to the contributing guidelines in each submodule repository.
