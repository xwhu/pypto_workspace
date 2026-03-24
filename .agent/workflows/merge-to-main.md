---
description: 将代码合并到 main 分支 — 创建 PR、等待 CI、合并
---

# 合并到 main 工作流 (merge-to-main)

将通过测试的 dev 分支代码合并到受保护的 `main` 分支。

## 前置条件

- 当前分支的所有测试已通过（通过 `/dev-test` 工作流验证）
- 所有代码已提交并推送

## 工作流程

### 1. 确认测试结果

确保最新的远程测试已通过：

```bash
// turbo
ssh 996server 'bash /data/h00546948/pypto_workspace/scripts/remote-test.sh -b dev/feature-name'
```

### 2. 创建或更新 Pull Request

```bash
# 创建 PR（首次）
gh pr create --base main --head dev/feature-name --title "feat: description" --body "## Changes\n- item 1\n- item 2\n\n## Test Results\n- All tests passed on remote server"
```

如果 PR 已存在，直接推送更新即可：

```bash
git push origin dev/feature-name
```

### 3. 等待 CI 检查通过

```bash
# 查看 PR 检查状态
// turbo
gh pr checks
```

如果 CI 失败，修复代码并推送更新，然后重新检查。

### 4. 合并 PR

CI 通过后，合并 PR：

```bash
# Squash merge 到 main
gh pr merge --squash --delete-branch
```

### 5. 本地同步

合并完成后，更新本地 main 分支：

```bash
// turbo
git checkout main && git pull origin main
```

## 注意事项

- `main` 分支受保护，不允许直接 push
- PR 必须通过 CI 检查才能合并
- 推荐使用 squash merge 保持 main 分支历史整洁
