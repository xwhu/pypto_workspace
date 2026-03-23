---
description: 开发测试流程 — 在本地开发，在远程服务器构建和测试
---

# 开发测试工作流 (dev-test)

本工作流用于日常开发：本地编写代码，远程 996server 构建和测试，结果返回给 Agent。

## 前置条件

- 远程服务器 `996server` 可通过 SSH 访问
- 远程服务器已安装 Rust 工具链
- 远程仓库已克隆到 `/data/h00546948/pypto_workspace`

## 工作流程

### 1. 确保在 dev 分支开发

所有开发工作必须在非 `main` 分支进行。使用 `dev/feature-name` 命名规范。

```bash
# 创建或切换到 dev 分支
git checkout -b dev/feature-name
```

### 2. 编写代码并提交

完成代码修改后，提交并推送到远程：

```bash
git add -A
git commit -m "feat: description of changes"
git push origin dev/feature-name
```

### 3. 在远程服务器执行构建和测试

推送后，通过 SSH 在远程服务器上运行构建和测试脚本：

```bash
# 完整构建 + 测试
// turbo
ssh 996server 'bash /data/h00546948/pypto_workspace/scripts/remote-test.sh -b dev/feature-name'
```

可选的快速检查（仅语法和 lint，不运行测试）：

```bash
# 快速检查
// turbo
ssh 996server 'bash /data/h00546948/pypto_workspace/scripts/remote-test.sh -b dev/feature-name --check'
```

测试特定 package：

```bash
# 测试指定 package
// turbo
ssh 996server 'bash /data/h00546948/pypto_workspace/scripts/remote-test.sh -b dev/feature-name -p package_name'
```

### 4. 解析结果

脚本输出结构化 JSON 结果，位于 `===RESULT_JSON_BEGIN===` 和 `===RESULT_JSON_END===` 标记之间：

```json
{
  "status": "pass|fail|error",
  "stage": "check|build|test|git|preflight",
  "message": "human readable message",
  "duration_seconds": 42,
  "branch": "dev/feature-name",
  "timestamp": "2026-03-23T06:00:00Z"
}
```

- **status=pass**: 所有检查通过，可以继续
- **status=fail**: 构建或测试失败，需要修复
- **status=error**: 环境问题（如 cargo 未安装），需要先解决

### 5. 修复问题（如有）

如果测试失败，根据输出修复代码，然后从步骤 2 重新开始。

### 6. 测试通过后，创建 PR

所有测试通过后，使用 `/merge-to-main` 工作流将代码合并到 `main` 分支。
