#!/usr/bin/env python3
"""Generate side-by-side AIC/AIV local tensor usage per mixed kernel."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path("/data/liaoheng/pypto_workspace/pypto-lib/examples")

PASS08_FILES = [
    ("decode_front", ROOT / "deepseek_v3_2_decode_front_dump/passes_dump/08_after_ExpandMixedKernel.py"),
    ("prefill_front", ROOT / "deepseek_v3_2_prefill_front_dump/passes_dump/08_after_ExpandMixedKernel.py"),
    ("decode_back", ROOT / "deepseek_v3_2_decode_back_dump/passes_dump/08_after_ExpandMixedKernel.py"),
    ("prefill_back", ROOT / "deepseek_v3_2_prefill_back_dump/passes_dump/08_after_ExpandMixedKernel.py"),
]

DTYPE_SIZE = {
    "BFLOAT16": 2,
    "FP16": 2,
    "FP32": 4,
    "FP8E4M3FN": 1,
    "INT32": 4,
    "INT64": 8,
    "INDEX": 4,
    "BOOL": 1,
}


def eval_int_expr(expr: str) -> int:
    node = ast.parse(expr, mode="eval").body

    def _eval(n: ast.AST) -> int:
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -_eval(n.operand)
        if isinstance(n, ast.BinOp):
            lhs = _eval(n.left)
            rhs = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return lhs + rhs
            if isinstance(n.op, ast.Sub):
                return lhs - rhs
            if isinstance(n.op, ast.Mult):
                return lhs * rhs
            if isinstance(n.op, ast.FloorDiv):
                return lhs // rhs
        raise ValueError(f"unsupported expr: {expr}")

    return _eval(node)


def parse_func_blocks(text: str) -> list[tuple[str, str]]:
    pat = re.compile(r"^    def\s+(\w+)\(.*?:\n(.*?)(?=^    def\s+\w+\(|\Z)", re.M | re.S)
    return [(m.group(1), m.group(2)) for m in pat.finditer(text)]


def bytes_from_create_tensor(func_body: str) -> int:
    total = 0
    pat = re.compile(r"tensor\.create\((\[[^\)]*?\]|__list__\([^\)]*?\)),\s*dtype=pl\.([A-Z0-9_]+)", re.S)
    for m in pat.finditer(func_body):
        dims_raw = m.group(1).strip()
        dtype = m.group(2)
        item_size = DTYPE_SIZE.get(dtype, 0)
        if item_size == 0:
            continue
        if dims_raw.startswith("__list__("):
            inner = dims_raw[len("__list__(") : -1]
            dims = [d.strip() for d in inner.split(",") if d.strip()]
        else:
            inner = dims_raw[1:-1]
            dims = [d.strip() for d in inner.split(",") if d.strip()]
        numel = 1
        for d in dims:
            numel *= eval_int_expr(d)
        total += numel * item_size
    return total


def format_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def main() -> None:
    out = ["# DeepSeek v3.2 Mixed Kernel Local Tensor Usage (AIC vs AIV)", ""]
    out.append("统计口径：基于 `08_after_ExpandMixedKernel.py` 中每个 `*_aic/*_aiv` 函数里 `create_tensor(...)` 的显式本地张量字节数。")
    out.append("")
    out.append("- 这是 **AIC/AIV 并排同口径** 的 local tensor 统计。")
    out.append("- 不再只看 AIV，也不做 front/back 跨函数“总 usage%”混算。")
    out.append("")

    for tag, path in PASS08_FILES:
        text = path.read_text(encoding="utf-8")
        blocks = parse_func_blocks(text)
        rows: dict[str, dict[str, int]] = {}
        for name, body in blocks:
            if not (name.endswith("_aic") or name.endswith("_aiv")):
                continue
            side = "aic" if name.endswith("_aic") else "aiv"
            base = name[: -len("_aic")] if side == "aic" else name[: -len("_aiv")]
            rows.setdefault(base, {"aic": 0, "aiv": 0})
            rows[base][side] = bytes_from_create_tensor(body)

        out.append(f"## {tag}")
        out.append("")
        out.append("| Mixed Kernel | AIC local tensor used | AIV local tensor used |")
        out.append("|---|---:|---:|")
        for base in sorted(rows.keys()):
            aic_b = rows[base]["aic"]
            aiv_b = rows[base]["aiv"]
            out.append(f"| `{base}` | {format_bytes(aic_b)} | {format_bytes(aiv_b)} |")
        out.append("")

    out_path = ROOT / "docs/deepseek_v3_2_mixed_kernel_local_usage_side_by_side.md"
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

