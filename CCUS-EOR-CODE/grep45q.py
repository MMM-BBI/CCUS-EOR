"""Small helper: print context around key 45Q patterns in the fetched texts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:                      # Jupyter 的 sys.stdout 是 ipykernel.OutStream，
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:         # 没有 reconfigure 方法时忽略即可
    pass

D = chr(36)
R = Path(__file__).parent / "45q_sources"     # 由 fetch_45q.py 下载/归档

PATS = [
    D + "17",
    D + "85",
    D + "180",
    D + "36",
    "multiplied by 5",
    "increased credit amount",
    "applicable dollar amount",
    "12-year period",
    "placed in service after the date of enactment",
    "Parity",
]


def show(path: Path, keys: list[str], before: int = 400, after: int = 400, cap: int = 5) -> None:
    if not path.exists():
        print("#" * 110)
        print(f"## {path.name}: 文件不存在（{path.parent.name}/ 目录为空）。")
        print("   请先运行 fetch_45q.py 联网下载 45Q 原文，再执行本检索。")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    print("#" * 110)
    print(f"## {path.name}  ({len(text)} chars)")
    for k in keys:
        ms = list(re.finditer(re.escape(k), text, flags=re.I))[:cap]
        print("=" * 30, k, f"({len(ms)} shown)")
        for m in ms:
            s = max(0, m.start() - before)
            e = min(len(text), m.end() + after)
            print("  >", text[s:e].replace("\n", " "))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "irs_form8933_instructions_2025-12.txt"
    show(R / target, PATS)
