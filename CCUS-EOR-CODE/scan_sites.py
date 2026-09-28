"""Print every code site that touches the 45Q credit so the three scripts stay in sync."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:                      # Jupyter 的 sys.stdout 是 ipykernel.OutStream，
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:         # 没有 reconfigure 方法时忽略即可
    pass

KEYS = [
    r"self\.pi = params",
    r"R_EOR = self\.S_EOR",
    r"\+ self\.S_EOR \* self\.Q_EOR",
    r"r_eor",
    r"claim_years",
    r"232\.4",
    r"335\.4",
    r"IRA_45Q_UPLIFT",
    r"S_EOR_multiplier",
    r"'pi':",
    r"'S_EOR'",
]

named = sys.argv[1:] or ["ccus_part1.py", "ccus_part2senario.py", "ccus_part3.py"]
for name in named:
    path = Path(__file__).parent / name
    lines = path.read_text(encoding="utf-8").splitlines()
    print("#" * 95)
    print(path.name)
    for i, line in enumerate(lines, start=1):
        if any(re.search(k, line) for k in KEYS):
            print(f"{i:>5}| {line}")
