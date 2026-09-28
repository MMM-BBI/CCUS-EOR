"""Static sanity check for run_all_ccus.ipynb: cell syntax, referenced scripts,
and whether the scripts themselves depend on sys.argv."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

try:                      # Jupyter 的 sys.stdout 是 ipykernel.OutStream，
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:         # 没有 reconfigure 方法时忽略即可
    pass

ROOT = Path(__file__).parent
NB = json.loads((ROOT / "run_all_ccus.ipynb").read_text(encoding="utf-8"))

print("== pipeline scripts ==")
for name in ("ccus_part1.py", "ccus_part2senario.py", "ccus_part3.py"):
    text = (ROOT / name).read_text(encoding="utf-8")
    print(f"  {name:<26} uses sys.argv: {'sys.argv' in text!s:<5} "
          f"os.environ refs: {text.count('os.environ')}")

print("== notebook cell syntax ==")
bad = 0
for i, cell in enumerate(NB["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    lines = ["pass\n" if re.match(r"^\s*%[a-zA-Z]", ln) else ln
             for ln in src.splitlines(True)]
    src = "".join(lines)
    if not src.strip():
        continue
    try:
        compile(src, f"cell{i}", "exec")
        print(f"  cell {i}: ok")
    except SyntaxError as exc:
        bad += 1
        print(f"  cell {i}: SYNTAX ERROR {exc}")
print("  ->", "ALL OK" if bad == 0 else f"{bad} problem(s)")

print("== scripts referenced by the notebook ==")
refs: set[str] = set()
for cell in NB["cells"]:
    src = "".join(cell["source"])
    refs |= set(re.findall(r'BASE_DIR / "([^"]+\.py)"', src))
    refs |= set(re.findall(r'run_py\("([^"]+\.py)"', src))
    refs |= set(re.findall(r'run_script\("([^"]+\.py)"', src))
for name in sorted(refs):
    print(f"  {name:<26} present: {(ROOT / name).exists()}")

print("== notebook-created artefacts (created at run time) ==")
for name in ("figures", "45q_sources"):
    exists = (ROOT / name).exists()
    note = "already present" if exists else "will be created when the notebook runs"
    print(f"  {name:<26} {note}")
