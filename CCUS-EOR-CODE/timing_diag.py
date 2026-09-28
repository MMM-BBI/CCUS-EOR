"""How saturated is the investment-timing response?"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).parent   # 与本文件同目录（情景 CSV 的输出目录）
_NEEDED = ["ccus_scenario_timing_distribution.csv", "ccus_scenario_results.csv"]
_missing = [n for n in _NEEDED if not (HERE / n).exists()]
if _missing:
    print("缺少情景结果文件：%s。请先运行 ccus_part2senario.py"
          "（notebook 第 2 个 cell）生成情景 CSV，再执行本诊断。" % "、".join(_missing))
    raise SystemExit(0)

dist = pd.read_csv(HERE / "ccus_scenario_timing_distribution.csv")
res = pd.read_csv(HERE / "ccus_scenario_results.csv")

singles = ["Technology incentives", "One-off investment subsidy",
           "Carbon trading", "Tax incentives"]
combos = [n for n in dist["Policy type"].unique()
          if n.startswith("Combined policy:") and n.count("+") == 1]

print("share of paths that invest immediately in 2026")
for n in singles + combos:
    d26 = dist[(dist["Policy type"] == n) & (dist["Investment year"] == 2026)]
    txt = "  ".join(f"{r['Policy strength (%)']}={r['Share of investing paths (%)']:.0f}%"
                    for _, r in d26.iterrows())
    print(f"  {n[:58]:<60} {txt}")

print()
print("median investment year")
for n in singles + combos:
    sub = res[res["Policy type"] == n]
    txt = "  ".join(f"{r['Policy strength (%)']}={r['Median investment year']:.0f}"
                    for _, r in sub.iterrows())
    print(f"  {n[:58]:<60} {txt}")
