"""Order-of-magnitude net-emission check for the case project."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 与脚本同目录：由 ccus_part1.py 生成（这样整个目录可以整体迁移/上传 GitHub）
CASH = Path(__file__).parent / "ccus_cashflow_breakdown.csv"
BBL_PER_T = 7.3          # 表 2 注
CO2_PER_BBL = 0.43       # 常规原油燃烧排放系数（吨 CO2/桶）
ZETA = 0.975
CAPTURE = 1.604023e6     # 吨/年

if not CASH.exists():
    print("ccus_cashflow_breakdown.csv 不存在：请先运行 ccus_part1.py"
          "（notebook 第 1 个 cell）生成该文件，再执行本核算。")
    raise SystemExit(0)

df = pd.read_csv(CASH)
df = df[pd.to_numeric(df["operating_year"], errors="coerce").notna()].copy()
df["operating_year"] = df["operating_year"].astype(float)
df["oil_production_t"] = df["oil_production_t"].astype(float)

years = len(df)
oil = df["oil_production_t"].sum()
comb = oil * BBL_PER_T * CO2_PER_BBL
cap = CAPTURE * years

print(f"运营年数                 : {years}")
print(f"全周期增油量 (t)         : {oil:,.0f}   (= {oil*BBL_PER_T:,.0f} bbl)")
print(f"增油燃烧排放 (tCO2)      : {comb:,.0f}")
print(f"同期捕集量 (tCO2)        : {cap:,.0f}")
print(f"净封存量 (ζ=0.975)       : {cap*ZETA:,.0f}")
print(f"净排放 = 燃烧 − 净封存   : {comb - cap*ZETA:,.0f}")
print(f"净减排比 = 净封存/燃烧   : {cap*ZETA/comb:.2f}")

peak = df.loc[df["operating_year"] == 8, "oil_production_t"]
if len(peak):
    p = float(peak.iloc[0])
    print()
    print(f"峰值年 (t=8) 增油 (t)    : {p:,.0f}")
    print(f"峰值年燃烧排放 (tCO2)    : {p*BBL_PER_T*CO2_PER_BBL:,.0f}")
    print(f"峰值年捕集量 (tCO2)      : {CAPTURE:,.0f}")

print()
print("若效率曲线改按方案B（峰值仍为0.444，但前期更低）：全周期增油量下降，"
      "净排放会进一步恶化；若按方案C（峰值0.764）则相反。")
