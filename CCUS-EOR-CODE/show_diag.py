"""Print the diagnostics results in a compact form."""

import json
import sys
from pathlib import Path

try:                      # Jupyter 的 sys.stdout 是 ipykernel.OutStream，
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:         # 没有 reconfigure 方法时忽略即可
    pass

SRC = Path(__file__).parent / "diagnostics_results.json"
if not SRC.exists():
    print("diagnostics_results.json 不存在：请先把 notebook 第 5 个 cell 里的 "
          "RUN_DIAGNOSTICS 设为 True 并运行一次诊断（约 30-45 分钟），"
          "诊断脚本会在本目录生成该文件。")
    raise SystemExit(0)

d = json.loads(SRC.read_text(encoding="utf-8"))

print("--- C. analytic validation ---")
for c in d["C_analytic"]["cases"]:
    print(f"  T={c['T']:.0f} steps={c['steps']} K/S={c['K']/c['S0']:.2f} "
          f"binomial={c['binomial']:.4f}")
    for k, v in c["baselines"].items():
        print(f"      {k:<9} {v['mean']:.4f}  sd={v['sd']:.4f}  "
              f"err={v['relative_error_pct']:+.2f}%")

print("--- E. CRN resolution (M=5000, seeds 7/42/2024) ---")
for k, v in d["E_resolution"]["variants"].items():
    print(f"  {k:<48} dV0={v['delta_mean']/1e6:+8.1f}  "
          f"sd={v['delta_sd']/1e6:5.1f}  level_sd={v['level_sd']/1e6:5.1f}  "
          f"noise cut {v['noise_reduction_factor']:.2f}x")

print("--- A. seeds ---")
a = d["A_seed"]
print("  seeds", [r["seed"] for r in a["seeds"]],
      f"mean={a['mean_v0']/1e6:,.1f} sd={a['sd_v0']/1e6:,.1f} "
      f"CI=[{a['ci95_low']/1e6:,.1f}, {a['ci95_high']/1e6:,.1f}]")

print("--- B. convergence ---")
for r in d["B_convergence"]["rows"]:
    print(f"  M={r['M']:>6}  V0={r['v0']/1e6:>9,.1f}  median={r['median_year']}  "
          f"modal={r['modal_year']}({r['modal_share']:.1f}%)  {r['seconds']:.0f}s")

print("--- D. controls ---")
dd = d["D_controls"]
for k, v in dd.items():
    if isinstance(v, dict) and "v0" in v:
        print(f"  {k:<32} V0={v['v0']/1e6:>9,.1f}  median={v['median_year']}  "
              f"premium={v['premium']/1e6:>8,.1f}")
print("  deterministic best year:", dd["deterministic_timing"]["best_year"],
      f"({dd['deterministic_timing']['best_npv']/1e6:,.1f} million CNY)")
print("  invest immediately:", f"{dd['invest_immediately']['mean_npv']/1e6:,.1f}",
      "share NPV>0 =", f"{dd['invest_immediately']['share_positive']*100:.1f}%")
