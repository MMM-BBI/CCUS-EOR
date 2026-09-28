"""Summarise the single-policy and combined-policy scenarios.

Key idea: because every tool is anchored at the same baseline (0% = baseline,
100% = that tool's policy ceiling), the effect of a package at full intensity
can be compared with the sum of its members' individual effects.  The
difference is the synergy of the combination.
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).parent
FULL = HERE / "ccus_scenario_results.csv"
BENCH = HERE / "benchmarks.json"

_missing = [p.name for p in (FULL, BENCH) if not p.exists()]
if _missing:
    print("缺少输入文件：%s。\n请按顺序先运行 ccus_part2senario.py"
          "（notebook 第 2 个 cell）与 decompose.py（第 4 个 cell），"
          "再执行本汇总。" % "、".join(_missing))
    raise SystemExit(0)

SINGLE = {
    "Carbon trading": "carbon",
    "One-off investment subsidy": "subsidy",
    "Tax incentives": "tax",
    "Technology incentives": "technology",
}


def load(path=FULL):
    df = pd.read_csv(path)
    df["strength"] = df["Policy strength (%)"].str.rstrip("%").astype(float)
    for col in ("Option value at decision, V0 (million CNY)",
                "Median investment year",
                "Share of investing paths (%)"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace("%", ""), errors="coerce")
    return df


def main():
    df = load()
    df["label"] = df["Policy type"].str.replace("Combined policy: ", "", regex=False)

    rows = []
    for label, grp in df.groupby("label", sort=False):
        grp = grp.sort_values("strength")
        lo = grp.iloc[0]
        hi = grp.iloc[-1]
        rows.append({
            "scenario": label,
            "V0_0": lo["Option value at decision, V0 (million CNY)"],
            "V0_100": hi["Option value at decision, V0 (million CNY)"],
            "dV0": hi["Option value at decision, V0 (million CNY)"]
                   - lo["Option value at decision, V0 (million CNY)"],
            "year_0": lo["Median investment year"],
            "year_100": hi["Median investment year"],
            "year_gain": lo["Median investment year"] - hi["Median investment year"],
        })
    summary = pd.DataFrame(rows)

    base_v0 = summary.loc[summary.scenario == "Carbon trading", "V0_0"].iloc[0]
    base_year = summary.loc[summary.scenario == "Carbon trading", "year_0"].iloc[0]
    summary["dV0_vs_base"] = summary["V0_100"] - base_v0

    # Individual tool effects, each evaluated at the SAME policy ceiling that the
    # tool reaches inside a combination (s = 100%):
    #   carbon     : carbon price +25%
    #   subsidy    : investment subsidy ratio 100%
    #   tax        : 25% relief of the 25% corporate income tax
    #   technology : 45Q utilisation credit raised from the baseline level
    #                (35 USD/t) to the current full credit (85 USD/t = 603.5 CNY/t),
    #                with the clean-electricity subsidy held at its baseline
    tool_effect = {}
    for name, short in SINGLE.items():
        if short == "technology":
            continue                      # handled from the dedicated benchmark run
        r = summary[summary.scenario == name].iloc[0]
        tool_effect[short] = r["dV0_vs_base"]

    bench_path = HERE / "benchmarks.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        tool_effect["technology"] = (
            bench["technology_ira"]["v0"] - bench["baseline"]["v0"]) / 1e6
        print("technology tool effect at the full 45Q ceiling: "
              f"{tool_effect['technology']:,.1f} million CNY "
              f"(benchmark V0 = {bench['technology_ira']['v0']/1e6:,.1f})")
        print()
    else:
        tool_effect["technology"] = 0.0
        print("!! benchmarks.json missing - run decompose.py first")

    def short_name(label):
        return [k for k, v in SINGLE.items() if v and v in label] or [label]

    def members(label):
        if label in SINGLE:
            return [SINGLE[label]]
        return [s for s in ("carbon", "subsidy", "tax", "technology")
                if re.search(rf"\b{s}", label) or s in label]

    def degree(label):
        return 1 if label in SINGLE else len(members(label))

    summary["degree"] = summary.scenario.map(degree)
    summary["members"] = summary.scenario.map(lambda s: "+".join(members(s)))
    summary["sum_individual"] = summary["members"].map(
        lambda m: sum(tool_effect.get(x, 0.0) for x in m.split("+")))
    summary["synergy"] = summary["dV0_vs_base"] - summary["sum_individual"]
    summary["synergy_ratio"] = summary["synergy"] / summary["sum_individual"]
    # 单一政策情景本身就是基准，不做协同分解
    single_mask = summary["degree"] == 1
    for col in ("sum_individual", "synergy", "synergy_ratio"):
        summary.loc[single_mask, col] = float("nan")

    cols = ["scenario", "degree", "V0_0", "V0_100", "dV0_vs_base",
            "sum_individual", "synergy", "synergy_ratio",
            "year_0", "year_100", "year_gain"]
    out = summary[cols].sort_values(["degree", "dV0_vs_base"], ascending=[True, False])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("baseline V0 (0%):", round(base_v0, 2), " million CNY; baseline median year:", base_year)
    print("individual tool effects at their combination ceiling (million CNY):")
    for k, v in tool_effect.items():
        print(f"    {k:<12} {v:,.1f}")
    print()
    for d in (1, 2, 3, 4):
        sub = out[out.degree == d]
        if sub.empty:
            continue
        print("=" * 100)
        print(f"{d}-instrument scenarios ({len(sub)})")
        show = sub.drop(columns=["degree"]).copy()
        show["synergy_ratio"] = (show["synergy_ratio"] * 100).round(1)
        print(show.to_string(index=False, float_format=lambda x: f"{x:,.1f}",
                             na_rep="-"))
        print()

    csv = out.copy()
    csv["synergy_ratio"] = (csv["synergy_ratio"] * 100).round(2)
    csv.to_csv(HERE / "ccus_combination_summary.csv", index=False, encoding="utf-8-sig")
    print("written", HERE / "ccus_combination_summary.csv")


if __name__ == "__main__":
    main()
