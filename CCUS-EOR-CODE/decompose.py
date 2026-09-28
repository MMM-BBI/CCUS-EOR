"""Run the extra benchmark needed for a consistent synergy decomposition.

The single "Technology incentives" scenario sweeps the 45Q utilisation credit
from zero up to the current full 45Q credit (85 USD/t = 603.5 CNY/t).  Inside a
combination the technology tool instead moves from the baseline 45Q level
(35 USD/t = 248.5 CNY/t) up to that same current full credit (85/35 of it).  To
decompose a package's gain into "sum of individual effects", the technology tool
therefore needs its own individual benchmark at the SAME ceiling used inside the
combinations: baseline 45Q -> current full 45Q.

This script loads only the class + parameter definitions from
ccus_part2senario.py (everything before the "run" section), so importing it does
not trigger the whole pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).parent
SRC = HERE / "ccus_part2senario.py"
OUT = HERE / "benchmarks.json"

text = SRC.read_text(encoding="utf-8")
prefix = text.split("# 运行基准模型")[0]
ns: dict = {}
exec(compile(prefix, str(SRC), "exec"), ns)          # noqa: S102

Model = ns["CCUSInvestmentModel"]
base_params = ns["params"]
TECH_45Q_UPLIFT = ns["TECH_45Q_UPLIFT"]


def run(params, seed=42):
    np.random.seed(seed)
    model = Model(params)
    res = model.lsmc_solver()
    base_year = params["base_year"]
    valid = [y for y in res.get("optimal_year_by_path", []) if y < base_year + params["t_v"]]
    return {
        "v0": float(np.mean(res["TIV"][:, 0])),
        "median_year": float(np.median(valid)) if valid else None,
        "invest_ratio": float(res.get("investment_ratio", 0.0)),
    }


def main():
    out = {}

    base = dict(base_params)
    out["baseline"] = run(base)
    print("baseline            V0 =", round(out["baseline"]["v0"] / 1e6, 2),
          " median year =", out["baseline"]["median_year"])

    tech = dict(base_params)
    tech["S_EOR"] = base_params["S_EOR"] * (1.0 + TECH_45Q_UPLIFT)
    # 只加码 45Q 抵免；清洁电价补贴保持基准水平 0.019 元/千瓦时，
    # 与 ccus_part2senario.py 组合情景中“技术激励”工具的口径完全一致。
    out["technology_ira"] = run(tech)
    print("technology @ full 45Q  V0 =", round(out["technology_ira"]["v0"] / 1e6, 2),
          " median year =", out["technology_ira"]["median_year"])

    out["ira_params"] = {
        "S_EOR": {"base": base_params["S_EOR"],
                  "full_45q": base_params["S_EOR"] * (1.0 + TECH_45Q_UPLIFT)},
        "S_clean": {"base": base_params["S_clean"],
                    "full_45q": base_params["S_clean"]},
    }
    out["ira_uplift"] = TECH_45Q_UPLIFT

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
