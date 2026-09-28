"""Quantify how the discontinuous displacement-efficiency curve affects results."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).parent
SRC = HERE / "ccus_part1.py"      # 与本文件同目录


def model_prefix(text: str) -> str:
    """Return the part of ccus_part1.py that defines the model class and the
    parameter dictionary, i.e. everything up to (and including) `params = {...}`.
    Brace matching is used so the split does not depend on any Chinese comment."""
    start = text.index("params = {")
    depth = 0
    for j in range(start + len("params = "), len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[: j + 1]
    raise ValueError("could not find the end of the params dictionary")


BASE = model_prefix(SRC.read_text(encoding="utf-8"))

SEG1 = "np.exp(0.791 * t - 4.662) / 6.93"
SEG2 = "-0.0145 * t + 0.560"
assert SEG1 in BASE and SEG2 in BASE

VARIANTS = {
    "现状（三段式，t=8 处不连续）": BASE,
    "方案B：调整第一段系数至 0.7233（值连续，峰值 0.4440）":
        BASE.replace(SEG1, "np.exp(0.7233 * t - 4.662) / 6.93"),
    "方案C：第二段改为连接线（0.7636→0.4006，峰值 0.7636）":
        BASE.replace(SEG2, "0.7636 - 0.1815 * (t - 8)"),
}


def show_curve(src):
    ns = {}
    exec(compile(src, "p1", "exec"), ns)
    np.random.seed(42)
    m = ns["CCUSInvestmentModel"](dict(ns["params"]))
    vals = [m.calculate_CE_replace(t) for t in range(0, 16)]
    return "  ".join(f"t{t}:{v:.4f}" for t, v in enumerate(vals))


def run(src):
    ns = {}
    exec(compile(src, "p1", "exec"), ns)
    params = dict(ns["params"])
    np.random.seed(42)
    m = ns["CCUSInvestmentModel"](params)
    r = m.lsmc_solver()
    base_year = params["base_year"]
    valid = [y for y in r["optimal_year_by_path"] if y < base_year + params["t_v"]]
    # year-8 oil revenue under the mean price path
    return {
        "v0": float(np.mean(r["TIV"][:, 0])) / 1e6,
        "npv0": float(np.mean(r["NPV"][:, 0])) / 1e6,
        "median": float(np.median(valid)) if valid else float("nan"),
    }


for name, src in VARIANTS.items():
    print("=" * 100)
    print(name)
    print("  效率曲线:", show_curve(src))
    r = run(src)
    print(f"  V0 = {r['v0']:,.1f}   立即投资 = {r['npv0']:,.1f}   中位投资年份 = {r['median']:.0f}")
