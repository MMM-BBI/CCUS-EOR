"""Standalone diagnostics and control experiments for the CCUS-EOR option model.

This script is deliberately SEPARATE from the three production scripts: it never
modifies ccus_part1/2/3.py and writes only its own outputs
(diagnostics_results.json, diag_convergence.png, 诊断与对照实验结果.md).

It answers two reviewer requests:

  (7) LSMC error diagnostics
      A. multiple random seeds  -> mean, standard deviation and 95% interval of V0
      B. path-count convergence -> V0 and the median investment year vs M
      C. analytic validation    -> the same LSMC routine applied to a plain
                                   American put, compared with a binomial tree

  (8) Method controls (same project, same seed)
      1. fixed displacement efficiency instead of the three-stage path
      2. fixed electricity price instead of the two-stage stochastic process
      3. no CAPM (risk-free discounting instead of the risk-adjusted rate)
      4. invest immediately at the first decision point
      5. deterministic optimal timing on the expected price path
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:                      # Jupyter 的 sys.stdout 是 ipykernel.OutStream，
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:         # 没有 reconfigure 方法时忽略即可
    pass

HERE = Path(__file__).parent
SRC = (HERE / "ccus_part1.py").read_text(encoding="utf-8")
CUT = SRC.index("# 首先分析CE_replace")

NS: dict = {"__name__": "ccus_model_module"}
exec(compile(SRC[:CUT], "ccus_part1_head", "exec"), NS)  # noqa: S102
BaseModel = NS["CCUSInvestmentModel"]
BASE_PARAMS = dict(NS["params"])
BASE_YEAR = int(BASE_PARAMS["base_year"])
N_DEC = int(BASE_PARAMS["t_v"])

RESULTS: dict = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def summarise(res: dict, params: dict) -> dict:
    v0 = float(np.mean(res["TIV"][:, 0]))
    npv0 = float(np.mean(res["NPV"][:, 0]))
    valid = [y for y in res.get("optimal_year_by_path", [])
             if y < BASE_YEAR + params["t_v"]]
    return {
        "v0": v0,
        "npv0": npv0,
        "premium": v0 - npv0,
        "premium_pct": (v0 - npv0) / v0 * 100 if v0 else float("nan"),
        "median_year": float(np.median(valid)) if valid else None,
        "modal_year": res.get("optimal_timing_modal_year"),
        "modal_share": float(res.get("optimal_timing_modal_share", 0.0)),
        "invest_ratio": float(res.get("investment_ratio", 0.0)),
        "distribution": {int(k): int(v) for k, v in
                         (res.get("timing_distribution") or {}).items()},
        "investing_paths": len(valid),
    }


def solve(params: dict, label: str = "", quiet: bool = True) -> dict:
    model = BaseModel(params)
    if quiet:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            res = model.lsmc_solver()
    else:
        res = model.lsmc_solver()
    out = summarise(res, params)
    if label:
        print(f"  {label:<46} V0={out['v0']/1e6:>9,.1f}  "
              f"invest-now={out['npv0']/1e6:>9,.1f}  "
              f"median={out['median_year']}  "
              f"modal={out['modal_year']}({out['modal_share']:.1f}%)")
    return out


def with_params(**over) -> dict:
    p = dict(BASE_PARAMS)
    p.update(over)
    return p


# ---------------------------------------------------------------------------
# A. multiple random seeds
# ---------------------------------------------------------------------------
def run_seed_experiment(M: int, seeds: list[int]) -> dict:
    print("=" * 100)
    print(f"A. Multiple random seeds (M = {M} paths each)")
    print("   Note: each run uses its own model instance, so the module-level "
          "SEED of the production scripts is never modified.")
    rows = []
    for s in seeds:
        out = solve_with_seed(with_params(M=M), s)
        rows.append({"seed": s, **{k: out[k] for k in
                                   ("v0", "npv0", "premium", "median_year",
                                    "modal_year", "modal_share")}})
        print(f"  seed {s:>5}: V0={out['v0']/1e6:>9,.1f}  "
              f"median={out['median_year']}  modal={out['modal_year']}")
    v0s = np.array([r["v0"] for r in rows])
    mean, sd = float(v0s.mean()), float(v0s.std(ddof=1))
    half = 1.96 * sd / np.sqrt(len(v0s)) if len(v0s) > 1 else float("nan")
    summary = {
        "M": M,
        "seeds": rows,
        "mean_v0": mean,
        "sd_v0": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "range_v0": [float(v0s.min()), float(v0s.max())],
    }
    print(f"  mean V0 = {mean/1e6:,.1f} million CNY, sd = {sd/1e6:,.1f}, "
          f"95% CI = [{summary['ci95_low']/1e6:,.1f}, {summary['ci95_high']/1e6:,.1f}]")
    return summary


class _SeedModel(BaseModel):
    """Model whose simulation seed can be set per instance."""

    seed = 42

    def simulate_uncertainties(self):
        """Identical to the production routine but with an instance-level seed."""
        dt = 1
        total = self.T
        Pc = np.zeros((self.M, total))
        Poil = np.zeros((self.M, total))
        Pc[:, 0] = self.Pc0
        Poil[:, 0] = self.Poil0
        np.random.seed(self.seed)
        Z_c = np.random.standard_normal((self.M, total - 1))
        Z_oil = np.random.standard_normal((self.M, total - 1))
        for t in range(1, total):
            Pc[:, t] = Pc[:, t - 1] * np.exp(
                (self.mu_c - 0.5 * self.sigma_c ** 2) * dt
                + self.sigma_c * np.sqrt(dt) * Z_c[:, t - 1])
            Poil[:, t] = Poil[:, t - 1] * np.exp(
                (self.mu_oil - 0.5 * self.sigma_oil ** 2) * dt
                + self.sigma_oil * np.sqrt(dt) * Z_oil[:, t - 1])
        # 与生产脚本一致：把碳价随机增量交给电价模块，保证碳价-电价相关性生效
        self.Z_c_paths = Z_c
        Pe = self.simulate_electricity_price(Pc)
        return Pc, Poil, Pe

    def simulate_electricity_price(self, Pc_paths):
        M, total_years = Pc_paths.shape
        Pe = np.zeros((M, total_years))
        Pe[:, 0] = self.Pe0
        rng = np.random.default_rng(self.seed + 1)
        Z_e = rng.standard_normal((M, total_years - 1))
        Z_c_shock = getattr(self, "Z_c_paths", None)
        if Z_c_shock is None:
            Z_c_shock = rng.standard_normal((M, total_years - 1))
        Z = self.rho_e_c * Z_c_shock + np.sqrt(1 - self.rho_e_c ** 2) * Z_e
        for t in range(1, total_years):
            if t < self.stage1_years:
                g = self.Pe_growth_rate + (t - 1) * self.Pe_growth_increment
                Pe[:, t] = Pe[:, t - 1] * (1 + g)
            else:
                prev = Pe[:, t - 1]
                d = self.alpha_e * (self.mu_e - np.log(prev)) + self.sigma_e * Z[:, t - 1]
                Pe[:, t] = prev * np.exp(d)
        self.Pe_paths = Pe
        return Pe


def solve_with_seed(params: dict, seed: int) -> dict:
    _SeedModel.seed = seed
    model = _SeedModel(params)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        res = model.lsmc_solver()
    return summarise(res, params)


# ---------------------------------------------------------------------------
# 0. reproduction check: the diagnostics baseline must equal ccus_part1.py
# ---------------------------------------------------------------------------
def check_baseline_reproduction(M: int) -> dict:
    """The baseline solved here (module SEED, no overrides) has to reproduce the
    headline V0 written by ccus_part1.py, otherwise the diagnostic blocks would
    describe a different model."""
    print("=" * 100)
    print(f"0. Reproduction check against ccus_part1_results.csv (M = {M})")
    out = solve(with_params(M=M), label="diagnostics baseline")
    ref = None
    csv = HERE / "ccus_part1_results.csv"
    if csv.exists():
        import csv as _csv
        with csv.open(encoding="utf-8-sig") as fh:
            for row in _csv.DictReader(fh):
                if row.get("item") == "option_value_at_decision_V0":
                    ref = float(row["value"])          # million CNY
                    break
    if ref is None:
        print("   (ccus_part1_results.csv 未找到，跳过比对)")
        return {"diagnostics_v0": out["v0"], "reference_v0": None, "match": None}
    diff = out["v0"] / 1e6 - ref
    print(f"   ccus_part1_results.csv V0 = {ref:,.2f} million CNY; "
          f"difference = {diff:+.4f} million CNY")
    return {"diagnostics_v0": out["v0"], "reference_v0": ref,
            "difference_million_cny": diff, "match": abs(diff) < 1e-6}


# ---------------------------------------------------------------------------
# B. convergence in the number of paths
# ---------------------------------------------------------------------------
def run_convergence(path_counts: list[int]) -> dict:
    print("=" * 100)
    print("B. Convergence in the number of simulation paths (seed 42)")
    rows = []
    for M in path_counts:
        t0 = time.perf_counter()
        out = solve(with_params(M=M), label=f"M = {M:>6}")
        row = {"M": M, "v0": out["v0"], "median_year": out["median_year"],
               "modal_year": out["modal_year"], "modal_share": out["modal_share"],
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        print(f"           ({row['seconds']:.1f} s, std. err. of the mean "
              f"~ {out['v0']/1e6/np.sqrt(M):.1f} million CNY)")
    return {"rows": rows}


# ---------------------------------------------------------------------------
# C. analytic validation of the LSMC routine on a plain American put
# ---------------------------------------------------------------------------
def _crr_american_put(S0, K, r, sigma, T, steps, kind="put"):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S0 * u ** (2 * j - steps)
    if kind == "put":
        V = np.maximum(K - ST, 0.0)
    else:
        V = np.maximum(ST - K, 0.0)
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        Si = S0 * u ** (2 * j - i)
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if kind == "put":
            V = np.maximum(V, K - Si)
        else:
            V = np.maximum(V, Si - K)
    return float(V[0])


def _basis(x, kind):
    """Regression basis for the LSMC continuation value."""
    if kind == "poly2":
        return np.vander(x, 3, increasing=True)
    if kind == "poly3":
        return np.vander(x, 4, increasing=True)
    if kind == "laguerre":
        # Laguerre polynomials L0..L3 (the basis used in Longstaff-Schwartz)
        L0 = np.ones_like(x)
        L1 = 1.0 - x
        L2 = 1.0 - 2.0 * x + 0.5 * x ** 2
        L3 = 1.0 - 3.0 * x + 1.5 * x ** 2 - x ** 3 / 6.0
        return np.column_stack([L0, L1, L2, L3])
    raise ValueError(kind)


def _lsmc_american_put(S0, K, r, sigma, T, n_steps, n_paths, seed, kind="poly3"):
    """Least-squares Monte Carlo for an American put, exactly the algorithm used
    in the CCUS model (regress the discounted continuation value on a polynomial
    of the state variables, using in-the-money paths only)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    logS = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt
                                  + sigma * np.sqrt(dt) * Z, axis=1)
    S = np.exp(np.column_stack([np.full(n_paths, np.log(S0)), logS]))
    payoff = np.maximum(K - S, 0.0)
    cashflow = payoff[:, -1].copy()
    exercise = np.full(n_paths, n_steps)
    disc1 = np.exp(-r * dt)
    for t in range(n_steps - 1, 0, -1):
        itm = payoff[:, t] > 0
        if itm.sum() < 8:
            continue
        y = cashflow[itm] * np.exp(-r * dt * (exercise[itm] - t))
        x = S[itm, t] / K
        A = _basis(x, kind)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        cont = A @ coef
        ex_now = payoff[itm, t] > cont
        idx = np.where(itm)[0][ex_now]
        cashflow[idx] = payoff[idx, t]
        exercise[idx] = t
    price = float(np.mean(cashflow * np.exp(-r * dt * exercise)))
    return price


def run_analytic_validation() -> dict:
    print("=" * 100)
    print("C. LSMC validation on an American put with a known binomial price")
    cases = [
        dict(S0=100, K=100, r=0.05, sigma=0.20, T=3.0, steps=3),
        dict(S0=100, K=110, r=0.05, sigma=0.25, T=5.0, steps=5),
        dict(S0=100, K=100, r=0.05, sigma=0.20, T=5.0, steps=10),
    ]
    rows = []
    for c in cases:
        bench = _crr_american_put(c["S0"], c["K"], c["r"], c["sigma"], c["T"], 4000)
        row = {**c, "binomial": bench, "baselines": {}}
        for kind in ("poly2", "poly3", "laguerre"):
            estimates = [_lsmc_american_put(c["S0"], c["K"], c["r"], c["sigma"],
                                            c["T"], c["steps"], 200_000,
                                            1000 + 37 * i, kind)
                         for i in range(5)]
            est_mean = float(np.mean(estimates))
            est_sd = float(np.std(estimates, ddof=1))
            rel = (est_mean - bench) / bench * 100
            row["baselines"][kind] = {
                "mean": est_mean, "sd": est_sd, "replicates": estimates,
                "relative_error_pct": rel,
            }
            print(f"  T={c['T']:.0f}y, K/S={c['K']/c['S0']:.2f}, "
                  f"{c['steps']:>2} exercise dates, basis={kind:<9}: "
                  f"binomial={bench:.4f}, LSMC={est_mean:.4f} "
                  f"(sd {est_sd:.4f}), relative error={rel:+.2f}%")
        rows.append(row)
    return {"cases": rows}


# ---------------------------------------------------------------------------
# D. method controls
# ---------------------------------------------------------------------------
def run_controls(M: int) -> dict:
    print("=" * 100)
    print(f"D. Method controls (M = {M}, seed 42)")

    base = solve(with_params(M=M), label="baseline (published specification)")

    prof = [BaseModel(BASE_PARAMS).calculate_CE_replace(t)
            for t in range(1, BASE_PARAMS["T"] - BASE_PARAMS["t_c"])]
    ce_bar = float(np.mean(prof))

    class FixedCE(BaseModel):
        level = ce_bar

        def calculate_CE_replace(self, t):
            return self.level

    class FixedPe(BaseModel):
        def simulate_electricity_price(self, Pc_paths):
            M_, T_ = Pc_paths.shape
            Pe = np.full((M_, T_), self.Pe0)
            self.Pe_paths = Pe
            return Pe

    out = {"baseline": base, "mean_displacement_efficiency": ce_bar}

    fixed_ce = FixedCE(with_params(M=M))
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        r1 = summarise(fixed_ce.lsmc_solver(), with_params(M=M))
    out["fixed_displacement_efficiency"] = r1
    print(f"  fixed displacement efficiency (CE = {ce_bar:.4f})  V0={r1['v0']/1e6:>9,.1f}"
          f"  median={r1['median_year']}  dV0={(r1['v0']-base['v0'])/1e6:+,.1f}")

    fixed_pe = FixedPe(with_params(M=M))
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = summarise(fixed_pe.lsmc_solver(), with_params(M=M))
    out["fixed_electricity_price"] = r2
    print(f"  fixed electricity price (Pe = {BASE_PARAMS['Pe0']:.2f})       "
          f"V0={r2['v0']/1e6:>9,.1f}  median={r2['median_year']}  "
          f"dV0={(r2['v0']-base['v0'])/1e6:+,.1f}")

    no_capm = solve(with_params(M=M, beta_risk=0.0), label="no CAPM (r = risk-free 2.64%)")
    out["no_capm"] = no_capm

    # 4. invest immediately at the first decision point
    p0 = with_params(M=M)
    m0 = BaseModel(p0)
    Pc, Poil, Pe = m0.simulate_uncertainties()
    now = np.array([m0.calculate_project_value(0, Pc[j], Poil[j], Pe[j])
                    for j in range(p0["M"])])
    out["invest_immediately"] = {
        "mean_npv": float(now.mean()), "sd_npv": float(now.std(ddof=1)),
        "share_positive": float((now > 0).mean()),
    }
    print(f"  invest immediately (t = 0)                         "
          f"mean NPV={out['invest_immediately']['mean_npv']/1e6:>9,.1f}  "
          f"share NPV>0 = {out['invest_immediately']['share_positive']*100:.1f}%")

    # 5. deterministic optimal timing on the expected (mean-path) prices
    ce_curve = np.array([m0.calculate_CE_replace(t)
                         for t in range(1, BASE_PARAMS["T"] + 1)])
    det = {}
    for t in range(N_DEC):
        year = BASE_YEAR + t
        op = m0.calculate_operating_period(t)
        if op <= 0:
            continue
        Epc, Eoil, Epe = m0.expected_price_paths(
            t,
            float(np.mean(Pc[:, min(t, Pc.shape[1] - 1)])),
            float(np.mean(Poil[:, min(t, Poil.shape[1] - 1)])),
            float(np.mean(Pe[:, min(t, Pe.shape[1] - 1)])),
        )
        val = 0.0
        for op_year in range(1, op + 1):
            cal = t + m0.t_c + op_year - 1
            if cal >= len(Epc):
                break
            revenue = (Epc[cal] * m0.Q_CO2 + m0.S_clean * m0.Q_e
                       + Eoil[cal] * ce_curve[op_year - 1] * m0.Q_EOR
                       + (m0.S_EOR * m0.Q_EOR
                          if op_year <= m0.claim_years else 0.0)
                       + m0.pi * m0.zeta * m0.Q_CO2)
            cost = (m0.Q_CO2 * m0.UTC_CO2 + m0.zeta * m0.Q_CO2 * m0.UUC_CO2
                    + m0.Q_CO2 * m0.USC_CO2 + Epe[cal] * m0.Q_loss
                    + m0.M0 * np.exp(-m0.beta * (t + m0.t_c + op_year)))
            val += (revenue - cost) * (1 - m0.tax) * np.exp(-m0.r * (m0.t_c + op_year))
        val -= m0.I0 * np.exp(-m0.alpha * t)
        det[year] = val
    best_year = max(det, key=det.get)
    out["deterministic_timing"] = {
        "npv_by_year": {k: float(v) for k, v in det.items()},
        "best_year": int(best_year),
        "best_npv": float(det[best_year]),
    }
    print("  deterministic NPV by investment year: " +
          ", ".join(f"{y}:{v/1e6:,.0f}" for y, v in sorted(det.items())))
    print(f"  deterministic optimum: invest in {best_year}")
    return out


# ---------------------------------------------------------------------------
# E. common-random-numbers resolution test
# ---------------------------------------------------------------------------
def run_resolution(M: int, seeds: list[int]) -> dict:
    """The scenarios are evaluated on the SAME simulated paths (common random
    numbers), so what matters for the credibility of the synergy decomposition is
    the Monte Carlo noise of the *difference*, not of the level."""
    print("=" * 100)
    print(f"E. Common-random-numbers resolution test (M = {M}, seeds = {seeds})")
    variants = {
        "carbon price +25%": {"Pc0": BASE_PARAMS["Pc0"] * 1.25},
        "45Q credit at the current full level (85 USD/t)": {"S_EOR": 85.0 * 7.1},
    }
    per_variant: dict[str, dict] = {}
    base_by_seed = {}
    for s in seeds:
        base_by_seed[s] = solve_with_seed(with_params(M=M), s)["v0"]
    for name, over in variants.items():
        diffs = []
        levels = []
        for s in seeds:
            v = solve_with_seed(with_params(M=M, **over), s)["v0"]
            levels.append(v)
            diffs.append(v - base_by_seed[s])
        d = np.array(diffs)
        lv = np.array(levels)
        bv = np.array([base_by_seed[s] for s in seeds])
        per_variant[name] = {
            "delta_per_seed": diffs,
            "delta_mean": float(d.mean()),
            "delta_sd": float(d.std(ddof=1)),
            "delta_range": [float(d.min()), float(d.max())],
            "level_sd": float(lv.std(ddof=1)),
            "baseline_level_sd": float(bv.std(ddof=1)),
            "noise_reduction_factor": float(bv.std(ddof=1) / d.std(ddof=1))
            if d.std(ddof=1) > 0 else None,
        }
        print(f"  {name:<48} dV0 = {d.mean()/1e6:>+8,.1f} "
              f"(sd {d.std(ddof=1)/1e6:>6,.1f}); level sd = "
              f"{bv.std(ddof=1)/1e6:,.1f} million CNY -> noise cut by "
              f"{bv.std(ddof=1)/max(d.std(ddof=1), 1e-9):.1f}x")
    return {"M": M, "seeds": seeds, "variants": per_variant}


def make_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = RESULTS.get("B_convergence", {}).get("rows", [])
    if not rows:
        return
    Ms = [r["M"] for r in rows]
    v0 = [r["v0"] / 1e6 for r in rows]
    med = [r["median_year"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(Ms, v0, marker="o", color="#4E79A7", lw=1.4, ms=4)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Number of simulated paths", fontsize=9)
    ax[0].set_ylabel("Option value at the decision point\n(million CNY)", fontsize=9)
    ax[1].plot(Ms, med, marker="s", color="#CC5A6A", lw=1.4, ms=4)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Number of simulated paths", fontsize=9)
    ax[1].set_ylabel("Median investment year", fontsize=9)
    for a in ax:
        a.tick_params(labelsize=8)
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(HERE / "diag_convergence.png", dpi=600)
    print("[figure] saved diag_convergence.png")


def main() -> None:
    # Robust argument handling: the script must also work when it is executed
    # from the notebook through runpy, where sys.argv belongs to the Jupyter
    # kernel (e.g. ["...ipykernel_launcher.py", "-f", "kernel.json"]).
    M_seed = 10000
    for _a in sys.argv[1:]:
        try:
            M_seed = int(_a)
            break
        except ValueError:
            continue
    quick = ("--quick" in sys.argv) or bool(os.environ.get("CCUS_QUICK_TEST"))
    print(f"diagnostics: M = {M_seed}, quick = {quick}")

    # 让整个诊断脚本先对齐主结果：本次基准必须与 ccus_part1_results.csv 完全一致
    RESULTS["Z_reproduction"] = check_baseline_reproduction(M_seed)
    RESULTS["A_seed"] = run_seed_experiment(
        M_seed, [1, 7, 42, 123, 2024] if not quick else [42, 123])
    RESULTS["B_convergence"] = run_convergence(
        [1000, 2500, 5000, 10000] if not quick else [500, 1000])
    RESULTS["C_analytic"] = run_analytic_validation()
    RESULTS["D_controls"] = run_controls(M_seed)
    RESULTS["E_resolution"] = run_resolution(
        5000 if not quick else 1000, [7, 42, 2024] if not quick else [42])
    make_figures()
    (HERE / "diagnostics_results.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwritten diagnostics_results.json")


if __name__ == "__main__":
    main()
