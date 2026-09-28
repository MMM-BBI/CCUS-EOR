"""Experiment: how the 2025 45Q reform changes the model's headline results.

Loads the model class and the parameter dictionary from ccus_part1.py by
executing only the part of the script that defines them (everything above the
first top-level analysis block), so no figures or long simulation blocks run.

Then evaluates several 45Q parameterisations:
  * the published baseline (S_EOR = 232.4 CNY/t, i.e. 35 USD/t at 6.64)
  * 60 USD/t (IRA, utilisation pathway) at 7.1 CNY/USD  -> 426.2 CNY/t
  * 85 USD/t (post-OBBBA standardised credit) at 7.1     -> 603.5 CNY/t
  * each of the above with and without the statutory 12-year claim window
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:                      # Jupyter 的 sys.stdout 是 ipykernel.OutStream，
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:         # 没有 reconfigure 方法时忽略即可
    pass

HERE = Path(__file__).parent
SRC = (HERE / "ccus_part1.py").read_text(encoding="utf-8")
CUT = SRC.index("# 首先分析CE_replace")

NAMESPACE: dict = {"__name__": "ccus_model_module"}
exec(compile(SRC[:CUT], "ccus_part1_head", "exec"), NAMESPACE)  # noqa: S102

Base = NAMESPACE["CCUSInvestmentModel"]
PARAMS = dict(NAMESPACE["params"])
FX = 7.1


class Windowed(Base):
    """Same model, but the 45Q-style credit is claimable for `claim_years`
    operating years only (26 U.S.C. 45Q(a)(3)(A): the 12-year period beginning
    on the date the equipment was originally placed in service)."""

    def _credit(self, operation_year: int) -> float:
        limit = getattr(self, "claim_years", 10 ** 6)
        return self.S_EOR * self.Q_EOR if operation_year <= limit else 0.0

    def calculate_cash_flow(self, investment_year, operation_year, Pc_t, Poil_t, Pe_t):
        total_years = investment_year + self.t_c + operation_year
        M_t_CCUS = self.M0 * np.exp(-self.beta * total_years)
        TC_CO2 = self.Q_CO2 * self.UTC_CO2
        UC_CO2 = self.zeta * self.Q_CO2 * self.UUC_CO2
        SC_CO2 = self.Q_CO2 * self.USC_CO2
        C_loss = Pe_t * self.Q_loss
        R_c = Pc_t * self.Q_CO2
        R_e = self.S_clean * self.Q_e
        ce = self.calculate_CE_replace(operation_year)
        Q_oil = ce * self.Q_EOR
        R_oil = Poil_t * Q_oil
        R_EOR = self._credit(operation_year)
        R_SC = self.pi * self.zeta * self.Q_CO2
        total_revenue = R_c + R_e + R_oil + R_EOR + R_SC
        total_cost = TC_CO2 + UC_CO2 + SC_CO2 + C_loss + M_t_CCUS
        return (total_revenue - total_cost) * (1 - self.tax), ce

    def expected_project_value(self, t, Pc_t, Poil_t, Pe_t):
        investment_cost = (1 - self.lambda_) * self.I0 * np.exp(-self.alpha * t)
        operating_period = self.calculate_operating_period(t)
        if operating_period <= 0:
            return -investment_cost
        Epc, Eoil, Epe = self.expected_price_paths(t, Pc_t, Poil_t, Pe_t)
        value = 0.0
        for op_year in range(1, operating_period + 1):
            cal = t + self.t_c + op_year - 1
            if cal >= len(Epc):
                break
            ce = self.calculate_CE_replace(op_year)
            revenue = (Epc[cal] * self.Q_CO2
                       + self.S_clean * self.Q_e
                       + Eoil[cal] * ce * self.Q_EOR
                       + self._credit(op_year)
                       + self.pi * self.zeta * self.Q_CO2)
            cost = (self.Q_CO2 * self.UTC_CO2
                    + self.zeta * self.Q_CO2 * self.UUC_CO2
                    + self.Q_CO2 * self.USC_CO2
                    + Epe[cal] * self.Q_loss
                    + self.M0 * np.exp(-self.beta * (t + self.t_c + op_year)))
            value += (revenue - cost) * (1 - self.tax) * np.exp(-self.r * (self.t_c + op_year))
        return value - investment_cost


def run(label, M=2000, **over):
    p = dict(PARAMS)
    p["M"] = M
    p.update({k: v for k, v in over.items() if k != "claim_years"})
    model = Windowed(p)
    model.claim_years = over.get("claim_years", 10 ** 6)
    res = model.lsmc_solver()
    v0 = float(np.mean(res["TIV"][:, 0])) / 1e6
    npv0 = float(np.mean(res["NPV"][:, 0])) / 1e6
    print(f"  -> {label}: V0={v0:,.0f}  invest-now={npv0:,.0f}  "
          f"premium={v0 - npv0:,.0f} ({(v0 - npv0) / v0 * 100:.1f}%)  "
          f"median year={res['optimal_timing_median_year']}  "
          f"modal={res['optimal_timing_modal_year']} "
          f"({res['optimal_timing_modal_share']:.1f}%)  "
          f"invest share={res['investment_ratio'] * 100:.1f}%")
    return dict(label=label, v0=v0, npv0=npv0, median=res["optimal_timing_median_year"],
                modal=res["optimal_timing_modal_year"],
                modal_share=res["optimal_timing_modal_share"],
                dist=dict(res["timing_distribution"]))


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = []
    print("=" * 100)
    print("A. 45Q utilisation-credit level, no claim-window limit")
    for label, s in [("35 USD/t (=232.4, published baseline)", 35 * 6.64),
                     ("60 USD/t (IRA utilisation)", 60 * FX),
                     ("85 USD/t (post-OBBBA standardised)", 85 * FX)]:
        out.append(run(label, M=M, S_EOR=s))

    print("=" * 100)
    print("B. Same levels with the statutory 12-year claim window")
    for label, s in [("35 USD/t", 35 * 6.64),
                     ("60 USD/t (IRA utilisation)", 60 * FX),
                     ("85 USD/t (post-OBBBA standardised)", 85 * FX)]:
        out.append(run(label + " [12y]", M=M, S_EOR=s, claim_years=12))

    print("=" * 100)
    print("C. Statutory 12-year window at the base applicable amount and at the FX-corrected 35 USD/t")
    for label, s in [("17 USD/t x 7.1 (=120.7, base amount)", 17 * FX),
                     ("35 USD/t x 7.1 (=248.5, FX corrected)", 35 * FX)]:
        out.append(run(label + " [12y]", M=M, S_EOR=s, claim_years=12))

    print("=" * 100)
    for r in out:
        print(f"{r['label']:<44} V0={r['v0']:>9,.0f}  median={r['median']}  "
              f"modal={r['modal']}({r['modal_share']:.1f}%)  {r['dist']}")
