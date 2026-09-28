# CCUS-EOR
# Real-Option Investment Decision Model for CCUS-EOR Projects under Electricity-Price Uncertainty

Code and reproducible results for the paper *"Investment decision analysis of CCUS-EOR
projects considering the dynamic evolution of oil displacement efficiency and electricity
price uncertainty"*.

The model values a new coal-fired power plant with integrated carbon capture, utilisation
and storage for enhanced oil recovery (CCUS-EOR) as an **American call option**, and asks
*when* — not merely *whether* — the project should be built. It combines real-option theory,
stochastic price processes and the capital asset pricing model (CAPM), solves the optimal
stopping problem by least-squares Monte Carlo (LSMC), and compares four single policy
instruments together with two-, three- and four-instrument policy packages.

---

## 1. Case study and model in brief

| Item | Setting |
|---|---|
| Case | New 600 MW coal-fired unit in China with post-combustion capture + CO2-EOR |
| Plant life / decision window | 30 years / 2026–2035 (one irreversible investment, 1-year construction) |
| Annual CO2 captured and injected | 1.604 Mt |
| Uncertainty | Carbon price and oil price (geometric Brownian motion); electricity price (5-year deterministic growth, then mean reversion to 0.60 CNY/kWh, correlated with the carbon price, ρ = 0.4) |
| Oil displacement efficiency | Deterministic three-stage trajectory (rise–peak–decline) defined on annual operating steps; peak 0.444 t oil/t CO2 (≈3.24 bbl/t CO2) |
| Systematic risk | CAPM risk-adjusted discount rate 7.44% (r_f = 2.64%, β = 0.8, equity risk premium 6%) |
| Solution method | Least-squares Monte Carlo, backward induction, 10,000 paths, common random numbers |
| Policy benchmark | 45Q utilisation credit: baseline 248.5 CNY/t (35 USD/t); policy ceiling 603.5 CNY/t (85 USD/t); credit claimable for the first 12 operating years only |

## 2. Headline results

* **Baseline**: option value at the decision point **V₀ = 8,686 million CNY**, immediate-investment
  NPV = 7,864 million CNY, **deferral-option premium 822 million CNY (9.5 %)**, median optimal
  investment year **2033** (mode 2035, 26.8 % of paths).
* **Single instruments** (0 → 100 % intensity): technology incentives 7,353 → **10,895** million CNY
  (+25.4 %, median year 2035 → 2026), one-off investment subsidy 8,686 → **10,264** (+18.2 %,
  2033 → 2026), income-tax relief 8,686 → **9,510** (+9.5 %, median year unchanged),
  carbon trading 8,686 → **8,928** (+2.8 %, median year unchanged).
* **Policy packages**: synergy of 15–822 million CNY (1.3–21.7 %) for two-instrument packages,
  177–1,106 million CNY (6.7–24.0 %) for three-instrument packages, and
  1,252 million CNY (25.8 %) for the four-instrument package (V₀ → **14,791** million CNY).
  The *upper bound* of the synergy ratio rises with the number of instruments.
* **Net emissions (project boundary)**: life-cycle net storage ≈ **1.9 ×** the combustion
  emissions of the incremental oil, i.e. a net reduction of ≈ **17 Mt CO2**; the peak
  operating year is emission-positive, so the life-cycle convention is the relevant one.
* **Numerical validation**: LSMC reproduces an analytic American-put benchmark with a *downward*
  bias of 2.7–6.0 % (which grows smaller as the number of exercise dates increases); across five
  random seeds V₀ = 8,686–8,821 million CNY; the median investment year is stable at 2033–2035
  for 2,500–10,000 paths.

## 3. Repository structure

### 3.1 Core pipeline (produces the paper's numbers)

| File | Role |
|---|---|
| `ccus_part1.py` | **Baseline model.** Parameter set (Table 2), simulation of carbon/oil/electricity price paths, annual cash-flow model, LSMC solver, baseline figures (Figs. 2–4) and the result CSVs. |
| `ccus_part2senario.py` | **Policy scenarios.** Four single instruments plus all two-, three- and four-instrument packages; scenario figures (Figs. 5–8) and the scenario CSVs (Tables 4–5). |
| `ccus_part3.py` | **Parameter sensitivity.** One-at-a-time scans of ten key parameters (value, timing and investment-probability sensitivity indices) and the sensitivity figures. |
| `run_all_ccus.ipynb` | **One-click entry point.** Runs the three scripts above in order, then the combination summary, the diagnostics, the net-emission accounting and the 45Q source audit. Start here. |

### 3.2 Policy-package and diagnostic analyses

| File | Role |
|---|---|
| `decompose.py` | Runs the extra benchmark required for a consistent synergy decomposition (technology tool at its own policy ceiling). Writes `benchmarks.json`. |
| `analyze_combos.py` | Decomposes each package into "sum of individual effects" + synergy, and writes `ccus_combination_summary.csv`. |
| `diagnostics.py` | Standalone diagnostics: (A) multiple random seeds, (B) path-count convergence, (C) analytic American-option validation, (D) method controls (fixed efficiency, fixed electricity price, no CAPM, immediate investment, deterministic timing), (E) common-random-numbers resolution test. Writes `diagnostics_results.json` and `diag_convergence.png`. |
| `diag_extra.py` | Optional top-up: re-runs only the analytic-validation and resolution blocks and merges them into `diagnostics_results.json`. |
| `show_diag.py` | Prints `diagnostics_results.json` as a compact table. |

### 3.3 Parameter, policy and emission checks

| File | Role |
|---|---|
| `exp_45q.py` | Sensitivity of the results to the 45Q credit level (17 / 35 / 60 / 85 USD/t) and to the 12-year claim window; reproduces the baseline V₀ exactly. |
| `fetch_45q.py` | Downloads and archives the authoritative 45Q sources (IRS Form 8933 instructions, 26 U.S.C. §45Q) into `45q_sources/`. |
| `grep45q.py` | Searches the archived 45Q texts for the key clauses (12-year claim period, applicable dollar amount, ×5 increased credit). |
| `net_emissions.py` | Project-level life-cycle net-emission accounting from the cash-flow table (incremental oil, combustion emissions, net storage, net reduction). |

### 3.4 Quality-assurance utilities

| File | Role |
|---|---|
| `compare_params.py` | Verifies that the three pipeline scripts use identical parameters and identical model methods (AST-level comparison). |
| `check_notebook.py` | Static check of the notebook: cell syntax, referenced scripts present, artefacts created at run time. |
| `test_jupyter_safety.py` | Runs every script's module-level code against a Jupyter-style stdout object, so the pipeline cannot break inside a notebook. |
| `scan_sites.py` | Lists every code site that touches the 45Q credit (helps keep the three copies in sync). |
| `timing_diag.py` | Checks how saturated the investment-timing response is across scenarios. |
| `test_ce_variants.py` | Compares three specifications of the displacement-efficiency trajectory. |

## 4. How to run

```bash
pip install -r requirements.txt
jupyter notebook run_all_ccus.ipynb      # run the cells from top to bottom
```

Everything is regenerated by the notebook: result CSVs, `figures/*.png` (1000 dpi),
`benchmarks.json`, `diagnostics_results.json`, `diag_convergence.png` and `45q_sources/*.txt`.
Individual scripts can also be run directly, in this order:

```bash
python ccus_part1.py
python ccus_part2senario.py
python ccus_part3.py
python decompose.py
python analyze_combos.py
python diagnostics.py
python net_emissions.py
```

Notes

* Set `MPLBACKEND=Agg` when running head-less (otherwise `plt.show()` may block).
* `CCUS_QUICK_TEST=1` runs the whole pipeline with 300 paths (≈2 minutes) for a smoke test;
  these results are for checking only, not for the paper.
* `fetch_45q.py` needs internet access; the archived texts in `45q_sources/` are used otherwise.
* The three pipeline scripts each carry a copy of the model class so that they can be run
  independently; any parameter or formula change must be applied to all three — run
  `python compare_params.py` afterwards, which reports any divergence.

## 5. Outputs

| Output | Content |
|---|---|
| `ccus_part1_results.csv` | Baseline indicators (V₀, immediate-investment NPV, deferral premium, timing distribution, price-path statistics, efficiency trajectory) |
| `ccus_cashflow_breakdown.csv` | Annual cash-flow decomposition of the baseline case |
| `ccus_exercise_boundary.csv` | Per-decision-year investment share, oil-price statistics and break-even oil price |
| `ccus_scenario_results.csv`, `ccus_scenario_summary.csv`, `ccus_scenario_timing_distribution.csv` | All single-instrument and package scenarios, level by level |
| `ccus_combination_summary.csv` | Value increments, synergy and synergy ratio of every policy package (Table 5) |
| `ccus_sensitivity_comprehensive_results.csv`, `ccus_sensitivity_ranking.csv` | Parameter sensitivity results and ranking |
| `diagnostics_results.json`, `diag_convergence.png` | Numerical diagnostics and the path-count convergence figure |
| `benchmarks.json` | Benchmark runs used by the synergy decomposition |
| `figures/` | All manuscript and supplementary figures (1000 dpi PNG) |

## 6. Reproducibility

* Fixed random seed (`SEED = 42`) in all three scripts; policy scenarios and the baseline share
  the same simulated paths (common random numbers), so scenario differences are not sampling noise.
* All monetary values are in constant 2026 prices; price drifts are physical drifts and the
  discount rate is the CAPM required return on the corresponding cash flow (the two do not
  double-count risk).
* The diagnostic script first re-solves the baseline and compares it with
  `ccus_part1_results.csv`; the two agree to the last decimal.

## 7. Data sources

All parameters are taken from public sources and are documented item by item in Table 2 of the
paper (IEA, National Bureau of Statistics of China, National Energy Administration, Shanghai
Environment and Energy Exchange, IRS/26 U.S.C. §45Q, and the peer-reviewed literature).
No proprietary data are used.

## 8. Citation

```bibtex
@article{ccus_eor_realoption,
  title   = {Investment decision analysis of CCUS-EOR projects considering oil displacement efficiency and electricity price uncertainty},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

## 9. Contact

Questions about the code or the model: <corresponding mzisong@yeah.net>.
