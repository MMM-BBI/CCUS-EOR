import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Set CCUS_QUICK_TEST=1 to run the whole pipeline with 300 paths (fast smoke test).
QUICK_TEST = bool(os.environ.get("CCUS_QUICK_TEST"))

# Same seed and same number of paths as ccus_part1.py: every scenario is
# evaluated on the same simulated price paths (common random numbers), so the
# scenario differences are not Monte Carlo noise.
SEED = 42

# ---------------------------------------------------------------------------
# 45Q 政策工具的参考水平（政策强度 100% 对应的取值）。
#
# 现行法案（Pub. L. 119-21, 2025-07-04, §70522）已将“安全地质封存”与“驱油
# 利用/利用”两条路径并档：2025 年 7 月 4 日之后投用的设施适用同一适用美元金额，
# 基础抵免额 17 美元/吨，满足现行工资与学徒制要求后乘以 5，即 85 美元/吨
# （IRS Form 8933 (12/2025)）。按论文表注的 7.1 元/美元折算为 603.5 元/吨。
#
# 因此技术激励情景的政策上限（100%）取现行法案的全额抵免 603.5 元/吨，
# 基准情景则取 45Q 驱油利用抵免的基准水平 35 美元/吨 × 7.1 = 248.5 元/吨；
# 与之对应，各补贴工具的“强度”必须相对 45Q 的参考上限定义，而不能相对基准
# 参数（否则 pi 的基准值为 0，无法按比例扫描）。
# ---------------------------------------------------------------------------
FULL_45Q_CNY = 85.0 * 7.1                        # 603.5 元/吨（现行法案全额抵免）
BASE_45Q_CNY = 35.0 * 7.1                        # 248.5 元/吨（基准抵免水平）

CREDIT_REFERENCE = {
    'S_EOR': FULL_45Q_CNY,   # 技术激励情景 100% = 现行 45Q 全额抵免（85 美元/吨）
    'pi': FULL_45Q_CNY,      # 现行法案已并档，封存与驱油利用抵免同额
    'S_clean': 0.019,        # 清洁电价补贴，参考脱硫电价补贴标准
}

# The paper reports the four policy scenarios only; the two market scenarios
# (on-grid electricity price, initial oil price) are optional supplementary runs.
INCLUDE_MARKET_SCENARIOS = False

# ---------------------------------------------------------------------------
# 组合政策情景（本文研究缺口：现有文献缺少对政策工具组合协同效应的分析）
# ---------------------------------------------------------------------------
# 统一政策强度 s 取 0/25/50/75/100%，含义是"参与组合的每一项政策工具同时
# 达到其自身政策区间的同一相对档位"。各工具均以**基准情景为 s=0 的共同起点**，
# 因此不同组合在同一强度下可以直接横向比较：
#   碳交易     : 碳价相对基准上浮 0 → +25%        Pc0 × (1 + 0.25 s)
#   一次性补贴 : 补贴率 0 → 100%                  lambda_ = 1.00 s
#   税收优惠   : 所得税减免 0 → 25%               tax = 0.25 × (1 − 0.25 s)
#   技术激励   : 45Q 驱油利用抵免由基准水平提升至现行法案全额抵免
#                （35 → 85 美元/吨，倍数 85/35 ≈ 2.4286），即 × (1 + 1.4286 s)
# 前三项的上下限与单一政策情景完全一致；技术激励的上限取自美国现行 45Q 法案
# （Pub. L. 119-21, 2025）对驱油利用抵免的标准额（85 美元/吨，与地质封存同档），
# 未引入任何新的自由参数。
INCLUDE_COMBINATION_SCENARIOS = True

# 组合的元数：两两组合 6 个 + 三工具组合 4 个 + 四工具全组合 1 个 = 11 个情景。
# 若只想保留两两组合，改回 (2,) 即可。
COMBINATION_DEGREES = (2, 3, 4)

# 统一强度轴的档位。档位越多曲线越平滑，但运行时间线性增加。
COMBINATION_LEVELS = [0.00, 0.25, 0.50, 0.75, 1.00]

# 组合情景中技术激励工具在基准之上继续加码的幅度：由基准抵免水平（35 美元/吨）
# 提高到现行法案的全额抵免（85 美元/吨，与地质封存同档），倍数 85/35 ≈ 2.4286。
TECH_45Q_UPLIFT = 85.0 / 35.0 - 1.0         # ≈ 1.4286

# 组合情景的要素：四种政策工具
COMBINATION_TOOLS = [
    'carbon_trading',
    'investment_subsidy',
    'tax_incentive',
    'technology_incentive',
]
COMBINATION_TOOL_SHORT = {
    'carbon_trading': 'carbon',
    'investment_subsidy': 'subsidy',
    'tax_incentive': 'tax',
    'technology_incentive': 'technology',
}
COMBINATION_TOOL_LABEL = {
    'carbon_trading': 'carbon trading',
    'investment_subsidy': 'one-off investment subsidy',
    'tax_incentive': 'tax incentives',
    'technology_incentive': 'technology incentives',
}

import warnings
warnings.filterwarnings('ignore')

# 图内文字统一使用 Times New Roman（英文图，西文期刊惯例）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'   # 公式与角标同样使用 Times 风格
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['savefig.dpi'] = 1000     # save every figure at 1000 dpi
plt.rcParams['savefig.format'] = 'png'  # save every figure as PNG

# ---------------------------------------------------------------------------
# Figure export: every figure is written as a 1000-dpi PNG under ./figures/
# ---------------------------------------------------------------------------
import re as _re
from pathlib import Path as _Path

FIGURE_DIR = _Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


def _slug(text):
    """Turn a panel title into a file-name friendly slug."""
    slug = _re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return slug[:60] or "figure"


def save_and_show(name=None, dpi=1000):
    """Save the current figure as a 1000-dpi PNG, then display it.

    图内不再显示标题，因此由调用处显式给出文件名；未给出时回退到原来的按标题命名。
    """
    fig = plt.gcf()
    if name is None:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        name = _slug(titles[0]) if titles else "figure"
    path = FIGURE_DIR / (name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=dpi)
    print(f"[figure] saved {path} at {dpi} dpi")
    plt.show()


class CCUSInvestmentModel:
    def __init__(self, params):
        # 模型参数初始化
        self.IC = params['IC']  # 电厂装机容量 (kW)
        self.RT = params['RT']  # 年运行时间 (小时)
        self.EF = params['EF']  # 二氧化碳排放因子 (吨/kWh)
        self.CE = params['CE']  # 二氧化碳捕集效率
        self.phi = params['phi']  # 机组效率
        
        self.I0 = params['I0']  # 初始投资成本 (元)
        self.M0 = params['M0']  # 初始运营维护成本 (元)
        self.alpha = params['alpha']  # 技术进步对投资成本影响参数
        self.beta = params['beta']  # 技术进步对运营维护成本影响参数
        
        self.UTC_CO2 = params['UTC_CO2']  # 二氧化碳单位运输成本 (元/吨)
        self.UUC_CO2 = params['UUC_CO2']  # 单位驱油封存成本 (元/吨)
        self.USC_CO2 = params['USC_CO2']  # 单位封存成本 (元/吨)
        
        self.zeta = params['zeta']  # 二氧化碳封存率
        self.eta_loss = params['eta_loss']  # 发电效率损失因子
        
        # 电价相关参数设置
        self.Pe0 = params.get('Pe0', 0.35)  # 初始电价 (元/kWh)
        self.mu_e = params.get('mu_e', np.log(0.6))  # 长期均衡电价的ln值 (ln(0.6))
        self.alpha_e = params.get('alpha_e', 0.45564)  # 均值回复速度
        self.sigma_e = params.get('sigma_e', 0.1)  # 电价波动率
        self.rho_e_c = params.get('rho_e_c', 0.4)  # 电价与碳价相关系数        
        # 电价增长阶段参数
        self.stage1_years = params.get('stage1_years', 5)  # 第一阶段年数
        self.Pe_growth_rate = params.get('Pe_growth_rate', 0.024)  # 初始年增长率
        self.Pe_growth_increment = params.get('Pe_growth_increment', 0.0012)  # 年增长率增量        
        # 电价路径存储
        self.Pe_paths = None
        
        self.S_clean = params['S_clean']  # 清洁电价补贴 (元/kWh)
        # 动态计算CE_replace参数
        self.CE_replace_params = params.get('CE_replace_params', {})
        self.S_EOR = params['S_EOR']  # EOR补贴 (元/吨)
        self.pi = params['pi']  # 单位二氧化碳封存补贴 (元/吨)
        # 45Q 抵免的申领期：26 U.S.C. §45Q(a)(3)(A) 规定抵免只能在"设备原始投用日起
        # 12 年"内申领（IRS Form 8933 (12/2025) 同），即抵免仅覆盖投产后前 12 个运营年。
        self.claim_years = params.get('claim_years', 12)
        
        self.tax = params['tax']  # 企业所得税率
        self.lambda_ = params['lambda_']  # 投资补贴系数
        
        # 时间参数
        self.base_year = params.get('base_year', 2026)  # 基准年份
        self.T = params['T']  # 电厂寿命 (年)
        self.t_v = params['t_v']  # 投资决策期 (年)
        self.t_c = params['t_c']  # 建设期 (年)
        
        # 金融参数
        self.r_f = params['r_f']  # 无风险利率
        self.beta_risk = params['beta_risk']  # 系统性风险系数
        self.r_m_rf = params['r_m_rf']  # 系统性风险溢价
        self.r = self.r_f + self.beta_risk * self.r_m_rf  # 风险调整后贴现率
        
        # 不确定性参数
        self.Pc0 = params['Pc0']  # 初始碳价 (元/吨)
        self.mu_c = params['mu_c']  # 碳价漂移率
        self.sigma_c = params['sigma_c']  # 碳价波动率
        
        self.Poil0 = params['Poil0']  # 初始油价 (元/吨)
        self.mu_oil = params['mu_oil']  # 油价漂移率
        self.sigma_oil = params['sigma_oil']  # 油价波动率
        
        # 模拟参数
        self.M = params['M']  # 模拟路径数
        if QUICK_TEST:
            self.M = 300
            print(f"[quick test] simulation paths reduced to {self.M}")
        self.N = self.t_v  # 决策点数
        
        # 计算固定参数
        self.Q_e = self.IC * self.RT * self.phi  # 年发电量 (kWh)
        
        # 年二氧化碳排放量 (吨)
        annual_emissions = self.IC * self.RT * self.EF
        self.Q_CO2 = annual_emissions * self.CE * self.phi  # 年二氧化碳捕集量 (吨)
        
        # 年发电损失量 (kWh)
        self.Q_loss = self.eta_loss * self.phi * self.IC * self.RT
        
        # 年二氧化碳驱油使用量 (吨)
        self.Q_EOR = self.Q_CO2  # CO2 injected for EOR (t); all captured CO2 is injected, as stated after Eq. (13)
        
        print(f"Model initialised: ")
        print(f"  Annual generation: {self.Q_e:,.0f} kWh")
        print(f"Electricity price model parameters:")
        print(f"  Initial electricity price: {self.Pe0:.3f} CNY/kWh")
        print(f"  Long-run equilibrium price: {np.exp(self.mu_e):.3f} CNY/kWh")
        print(f"  Mean reversion speed: {self.alpha_e:.4f}")
        print(f"  Electricity price volatility: {self.sigma_e:.3f}")
        print(f"  Correlation with the carbon price: {self.rho_e_c:.1f}")
        print(f"  Years in stage one: {self.stage1_years}")        
        print(f"  Annual CO2 capture: {self.Q_CO2:,.0f} t")
        print(f"  Annual CO2 injected for EOR: {self.Q_EOR:,.0f} t")
        print(f"  Annual generation loss: {self.Q_loss:,.0f} kWh")
        print(f"  Risk-adjusted discount rate: {self.r:.3f}")
        print(f"  Time structure: plant life {self.T} years = decision window {self.t_v} years + construction {self.t_c} year(s) + dynamic operating period")
        
    def calculate_operating_period(self, investment_year):
        """
        Operating period implied by the investment year.
        Operating period = plant life - investment year - construction period
        """
        operating_period = self.T - investment_year - self.t_c
        return max(0, operating_period)  # 确保非负
        
    def calculate_CE_replace(self, t):
        """
        Time-varying oil displacement efficiency (piecewise function):
        - t < 8: CE_replace = exp(0.791t - 4.662) / 6.93
        - 8 <= t <= 10: CE_replace = -0.0145t + 0.560
        - t > 10: CE_replace = exp(-0.183t + 2.851) / 6.93
        """
        if t < 8:
            # 快速增长阶段
            ce_replace = np.exp(0.791 * t - 4.662) / 6.93
        elif t <= 10:
            # 线性下降阶段
            ce_replace = -0.0145 * t + 0.560
        else:
            # 缓慢下降阶段
            ce_replace = np.exp(-0.183 * t + 2.851) / 6.93
        
        # 确保效率在合理范围内 (0到1之间)
        ce_replace = max(0, min(ce_replace, 1.0))
        
        return ce_replace
    
    def simulate_electricity_price(self, Pc_paths):
        """Simulate electricity price paths (two-stage model).

        The stage-two innovation is correlated with the CARBON price innovation
        with coefficient rho_e_c, as assumed in the paper.  The carbon shocks
        are handed over by simulate_uncertainties() so that the two processes
        really share the same random driver.
        """
        print("Simulating electricity price paths...")
        M, total_years = Pc_paths.shape
        
        # 初始化电价路径
        Pe_paths = np.zeros((M, total_years))
        Pe_paths[:, 0] = self.Pe0
        
        # 独立的电价随机扰动（用与碳价/油价不同的随机流，避免重复抽样）
        rng = np.random.default_rng(SEED + 1)
        Z_e = rng.standard_normal((M, total_years - 1))

        # 与碳价随机增量相关的电价随机增量
        Z_c_shock = getattr(self, "Z_c_paths", None)
        if Z_c_shock is None:
            Z_c_shock = rng.standard_normal((M, total_years - 1))
        Z_e_correlated = self.rho_e_c * Z_c_shock + np.sqrt(1 - self.rho_e_c**2) * Z_e
        
        # 模拟电价路径
        for t in range(1, total_years):
            if t < self.stage1_years:
                # 第一阶段：确定性增长
                # 增长率 = 初始增长率 + (t-1) * 年增量
                growth_rate = self.Pe_growth_rate + (t-1) * self.Pe_growth_increment
                Pe_paths[:, t] = Pe_paths[:, t-1] * (1 + growth_rate)
            else:
                # 第二阶段：均值回复随机过程
                for j in range(M):
                    # 均值回复过程离散形式
                    Pe_prev = Pe_paths[j, t-1]
                    ln_Pe_prev = np.log(Pe_prev)
                    
                    # 均值回复公式
                    d_ln_Pe = self.alpha_e * (self.mu_e - ln_Pe_prev) + \
                              self.sigma_e * Z_e_correlated[j, t-1]
                    
                    Pe_paths[j, t] = Pe_prev * np.exp(d_ln_Pe)
        
        self.Pe_paths = Pe_paths
        return Pe_paths
    
    def simulate_uncertainties(self):
        """Simulate uncertainty paths for carbon, oil and electricity prices"""
        dt = 1
        
        # 需要模拟的总年数 = 电厂总寿命
        total_simulation_years = self.T
        
        # 初始化价格路径
        Pc_paths = np.zeros((self.M, total_simulation_years))
        Poil_paths = np.zeros((self.M, total_simulation_years))
        
        # 初始价格
        Pc_paths[:, 0] = self.Pc0
        Poil_paths[:, 0] = self.Poil0
        
        # 生成随机数（使用相关随机过程）
        np.random.seed(SEED)
        
        # 基础随机数
        Z_c_base = np.random.standard_normal((self.M, total_simulation_years-1))
        Z_oil_base = np.random.standard_normal((self.M, total_simulation_years-1))
        
        # 假设碳价和油价独立（如果文献没提相关性）
        Z_c = Z_c_base
        Z_oil = Z_oil_base
        # 保存碳价随机增量，供电价模型产生相关系数为 rho_e_c 的随机增量
        self.Z_c_paths = Z_c
        
        # 模拟碳价路径
        for t in range(1, total_simulation_years):
            Pc_paths[:, t] = Pc_paths[:, t-1] * np.exp(
                (self.mu_c - 0.5 * self.sigma_c**2) * dt + 
                self.sigma_c * np.sqrt(dt) * Z_c[:, t-1]
            )
            
            Poil_paths[:, t] = Poil_paths[:, t-1] * np.exp(
                (self.mu_oil - 0.5 * self.sigma_oil**2) * dt + 
                self.sigma_oil * np.sqrt(dt) * Z_oil[:, t-1]
            )
        
        # 模拟电价路径（与碳价相关）
        Pe_paths = self.simulate_electricity_price(Pc_paths)
        
        return Pc_paths, Poil_paths, Pe_paths
    
    def calculate_cash_flow(self, investment_year, operation_year, Pc_t, Poil_t, Pe_t):
        """Compute cash flows with the dynamic electricity price"""
        try:
            # 年运营维护成本
            total_years = investment_year + self.t_c + operation_year
            M_t_CCUS = self.M0 * np.exp(-self.beta * total_years)
            
            # 运输成本
            TC_CO2 = self.Q_CO2 * self.UTC_CO2
            
            # 驱油封存成本（ζQ × 单位驱油封存成本）
            UC_CO2 = self.zeta * self.Q_CO2 * self.UUC_CO2
            # 封存成本（Q × 单位封存成本，与驱油封存成本并列）
            SC_CO2 = self.Q_CO2 * self.USC_CO2
            
            # 损失成本 - 使用动态电价
            C_loss = Pe_t * self.Q_loss
            
            # 收入计算
            # 碳交易收入
            R_c = Pc_t * self.Q_CO2
            
            # 清洁电价补贴 - 基于动态电价
            R_e = self.S_clean * self.Q_e
            
            # 发电收入 = 电价 × 发电量
            # R_generation = Pe_t * self.Q_e  # 参考文献中没有考虑发电收入
            
            # 原油销售收入
            current_ce_replace = self.calculate_CE_replace(operation_year)
            Q_oil = current_ce_replace * self.Q_EOR
            R_oil = Poil_t * Q_oil
            
            # EOR补贴
            # EOR补贴：45Q 仅对设备投用日起 12 个运营年内的合格二氧化碳计提
            if operation_year <= self.claim_years:
                R_EOR = self.S_EOR * self.Q_EOR
            else:
                R_EOR = 0.0
            
            # 封存补贴
            # 封存补贴（若启用）同样受 12 年申领期限制
            if operation_year <= self.claim_years:
                R_SC = self.pi * self.zeta * self.Q_CO2
            else:
                R_SC = 0.0
            
            # 总收入 - 加入发电收入
            total_revenue = R_c + R_e + R_oil + R_EOR + R_SC   # R_generation
            
            # 总运营成本
            total_operating_cost = TC_CO2 + UC_CO2 + SC_CO2 + C_loss + M_t_CCUS
            
            # 税前运营现金流
            CF_before_tax = total_revenue - total_operating_cost
            
            # 税后运营现金流
            CF_after_tax = CF_before_tax * (1 - self.tax)
            
            return CF_after_tax, current_ce_replace
            
        except Exception as e:
            print(f"Cash flow error: {e}")
            return 0, 0
    
    def calculate_project_value(self, investment_year, Pc_path, Poil_path, Pe_path):
        """
        Project value using the simulated electricity price path
        """
        try:
            # 投资成本
            investment_cost = (1 - self.lambda_) * self.I0 * np.exp(-self.alpha * investment_year)
            
            # 计算实际运营期
            operating_period = self.calculate_operating_period(investment_year)
            
            if operating_period <= 0:
                return -investment_cost
            
            # 计算运营期各年的现金流现值
            operating_value = 0
            ce_replace_values = []
            
            for op_year in range(1, operating_period + 1):
                total_year = investment_year + self.t_c + op_year - 1
                
                if total_year >= len(Pc_path):
                    break
                
                CF, ce_replace = self.calculate_cash_flow(
                    investment_year, 
                    op_year,
                    Pc_path[total_year],
                    Poil_path[total_year],
                    Pe_path[total_year]  # 传递电价
                )
                
                ce_replace_values.append(ce_replace)
                
                # 贴现到投资决策时点
                discount_years = self.t_c + op_year
                discount_factor = np.exp(-self.r * discount_years)
                operating_value += CF * discount_factor
            
            # 项目价值
            project_value = operating_value - investment_cost
            
            return project_value
            
        except Exception as e:
            print(f"Project value error: {e}")
            return -1e9

    def expected_price_paths(self, t, Pc_t, Poil_t, Pe_t):
        """从决策时点 t 的当期价格出发，递推未来各年价格的**条件期望**。

        只用到 t 时刻可观测的状态，不使用任何未来信息。
        """
        Epc = np.zeros(self.T)
        Eoil = np.zeros(self.T)
        Epe = np.zeros(self.T)
        Epc[t], Eoil[t], Epe[t] = Pc_t, Poil_t, Pe_t
        m_pe = np.log(Pe_t)
        v_pe = 0.0
        for k in range(t + 1, self.T):
            Epc[k] = Epc[k - 1] * np.exp(self.mu_c)
            Eoil[k] = Eoil[k - 1] * np.exp(self.mu_oil)
            if k < self.stage1_years:
                growth = self.Pe_growth_rate + (k - 1) * self.Pe_growth_increment
                Epe[k] = Epe[k - 1] * (1 + growth)
                m_pe, v_pe = np.log(Epe[k]), 0.0
            else:
                m_pe = (1 - self.alpha_e) * m_pe + self.alpha_e * self.mu_e
                v_pe = (1 - self.alpha_e) ** 2 * v_pe + self.sigma_e ** 2
                Epe[k] = np.exp(m_pe + 0.5 * v_pe)
        return Epc, Eoil, Epe

    def expected_project_value(self, t, Pc_t, Poil_t, Pe_t):
        """式(25)的投资期望值：在决策时点 t、仅依据当期价格状态，立即投资的期望净现值。"""
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
                       + (self.S_EOR * self.Q_EOR
                          if op_year <= self.claim_years else 0.0)
                       + (self.pi * self.zeta * self.Q_CO2
                          if op_year <= self.claim_years else 0.0))
            cost = (self.Q_CO2 * self.UTC_CO2
                    + self.zeta * self.Q_CO2 * self.UUC_CO2
                    + self.Q_CO2 * self.USC_CO2
                    + Epe[cal] * self.Q_loss
                    + self.M0 * np.exp(-self.beta * (t + self.t_c + op_year)))
            value += (revenue - cost) * (1 - self.tax) * np.exp(-self.r * (self.t_c + op_year))
        return value - investment_cost

    
    def lsmc_solver(self):
        """Solve by least-squares Monte Carlo simulation"""
        print("Simulating uncertainty paths...")
        Pc_paths, Poil_paths, Pe_paths = self.simulate_uncertainties()
        
        print("Running the least-squares Monte Carlo solver...")
        # 初始化矩阵
        TIV = np.zeros((self.M, self.N))  # 总投资价值
        NPV = np.zeros((self.M, self.N))  # 净现值
        EX = np.zeros((self.M, self.N))   # 当期信息下的投资期望价值（决策依据）
        decisions = np.zeros((self.M, self.N), dtype=int)  # 投资决策
        exercise_values = np.zeros((self.M, self.N))  # 立即执行价值
        
        # 预计算所有路径所有时间的NPV（立即执行价值）
        print("Pre-computing the value matrices...")
        report_every = max(1, self.M // 10)
        for j in range(self.M):
            if (j + 1) % report_every == 0:
                print(f"  value matrices: {j + 1}/{self.M} paths priced")
            for t in range(self.N):
                NPV[j, t] = self.calculate_project_value(t, Pc_paths[j], Poil_paths[j], Pe_paths[j])
                EX[j, t] = self.expected_project_value(t, Pc_paths[j, t], Poil_paths[j, t], Pe_paths[j, t])
        
        # 步骤1: 最后决策点的投资决策
        t_final = self.N - 1
        decisions[:, t_final] = (EX[:, t_final] >= 0).astype(int)
        TIV[:, t_final] = np.where(decisions[:, t_final] == 1, NPV[:, t_final], 0.0)
        exercise_values[:, t_final] = EX[:, t_final]
        
        print(f"Final decision point: {np.sum(decisions[:, t_final])} paths choose to invest")
        
        # 步骤2: 逆向动态规划
        for t in range(self.N-2, -1, -1):
            print(f"Processing decision point t={t} (year {self.base_year + t})")
            
            # 倒推至时点 t 时，各路径在该时点的决策尚未做出（decisions[:, t] 全为 0），
            # 因此所有路径都仍然可以选择继续持有期权，回归样本为全部路径。
            continuation_paths = np.where(decisions[:, t] == 0)[0]
            
            if len(continuation_paths) > 10:
                # 准备回归数据：使用当前状态变量作为特征
                # 准备回归数据：特征全部取自 t 时刻可观测的状态（不增加任何新变量），
                # 回归前统一标准化——几个变量的量级相差很大（电价约 0.4、油价约 3500、
                # 期望价值约 10^4），标准化可避免数值条件数过差导致的系数抖动。
                raw_X = np.column_stack([
                    EX[continuation_paths, t],           # 当期状态下的投资期望价值
                    Pc_paths[continuation_paths, t],     # 当前碳价
                    Poil_paths[continuation_paths, t],   # 当前油价
                    Pe_paths[continuation_paths, t]      # 当前电价
                ])
                mu_X = raw_X.mean(axis=0)
                sd_X = raw_X.std(axis=0)
                sd_X[sd_X == 0] = 1.0
                X = (raw_X - mu_X) / sd_X
                
                # 目标变量：下一期折现后的投资价值
                Y = TIV[continuation_paths, t+1] * np.exp(-self.r)
                
                # 使用线性回归（避免过拟合）
                try:
                    model = LinearRegression()
                    model.fit(X, Y)
                    continuation_estimates = model.predict(X)
                    
                    # 更新继续持有价值
                    for idx, path_idx in enumerate(continuation_paths):
                        immediate_value = EX[path_idx, t]
                        realized_value = NPV[path_idx, t]
                        cont_value = continuation_estimates[idx]
                        
                        # 决策规则
                        if immediate_value > 0 and immediate_value >= cont_value:
                            decisions[path_idx, t] = 1  # 立即投资
                            TIV[path_idx, t] = realized_value
                        elif immediate_value <= 0 and cont_value > 0:
                            decisions[path_idx, t] = 0  # 延迟投资
                            TIV[path_idx, t] = TIV[path_idx, t+1] * np.exp(-self.r)
                        elif immediate_value <= 0 and cont_value <= 0:
                            # NPV<0 且 TIV=0
                            decisions[path_idx, t] = 0  # 放弃投资
                            TIV[path_idx, t] = 0
                        else:
                            decisions[path_idx, t] = 0  # 延迟投资
                            TIV[path_idx, t] = TIV[path_idx, t+1] * np.exp(-self.r)
                            
                except Exception as e:
                    print(f"Regression failed at t={t}: {e}, using a simple rule")
                    # 回归失败时的备用方案
                    for path_idx in continuation_paths:
                        immediate_value = EX[path_idx, t]
                        realized_value = NPV[path_idx, t]
                        cont_value = TIV[path_idx, t+1] * np.exp(-self.r)
                        
                        # 应用相同的决策规则
                        if immediate_value > 0 and immediate_value >= cont_value:
                            decisions[path_idx, t] = 1
                            TIV[path_idx, t] = realized_value
                        elif immediate_value <= 0 and cont_value > 0:
                            decisions[path_idx, t] = 0
                            TIV[path_idx, t] = TIV[path_idx, t+1] * np.exp(-self.r)
                        elif immediate_value <= 0 and cont_value <= 0:
                            decisions[path_idx, t] = 0
                            TIV[path_idx, t] = 0
                        else:
                            decisions[path_idx, t] = 0
                            TIV[path_idx, t] = TIV[path_idx, t+1] * np.exp(-self.r)
            else:
                # 样本不足，使用简单规则
                for path_idx in continuation_paths:
                    immediate_value = EX[path_idx, t]
                    realized_value = NPV[path_idx, t]
                    cont_value = TIV[path_idx, t+1] * np.exp(-self.r)
                    
                    # 应用相同的决策规则
                    if immediate_value > 0 and immediate_value >= cont_value:
                        decisions[path_idx, t] = 1
                        TIV[path_idx, t] = realized_value
                    elif immediate_value <= 0 and cont_value > 0:
                        decisions[path_idx, t] = 0
                        TIV[path_idx, t] = TIV[path_idx, t+1] * np.exp(-self.r)
                    elif immediate_value <= 0 and cont_value <= 0:
                        decisions[path_idx, t] = 0
                        TIV[path_idx, t] = 0
                    else:
                        decisions[path_idx, t] = 0
                        TIV[path_idx, t] = TIV[path_idx, t+1] * np.exp(-self.r)
        
        # 步骤3: 确定最优投资时机和投资价值
        print("Determining the optimal investment timing...")
        optimal_timing_by_path = []
        optimal_value_by_path = []
        optimal_year_by_path = []
        
        for j in range(self.M):
            investment_times = np.where(decisions[j, :] == 1)[0]
            if len(investment_times) > 0:
                t_star = investment_times[0]  # 首次投资时间
                optimal_timing_by_path.append(t_star)
                optimal_value_by_path.append(TIV[j, t_star])
                optimal_year_by_path.append(self.base_year + t_star)
            else:
                optimal_timing_by_path.append(self.N)  # 无投资
                optimal_value_by_path.append(0)
                optimal_year_by_path.append(self.base_year + self.N)
        
        # 计算统计量 - 只考虑在决策期内的投资
        valid_investments = [(t, v, y) for t, v, y in zip(optimal_timing_by_path, optimal_value_by_path, optimal_year_by_path) if t < self.N]
        
        if valid_investments:
            investment_timings, investment_values, investment_years = zip(*valid_investments)
            avg_optimal_timing = np.mean(investment_timings)
            median_optimal_timing = float(np.median(investment_timings))
            avg_optimal_value = np.mean(investment_values)
            avg_optimal_year = np.mean(investment_years)
            median_optimal_year = self.base_year + median_optimal_timing
            investment_count = len(valid_investments)
            
            # 投资年份分布
            year_counts = {}
            for year in investment_years:
                year_int = int(year)
                year_counts[year_int] = year_counts.get(year_int, 0) + 1
            # 出现频率最高的投资年份（论文 4.4.3 节的“全局最优投资时机”）
            modal_year = max(sorted(year_counts), key=lambda y: year_counts[y])
            modal_share = year_counts[modal_year] / len(valid_investments) * 100
        else:
            avg_optimal_timing = np.nan
            median_optimal_timing = np.nan
            avg_optimal_value = 0
            avg_optimal_year = np.nan
            median_optimal_year = np.nan
            modal_year = None
            modal_share = 0.0
            investment_count = 0
            year_counts = {}
        
        print("\nInvestment decision statistics:")
        for year in sorted(year_counts.keys()):
            percentage = year_counts[year] / len(valid_investments) * 100 if valid_investments else 0
            op_period = self.calculate_operating_period(year - self.base_year)
            print(f"  {year}: {year_counts[year]} paths ({percentage:.1f}%), operating period {op_period} years")
        
        return {
            'optimal_timing': avg_optimal_timing,
            'optimal_timing_median': median_optimal_timing,
            'optimal_timing_median_year': median_optimal_year,
            'optimal_timing_modal_year': modal_year,
            'optimal_timing_modal_share': modal_share,
            'optimal_value': avg_optimal_value,
            'optimal_timing_year': avg_optimal_year,
            'decisions': decisions,
            'TIV': TIV,
            'NPV': NPV,
            'Pc_paths': Pc_paths,
            'Poil_paths': Poil_paths,
            'investment_count': investment_count,
            'investment_ratio': investment_count / self.M,
            'timing_distribution': year_counts,
            'optimal_timing_by_path': optimal_timing_by_path,
            'optimal_year_by_path': optimal_year_by_path,
            'operating_period_info': {t: self.calculate_operating_period(t) for t in range(self.N)}
        }

# 情景分析类
class ScenarioAnalyzer:
    def __init__(self, base_params):
        self.base_params = base_params.copy()
        
    def create_scenarios(self):
        """Create the five policy scenarios.

        The baseline already carries the 45Q-style utilisation credit
        (S_EOR = 248.5 CNY/t, i.e. 35 USD/t at 7.1 CNY/USD) but no separate
        geological storage credit, because the credit is allowed once per tonne
        of CO2 and the project injects all captured CO2 for EOR.  Under the
        current statute (Pub. L. 119-21, 2025) the utilisation and geological
        storage pathways carry the *same* applicable dollar amount, so the
        storage pathway is not an alternative with a higher credit rate.
        """
        scenarios = {}


        # 情景1: 碳交易机制情景
        scenarios['carbon_trading'] = {
            'name': 'Carbon trading',
            'description': 'Carbon trading revenue scaled up step by step',
            'x_label': 'Increase in the carbon price (%)',
            'parameter_changes': [
                {'Pc0_multiplier': 1.00},  # 基准
                {'Pc0_multiplier': 1.05},  # +5%
                {'Pc0_multiplier': 1.10},  # +10%
                {'Pc0_multiplier': 1.15},  # +15%
                {'Pc0_multiplier': 1.20},  # +20%
                {'Pc0_multiplier': 1.25},  # +25%
            ]
        }
        
        # 情景2: 一次性投资补贴情景
        scenarios['investment_subsidy'] = {
            'name': 'One-off investment subsidy',
            'description': 'Investment subsidy ratio scaled up step by step',
            'x_label': 'Investment subsidy ratio (%)',
            'parameter_changes': [
                {'lambda_': 0.00},  # 0
                {'lambda_': 0.10},  # 10%
                {'lambda_': 0.20},  # 20%
                {'lambda_': 0.30},  # 30%
                {'lambda_': 0.40},  # 40%
                {'lambda_': 0.50},  # 50%
                {'lambda_': 0.60},  # 60%
                {'lambda_': 0.70},  # 70%
                {'lambda_': 0.80},  # 80%
                {'lambda_': 0.90},  # 90%
                {'lambda_': 1.00},  # 100%
            ]
        }
        
        # 情景3: 技术激励情景（45Q 驱油利用抵免 + 清洁电价补贴，
        #          100% 即基准情景；不含封存抵免，避免对同一吨 CO2 重复计补）
        scenarios['technology_incentive'] = {
            'name': 'Technology incentives',
            'description': '45Q utilisation credit scaled up from zero to the full level',
            'x_label': 'Technology subsidy, % of the 45Q level',
            'parameter_changes': [
                {'S_EOR_multiplier': 0.00, 'S_clean_multiplier': 0.00},
                {'S_EOR_multiplier': 0.15, 'S_clean_multiplier': 0.15},
                {'S_EOR_multiplier': 0.30, 'S_clean_multiplier': 0.30},
                {'S_EOR_multiplier': 0.45, 'S_clean_multiplier': 0.45},
                {'S_EOR_multiplier': 0.60, 'S_clean_multiplier': 0.60},
                {'S_EOR_multiplier': 0.75, 'S_clean_multiplier': 0.75},
                {'S_EOR_multiplier': 1.00, 'S_clean_multiplier': 1.00},
            ]
        }

        # 说明：45Q 对同一吨 CO2 只给予一次抵免；现行法案（2025）已将“安全地质封存”
        # 与“驱油利用/利用”并档为同一适用美元金额（基础额 17 美元/吨，满足工资与
        # 学徒制要求后乘以 5，即 85 美元/吨），两条路径不再存在抵免率差异。本项目捕集
        # 的 CO2 全部用于驱油，因此不设“封存抵免路径”情景；参数 pi 保留在模型中，
        # 以便把模型推广到咸水层封存类项目（届时取 603.5 元/吨）。
        
        # 情景4: 税收优惠情景
        scenarios['tax_incentive'] = {
            'name': 'Tax incentives',
            'description': 'Tax relief scaled up step by step',
            'x_label': 'Tax relief (%)',
            'parameter_changes': [
                {'tax': 0.25},      # 基准25%
                {'tax': 0.2375},    # 5%优惠
                {'tax': 0.2250},    # 10%优惠
                {'tax': 0.2125},    # 15%优惠
                {'tax': 0.2000},    # 20%优惠
                {'tax': 0.1875},    # 25%优惠
            ]
        }
        
        # 情景5: 上网电价情景 
        scenarios['electricity_price'] = {
            'name': 'On-grid electricity price',
            'description': 'On-grid electricity price raised step by step',
            'x_label': 'Increase in the on-grid electricity price (%)',
            'parameter_changes': [
                {'Pe_multiplier': 1.00},  # 基准
                {'Pe_multiplier': 1.05},  # +5%
                {'Pe_multiplier': 1.10},  # +10%
                {'Pe_multiplier': 1.15},  # +15%
                {'Pe_multiplier': 1.20},  # +20%
                {'Pe_multiplier': 1.25},  # +25%
            ]
        }
        
        # 情景6: 初始油价情景
        scenarios['oil_price'] = {
            'name': 'Initial oil price',
            'description': 'Initial oil price raised step by step',
            'x_label': 'Increase in the oil price (%)',
            'parameter_changes': [
                {'Poil0_multiplier': 1.00},  # 基准
                {'Poil0_multiplier': 1.05},  # +5%
                {'Poil0_multiplier': 1.10},  # +10%
                {'Poil0_multiplier': 1.15},  # +15%
                {'Poil0_multiplier': 1.20},  # +20%
                {'Poil0_multiplier': 1.25},  # +25%
            ]
        }
        
        if not INCLUDE_MARKET_SCENARIOS:
            scenarios.pop('electricity_price', None)
            scenarios.pop('oil_price', None)

        # 组合政策情景：把参与组合的工具同时置于统一强度轴上
        if INCLUDE_COMBINATION_SCENARIOS:
            from itertools import combinations as _combos

            for degree in COMBINATION_DEGREES:
                for tools in _combos(COMBINATION_TOOLS, degree):
                    key = 'combo_' + '_'.join(
                        COMBINATION_TOOL_SHORT[t] for t in tools)
                    label = ' + '.join(
                        COMBINATION_TOOL_LABEL[t] for t in tools)
                    scenarios[key] = {
                        'name': f'Combined policy: {label}',
                        'description': 'Policy intensity scaled up step by step',
                        'x_label': 'Policy intensity (%)',
                        'parameter_changes': [
                            self._combination_change(tools, level)
                            for level in COMBINATION_LEVELS
                        ],
                        # 组合情景数量多，显式指定文件名，避免 60 字符截断后重名
                        'figure_name': f'{key}_policy_intensity.png',
                        'is_combination': True,
                    }

        return scenarios

    def _combination_change(self, tools, level):
        """Return the parameter change for a policy package at intensity `level`.

        `level` runs from 0 (every participating tool at its baseline value)
        to 1 (every participating tool at its own policy ceiling), so all
        combinations share the same 0% anchor and are directly comparable.
        """
        change = {'combo_level': float(level)}
        if 'carbon_trading' in tools:
            change['Pc0_multiplier'] = 1.0 + 0.25 * level
        if 'investment_subsidy' in tools:
            change['lambda_'] = 1.00 * level
        if 'tax_incentive' in tools:
            base_tax = self.base_params.get('tax', 0.25)
            change['tax'] = base_tax * (1.0 - 0.25 * level)
        if 'technology_incentive' in tools:
            # 相对**基准参数**加码（而非相对 CREDIT_REFERENCE，后者已是现行全额抵免）：
            # 只加码 45Q 抵免——由基准水平 248.5 元/吨线性提升至现行全额 603.5 元/吨。
            # 清洁电价补贴 S_clean 在全部情景中均保持基准水平 0.019 元/千瓦时：
            # 该值参照脱硫电价补贴标准，不存在将其在组合情景中单独放大的政策依据，
            # 因此与单一技术激励情景的上限口径保持一致（603.5 元/吨 + 0.019 元/千瓦时）。
            change['S_EOR_of_baseline'] = 1.0 + TECH_45Q_UPLIFT * level
        return change
    
    def run_scenario_analysis(self, scenarios, M_scenario=2000):
        """Run the scenario analysis"""
        print("=" * 70)
        print("Starting the scenario analysis")
        print("=" * 70)
        
        # 获取基准年份
        base_year = self.base_params.get('base_year', 2026)
        
        scenario_results = {}
        
        for scenario_key, scenario_config in scenarios.items():
            print(f"\n Analysing: {scenario_config['name']}")
            print(f" description: {scenario_config['description']}")
            print("-" * 50)
            
            scenario_results[scenario_key] = {
                'name': scenario_config['name'],
                'description': scenario_config['description'],
                'x_label': scenario_config['x_label'],
                'figure_name': scenario_config.get('figure_name'),
                'is_combination': scenario_config.get('is_combination', False),
                'sub_scenarios': []
            }
            
            for i, param_change in enumerate(scenario_config['parameter_changes']):
                print(f"  Sub-scenario {i+1}: {param_change}")
                
                # 创建修改后的参数
                modified_params = self.base_params.copy()
                modified_params['M'] = M_scenario   #减少模拟次数以提高速度

                # 应用参数变化
                for param, value in param_change.items():
                    if param.endswith('_multiplier'):
                        # 价格类工具相对基准值缩放；补贴类工具相对 45Q 参考水平
                        # 缩放（基准情景中 pi = 0，不能乘基准值）
                        name = param[: -len('_multiplier')]
                        key = {'Pe': 'Pe0'}.get(name, name)
                        reference = CREDIT_REFERENCE.get(
                            key, self.base_params.get(key, 0.0))
                        modified_params[key] = reference * value
                    elif param.endswith('_of_baseline'):
                        # 组合情景专用：相对基准参数按倍数加码
                        name = param[: -len('_of_baseline')]
                        modified_params[name] = self.base_params[name] * value
                    elif param == 'lambda_':
                        modified_params['lambda_'] = value
                    elif param == 'tax':
                        modified_params['tax'] = value
                
                # 运行模型
                try:
                    model = CCUSInvestmentModel(modified_params)
                    results = model.lsmc_solver()
                    
                    # 存储结果
                    valid_years = [y for y in results.get('optimal_year_by_path', [])
                                   if y < base_year + modified_params['t_v']]
                    sub_scenario_result = {
                        'parameters': param_change,
                        'optimal_timing_year': results.get('optimal_timing_year', np.nan),
                        'optimal_value': results.get('optimal_value', 0),
                        'option_value_at_decision': float(
                            np.mean(results['TIV'][:, 0])) if 'TIV' in results else 0.0,
                        'mean_investment_year': float(np.mean(valid_years))
                        if valid_years else np.nan,
                        'median_investment_year': float(np.median(valid_years))
                        if valid_years else np.nan,
                        'investment_ratio': results.get('investment_ratio', 0),
                        'timing_distribution': results.get('timing_distribution', {})
                    }
                    
                    scenario_results[scenario_key]['sub_scenarios'].append(sub_scenario_result)
                    
                    # 显示实际年份而不是相对年份
                    optimal_year = results.get('optimal_timing_year', np.nan)
                    if not np.isnan(optimal_year):
                        optimal_year_int = round(optimal_year)
                        print(f"    Result: optimal year {optimal_year_int}, "
                              f"investment value {results.get('optimal_value', 0)/1e6:.2f} million CNY, "
                              f"investment share {results.get('investment_ratio', 0):.1%}")
                    else:
                        print(f"    Result: no suitable investment window in {base_year}-{base_year+modified_params['t_v']-1}, "
                              f"investment share {results.get('investment_ratio', 0):.1%}")
                              
                except Exception as e:
                    print(f"     error: {e}")
                    scenario_results[scenario_key]['sub_scenarios'].append({
                        'parameters': param_change,
                        'error': str(e)
                    })
        
        return scenario_results
    
    def plot_individual_scenario(self, scenario_results, scenario_key):
        """Plot one scenario: optimal decision and investment-year distribution"""
        if scenario_key not in scenario_results:
            print(f" Scenario '{scenario_key}' does not exist")
            return
        
        scenario_data = scenario_results[scenario_key]
        sub_scenarios = scenario_data['sub_scenarios']
        
        if not sub_scenarios or 'error' in sub_scenarios[0]:
            print(f" Scenario '{scenario_data['name']}' no valid data")
            return
        
        # 准备数据
        policy_strengths = []
        timing_values = []
        value_values = []
        timing_distributions = []  # 存储投资年份分布数据
        
        for i, sub_scenario in enumerate(sub_scenarios):
            if 'error' not in sub_scenario:
                # 计算政策强度百分比
                policy_strength = self._calculate_policy_strength(scenario_key, sub_scenario['parameters'])
                policy_strengths.append(policy_strength)

                # 时机用中位投资年份、价值用决策时点期权价值 V0（跨情景可比）
                timing_values.append(sub_scenario.get('median_investment_year', np.nan))
                value_values.append(sub_scenario.get('option_value_at_decision', 0.0) / 1e6)
                timing_distributions.append(sub_scenario.get('timing_distribution', {}))  # 存储分布数据
        
        if not policy_strengths:
            return
        
        # 创建图形 - 创建2个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 获取基准年份
        base_year = self.base_params.get('base_year', 2026)
        decision_period = self.base_params['t_v']
        
        # ============================
        # 子图1：投资价值和最优时机（左边）
        # ============================
        x = np.arange(len(policy_strengths))
        
        # 投资价值柱形图（决策时点期权价值 V0）
        bars = ax1.bar(x, value_values, width=0.6, color='#4E79A7', alpha=0.9,
                       label='Option value at decision point V0 (million CNY)')
        
        # 最优投资时机折线图（使用次坐标轴）
        ax1_twin = ax1.twinx()
        line = ax1_twin.plot(x, timing_values, 'o-', color='#A68095', linewidth=3,
                           markersize=10, label='Median optimal investment year',
                           markerfacecolor='white', markeredgewidth=2)
        
        # 设置标题和标签
        ax1.set_title(f'{scenario_data["name"]}\n{scenario_data["description"]}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel(scenario_data['x_label'], fontsize=12)
        ax1.set_ylabel('Option value at decision point V0 (million CNY)', fontsize=12,
                       color='black')
        ax1_twin.set_ylabel('Median optimal investment year', fontsize=12, color='black')
        
        # 设置x轴标签
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{ps:.0f}%' for ps in policy_strengths], fontsize=11)
        
        # 坐标轴文字统一用黑色
        ax1.tick_params(axis='y', labelcolor='black')
        ax1_twin.tick_params(axis='y', labelcolor='black')
        ax1.tick_params(axis='x', labelcolor='black')

        # 抬高左轴上限（柱状图仍从 0 起），为图例留出空白。
        # 若某情景的时机折线近似水平（如碳交易机制情景，中位年份全程 2033 年），
        # 折线标签会稳定停在某一高度，需要更大的上限把它与柱顶数字分开。
        v_max = max([v for v in value_values if not np.isnan(v)] or [1.0])
        span = max(v_max, 1.0)
        _tv = np.array([v for v in timing_values if not np.isnan(v)], dtype=float)
        _flat_timing = _tv.size > 1 and (float(_tv.max()) - float(_tv.min())) <= 1.5
        ax1.set_ylim(0, v_max * (1.75 if _flat_timing else 1.40))
        
        # 设置投资时机y轴范围
        ax1_twin.set_ylim(base_year - 0.5, base_year + decision_period - 0.5)
        ax1_twin.set_yticks(np.arange(base_year, base_year + decision_period, 1))
        ax1_twin.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        
        # 添加数值标签。柱顶标签在左轴，折线标签在右轴，两者若恰好处于同一高度
        # 会互相压字（45Q 情景 60% 档即如此），因此加入碰撞检测。
        _lo_l, _hi_l = ax1.get_ylim()
        _lo_r, _hi_r = ax1_twin.get_ylim()

        def _frac(lo, hi, v):
            return (v - lo) / (hi - lo) if hi > lo else 0.0

        for i, (value, timing) in enumerate(zip(value_values, timing_values)):
            bar_y = np.nan
            if not np.isnan(value) and value > 0:
                bar_y = value + span * 0.02
                # 仅“一次性投资补贴”情景的 20% 档：柱顶数字整体上移一点点，
                # 避开与该档位过近的投资年份折线；其余情景与档位一律保持原样。
                if (scenario_key == 'investment_subsidy'
                        and round(float(policy_strengths[i])) == 20):
                    bar_y += span * 0.06
                ax1.text(i, bar_y, f'{value:.0f}',
                         ha='center', va='bottom', fontsize=10,
                         fontweight='bold', color='black')

            if not np.isnan(timing):
                timing_int = round(timing)
                collide = (
                    not np.isnan(bar_y)
                    and abs(_frac(_lo_l, _hi_l, bar_y)
                            - _frac(_lo_r, _hi_r, timing + 0.1)) < 0.045)
                if collide:
                    # 折线标签下移到标记点下方（不加任何底色，避免数字带背景框）
                    ax1_twin.text(i, timing - 0.12, f'{timing_int}',
                                  ha='center', va='top', fontsize=10,
                                  fontweight='bold', color='black')
                else:
                    ax1_twin.text(i, timing + 0.1, f'{timing_int}',
                                  ha='center', va='bottom', fontsize=10,
                                  fontweight='bold', color='black')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        # 图例位置自适应：若时机折线的最高点落在左半部分（左上角被 2035 等年份标签占用），
        # 图例放右上角；否则放左上角
        tv = np.array([v for v in timing_values if not np.isnan(v)], dtype=float)
        legend_loc = 'upper left'
        if tv.size:
            n_left = max(1, int(np.ceil(len(timing_values) * 0.5)))
            left_vals = [v for v in timing_values[:n_left] if not np.isnan(v)]
            if left_vals and max(left_vals) >= tv.max() - 1e-9:
                legend_loc = 'upper right'
        ax1.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc,
                   frameon=True, fontsize=10)
        
        # 添加网格
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # ============================
        # 子图2：投资年份分布折线图（右边）
        # ============================
        ax2.set_title('Distribution of investment years', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Investment year', fontsize=12)
        ax2.set_ylabel('Share of investment paths (%)', fontsize=12)
        
        # 准备投资年份
        all_years = list(range(base_year, base_year + decision_period))
        
        # 定义颜色和线型 - 6种不同的样式对应6个子情景
        colors = plt.cm.Set3(np.linspace(0, 1, len(policy_strengths)))
        line_styles = ['-', '--', '-.', ':', '-', '--']
        markers = ['o', 's', '^', 'D', 'v', '*']
        
        # 绘制每条政策强度的投资年份分布
        for i, (policy_strength, timing_dist) in enumerate(zip(policy_strengths, timing_distributions)):
            # 计算每个年份的投资比例
            year_ratios = []
            for year in all_years:
                count = timing_dist.get(year, 0)
                total_paths = sum(timing_dist.values())
                ratio = (count / total_paths * 100) if total_paths > 0 else 0
                year_ratios.append(ratio)
            
            # 绘制折线图
            ax2.plot(all_years, year_ratios, line_styles[i % len(line_styles)], 
                    color=colors[i], linewidth=2.5, marker=markers[i % len(markers)], 
                    markersize=6, label=f'strength {policy_strength:.0f}%')
        
        # 设置x轴
        ax2.set_xticks(all_years)
        ax2.set_xticklabels([str(year) for year in all_years], rotation=45, fontsize=10)
        
        # 设置y轴：按实际数据的最大值向上取整到十位（例如最大 44% → 上限 50%），
        # 下限固定为 0，保证所有投资年份占比都完整显示
        ratio_max = 0.0
        for timing_dist in timing_distributions:
            total_paths = sum(timing_dist.values()) if timing_dist else 0
            if total_paths > 0:
                ratio_max = max(ratio_max,
                                max(timing_dist.values()) / total_paths * 100)
        y_top = max(10, int(np.ceil(ratio_max / 10.0) * 10))
        ax2.set_ylim(0, y_top)
        y_step = 50
        for cand in (5, 10, 20, 25, 50):
            if y_top / cand <= 6:
                y_step = cand
                break
        ax2.set_yticks(np.arange(0, y_top + y_step, y_step))
        
        # 添加图例
        ax2.legend(title='Policy strength', loc='best', frameon=True, fontsize=10, ncol=2)
        
        # 添加网格和参考线
        ax2.grid(True, alpha=0.3, linestyle='--')
        for ref in (25, 50, 75):
            if ref < y_top:
                ax2.axhline(y=ref, color='gray' if ref == 50 else 'lightgray',
                            linestyle=':', alpha=0.5, linewidth=1)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        
        figure_name = scenario_data.get('figure_name')
        if not figure_name:
            figure_name = _slug(scenario_data['name'] + ' ' +
                                scenario_data['description']) + '.png'
        filename = FIGURE_DIR / figure_name
        fig.savefig(filename, dpi=1000, bbox_inches='tight')
        print(f' figure saved: {filename}')
        plt.show()
    
    def plot_all_scenarios_separately(self, scenario_results):
        """Plot the results of every scenario"""
        print("\n" + "=" * 70)
        print(" Generating the scenario figures")
        print("=" * 70)
        
        for scenario_key in scenario_results.keys():
            print(f"\n Generating: {scenario_results[scenario_key]['name']}")
            self.plot_individual_scenario(scenario_results, scenario_key)
        
        print("\n All scenario figures generated.")
    
    def plot_scenario_comparison(self, scenario_results):
        """Plot the scenario comparison"""
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        # 获取基准年份
        base_year = self.base_params.get('base_year', 2026)
        decision_period = self.base_params['t_v']
        
        plotted = 0
        for scenario_key, scenario_data in scenario_results.items():
            # 对比图只呈现四类单一政策情景；组合情景另有各自的柱状折线图
            if scenario_data.get('is_combination'):
                continue
            if plotted >= len(axes):
                break
            idx = plotted
            plotted += 1
                
            ax = axes[idx]
            sub_scenarios = scenario_data['sub_scenarios']
            
            policy_strengths = []
            timing_values = []
            
            for i, sub_scenario in enumerate(sub_scenarios):
                if 'error' not in sub_scenario:
                    policy_strength = self._calculate_policy_strength(scenario_key, sub_scenario['parameters'])
                    policy_strengths.append(policy_strength)
                    timing_values.append(sub_scenario.get('median_investment_year', np.nan))
            
            if policy_strengths:
                x = np.arange(len(policy_strengths))
                
                # 绘制最优投资时机
                bars = ax.bar(x, timing_values, width=0.6, color=plt.cm.Set3(idx/len(scenario_results)), 
                            alpha=0.8, edgecolor='black', linewidth=1)
                
                ax.set_xlabel(scenario_data['x_label'], fontsize=10)
                ax.set_ylabel('Optimal investment year', fontsize=10)
                
                ax.set_xticks(x)
                ax.set_xticklabels([f'{ps:.0f}%' for ps in policy_strengths])
                
                # 设置y轴范围
                ax.set_ylim(base_year - 0.5, base_year + decision_period - 0.5)
                ax.set_yticks(np.arange(base_year, base_year + decision_period, 2))
                
                # 添加数值标签
                for i, timing in enumerate(timing_values):
                    if not np.isnan(timing):
                        timing_int = round(timing)
                        ax.text(i, timing + 0.1, f'{timing_int}', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 隐藏多余的子图
        for idx in range(plotted, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Optimal investment timing across policy scenarios', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(FIGURE_DIR / 'scenario_comparison.png', dpi=1000, bbox_inches='tight')
        save_and_show('optimal_investment_timing_across_policy_scenarios')
    
    def _calculate_policy_strength(self, scenario_key, parameters):
        """Compute the policy strength in percent"""
        if scenario_key.startswith('combo_'):
            # 组合情景：统一强度轴（0% = 全部参与工具位于基准水平）
            return float(parameters.get('combo_level', 0.0)) * 100.0

        if scenario_key == 'carbon_trading':
            # 碳价增长百分比
            multiplier = parameters.get('Pc0_multiplier', 1.0)
            return (multiplier - 1.0) * 100
            
        elif scenario_key == 'investment_subsidy':
            # 投资补贴比例直接转换为百分比
            return parameters.get('lambda_', 0) * 100
            
        elif scenario_key == 'technology_incentive':
            # 技术补贴相对现行 45Q 全额抵免的强度
            # （0% = 无补贴，100% = 85 美元/吨 = 603.5 元/吨）
            multipliers = [parameters.get('S_EOR_multiplier', 1.0),
                           parameters.get('S_clean_multiplier', 1.0)]
            return float(np.mean(multipliers)) * 100

        elif scenario_key == 'tax_incentive':
            # 税收优惠百分比
            base_tax = self.base_params['tax']
            current_tax = parameters.get('tax', base_tax)
            tax_reduction = (base_tax - current_tax) / base_tax * 100
            return tax_reduction
            
        elif scenario_key == 'electricity_price':  # 新增
            # 上网电价增长百分比
            multiplier = parameters.get('Pe_multiplier', 1.0)
            return (multiplier - 1.0) * 100
            
        elif scenario_key == 'oil_price':  # 新增
            # 油价增长百分比
            multiplier = parameters.get('Poil0_multiplier', 1.0)
            return (multiplier - 1.0) * 100
        
        return 0
    
    def generate_scenario_report(self, scenario_results):
        """Generate the scenario report"""
        print("\n" + "=" * 70)
        print(" Scenario analysis report")
        print("=" * 70)
        
        # 获取基准年份
        base_year = self.base_params.get('base_year', 2026)
        decision_period = self.base_params['t_v']
        
        report_data = []
        
        for scenario_key, scenario_data in scenario_results.items():
            print(f"\n {scenario_data['name']}: {scenario_data['description']}")
            print("-" * 60)
            
            for i, sub_scenario in enumerate(scenario_data['sub_scenarios']):
                if 'error' not in sub_scenario:
                    timing = sub_scenario.get('optimal_timing_year', np.nan)
                    value = sub_scenario.get('optimal_value', 0) / 1e6
                    v0 = sub_scenario.get('option_value_at_decision', 0) / 1e6
                    mean_year = sub_scenario.get('mean_investment_year', np.nan)
                    median_year = sub_scenario.get('median_investment_year', np.nan)
                    ratio = sub_scenario.get('investment_ratio', 0) * 100
                    
                    # 计算政策强度
                    policy_strength = self._calculate_policy_strength(scenario_key, sub_scenario['parameters'])
                    
                    if not np.isnan(timing):
                        timing_int = round(timing)
                        print(f"  Policy strength {policy_strength:.0f}%: "
                              f"optimal year {timing_int}, "
                              f"investment value {value:.2f} million CNY, "
                              f"investment share {ratio:.1f}%")
                    else:
                        print(f"  Policy strength {policy_strength:.0f}%: "
                              f"No suitable investment window in {base_year}-{base_year+decision_period-1}, "
                              f"investment share {ratio:.1f}%")
                    
                    report_data.append({
                        'Policy type': scenario_data['name'],
                        'Policy strength (%)': f'{policy_strength:.0f}%',
                        'Optimal investment year': f'{round(timing)}' if not np.isnan(timing) else f'{base_year}-{base_year+decision_period-1} - no investment',
                        'Mean investment year': f'{mean_year:.1f}' if not np.isnan(mean_year) else '',
                        'Median investment year': f'{median_year:.0f}' if not np.isnan(median_year) else '',
                        'Option value at decision, V0 (million CNY)': f'{v0:.2f}',
                        'Value at exercise date, not comparable (million CNY)': f'{value:.2f}',
                        'Share of investing paths (%)': f'{ratio:.1f}%',
                        'Parameter change': str(sub_scenario['parameters'])
                    })
                else:
                    print(f"  Sub-scenario{i+1}: run error - {sub_scenario['error']}")
        
        # 创建DataFrame用于进一步分析
        df_report = pd.DataFrame(report_data)
        return df_report

# ---------------------------------------------------------------------------
# 初始投资成本 I0（元）：必须与 ccus_part1.py、ccus_part3.py 保持一致。
# CCUS-EOR 全流程增量投资口径：捕集系统 1.6e9 + 压缩输送管道 0.69e9
# + 驱油地面工程 ≈0.11e9 = 2.4e9 元（依据见 ccus_part1.py 与参数说明）。
# ---------------------------------------------------------------------------
I0_CNY = 2.4e9

# 模型参数
params = {
    # 技术参数
    'base_year': 2026, # 基准年份
    'IC': 600000,    # 电厂装机容量 (kW) - 600MW
    # 2025 年火电平均利用小时 4147 小时（国家能源局 2025 年全国电力工业统计）
    'RT': 4147,      # 年运行时间 (小时)
    'EF': 0.000762,    # 二氧化碳排放因子 (吨/kWh)
    'CE': 0.9,       # 二氧化碳捕集效率
    'phi': 0.94,     # 机组效率

    # 成本参数
            'I0': I0_CNY,
    'M0': 3.774e7,   # 初始运营维护成本 (元)
    'alpha': 0.0202, # 技术进步对投资成本影响参数
    'beta': 0.057,   # 技术进步对运营维护成本影响参数

    'UTC_CO2': 100,  # 二氧化碳单位运输成本 (元/吨)
    'USC_CO2': 50,   # 单位封存成本 (元/吨)
    'UUC_CO2': 106.5, # 单位驱油封存成本 (元/吨)

    # 技术效率参数
    'zeta': 0.975,   # 二氧化碳封存率
    'eta_loss': 0.32, # 发电效率损失因子

    # 电价模型参数（基于文献）
    'Pe0': 0.35,                      # 初始电价 (元/kWh) - 根据文献约0.45-0.6RMB/kWh
    'mu_e': np.log(0.6),              # 长期均衡电价的ln值 (ln(0.6))
    'alpha_e': 0.45564,               # 均值回复速度
    'sigma_e': 0.1,                   # 电价波动率
    'rho_e_c': 0.4,                   # 电价与碳价相关系数

    # 第一阶段增长参数
    'stage1_years': 5,                # 第一阶段年数
    'Pe_growth_rate': 0.024,          # 初始年增长率
    'Pe_growth_increment': 0.0012,    # 年增长率增量

    # 价格和补贴参数
    'S_clean': 0.019, # 清洁电价补贴 (元/kWh)
    # 45Q 驱油利用抵免：35 美元/吨 × 7.1 = 248.5 元/吨（见 ccus_part1.py 的参数注释）
    'S_EOR': 248.5,
    'claim_years': 12,   # 45Q 抵免可申领的运营年数（§45Q(a)(3)(A)）
    # 现行 45Q 法案已对地质封存与驱油利用路径并档，基准情景只计驱油利用抵免
    'pi': 0.0,        # 单位二氧化碳封存补贴 (元/吨)（封存抵免路径在情景中单列）

    # 财税参数
    'tax': 0.25,     # 企业所得税率
    'lambda_': 0,  # 投资补贴系数

    # 时间参数
    'T': 30,         # 电厂总寿命 (年)
    't_v': 10,       # 投资决策期 (年)
    't_c': 1,        # construction period (years) - Table 2

    # 金融参数
    'r_f': 0.0264,     # 无风险利率  *0.0264
    'beta_risk': 0.8, # 系统性风险系数
    'r_m_rf': 0.06,  # 系统性风险溢价

    # 不确定性参数
    'Pc0': 62.36,    # 初始碳价 (元/吨)  原：50.15   * 62.36
    'mu_c': 0.05,    # 碳价漂移率
    'sigma_c': 0.07, # 碳价波动率

    'Poil0': 3548, # 初始油价 (元/吨)   原：1816.6  *3548
    'mu_oil': 0.06,  # 油价漂移率
    'sigma_oil': 0.11, # 油价波动率

    # 模拟参数
    'M': 10000        # 模拟路径数
}

# 运行基准模型
print("=" * 70)
print(" Initialising the baseline CCUS model...")
print("=" * 70)
model_baseline = CCUSInvestmentModel(params)
results_baseline = model_baseline.lsmc_solver()

print("\n" + "=" * 70)
print(" Baseline model results")
print("=" * 70)
if not np.isnan(results_baseline['optimal_timing']):
    print(f" optimal investment year (median): "
          f"{results_baseline['optimal_timing_median_year']:.0f}")
    print(f" optimal investment year (modal):  "
          f"{results_baseline['optimal_timing_modal_year']} "
          f"({results_baseline['optimal_timing_modal_share']:.1f}% of investing paths)")
else:
    print("  Optimal investment year: none within the decision window")
print(f" investment path share: {results_baseline['investment_ratio']:.1%}")
print(f" option value at the decision point (V0): "
      f"{np.mean(results_baseline['TIV'][:, 0])/1e6:.2f} million CNY")
print(f" invest-now NPV at the decision point: "
      f"{np.mean(results_baseline['NPV'][:, 0])/1e6:.2f} million CNY")

# 运行情景分析
print("\n" + "=" * 70)
print(" Starting the policy scenario analysis")
print("=" * 70)

analyzer = ScenarioAnalyzer(params)
scenarios = analyzer.create_scenarios()

# 运行所有情景分析
# Same number of paths as the baseline run so that the 0% rows reproduce the
# baseline of ccus_part1.py exactly.
M_SCENARIO = 300 if QUICK_TEST else 10000
scenario_results = analyzer.run_scenario_analysis(scenarios, M_scenario=M_SCENARIO)

# 生成详细报告
df_report = analyzer.generate_scenario_report(scenario_results)

base_year_reported = params['base_year']

# 保存结果（先落盘，再绘图：绘图失败也不会丢失 CSV）
try:
    df_report.to_csv('ccus_scenario_results.csv', index=False, encoding='utf-8-sig')
    print("\n Detailed results saved to: ccus_scenario_results.csv")

    # 各情景各强度下的投资年份分布（论文表 5 的时机对比用）
    import csv as _csv
    with open('ccus_scenario_timing_distribution.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        writer = _csv.writer(fh)
        writer.writerow(['Policy type', 'Policy strength (%)', 'Investment year',
                         'Share of investing paths (%)'])
        for scenario_key, scenario_data in scenario_results.items():
            for sub in scenario_data['sub_scenarios']:
                if 'error' in sub:
                    continue
                strength = analyzer._calculate_policy_strength(scenario_key,
                                                               sub['parameters'])
                dist = sub.get('timing_distribution', {}) or {}
                total = sum(dist.values()) or 1
                for year in range(base_year_reported, base_year_reported + params['t_v']):
                    writer.writerow([scenario_data['name'], f'{strength:.0f}%', year,
                                     round(dist.get(year, 0) / total * 100, 2)])
    print(" Timing distributions saved to ccus_scenario_timing_distribution.csv")
    
    # 保存汇总统计
    summary_stats = df_report.groupby('Policy type').agg({
        'Option value at decision, V0 (million CNY)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'Mean investment year': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'Median investment year': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'Share of investing paths (%)': lambda x: pd.to_numeric(x.str.replace('%', ''), errors='coerce').mean()
    }).round(2)
    
    summary_stats.to_csv('ccus_scenario_summary.csv', encoding='utf-8-sig')
    print(" Summary statistics saved to: ccus_scenario_summary.csv")
    
except Exception as e:
    print(f"\n Error while saving results: {e}")

# 分别绘制各个情景的图表
analyzer.plot_all_scenarios_separately(scenario_results)

# 绘制情景对比图
print("\n" + "=" * 70)
print(" Plotting the scenario comparison")
print("=" * 70)
analyzer.plot_scenario_comparison(scenario_results)

print("\n" + "=" * 70)
print(" Scenario analysis completed")
print("=" * 70)
print(" Summary of policy scenarios:")
print("1. Carbon trading: raises the carbon price and shifts investment timing")
print("2. One-off investment subsidy: lowers the initial investment cost")
print("3. Technology incentives: raise subsidy revenue during operation") 
print("4. Tax incentives: lower the tax burden and raise net returns")
print("5. On-grid electricity price: raises generation revenue")
print("6. Initial oil price: affects EOR revenue")
if INCLUDE_COMBINATION_SCENARIOS:
    print(f"7. Combined policy scenarios: {len(COMBINATION_DEGREES)}-way packages "
          f"({sum(1 for k in scenario_results if k.startswith('combo_'))} scenarios, "
          f"each at policy intensity {COMBINATION_LEVELS})")
