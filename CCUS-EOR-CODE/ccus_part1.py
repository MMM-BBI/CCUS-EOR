import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Set CCUS_QUICK_TEST=1 to run the whole pipeline with 300 paths (fast smoke test).
QUICK_TEST = bool(os.environ.get("CCUS_QUICK_TEST"))

# Single random seed shared by all three scripts so that the baseline, the
# policy scenarios and the sensitivity runs are directly comparable
# (common random numbers).
SEED = 42

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
        # 12 年"内申领（IRS Form 8933 (12/2025) 同），即抵免仅覆盖投产后前 12 个运营年，
        # 而非电厂全寿命期。该限制同样适用于驱油利用（tertiary injectant）路径。
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

        The random innovation of stage two is correlated with the CARBON price
        innovation with coefficient rho_e_c, as assumed in the paper: a carbon
        price shock is passed through to the electricity price by marginal
        pricing.  The carbon shocks are handed over by simulate_uncertainties()
        so that the two processes really share the same random driver.
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
            
            # EOR补贴：45Q 仅对设备投用日起 12 个运营年内的合格二氧化碳计提
            if operation_year <= self.claim_years:
                R_EOR = self.S_EOR * self.Q_EOR
            else:
                R_EOR = 0.0
            
            # 封存补贴（若启用）：与驱油利用抵免一样，45Q 抵免同样只能在
            # 设备投用日起 12 年内申领，故用同一 claim_years 限制
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

        碳价与油价服从几何布朗运动，E[P_s | P_t] = P_t·exp(mu·(s-t))；
        电价在增长段为确定性路径，在均值回复段满足对数正态的线性递推
        （E[Pe_k] = exp(m_k + 0.5·v_k)，m、v 分别为 ln Pe 的条件均值与方差）。
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
        """式(25)的投资期望值：在决策时点 t、仅依据当期价格状态，立即投资的期望净现值。

        这是投资者在 t 时刻**真正可以计算**的价值（不使用未来信息），
        与 calculate_project_value() 的已实现路径价值不同。
        """
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
        
        # 预计算两个价值矩阵：
        #   NPV(j,t)：沿路径 j 的已实现净现值，只用于**评价**决策结果；
        #   EX(j,t) ：在时点 t 仅依据当期价格状态立即投资的期望价值（式(25)），
        #             是投资者在 t 时刻真正掌握的信息，决策只能用它。
        # 说明：EX 不含未来信息，因此同一时点相同价格状态下的决策必然一致
        #（例如 t=0 时所有路径的初始价格相同，决策结果完全相同）。
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
                        immediate_value = EX[path_idx, t]   # 决策依据：当期信息下的期望价值
                        realized_value = NPV[path_idx, t]   # 若执行，实际得到的是该路径的实现值
                        cont_value = continuation_estimates[idx]
                        
                        # 决策规则
                        if immediate_value > 0 and immediate_value >= cont_value:
                            # NPV>0 且 TIV = NPV (立即执行价值 >= 继续持有价值)
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

# ---------------------------------------------------------------------------
# 初始投资成本 I0（元）——CCUS-EOR 全流程增量投资口径
#
# 分项构成（详见参数说明）：
#   ① 燃烧后捕集系统        1.6e9 元   600 MW 机组，文献值（林伯强团队 / 川大博士论文表 4-4）
#   ② CO2 压缩与输送管道    0.69e9 元  12 英寸管道 8.64 百万元/公里 × 80 公里
#                                      （川大博士论文表 5-6；管道长度的现实性参照齐鲁石化—
#                                        胜利油田项目 109 公里的百万吨级输送管道）
#   ③ 驱油地面工程（注入井、集输与循环系统）
#                           ≈0.11e9 元 50～60 元/吨 CO2（川大博士论文表 5-5，
#                                      引自《中国 CCUS 年度报告》与 ADB 2022 路线图）
#   合计 ≈ 2.4e9 元，单位投资强度约 1500 元/(吨 CO2·年)，处于中国已投运示范项目
#   （国能锦界 400 万吨/年项目 19.9 亿元，约 500 元/(吨·年)）与国外首台套项目
#   （Petra Nova 约 10 亿美元/1.4 Mt，约 5000 元/(吨·年)）之间。
# 三个脚本必须保持一致，表 2 同步更新并标注出处。
# ---------------------------------------------------------------------------
I0_CNY = 2.4e9

# 模型参数
params = {
    # 技术参数
    'base_year': 2026, # 基准年份
    'IC': 600000,    # 电厂装机容量 (kW) - 600MW
    # 年运行时间：2025 年全国 6000 千瓦及以上电厂发电设备平均利用小时为 3119 小时，
    # 其中火电为 4147 小时（国家能源局 2025 年全国电力工业统计）。本文算例为煤电机组，
    # 故取火电口径 4147 小时，而非全口径平均的 3119 小时。
    'RT': 4147,      # 年运行时间 (小时)
    'EF': 0.000762,    # 二氧化碳排放因子 (吨/kWh)
    'CE': 0.9,       # 二氧化碳捕集效率
    'phi': 0.94,     # 机组效率
    
    # 成本参数
    # Initial investment cost (CNY). Documented value from Refs. [41,66,67];
    # replace I0_CNY above once a source gives the full-chain project total.
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
    # EOR补贴（45Q 驱油利用抵免）：35 美元/吨 × 7.1（论文表注的美元折算汇率）
    # = 248.5 元/吨。基准情景取 45Q 驱油利用抵免的基准水平；技术激励情景的政策
    # 上限取现行法案的全额抵免 85 美元/吨 = 603.5 元/吨（见 ccus_part2senario.py）。
    #
    # 现行法案要点（Pub. L. 119-21, 2025-07-04, §70522；IRS Form 8933 (12/2025)）：
    #   ① 并档：2025 年 7 月 4 日之后投用的设施，安全地质封存（line 1g）与驱油利用/
    #      利用（lines 2g、3i）适用同一“适用美元金额”（基础额 17 美元/吨，2026 年后
    #      随通胀调整），此前 17/12 美元的差异化双轨已废止；
    #   ② 加成：满足现行工资与学徒制要求的项目，抵免额乘以 5，即 85 美元/吨；
    #   ③ 申领期：抵免只能在设备原始投用日起 12 年内申领（见 claim_years）。
    # 本项目捕集的 CO2 全部用于驱油，基准情景只计驱油利用抵免（S_EOR），
    # 封存抵免（pi）在基准中取 0，仅在“封存抵免路径”情景中作为替代方案考察。
    'S_EOR': 248.5,
    'claim_years': 12,  # 45Q 抵免可申领的运营年数（26 U.S.C. §45Q(a)(3)(A)）
    'pi': 0.0,        # 单位二氧化碳封存补贴 (元/吨)（基准情景不适用，见情景分析）
    
    # 财税参数
    'tax': 0.25,     # 企业所得税率
    'lambda_': 0,  # 投资补贴系数
    
    # 时间参数
    'T': 30,         # 电厂总寿命 (年)
    't_v': 10,       # 投资决策期 (年)
    't_c': 1,        # 建设期 (年)
    
    # 金融参数
    'r_f': 0.0264,     # 无风险利率（更新）
    'beta_risk': 0.8, # 系统性风险系数
    'r_m_rf': 0.06,  # 系统性风险溢价
    
    # 不确定性参数
    'Pc0': 62.36,    # 初始碳价 (元/吨)（更新）
    'mu_c': 0.05,    # 碳价漂移率
    'sigma_c': 0.07, # 碳价波动率
    
    'Poil0': 3548, # 初始油价 (元/吨)（更新）
    'mu_oil': 0.06,  # 油价漂移率
    'sigma_oil': 0.11, # 油价波动率
    
    # 模拟参数
    'M': 10000        # 模拟路径数
}

# 首先分析CE_replace的变化趋势
print("Oil displacement efficiency over time:")
years = np.arange(1, 41)  # 1到40年
ce_replace_values = []

model_temp = CCUSInvestmentModel(params)
for year in years:
    ce_replace = model_temp.calculate_CE_replace(year)
    ce_replace_values.append(ce_replace)
    print(f"  Year {year}: CE_replace = {ce_replace:.4f}")

# 分析运营期变化
print("\nOperating period implied by each investment year:")
for t in range(params['t_v']):
    op_period = model_temp.calculate_operating_period(t)
    print(f"  Investment year {t}: operating period = {op_period} years (plant life {params['T']} - investment year {t} - construction {params['t_c']})")

# 可视化CE_replace变化
plt.figure(figsize=(10, 6))
plt.plot(years, ce_replace_values, '-o', color='#376B9E', linewidth=2, markersize=4)
plt.xlabel('Operating year')
plt.ylabel('CE_replace')
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 41, 2))  # 从0到24，步长为2
plt.axvline(x=8, color='#E5B5B5', linestyle='--', alpha=0.9, label='Stage boundary t = 8')
plt.axvline(x=10, color='#B22222', linestyle='--', alpha=0.9, label='Stage boundary t = 10')
plt.legend()
save_and_show('oil_displacement_efficiency_ce_replace_over_operating_years')

# 电价分析图表
def plot_electricity_price_analysis(Pe_paths, Pc_paths, params):
    """Electricity price analysis"""
    total_years = params['T']
    years_range = range(params['base_year'], params['base_year'] + total_years)
    # 用实际模拟出的路径数（QUICK_TEST 时 model.M 会被降到 300）
    sample_paths = min(50, Pe_paths.shape[0])
    
    # 1. 电价路径图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for j in range(sample_paths):
        plt.plot(years_range, Pe_paths[j, :], alpha=0.1, color='green', linewidth=0.8)
    mean_pe = np.mean(Pe_paths, axis=0)
    plt.plot(years_range, mean_pe, 'g-', linewidth=2, label='Mean path')
    plt.axvline(x=params['base_year'] + params['stage1_years'], 
               color='red', linestyle='--', alpha=0.7, label='Stage transition')
    plt.text(0.02, 0.97, 'a', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    plt.xlabel('Year')
    plt.ylabel('Electricity price (CNY/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 电价与碳价相关性
    plt.subplot(2, 2, 2)
    n_sim_paths = Pc_paths.shape[0]
    sample_size = min(500, n_sim_paths)
    sample_indices = np.random.choice(n_sim_paths, sample_size, replace=False)
    
    # 取最后一年价格观察相关性
    pc_final = Pc_paths[sample_indices, -1]
    pe_final = Pe_paths[sample_indices, -1]
    
    plt.scatter(pc_final, pe_final, alpha=0.6, s=20, color='purple')
    plt.text(0.02, 0.97, 'b', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    plt.xlabel('Carbon price (CNY/t)')
    plt.ylabel('Electricity price (CNY/kWh)')
    plt.grid(True, alpha=0.3)
    
    # 计算并显示相关系数
    correlation = np.corrcoef(pc_final, pe_final)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 电价分位数
    plt.subplot(2, 2, 3)
    pe_quantiles = np.percentile(Pe_paths, [10, 50, 90], axis=0)
    plt.plot(years_range, pe_quantiles[1], 'g-', label='Median', linewidth=2)
    plt.fill_between(years_range, pe_quantiles[0], pe_quantiles[2], 
                     alpha=0.3, color='green', label='10th-90th percentile')
    plt.axvline(x=params['base_year'] + params['stage1_years'], 
               color='red', linestyle='--', alpha=0.7, label='Stage transition')
    plt.xlabel('Year')
    plt.ylabel('Electricity price (CNY/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 电价增长率
    plt.subplot(2, 2, 4)
    pe_growth = np.diff(mean_pe) / mean_pe[:-1]
    plt.plot(years_range[1:], pe_growth * 100, 'b-', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=params['base_year'] + params['stage1_years'], 
               color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('Growth rate (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
        # 该图未在论文中使用，已在最终版中停止输出
    
    # 打印电价统计信息
    print("\n=== Electricity price statistics ===")
    print(f"Initial electricity price: {params['Pe0']:.3f} CNY/kWh")
    print(f"Long-run equilibrium price: {np.exp(params['mu_e']):.3f} CNY/kWh")
    print(f"Electricity price in the final year:")
    print(f"  Mean: {np.mean(Pe_paths[:, -1]):.3f} CNY/kWh")
    print(f"  Standard deviation: {np.std(Pe_paths[:, -1]):.3f} CNY/kWh")
    print(f"  10th percentile: {np.percentile(Pe_paths[:, -1], 10):.3f} CNY/kWh")
    print(f"  90th percentile: {np.percentile(Pe_paths[:, -1], 90):.3f} CNY/kWh")
    print(f"  Correlation with the carbon price: {correlation:.3f}")

# 运行模型
print("Initialising the CCUS investment decision model...")
model_corrected = CCUSInvestmentModel(params)
results_corrected = model_corrected.lsmc_solver()

print("\n=== Baseline model results ===")
if not np.isnan(results_corrected['optimal_timing']):
    print(f"Optimal investment year (median): {results_corrected['optimal_timing_median_year']:.0f}")
    print(f"Optimal investment year (modal):  {results_corrected['optimal_timing_modal_year']:.0f} "
          f"({results_corrected['optimal_timing_modal_share']:.1f}% of investing paths)")
    print(f"Optimal investment year (mean):   {results_corrected['optimal_timing_year']:.0f}")
    v0_print = float(np.mean(results_corrected['TIV'][:, 0])) / 1e6
    npv0_print = float(np.mean(results_corrected['NPV'][:, 0])) / 1e6
    print(f"Option value at the decision point V0: {v0_print:.2f} million CNY")
    print(f"Invest-now NPV at the decision point:  {npv0_print:.2f} million CNY "
          f"(deferral premium {(v0_print - npv0_print) / v0_print * 100:.1f}%)")
    print(f"Value at the exercise date: {results_corrected['optimal_value']/1e6:.2f} million CNY "
          f"(not comparable across scenarios)")
    # 显示对应的运营期
    op_period = model_corrected.calculate_operating_period(int(results_corrected['optimal_timing_median']))
    print(f"operating period: {op_period} years")
else:
    print("Optimal investment year: none within the decision window")
print(f"Share of investment paths: {results_corrected['investment_ratio']:.1%}")
# 电价分析
print("\n=== Electricity price model ===")
if hasattr(model_corrected, 'Pe_paths') and model_corrected.Pe_paths is not None:
    plot_electricity_price_analysis(
        model_corrected.Pe_paths, 
        results_corrected['Pc_paths'], 
        params
    )

# 详细分析投资时机分布
print("\n=== Distribution of investment timing ===")
timing_by_path = results_corrected['optimal_timing_by_path']
valid_timings = [t for t in timing_by_path if t < params['t_v']]

if valid_timings:
    timing_array = np.array(valid_timings)
    print(f"Investment timing statistics:")
    print(f"  Earliest investment: {2026 + timing_array.min():.0f}")
    print(f"  Latest investment: {2026 + timing_array.max():.0f}")
    print(f"  Mean investment year: {2026 + timing_array.mean():.1f}")
    print(f"  Median investment year: {2026 + np.median(timing_array):.0f}")
    
    # 投资年份分布
    year_counts = {}
    for timing in valid_timings:
        year = 2026 + int(timing)
        year_counts[year] = year_counts.get(year, 0) + 1
    
    print("  Year distribution:")
    for year in sorted(year_counts.keys()):
        percentage = year_counts[year] / len(valid_timings) * 100
        op_period = model_corrected.calculate_operating_period(year - 2026)
        print(f"    {year} year: {year_counts[year]} paths ({percentage:.1f}%), operating period {op_period} years")
else:
    print("No valid investment path")


# ---------------------------------------------------------------------------
# Export the key numbers to a CSV so the long console log can be ignored.
# Everything the manuscript quotes (Table 2 derived quantities, sections 6.1,
# abstract and conclusions) is written to ccus_part1_results.csv.
# ---------------------------------------------------------------------------
def write_results_csv(model, results, params, ce_years, ce_values,
                      path="ccus_part1_results.csv"):
    """Save the key numbers of the baseline run as item/value/unit rows."""
    import csv

    rows = []

    def add(item, value, unit="", note=""):
        if isinstance(value, float):
            value = round(value, 6)
        rows.append([item, value, unit, note])

    # 1) quantities derived from the parameters (Table 2)
    add("Q_e_annual_generation", model.Q_e, "kWh", "IC x RT x phi")
    add("Q_CO2_annual_capture", model.Q_CO2, "t", "IC x RT x EF x CE x phi")
    add("Q_EOR_annual_injection", model.Q_EOR, "t", "= Q_CO2")
    add("Q_loss_annual_generation_loss", model.Q_loss, "kWh", "eta_loss x phi x IC x RT")
    add("r_risk_adjusted_discount_rate", model.r, "-", "r_f + beta_risk x (r_m - r_f)")
    add("simulation_paths", model.M, "-", "paths used in this run")

    # 2) baseline decision (abstract, section 6.1, conclusions)
    # Headline value metric: the option value measured at the decision point
    # (2026 money). It is the only value that is comparable across scenarios
    # with different optimal timing.
    v0 = float(np.mean(results["TIV"][:, 0])) / 1e6
    npv0 = float(np.mean(results["NPV"][:, 0])) / 1e6
    add("option_value_at_decision_V0", v0, "million CNY",
        "headline value metric: mean TIV in the first decision year (2026 money)")
    add("invest_now_NPV_at_decision", npv0, "million CNY",
        "mean NPV if the project were executed in the first decision year")
    add("deferral_premium", v0 - npv0, "million CNY", "V0 minus invest-now NPV")
    add("deferral_premium_pct", (v0 - npv0) / v0 * 100, "%",
        "value of the option to wait, relative to V0")
    add("optimal_investment_year_median", results["optimal_timing_median_year"], "year",
        "median first-exercise year across paths (headline timing indicator)")
    add("optimal_investment_year_mean", results["optimal_timing_year"], "year",
        "mean first-exercise year across paths")
    add("optimal_investment_year_modal", results["optimal_timing_modal_year"], "year",
        "most frequent first-exercise year across paths")
    add("optimal_investment_year_modal_share", results["optimal_timing_modal_share"], "%",
        "share of investing paths that pick the modal year")
    add("optimal_investment_value_at_exercise", results["optimal_value"] / 1e6,
        "million CNY", "value at the exercise date; NOT comparable across scenarios")
    add("investment_path_share", results["investment_ratio"] * 100, "%",
        "share of paths that invest inside the decision window")
    if not np.isnan(results["optimal_timing"]):
        add("operating_period_at_optimum",
            model.calculate_operating_period(int(results["optimal_timing"])), "years", "")

    # 3) distribution of the optimal investment year across paths
    valid = np.array([t for t in results["optimal_timing_by_path"] if t < params["t_v"]],
                     dtype=float)
    if valid.size:
        add("earliest_investment_year", params["base_year"] + valid.min(), "year", "")
        add("latest_investment_year", params["base_year"] + valid.max(), "year", "")
        add("mean_investment_year", params["base_year"] + valid.mean(), "year", "")
        add("median_investment_year", params["base_year"] + np.median(valid), "year", "")
    for year in sorted(results["timing_distribution"]):
        share = results["timing_distribution"][year] / max(1, valid.size) * 100
        add("timing_share_%d" % year, share, "%", "share of investing paths")

    # 4) NPV, TIV and the deferral option value (section 6.1.3)
    all_tiv = results["TIV"].flatten()
    pos_tiv = all_tiv[all_tiv > 0]
    npv = results["NPV"].flatten()
    pos_npv = npv[npv > 0]
    option = (results["TIV"] - results["NPV"]).flatten()
    pos_option = option[option > 1e-6]
    if pos_npv.size:
        add("NPV_mean_positive", pos_npv.mean() / 1e6, "million CNY", "")
        add("NPV_max", pos_npv.max() / 1e6, "million CNY", "")
    if pos_tiv.size:
        add("TIV_mean_positive", pos_tiv.mean() / 1e6, "million CNY", "")
        add("TIV_max", pos_tiv.max() / 1e6, "million CNY", "")
        add("TIV_share_positive", pos_tiv.size / all_tiv.size * 100, "%", "")
    if pos_option.size:
        add("option_value_mean_positive", pos_option.mean() / 1e6, "million CNY", "")
        add("option_value_max", pos_option.max() / 1e6, "million CNY", "")
        add("option_value_share_positive", pos_option.size / option.size * 100, "%", "")
    mask = option > 1e-6
    if mask.sum() > 2 and np.std(option[mask]) > 0:
        add("corr_option_value_NPV", np.corrcoef(npv[mask], option[mask])[0, 1], "-",
            "dispersion of the deferral option value")

    # 5) simulated price paths year by year (section 6.1.1)
    Pc = results["Pc_paths"]
    Poil = results["Poil_paths"]
    Pe = getattr(model, "Pe_paths", None)
    for idx in range(Pc.shape[1]):
        year = params["base_year"] + idx
        add("carbon_price_mean_%d" % year, Pc[:, idx].mean(), "CNY/t", "")
        add("carbon_price_std_%d" % year, Pc[:, idx].std(), "CNY/t", "")
        if idx < Poil.shape[1]:
            add("oil_price_mean_%d" % year, Poil[:, idx].mean(), "CNY/t", "")
            add("oil_price_std_%d" % year, Poil[:, idx].std(), "CNY/t", "")
        if Pe is not None and idx < Pe.shape[1]:
            add("electricity_price_mean_%d" % year, Pe[:, idx].mean(), "CNY/kWh", "")
    if Pe is not None:
        add("electricity_price_final_mean", Pe[:, -1].mean(), "CNY/kWh", "")
        add("electricity_price_final_p10", np.percentile(Pe[:, -1], 10), "CNY/kWh", "")
        add("electricity_price_final_p90", np.percentile(Pe[:, -1], 90), "CNY/kWh", "")
        add("corr_electricity_carbon_final", np.corrcoef(Pe[:, -1], Pc[:, -1])[0, 1], "-", "")

    # 6) oil displacement efficiency (section 6.1.2)
    for year, value in zip(ce_years, ce_values):
        add("CE_replace_year_%d" % year, value, "-", "")
    if len(ce_values):
        peak = int(np.argmax(ce_values))
        add("CE_replace_peak_operating_year", int(ce_years[peak]), "operating year", "")
        add("CE_replace_peak_value", ce_values[peak], "-", "")

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "value", "unit", "note"])
        writer.writerows(rows)
    print("[results] %d key numbers written to %s" % (len(rows), path))


write_results_csv(model_corrected, results_corrected, params, years, ce_replace_values)


def write_exercise_boundary_csv(model, results, params,
                                path="ccus_exercise_boundary.csv"):
    """Descriptive statistics of the exercise pattern plus the break-even oil price.

    For every decision year the table reports
      * how many paths decide to invest in that year,
      * the median oil price of those paths and of the paths that keep waiting
        (descriptive statistics of the realised paths), and
      * the **break-even oil price**: the oil price at which the expected value of
        investing immediately at that decision point equals zero,
        i.e. E[NPV | invest at t, P_oil = P*] = 0, solved by bisection with the
        carbon and electricity prices fixed at their cross-sectional medians.

    The break-even price answers "how high must the oil price be for investing
    now to be worthwhile at all".  It is *not* the optimal-stopping boundary:
    it ignores the value of waiting, which is why it must be read together with
    the LSMC optimal timing (median optimal investment year).
    """
    import csv

    N = params["t_v"]
    first = np.array([t if t < N else -1
                      for t in results["optimal_timing_by_path"]])
    Poil = results["Poil_paths"]
    Pc = results["Pc_paths"]
    Pe = getattr(model, "Pe_paths", None)

    def break_even(t, Pc_t, Pe_t, lo=200.0, hi=40000.0):
        """Oil price at which the expected value of immediate investment is 0.

        expected_project_value() is strictly increasing in the oil price, so a
        bisection on the sign of the value is well defined.
        """
        f_lo = model.expected_project_value(t, Pc_t, lo, Pe_t)
        f_hi = model.expected_project_value(t, Pc_t, hi, Pe_t)
        if f_lo > 0 or f_hi < 0:          # no sign change inside the interval
            return float("nan")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if model.expected_project_value(t, Pc_t, mid, Pe_t) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    rows = []
    for t in range(N):
        invest = first == t
        wait = first > t
        if invest.sum() < 5:
            continue
        price_t = Poil[:, t]
        pc_t = float(np.median(Pc[:, t]))
        pe_t = float(np.median(Pe[:, t])) if Pe is not None else params["Pe0"]
        be = break_even(t, pc_t, pe_t)
        rows.append([
            t + 1, params["base_year"] + t,
            round(invest.mean() * 100, 2),
            round(float(np.median(Poil[invest, t])), 1),
            round(float(np.median(Poil[wait, t])), 1) if wait.sum() else "",
            round(be, 1) if not np.isnan(be) else "",
            round(be / params["Poil0"], 2) if not np.isnan(be) else "",
        ])

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["decision_index", "year", "share_investing_%",
                         "median_oil_price_of_investing_paths_CNY_t",
                         "median_oil_price_of_waiting_paths_CNY_t",
                         "break_even_oil_price_CNY_t",
                         "break_even_relative_to_initial_oil_price"])
        writer.writerows(rows)
    print("[results] exercise pattern and break-even oil price written to %s" % path)


write_exercise_boundary_csv(model_corrected, results_corrected, params)


def write_cashflow_breakdown_csv(model, results, params,
                                 path="ccus_cashflow_breakdown.csv"):
    """Annual cash-flow decomposition at the optimal investment year.

    The table is built along the expected (mean) simulated price paths, so it
    describes the structure of the project cash flow rather than one random
    path. It is the evidence base for the revenue-window discussion in
    section 6.1.
    """
    import csv

    t_star = results.get("optimal_timing")
    t_star = 0 if (t_star is None or np.isnan(t_star)) else int(t_star)

    Pc = results["Pc_paths"].mean(axis=0)
    Poil = results["Poil_paths"].mean(axis=0)
    Pe = model.Pe_paths.mean(axis=0)

    op_years = model.calculate_operating_period(t_star)
    rows = []

    for i in range(1, op_years + 1):
        idx = min(t_star + model.t_c + i - 1, len(Pc) - 1)
        ce = model.calculate_CE_replace(i)
        oil_t = ce * model.Q_EOR

        r_carbon = Pc[idx] * model.Q_CO2
        r_clean = model.S_clean * model.Q_e
        r_oil = Poil[idx] * oil_t
        # 45Q 抵免只在投产后前 claim_years 个运营年计提
        r_eor = (model.S_EOR * model.Q_EOR
                 if i <= getattr(model, "claim_years", 12) else 0.0)
        # 封存补贴同样受 12 年申领期限制（π=0 时该行为 0，不影响基准结果）
        r_store = (model.pi * model.zeta * model.Q_CO2
                   if i <= getattr(model, "claim_years", 12) else 0.0)
        revenue = r_carbon + r_clean + r_oil + r_eor + r_store

        c_transport = model.Q_CO2 * model.UTC_CO2
        c_utilise = model.zeta * model.Q_CO2 * model.UUC_CO2
        c_store = model.Q_CO2 * model.USC_CO2
        c_loss = Pe[idx] * model.Q_loss
        c_om = model.M0 * np.exp(-model.beta * (t_star + model.t_c + i))
        cost = c_transport + c_utilise + c_store + c_loss + c_om

        cf_pre = revenue - cost
        cf_after = cf_pre * (1 - model.tax)
        discount = np.exp(-model.r * (model.t_c + i))

        rows.append([
            i, params["base_year"] + t_star + model.t_c + i - 1, round(ce, 6),
            round(oil_t, 2), round(r_carbon / 1e6, 2), round(r_clean / 1e6, 2),
            round(r_oil / 1e6, 2), round(r_eor / 1e6, 2), round(r_store / 1e6, 2),
            round(revenue / 1e6, 2), round(c_transport / 1e6, 2),
            round(c_utilise / 1e6, 2), round(c_store / 1e6, 2),
            round(c_loss / 1e6, 2), round(c_om / 1e6, 2), round(cost / 1e6, 2),
            round(cf_pre / 1e6, 2), round(cf_after / 1e6, 2),
            round(discount, 6), round(cf_after * discount / 1e6, 2),
        ])

    header = ["operating_year", "calendar_year", "CE_replace",
              "oil_production_t", "carbon_revenue", "clean_electricity_subsidy",
              "oil_revenue", "EOR_subsidy", "storage_subsidy", "total_revenue",
              "transport_cost", "utilization_cost", "storage_cost",
              "generation_loss_cost", "om_cost", "total_cost",
              "pretax_cashflow", "after_tax_cashflow", "discount_factor",
              "discounted_cashflow"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["all monetary values in million CNY; investment year "
                         "%d; prices follow the mean simulated paths"
                         % (params["base_year"] + t_star)])
    print("[results] annual cash-flow decomposition written to %s" % path)


write_cashflow_breakdown_csv(model_corrected, results_corrected, params)

# 绘制结果图表
def plot_corrected_results(results, params):
    """Plot model results"""
    
    # 1. 绘制碳价和油价并列图
    plt.figure(figsize=(15, 6))
    
    total_years = params['T']  # 35年
    years_range = range(2026, 2026 + total_years)
    # 用实际模拟出的路径数（QUICK_TEST 时 model.M 会被降到 300）
    sample_paths = min(50, results['Pc_paths'].shape[0])
    
    # 碳价图
    plt.subplot(1, 2, 1)
    for j in range(sample_paths):
        plt.plot(years_range, results['Pc_paths'][j, :], 
                 alpha=0.35, color='#AFC3D8', linewidth=0.8)
    mean_pc = np.mean(results['Pc_paths'], axis=0)
    plt.plot(years_range, mean_pc, '-', color='#376B9E', linewidth=2, label='Mean path')
    plt.text(0.02, 0.97, 'a', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    plt.xlabel('Year')
    plt.ylabel('Carbon price (CNY/t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 油价图
    plt.subplot(1, 2, 2)
    for j in range(sample_paths):
        plt.plot(years_range, results['Poil_paths'][j, :], 
                 alpha=0.35, color='#E5B5B5', linewidth=0.8)
    mean_oil = np.mean(results['Poil_paths'], axis=0)
    plt.plot(years_range, mean_oil, '-', color='#B22222', linewidth=2, label='Mean path')
    plt.text(0.02, 0.97, 'b', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    plt.xlabel('Year')
    plt.ylabel('Oil price (CNY/t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show('a_simulated_carbon_price_paths_full_project_cycle')
    
    # 2. 投资时机分布图
    plt.figure(figsize=(10, 6))
    timing_by_path = results['optimal_timing_by_path']
    valid_timings = [2026 + t for t in timing_by_path if t < params['t_v']]
    
    if valid_timings:
        plt.hist(valid_timings, bins=range(2026, 2037), alpha=0.7, 
                edgecolor='black', color='#4E79A7')
        plt.xlabel('Investment year')
        plt.ylabel('Number of paths')
# plt.grid(True, alpha=0.3)
        plt.xlim(2025.5, params['base_year'] + params['t_v'] - 0.5)
        
        # 添加统计信息
        avg_year = np.mean(valid_timings)
        plt.axvline(x=avg_year, color='#CC5A6A', linestyle='--', linewidth=2, 
                   label=f'Mean investment year: {avg_year:.1f}')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No investment within the decision window', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    save_and_show('c_distribution_of_optimal_investment_timing')
    
    # 3. NPV分布图
    plt.figure(figsize=(10, 6))
    positive_npv = results['NPV'][results['NPV'] > 0]
    if len(positive_npv) > 0:
        plt.hist(positive_npv / 1e6, bins=30, alpha=0.7, 
                color='lightgreen', edgecolor='black')
        plt.xlabel('NPV (million CNY)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_npv = np.mean(positive_npv) / 1e6
        plt.axvline(x=avg_npv, color='red', linestyle='--', 
                   label=f'Mean NPV: {avg_npv:.1f} million CNY')
        plt.legend()
    
    plt.tight_layout()
        # 该图未在论文中使用，已在最终版中停止输出
    
    # 4. TIV分布图
    plt.figure(figsize=(10, 6))
    # 获取所有投资决策点的TIV值
    all_tiv = results['TIV'].flatten()
    positive_tiv = all_tiv[all_tiv > 0]
    
    if len(positive_tiv) > 0:
        plt.hist(positive_tiv / 1e6, bins=30, alpha=0.7, 
                color='lightcoral', edgecolor='black')
        plt.xlabel('TIV (million CNY)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_tiv = np.mean(positive_tiv) / 1e6
        max_tiv = np.max(positive_tiv) / 1e6
        min_tiv = np.min(positive_tiv) / 1e6
        
        plt.axvline(x=avg_tiv, color='red', linestyle='--', 
                   label=f'Mean TIV: {avg_tiv:.1f} million CNY')
        plt.axvline(x=max_tiv, color='blue', linestyle=':', 
                   label=f'Max TIV: {max_tiv:.1f} million CNY', alpha=0.7)
        
        plt.legend()
    
        # 打印详细统计信息
        print(f"\nTIV statistics:")
        print(f"  Mean TIV: {avg_tiv:.2f} million CNY")
        print(f"  Max TIV: {max_tiv:.2f} million CNY") 
        print(f"  Min TIV: {min_tiv:.2f} million CNY")
        print(f"  Share of paths with positive TIV: {len(positive_tiv)/len(all_tiv):.1%}")
    else:
        plt.text(0.5, 0.5, 'No positive TIV', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
        # 该图未在论文中使用，已在最终版中停止输出

    # 5. 延迟期权价值分布图
    plt.figure(figsize=(12, 5))
    
    # 计算延迟期权价值 = TIV - NPV
    option_values = results['TIV'] - results['NPV']
    positive_option_values = option_values[option_values > 1e-6]  # 只考虑正值，避免浮点数误差
    
    if len(positive_option_values) > 0:
        # 子图1：延迟期权价值分布
        plt.subplot(1, 2, 1)
        plt.hist(positive_option_values / 1e6, bins=30, alpha=0.7, 
                 color='gold', edgecolor='black')
        plt.text(0.02, 0.97, 'a', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
        plt.xlabel('Deferral option value (million CNY)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_option = np.mean(positive_option_values) / 1e6
        max_option = np.max(positive_option_values) / 1e6
        
        plt.axvline(x=avg_option, color='red', linestyle='--', 
                    label=f'Mean: {avg_option:.1f} million CNY')
        plt.axvline(x=max_option, color='blue', linestyle=':', 
                    label=f'Max: {max_option:.1f} million CNY', alpha=0.7)
        plt.legend()
        
        # 子图2：延迟期权价值与NPV的关系
        plt.subplot(1, 2, 2)
        sample_size = min(1000, len(positive_option_values))
        sample_indices = np.random.choice(len(positive_option_values), sample_size, replace=False)
        
        sample_npv = results['NPV'].flatten()[option_values.flatten() > 1e-6][sample_indices] / 1e6
        sample_option = positive_option_values[sample_indices] / 1e6
        
        plt.scatter(sample_npv, sample_option, alpha=0.6, s=20, color='purple')
        plt.text(0.02, 0.97, 'b', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
        plt.xlabel('NPV (million CNY)')
        plt.ylabel('Deferral option value (million CNY)')
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(sample_npv) > 1:
            z = np.polyfit(sample_npv, sample_option, 1)
            p = np.poly1d(z)
            trend_x = np.array([min(sample_npv), max(sample_npv)])
            trend_y = p(trend_x)
            plt.plot(trend_x, trend_y, 'r--', alpha=0.8, 
                    label=f'Trend line: y = {z[0]:.2f}x + {z[1]:.1f}')
            plt.legend()
        
        # 打印详细统计信息
        print(f"\nDeferral option value statistics:")
        print(f"  Mean deferral option value: {avg_option:.2f} million CNY")
        print(f"  Max deferral option value: {max_option:.2f} million CNY")
        print(f"  Share of paths with positive option value: {len(positive_option_values)/len(option_values.flatten()):.1%}")
        print(f"  Option value as a share of total value: {np.sum(positive_option_values)/np.sum(results['TIV'][results['TIV'] > 0]):.1%}")
    
    else:
        plt.subplot(1, 2, 1)
        plt.text(0.5, 0.5, 'No positive deferral option value', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.text(0.02, 0.97, 'a', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
        
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, 'No positive deferral option value', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.text(0.02, 0.97, 'b', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    save_and_show('f_distribution_of_the_deferral_option_value')
    
    # 6. CE_replace变化趋势图
    plt.figure(figsize=(10, 6))
    years = np.arange(1, 40)
    ce_values = [model_corrected.calculate_CE_replace(year) for year in years]
    plt.plot(years, ce_values, 'g-o', linewidth=2, markersize=4)
    plt.xlabel('Operating year')
    plt.ylabel('CE_replace')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 41, 2))  # 从0到24，步长为2
    plt.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Stage boundary t = 8')
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Stage boundary t = 10')
    plt.legend()
    
    plt.tight_layout()
        # 该图未在论文中使用，已在最终版中停止输出
    
    # 5. 投资价值散点图
    # 说明：TIV(j,t) 是"投资时点当年货币"下的价值，直接按年份作散点会让晚期投资
    # 看起来天然更值钱（含通胀与折现的货币幻觉）。此处统一按风险调整贴现率折算回
    # 2026 年，与正文的决策时点期权价值 V0 口径一致，趋势线才有经济含义。
    plt.figure(figsize=(10, 6))
    discount_rate = params['r_f'] + params['beta_risk'] * params['r_m_rf']
    investment_years = []
    investment_values = []
    for j in range(min(1000, results['decisions'].shape[0])):
        investment_times = np.where(results['decisions'][j, :] == 1)[0]
        if len(investment_times) > 0:
            t_invest = investment_times[0]
            investment_years.append(2026 + t_invest)
            investment_values.append(
                results['TIV'][j, t_invest] * np.exp(-discount_rate * t_invest) / 1e6)
    
    if investment_years:
        plt.scatter(investment_years, investment_values, s=20,
                    facecolors='#83B4D9', edgecolors='none', alpha=0.6)
        # 图题保持不变，以免改变自动生成的文件名（i_investment_timing_and_investment_value.png）
        plt.xlabel('Investment year')
        plt.ylabel('Investment value discounted to 2026 (million CNY)')
        plt.grid(True, alpha=0.3)
        
        # 按执行年份画均值线：单一趋势线会被"被拖到窗口末端"的那批路径主导，
        # 掩盖真实的形态（各年份均值大致持平、决策窗口末端明显偏低）。
        years_sorted = sorted(set(investment_years))
        mean_by_year = [
            float(np.mean([v for y, v in zip(investment_years, investment_values) if y == yy]))
            for yy in years_sorted
        ]
        plt.plot(years_sorted, mean_by_year, '-o', color='#CC5A6A', linewidth=2,
                 markersize=6, label='Mean by exercise year')
        plt.legend()
    
    plt.tight_layout()
    save_and_show('i_investment_timing_and_investment_value')

# 详细价格统计图（单独显示）
def plot_detailed_price_stats(results, params):
    """Detailed price statistics"""
    plt.figure(figsize=(12, 8))
    
    total_years = params['T']  # 使用总寿命40年
    
    # 1. 碳价最终年份分布
    plt.subplot(2, 2, 1)
    pc_final_year = results['Pc_paths'][:, -1]
    plt.hist(pc_final_year, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(x=np.mean(pc_final_year), color='red', linestyle='--', 
                label=f'Mean: {np.mean(pc_final_year):.1f}')
    plt.xlabel('Carbon price (CNY/t)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 油价最终年份分布
    plt.subplot(2, 2, 2)
    oil_final_year = results['Poil_paths'][:, -1]
    plt.hist(oil_final_year, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=np.mean(oil_final_year), color='red', linestyle='--', 
                label=f'Mean: {np.mean(oil_final_year):.1f}')
    plt.xlabel('Oil price (CNY/t)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)


    # 3. 价格路径分位数
    plt.subplot(2, 2, 3)
    years_range = range(2026, 2026 + total_years)
    pc_quantiles = np.percentile(results['Pc_paths'], [10, 50, 90], axis=0)
    plt.plot(years_range, pc_quantiles[1], 'b-', label='Median', linewidth=2)
    plt.fill_between(years_range, pc_quantiles[0], pc_quantiles[2], 
                    alpha=0.3, color='blue', label='10th-90th percentile')
    plt.xlabel('Year')
    plt.ylabel('Carbon price (CNY/t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 油价路径分位数
    plt.subplot(2, 2, 4)
    oil_quantiles = np.percentile(results['Poil_paths'], [10, 50, 90], axis=0)
    plt.plot(years_range, oil_quantiles[1], 'r-', label='Median', linewidth=2)
    plt.fill_between(years_range, oil_quantiles[0], oil_quantiles[2], 
                    alpha=0.3, color='red', label='10th-90th percentile')
    plt.xlabel('Year')
    plt.ylabel('Oil price (CNY/t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.xticks(fontsize=18)  # x轴刻度数字字号
    plt.yticks(fontsize=18)  # y轴刻度数字字号
    plt.legend(fontsize=18)
        # 该图未在论文中使用，已在最终版中停止输出
    
    # 打印详细统计信息
    print("\n=== Detailed price statistics ===")
    print(f"Carbon price statistics (year {total_years}):")
    print(f"  Mean: {np.mean(pc_final_year):.2f} CNY/t")
    print(f"  Standard deviation: {np.std(pc_final_year):.2f} CNY/t")
    print(f"  10th percentile: {np.percentile(pc_final_year, 10):.2f} CNY/t")
    print(f"  90th percentile: {np.percentile(pc_final_year, 90):.2f} CNY/t")
    
    print(f"\nOil price statistics (year {total_years}):")
    print(f"  Mean: {np.mean(oil_final_year):.2f} CNY/t")
    print(f"  Standard deviation: {np.std(oil_final_year):.2f} CNY/t")
    print(f"  10th percentile: {np.percentile(oil_final_year, 10):.2f} CNY/t")
    print(f"  90th percentile: {np.percentile(oil_final_year, 90):.2f} CNY/t")


# 绘制修正结果
try:
    print("Plotting the carbon and oil price panels...")
    plot_corrected_results(results_corrected, params)
    
    # 绘制详细价格统计图：选择
    # 无控制台（服务器/notebook 自动运行）时可设 CCUS_NONINTERACTIVE=1，
    # 由环境变量 CCUS_SHOW_DETAILED（默认 y）决定是否绘制，避免 input() 阻塞。
    if os.environ.get("CCUS_NONINTERACTIVE"):
        show_detailed_stats = os.environ.get("CCUS_SHOW_DETAILED", "y").lower().strip()
        print("Show the detailed price statistics figures? -> "
              f"{show_detailed_stats}   [CCUS_NONINTERACTIVE]")
    else:
        try:
            show_detailed_stats = input(
                "Show the detailed price statistics figures? (y/n): ").lower().strip()
        except EOFError:
            show_detailed_stats = "n"
            print("Show the detailed price statistics figures? -> n   [no console input]")
    if show_detailed_stats == 'y':
        plot_detailed_price_stats(results_corrected, params)
        
except Exception as e:
    print(f"Plotting error: {e}")
    import traceback
    traceback.print_exc()

# 打印时间逻辑验证
print("\n=== Timing check ===")
print(f"Plant life: {params['T']} years (from {params['base_year']} to {params['base_year'] + params['T'] - 1} years)")
print(f"Investment decision window: {params['t_v']} years (from {params['base_year']} to {params['base_year'] + params['t_v'] - 1} years)")
print(f"Construction period: {params['t_c']} years")

print("\nTiming implied by each investment year:")
for t in range(min(5, params['t_v'])):
    investment_year = params['base_year'] + t
    construction_end = investment_year + params['t_c']
    op_period = model_corrected.calculate_operating_period(t)
    project_end = construction_end + op_period - 1
    
    print(f"  Investment year {investment_year}:")
    print(f"    - construction period: {investment_year}-{construction_end}")
    print(f"    - operating period: {construction_end}-{project_end} years ({op_period} years)")
    print(f"    - project horizon: {investment_year}-{project_end}")
