import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
try:                      # tqdm 只是可选的进度条依赖
    from tqdm import tqdm
except ImportError:       # 未安装时退化为恒等函数，不影响结果
    def tqdm(iterable, *args, **kwargs):
        return iterable
import os

# Set CCUS_QUICK_TEST=1 to run the whole pipeline with 300 paths (fast smoke test).
QUICK_TEST = bool(os.environ.get("CCUS_QUICK_TEST"))

# Same seed and path count as ccus_part1.py / ccus_part2senario.py so that every
# sensitivity run is comparable with the baseline (common random numbers).
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

# 敏感性分析类
class SensitivityAnalyzer:
    def __init__(self, base_params):
        self.base_params = base_params
        self.results = {}
        # 从基础参数获取基准年份
        self.base_year = base_params.get('base_year', 2026)
        self.decision_period = base_params['t_v']  # 投资决策期年数
    
    def run_sensitivity_analysis(self, param_ranges, num_simulations=2):
        """Run the sensitivity analysis"""
        print("Running the sensitivity analysis...")
        
        all_results = []
        
        for param_name, param_range in param_ranges.items():
            print(f"\nAnalysing parameter: {param_name}")
            param_results = []
            
            for param_value in tqdm(param_range):
                current_params = self.base_params.copy()
                current_params[param_name] = param_value
                
                timing_results = []
                value_results = []
                ratio_results = []
                
                for sim in range(num_simulations):
                    try:
                        model = CCUSInvestmentModel(current_params)
                        results = model.lsmc_solver()
                        
                        if not np.isnan(results['optimal_timing']):
                            # 确保年份在合理范围内
                            optimal_year = results['optimal_timing_median_year']
                            if optimal_year < self.base_year:
                                optimal_year = self.base_year
                            elif optimal_year > self.base_year + self.decision_period - 1:
                                optimal_year = self.base_year + self.decision_period - 1
                            
                            timing_results.append(optimal_year)
                            # 敏感性分析统一采用决策时点期权价值 V0（跨情景可比），
                            # 而不是投资执行时点的价值
                            value_results.append(float(np.mean(results['TIV'][:, 0])) / 1e6)
                        ratio_results.append(results['investment_ratio'])
                        
                    except Exception as e:
                        continue
                
                if timing_results:
                    # 使用平均值并四舍五入到最接近的整数年份
                    avg_timing = round(np.mean(timing_results))
                    avg_value = np.mean(value_results)
                    avg_ratio = np.mean(ratio_results)
                else:
                    avg_timing = np.nan
                    avg_value = 0
                    avg_ratio = 0
                
                param_results.append({
                    'parameter': param_name,
                    'parameter_value': param_value,
                    'optimal_timing': avg_timing,
                    'optimal_value': avg_value,
                    'investment_ratio': avg_ratio
                })
            
            self.results[param_name] = param_results
            all_results.extend(param_results)
        
        return pd.DataFrame(all_results)
    
    def plot_individual_combined_charts(self, df_results):
        """Summarise the results for each parameter"""
        parameters = df_results['parameter'].unique()
        
        for param in parameters:
            param_data = df_results[df_results['parameter'] == param].sort_values('parameter_value')
            
            if len(param_data) <= 1:
                continue
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            
            x_pos = np.arange(len(param_data))
            bar_width = 0.35
            
            optimal_values = param_data['optimal_value'].values
            optimal_timings = param_data['optimal_timing'].values
            param_values = param_data['parameter_value'].values
            
            # 柱状图 - 最优投资价值
            bars = ax1.bar(x_pos - bar_width/2, optimal_values, 
                          width=bar_width, alpha=0.7, color='#1f77b4', 
                          edgecolor='darkblue', linewidth=1, 
                          label='Optimal investment value')
            
            # 折线图 - 最优投资时间
            ax2.plot(x_pos + bar_width/2, optimal_timings, 
                    'ro-', linewidth=2, markersize=6, markerfacecolor='red', 
                    markeredgecolor='darkred', markeredgewidth=1, 
                    label='Optimal investment year')
            
            ax1.set_xlabel(f'{self.get_param_display_name(param)}')
            ax1.set_ylabel('Optimal investment value (million CNY)', color='blue')
            ax2.set_ylabel('Optimal investment year', color='red')
            
            x_labels = self.format_x_labels(param, param_values)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, rotation=45)
            
            # 设置y轴为整数年份，范围在基准年份范围内
            year_min = self.base_year - 1
            year_max = self.base_year + self.decision_period
            ax2.set_ylim(year_min, year_max)
            ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_axisbelow(True)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                      ncol=2, fontsize=10)
            
            plt.title(f'{self.get_param_display_name(param)} sensitivity analysis')
            plt.tight_layout()
            save_and_show(_slug(self.get_param_display_name(param) + ' sensitivity analysis'))
    
    def plot_sensitivity_curves(self, df_results):
        """Plot the sensitivity curves"""
        parameters = df_results['parameter'].unique()
        n_params = len(parameters)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 投资价值敏感性曲线
        for param in parameters:
            param_data = df_results[df_results['parameter'] == param].sort_values('parameter_value')
            if len(param_data) > 1:
                axes[0].plot(param_data['parameter_value'], param_data['optimal_value'], 
                            'o-', label=self.get_param_display_name(param), linewidth=2)
        axes[0].set_xlabel('Parameter value')
        axes[0].set_ylabel('Optimal investment value (million CNY)')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # 投资时间敏感性曲线
        for param in parameters:
            param_data = df_results[df_results['parameter'] == param].sort_values('parameter_value')
            if len(param_data) > 1:
                axes[1].plot(param_data['parameter_value'], param_data['optimal_timing'], 
                            's-', label=self.get_param_display_name(param), linewidth=2)
        axes[1].set_xlabel('Parameter value')
        axes[1].set_ylabel('Optimal investment year')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 设置y轴为整数年份，范围在基准年份范围内
        axes[1].set_ylim(self.base_year - 1, self.base_year + self.decision_period)
        axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        save_and_show(_slug('investment value sensitivity curve'))
    
    def calculate_sensitivity_metrics(self, df_results):
        """Compute the sensitivity indices"""
        sensitivity_metrics = {}
        
        for param in df_results['parameter'].unique():
            param_data = df_results[df_results['parameter'] == param].sort_values('parameter_value')
            
            if len(param_data) > 1:
                x = param_data['parameter_value'].values
                y_value = param_data['optimal_value'].values
                y_time = param_data['optimal_timing'].values
                y_ratio = param_data['investment_ratio'].values
                
                # 移除NaN值
                valid_idx = ~np.isnan(y_value) & ~np.isnan(y_time)
                if np.sum(valid_idx) > 1:
                    x = x[valid_idx]
                    y_value = y_value[valid_idx]
                    y_time = y_time[valid_idx]
                    y_ratio = y_ratio[valid_idx]
                    
                    # 计算敏感性指数（标准化斜率）
                    if len(x) > 1 and np.std(x) > 0:
                        # 价值敏感性
                        value_slope = np.polyfit(x, y_value, 1)[0]
                        value_sensitivity = abs(value_slope * np.std(x) / (np.std(y_value) + 1e-8))
                        
                        # 时间敏感性
                        time_slope = np.polyfit(x, y_time, 1)[0]
                        time_sensitivity = abs(time_slope * np.std(x) / (np.std(y_time) + 1e-8))
                        
                        # 投资概率敏感性
                        ratio_slope = np.polyfit(x, y_ratio, 1)[0]
                        ratio_sensitivity = abs(ratio_slope * np.std(x) / (np.std(y_ratio) + 1e-8))
                        
                        # 综合敏感性
                        combined_sensitivity = (value_sensitivity + time_sensitivity + ratio_sensitivity) / 3
                        
                        sensitivity_metrics[param] = {
                            'value_sensitivity': value_sensitivity,
                            'time_sensitivity': time_sensitivity,
                            'ratio_sensitivity': ratio_sensitivity,
                            'combined_sensitivity': combined_sensitivity,
                            'max_value': y_value.max(),
                            'min_value': y_value.min(),
                            'max_time': y_time.max(),
                            'min_time': y_time.min(),
                            'max_ratio': y_ratio.max(),
                            'min_ratio': y_ratio.min(),
                            'value_range': y_value.max() - y_value.min(),
                            'time_range': y_time.max() - y_time.min()
                        }
        
        return sensitivity_metrics
    
    def plot_sensitivity_ranking(self, sensitivity_metrics):
        """Plot the sensitivity ranking"""
        if not sensitivity_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 综合敏感性排名
        sorted_by_combined = sorted(sensitivity_metrics.items(), 
                                  key=lambda x: x[1]['combined_sensitivity'], reverse=True)
        params_combined = [self.get_param_display_name(x[0]) for x in sorted_by_combined]
        values_combined = [x[1]['combined_sensitivity'] for x in sorted_by_combined]
        
        axes[0,0].barh(range(len(params_combined)), values_combined, color='lightgreen')
        axes[0,0].set_yticks(range(len(params_combined)))
        axes[0,0].set_yticklabels(params_combined)
        axes[0,0].set_xlabel('Comprehensive sensitivity index')
        axes[0,0].set_title('Comprehensive sensitivity ranking', fontweight='bold')
        
        # 价值敏感性排名
        sorted_by_value = sorted(sensitivity_metrics.items(), 
                               key=lambda x: x[1]['value_sensitivity'], reverse=True)
        params_value = [self.get_param_display_name(x[0]) for x in sorted_by_value]
        values_value = [x[1]['value_sensitivity'] for x in sorted_by_value]
        
        axes[0,1].barh(range(len(params_value)), values_value, color='lightblue')
        axes[0,1].set_yticks(range(len(params_value)))
        axes[0,1].set_yticklabels(params_value)
        axes[0,1].set_xlabel('Value sensitivity index')
        axes[0,1].set_title('Value sensitivity ranking', fontweight='bold')
        
        # 时间敏感性排名
        sorted_by_time = sorted(sensitivity_metrics.items(), 
                              key=lambda x: x[1]['time_sensitivity'], reverse=True)
        params_time = [self.get_param_display_name(x[0]) for x in sorted_by_time]
        values_time = [x[1]['time_sensitivity'] for x in sorted_by_time]
        
        axes[1,0].barh(range(len(params_time)), values_time, color='lightcoral')
        axes[1,0].set_yticks(range(len(params_time)))
        axes[1,0].set_yticklabels(params_time)
        axes[1,0].set_xlabel('Timing sensitivity index')
        axes[1,0].set_title('Timing sensitivity ranking', fontweight='bold')
        
        # 投资概率敏感性排名
        sorted_by_ratio = sorted(sensitivity_metrics.items(), 
                               key=lambda x: x[1]['ratio_sensitivity'], reverse=True)
        params_ratio = [self.get_param_display_name(x[0]) for x in sorted_by_ratio]
        values_ratio = [x[1]['ratio_sensitivity'] for x in sorted_by_ratio]
        
        axes[1,1].barh(range(len(params_ratio)), values_ratio, color='gold')
        axes[1,1].set_yticks(range(len(params_ratio)))
        axes[1,1].set_yticklabels(params_ratio)
        axes[1,1].set_xlabel('Investment probability sensitivity index')
        axes[1,1].set_title('Investment probability sensitivity ranking', fontweight='bold')
        
        plt.tight_layout()
        save_and_show('investment_probability_sensitivity_ranking')
    
    def plot_tornado_chart(self, sensitivity_metrics, base_value, base_time):
        """Plot the tornado chart"""
        if not sensitivity_metrics:
            return
        
        # 计算每个参数对投资价值的影响范围
        impacts = []
        for param, metrics in sensitivity_metrics.items():
            impact_range = metrics['value_range']
            impacts.append({
                'parameter': param,
                'impact_range': impact_range,
                'display_name': self.get_param_display_name(param)
            })
        
        # 按影响范围排序
        impacts.sort(key=lambda x: x['impact_range'], reverse=True)
        
        # 取前10个参数
        top_impacts = impacts[:10]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        parameters = [impact['display_name'] for impact in top_impacts]
        impact_ranges = [impact['impact_range'] for impact in top_impacts]
        
        y_pos = np.arange(len(parameters))
        
        bars = ax.barh(y_pos, impact_ranges, color='skyblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(parameters)
        ax.set_xlabel('Change in investment value (million CNY)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, impact) in enumerate(zip(bars, impact_ranges)):
            ax.text(bar.get_width() + max(impact_ranges)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{impact:.1f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        save_and_show('investment_probability_sensitivity_ranking')
    
    def get_param_display_name(self, param):
        names = {
            'alpha': 'Technological progress (investment cost)',
            'beta': 'Technological progress (O&M cost)',
            'Pc0': 'Initial carbon price',
            'mu_c': 'Carbon price drift',
            'sigma_c': 'Carbon price volatility',
            'Poil0': 'Initial oil price',
            'mu_oil': 'Oil price drift',
            'sigma_oil': 'Oil price volatility',
            'r_f': 'Risk-free rate',
            'beta_risk': 'Systemic risk coefficient'
        }
        return names.get(param, param)
    
    def format_x_labels(self, param, values):
        if param in ['alpha', 'beta', 'mu_c', 'mu_oil', 'sigma_c']:
            return [f'{x:.2f}' for x in values]
        elif param == 'sigma_oil':
            return [f'{x:.2f}' for x in values]
        elif param in ['Pc0', 'Poil0']:
            return [f'{x:.0f}' for x in values]
        elif param == 'r_f':
            return [f'{x:.2f}' for x in values]
        elif param == 'beta_risk':
            return [f'{x:.1f}' for x in values]
        else:
            return [f'{x:.3f}' for x in values]
    
    def generate_analysis_report(self, df_results, sensitivity_metrics):
        """Generate the analysis report"""
        print("\n" + "="*80)
        print("                     CCUS investment decision: sensitivity report")
        print("="*80)
        
        # 基础统计
        base_data = df_results[df_results['parameter'] == 'alpha'].iloc[5]  # 取中间值作为基准
        print(f"\n Baseline statistics:")
        print(f"   Mean optimal investment value: {base_data['optimal_value']:.1f} million CNY")
        print(f"   Mean optimal investment year: {base_data['optimal_timing']:.1f}")
        print(f"   Mean investment probability: {base_data['investment_ratio']*100:.1f}%")
        
        # 完整的参数敏感性排序
        if sensitivity_metrics:
            print(f"\n Full parameter sensitivity ranking:")
            sorted_metrics = sorted(sensitivity_metrics.items(), 
                                  key=lambda x: x[1]['combined_sensitivity'], reverse=True)
            
            print(f"{'Rank':<4} {'Parameter':<20} {'Comprehensive sensitivity':<12} {'Value sensitivity':<12} {'Timing sensitivity':<12} {'Investment probability sensitivity':<12}")
            print("-" * 90)
            for i, (param, metrics) in enumerate(sorted_metrics, 1):
                print(f"{i:<4} {self.get_param_display_name(param):<20} {metrics['combined_sensitivity']:<12.4f} "
                      f"{metrics['value_sensitivity']:<12.4f} {metrics['time_sensitivity']:<12.4f} "
                      f"{metrics['ratio_sensitivity']:<12.4f}")
        
        # 按不同维度分别排序
        if sensitivity_metrics:
            # 价值敏感性排序
            print(f"\n Full value-sensitivity ranking:")
            sorted_value = sorted(sensitivity_metrics.items(), 
                                key=lambda x: x[1]['value_sensitivity'], reverse=True)
            print(f"{'Rank':<4} {'Parameter':<20} {'Value sensitivity':<12} {'Value range':<12}")
            print("-" * 60)
            for i, (param, metrics) in enumerate(sorted_value, 1):
                print(f"{i:<4} {self.get_param_display_name(param):<20} {metrics['value_sensitivity']:<12.4f} "
                      f"{metrics['value_range']:<12.1f}")
            
            # 时间敏感性排序
            print(f"\n Full timing-sensitivity ranking:")
            sorted_time = sorted(sensitivity_metrics.items(), 
                               key=lambda x: x[1]['time_sensitivity'], reverse=True)
            print(f"{'Rank':<4} {'Parameter':<20} {'Timing sensitivity':<12} {'Timing range':<12}")
            print("-" * 60)
            for i, (param, metrics) in enumerate(sorted_time, 1):
                print(f"{i:<4} {self.get_param_display_name(param):<20} {metrics['time_sensitivity']:<12.4f} "
                      f"{metrics['time_range']:<12.1f}")
            
            # 投资概率敏感性排序
            print(f"\n Full investment-probability ranking:")
            sorted_ratio = sorted(sensitivity_metrics.items(), 
                                key=lambda x: x[1]['ratio_sensitivity'], reverse=True)
            print(f"{'Rank':<4} {'Parameter':<20} {'Investment probability sensitivity':<12} {'Probability range':<12}")
            print("-" * 60)
            for i, (param, metrics) in enumerate(sorted_ratio, 1):
                print(f"{i:<4} {self.get_param_display_name(param):<20} {metrics['ratio_sensitivity']:<12.4f} "
                      f"{(metrics['max_ratio'] - metrics['min_ratio'])*100:<11.1f}%")
        
        # 关键驱动因素识别
        print(f"\n Key drivers:")
        if sensitivity_metrics:
            # 价值驱动因素
            value_drivers = sorted(sensitivity_metrics.items(), 
                                 key=lambda x: x[1]['value_sensitivity'], reverse=True)[:3]
            print(f"   Main value drivers:")
            for param, metrics in value_drivers:
                print(f"     • {self.get_param_display_name(param)} (sensitivity: {metrics['value_sensitivity']:.4f}, "
                      f"Value change: {metrics['value_range']:.1f} million)")
            
            # 时间驱动因素
            time_drivers = sorted(sensitivity_metrics.items(), 
                                key=lambda x: x[1]['time_sensitivity'], reverse=True)[:3]
            print(f"   Main timing drivers:")
            for param, metrics in time_drivers:
                print(f"     • {self.get_param_display_name(param)} (sensitivity: {metrics['time_sensitivity']:.4f}, "
                      f"Timing change: {metrics['time_range']:.1f} years)")
            
            # 投资概率驱动因素
            ratio_drivers = sorted(sensitivity_metrics.items(), 
                                 key=lambda x: x[1]['ratio_sensitivity'], reverse=True)[:3]
            print(f"   Main investment-probability drivers:")
            for param, metrics in ratio_drivers:
                prob_range = (metrics['max_ratio'] - metrics['min_ratio']) * 100
                print(f"     • {self.get_param_display_name(param)} (sensitivity: {metrics['ratio_sensitivity']:.4f}, "
                      f"Probability change: {prob_range:.1f}%)")
        
        # 参数分类分析
        print(f"\n Parameter classification:")
        if sensitivity_metrics:
            # 高敏感性参数（综合敏感性 > 0.5）
            high_sensitive = [(p, m) for p, m in sensitivity_metrics.items() 
                            if m['combined_sensitivity'] > 0.5]
            if high_sensitive:
                print(f"   High-sensitivity parameters (index > 0.5):")
                for param, metrics in high_sensitive:
                    print(f"     • {self.get_param_display_name(param)}: {metrics['combined_sensitivity']:.4f}")
            
            # 中等敏感性参数（0.2 < 综合敏感性 <= 0.5）
            medium_sensitive = [(p, m) for p, m in sensitivity_metrics.items() 
                              if 0.2 < m['combined_sensitivity'] <= 0.5]
            if medium_sensitive:
                print(f"   Medium-sensitivity parameters (0.2 < index <= 0.5):")
                for param, metrics in medium_sensitive:
                    print(f"     • {self.get_param_display_name(param)}: {metrics['combined_sensitivity']:.4f}")
            
            # 低敏感性参数（综合敏感性 <= 0.2）
            low_sensitive = [(p, m) for p, m in sensitivity_metrics.items() 
                           if m['combined_sensitivity'] <= 0.2]
            if low_sensitive:
                print(f"   Low-sensitivity parameters (index <= 0.2):")
                for param, metrics in low_sensitive:
                    print(f"     • {self.get_param_display_name(param)}: {metrics['combined_sensitivity']:.4f}")
        
        # 投资决策概率分析
        print(f"\n Investment probability details:")
        high_prob_scenarios = []
        for param in df_results['parameter'].unique():
            param_data = df_results[df_results['parameter'] == param]
            max_ratio = param_data['investment_ratio'].max()
            min_ratio = param_data['investment_ratio'].min()
            avg_ratio = param_data['investment_ratio'].mean()
            
            if max_ratio >= 0.5:  # 降低阈值以显示更多参数
                high_prob_scenarios.append({
                    'parameter': param,
                    'max_ratio': max_ratio,
                    'min_ratio': min_ratio,
                    'avg_ratio': avg_ratio,
                    'range': max_ratio - min_ratio
                })
        
        if high_prob_scenarios:
            # 按最高概率排序
            high_prob_scenarios.sort(key=lambda x: x['max_ratio'], reverse=True)
            print(f"   High-probability parameters (max probability >= 50%):")
            print(f"{'Parameter':<20} {'Max probability':<10} {'Min probability':<10} {'Mean probability':<10} {'Probability range':<10}")
            print("-" * 70)
            for scenario in high_prob_scenarios:
                print(f"{self.get_param_display_name(scenario['parameter']):<20} "
                      f"{scenario['max_ratio']*100:<9.1f}% {scenario['min_ratio']*100:<9.1f}% "
                      f"{scenario['avg_ratio']*100:<9.1f}% {scenario['range']*100:<9.1f}%")
        else:
            print("   No high-probability parameter combination found")
        
        # 最优参数组合分析
        print(f"\n Suggested parameter combination:")
        if sensitivity_metrics:
            # 找出每个参数的最优范围
            optimal_params = {}
            for param in df_results['parameter'].unique():
                param_data = df_results[df_results['parameter'] == param]
                # 找到投资价值最高的参数值
                best_value_row = param_data.loc[param_data['optimal_value'].idxmax()]
                optimal_params[param] = {
                    'best_value': best_value_row['parameter_value'],
                    'max_investment_value': best_value_row['optimal_value'],
                    'investment_ratio_at_best': best_value_row['investment_ratio']
                }
            
            print(f"   Suggested parameter levels:")
            for param, info in optimal_params.items():
                if param in ['alpha', 'beta', 'mu_c', 'mu_oil', 'sigma_c', 'sigma_oil', 'r_f']:
                    value_str = f"{info['best_value']:.3f}"
                elif param == 'beta_risk':
                    value_str = f"{info['best_value']:.1f}"
                else:
                    value_str = f"{info['best_value']:.0f}"
                
                print(f"     • {self.get_param_display_name(param)}: {value_str} "
                      f"(investment value: {info['max_investment_value']:.1f} million, "
                      f"Investment probability: {info['investment_ratio_at_best']*100:.1f}%)")
        
        print(f"\n Management implications:")
        print("   1. Focus on the high-sensitivity parameters, which drive the decision most")
        print("   2. Allocate parameter-estimation effort according to the sensitivity ranking")

# 定义敏感性分析参数范围
def create_parameter_ranges():
    param_ranges = {
        'alpha': np.arange(0.01, 0.11, 0.01),
        'beta': np.arange(0.01, 0.11, 0.01),
        'Pc0': np.arange(60, 361, 30),
        'mu_c': np.arange(0.01, 0.11, 0.01),
        'sigma_c': np.arange(0.01, 0.11, 0.01),
        'Poil0': np.arange(3500, 7101, 400),
        'mu_oil': np.arange(0.01, 0.11, 0.01),
        'sigma_oil': np.arange(0.05, 0.51, 0.05),
        'r_f': np.arange(0.01, 0.09, 0.01),
        'beta_risk': np.arange(0.5, 1.6, 0.1),
    }
    return param_ranges

def run_comprehensive_sensitivity_analysis(base_params):
    print("=== Sensitivity analysis of the CCUS investment decision model ===\n")
    
    analyzer = SensitivityAnalyzer(base_params)
    param_ranges = create_parameter_ranges()
    
    # 运行敏感性分析
    print("Running the sensitivity analysis...")
    # 随机种子固定，同一参数取值重复运行结果完全相同，故只运行一次
    sensitivity_df = analyzer.run_sensitivity_analysis(param_ranges, num_simulations=1)
    
    # 计算基准值：直接运行基准参数，保证龙卷风图的参考线与正文基准完全一致
    baseline_params = dict(base_params)
    baseline_model = CCUSInvestmentModel(baseline_params)
    baseline_results = baseline_model.lsmc_solver()
    base_value = float(np.mean(baseline_results['TIV'][:, 0])) / 1e6
    base_time = baseline_results['optimal_timing_median_year']

    # 计算敏感性指标（先算指标并落盘，绘图失败也不会丢失结果）
    print("Computing the sensitivity indices...")
    sensitivity_metrics = analyzer.calculate_sensitivity_metrics(sensitivity_df)

    try:
        import csv

        with open("ccus_sensitivity_ranking.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["parameter", "value_sensitivity", "time_sensitivity",
                             "ratio_sensitivity", "combined_sensitivity",
                             "value_range", "time_range"])
            for param, met in sorted(sensitivity_metrics.items(),
                                     key=lambda kv: kv[1]["combined_sensitivity"],
                                     reverse=True):
                writer.writerow([param,
                                 round(met["value_sensitivity"], 4),
                                 round(met["time_sensitivity"], 4),
                                 round(met["ratio_sensitivity"], 4),
                                 round(met["combined_sensitivity"], 4),
                                 round(met["value_range"], 2),
                                 round(met["time_range"], 3)])
        print("[results] sensitivity ranking written to ccus_sensitivity_ranking.csv")
    except Exception as exc:
        print("[results] could not write the sensitivity ranking:", exc)

    try:
        sensitivity_df.to_csv('ccus_sensitivity_comprehensive_results.csv',
                              index=False, encoding='utf-8-sig')
        print("[results] detailed sensitivity results written to "
              "ccus_sensitivity_comprehensive_results.csv")
    except Exception as e:
        print(f"[results] could not save the detailed results: {e}")
    
    # 绘制各种图表
    print("\nPlotting the combined panels...")
    analyzer.plot_individual_combined_charts(sensitivity_df)
    
    print("Plotting the sensitivity curves...")
    analyzer.plot_sensitivity_curves(sensitivity_df)
    
    print("Plotting the sensitivity ranking...")
    analyzer.plot_sensitivity_ranking(sensitivity_metrics)
    
    print("Plotting the tornado chart...")
    analyzer.plot_tornado_chart(sensitivity_metrics, base_value, base_time)
    
    # 生成分析报告
    analyzer.generate_analysis_report(sensitivity_df, sensitivity_metrics)
    
    return sensitivity_df, sensitivity_metrics

# 执行分析
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 初始投资成本 I0（元）：与 ccus_part1.py、ccus_part2senario.py 保持一致。
    # CCUS-EOR 全流程增量投资口径：捕集系统 1.6e9 + 压缩输送管道 0.69e9
    # + 驱油地面工程 ≈0.11e9 = 2.4e9 元。
    # ------------------------------------------------------------------
    I0_CNY = 2.4e9

    base_params = {
        'base_year': 2026,  # 基准年份
        'IC': 600000,  # 电厂装机容量 (kW) - 600MW
        'RT': 4147,  # 年运行时间 (小时)，2025 年火电平均利用小时（国家能源局）
        'EF': 0.000762,  # 二氧化碳排放因子 (吨/kWh)
        'CE': 0.9,  # 二氧化碳捕集效率
        'phi': 0.94,  # 机组效率

        # 成本参数
        'I0': I0_CNY,
        'M0': 3.774e7,  # 初始运营维护成本 (元)
        'alpha': 0.0202,  # 技术进步对投资成本影响参数
        'beta': 0.057,  # 技术进步对运营维护成本影响参数

        'UTC_CO2': 100,  # 二氧化碳单位运输成本 (元/吨)
        'USC_CO2': 50,  # 单位封存成本 (元/吨)
        'UUC_CO2': 106.5,  # 单位驱油封存成本 (元/吨)

        # 技术效率参数
        'zeta': 0.975,  # 二氧化碳封存率
        'eta_loss': 0.32,  # 发电效率损失因子

        # 电价模型参数
        'Pe0': 0.35,
        'mu_e': np.log(0.6),
        'alpha_e': 0.45564,
        'sigma_e': 0.1,
        'rho_e_c': 0.4,

        # 第一阶段增长参数
        'stage1_years': 5,
        'Pe_growth_rate': 0.024,
        'Pe_growth_increment': 0.0012,

        # 价格和补贴参数
        'S_clean': 0.019,
        # 45Q 驱油利用抵免：35 美元/吨 × 7.1 = 248.5 元/吨（见 ccus_part1.py 的参数注释）
        'S_EOR': 248.5,
        'claim_years': 12,   # 45Q 抵免可申领的运营年数（§45Q(a)(3)(A)）
        'pi': 0.0,   # 45Q 现行法案已对封存与驱油利用并档，基准只计驱油利用抵免

        # 财税参数
        'tax': 0.25,
        'lambda_': 0,

        # 时间参数
        'T': 30,
        't_v': 10,  # 改为15年决策期
        't_c': 1,

        # 金融参数
        'r_f': 0.0264,
        'beta_risk': 0.8,
        'r_m_rf': 0.06,

        # 不确定性参数
        'Pc0': 62.36,  # 会被覆盖
        'mu_c': 0.05,  # carbon price drift - Table 2
        'sigma_c': 0.07,

        'Poil0': 3548,
        'mu_oil': 0.06,
        'sigma_oil': 0.11,

        # 模拟参数
        # Same path count as the baseline and the scenario runs, so the baseline
        # row of the sensitivity table reproduces ccus_part1_results.csv exactly.
        'M': 10000
    }
    
    sensitivity_df, sensitivity_metrics = run_comprehensive_sensitivity_analysis(base_params)
