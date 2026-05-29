# 1D 单相数值模拟器

一维单相微可压缩流体在多孔介质中的隐式 FVM 模拟器。是 `sim_pinn` 学习项目的第一块——
仓库整体规划见根目录的 `CLAUDE.md`。

## 物理模型与数值方法

控制方程（一维单相、均匀常数物性、Darcy + 弱可压缩）：

```
φ·ct·∂P/∂t = (k/μ) · ∂²P/∂x² + q_src
```

- 离散：**单元中心有限体积**，`nx` 个等长格子，长度 `L`，截面积 `A`。
- 时间：**后向欧拉（隐式）**，每步组装三对角线性方程组 `A·p^{n+1} = b`。
- 求解：`scipy.linalg.solve_banded`，直接法。
- 单元间面传导率 `T = k·A/(μ·dx)`，Dirichlet 边界面距格心 `dx/2`，面传导率为 `2T`。

每步对每个 cell 列方程：

```
α·p_i^{n+1} + 2T·p_i^{n+1} − T·p_{i−1}^{n+1} − T·p_{i+1}^{n+1} = α·p_i^n + 井项
α := φ·ct·V/Δt,    V = A·dx
```

边界 cell（左/右）的 `main[i]` 与 `b[i]` 增量：

| 边界类型 | `main` 增量 | `b` 增量 |
| --- | --- | --- |
| `noflow` | 0 | 0 |
| `dirichlet(P_bc)` | `+2T` | `+2T·P_bc` |

井项写法见下一节。该格式**离散质量守恒严格成立**（测试中 `rtol=1e-10`）。

## 配置 `Config`

全部参数集中在 `simulator/config.py` 的 `Config` 数据类里。SI 单位贯穿。

### 默认场景

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `nx`, `L`, `A` | 15, 150 m, 100 m² | 15 个格子，每格 10 m |
| `k`, `mu`, `phi`, `ct` | 1e-13 m², 5e-3 Pa·s, 0.2, 1e-9 Pa⁻¹ | ≈100 mD、5 cp |
| `P0` | 2e7 Pa | 初始压力 20 MPa |
| `left_bc` | `BoundarySpec("dirichlet", 2.0e7)` | 左 20 MPa 定压 |
| `right_bc` | `BoundarySpec("dirichlet", 1.9e7)` | 右 19 MPa 定压 |
| `wells` | `[BHPWellSpec(7, DEFAULT_BHP_SCHEDULE, kind="producer")]` | 中间格 1 口定流压采出井，20 步 BHP 调度（17 MPa ± 1 MPa 正弦两周期） |
| `dt`, `n_steps` | 86400 s, 20 | 1 天/步，模拟 20 天 |

### 边界条件

`BoundarySpec(kind, pressure)`：

- `BoundarySpec("dirichlet", P_bc)` —— 定压边界，需要 `pressure`。
- `BoundarySpec("noflow")` —— 封闭边界（无通量），`pressure` 不用。

左右边界各自独立指定。

### 井：多井 + 注/采 + 定流量/定流压

`wells` 字段是 `WellSpec` 列表。两种规格：

**`RateWellSpec(cell_index, rate)`** — 定流量井
- `rate > 0`：注入井，`rate < 0`：采出井，单位 m³/s。
- `rate` 可以是**标量**（每步同值）或**长度等于 `n_steps` 的序列**（按步取值）。

**`BHPWellSpec(cell_index, p_wf, kind, rw=0.1)`** — 定流压井
- `p_wf`：井底压力（Pa），同样可以是标量或长度 `n_steps` 的序列。
- `kind`：**必填**，`"producer"` 或 `"injector"`，显式声明井类型。
- `rw`：井筒半径（m），决定产能指数 `PI`。
- `q = PI · (P_wf − P_cell)`：当 `P_cell > P_wf` 时为采出。
- `PI` 由 1D 类 Peaceman 公式从几何参数和井半径算出：
  - `re = 0.14·√(dx² + A)`
  - `PI = 2π · k · √A / (μ · ln(re/rw))`

> **校验**：
> - 如果 `rate` / `p_wf` 是序列，长度必须等于 `n_steps`，否则 `Config` 实例化会抛错。
> - BHP 井**运行期**每步检查 `sign(q)` 是否与 `kind` 一致；如果 `producer` 但 `P_wf > P_cell^(n+1)`（实际变成注入），
>   或 `injector` 但 `P_wf < P_cell^(n+1)`（实际变成采出），`run()` 会抛 `RuntimeError` 终止模拟，
>   并在错误信息里给出步号、cell、p_wf、p_cell、实际 q。

### 自定义示例

```python
from simulator.config import Config, BoundarySpec
from simulator.well import RateWellSpec, BHPWellSpec

cfg = Config(
    left_bc=BoundarySpec("noflow"),
    right_bc=BoundarySpec("dirichlet", 1.8e7),
    wells=[
        RateWellSpec(cell_index=3,  rate=+3e-5),                       # 注水：每步同值
        BHPWellSpec (cell_index=11,
                     p_wf=[1.5e7]*100 + [1.0e7]*100,                   # 流压调度：前 100 步 15 MPa、后 100 步 10 MPa
                     kind="producer", rw=0.1),
    ],
    n_steps=200,
)
```

## 运行

Python 用 conda 的 `tor` 环境，但**直接走绝对路径**（不套 `conda run`）。所有命令在 `1D_1phase/` 目录下执行（包导入是 `simulator.X`）。

```powershell
D:\WorkSoftware\anaconnda\envs\tor\python.exe -m simulator.main
```

产物落到 `1D_1phase/output/simulator/`：

- `pressure.npy` — `(n_steps+1, nx)` 全张量
- `pressure.csv` — `time_day, cell_0, ..., cell_{nx-1}`，给 surrogate / PINN 对照等使用
- `rates.csv` — `time_day, well_0_cell{X}, well_1_cell{Y}, ...`，每口井每步的实际流量（含 BHP 井算出的 q）
- `config.json` — 配置快照（含派生量）
- `profiles.png` — 几个时刻的压力剖面叠加
- `heatmap.png` — `(x, t)` 时空热图
- `well_rates.png` — 每口井实际流量曲线（横轴 day，纵轴 m³/s）

主程序还会在终端打印各口井的最终格压力，对 BHP 井额外输出平均流量与累计体积。

## 测试

```powershell
D:\WorkSoftware\anaconnda\envs\tor\Scripts\pytest.exe
```

测试都集中在数学/物理上的关键不变量：

- `test_solver.py` — **单步离散质量守恒**，参数化覆盖 5 种「边界 × 井」组合。
- `test_simulator.py` — 全流程稳态参考：平衡态保持、两侧定压无井的线性稳态、BHP 井稳态产量与
  `PI·(P_cell − P_wf)` 自洽、注采平衡场景下整体压力不漂移。
- `test_well.py` — 井类型工厂、Peaceman PI 公式自洽。

> 写新代码时，数学/物理意义复杂的函数写测试，trivial 的拼装层（IO、画图、dataclass）跳过。

## 目录结构

```
1D_1phase/
├── simulator/
│   ├── config.py     # Config + BoundarySpec
│   ├── well.py       # Well Protocol, ConstantRateWell, BHPWell,
│   │                 # RateWellSpec, BHPWellSpec, make_wells, peaceman_pi_1d
│   ├── solver.py     # assemble_and_solve（单步隐式）
│   ├── core.py       # PressureHistory + run（时间循环）
│   ├── io.py         # save_results / load_pressure
│   ├── plot.py       # plot_profiles / plot_heatmap
│   ├── main.py       # 入口
│   └── tests/
├── output/simulator/ # 模拟产物（gitignored）
├── pinn/             # 推进中：本系统的 PINN（单场景纯物理损失；无井+定流量井已驯服，
│                     # 软约束 IC + 自适应权重；解析真解校验 + 模拟器交叉验证）。架构见 CLAUDE.md
├── surrogate/        # 计划中：参数化代理模型（扰动井控生成数据 + 物理损失正则）
└── README.md
```

`pinn/` 与 `surrogate/` 是**两条平行的方向**，不是 PINN 的两个阶段。详细约定见仓库根目录的
`CLAUDE.md`。这两个文件夹里写脚本时，**可以直接** `from simulator.config import Config;
from simulator.core import run` 来在内存里调用模拟器拿参考解 / 训练数据（包根都是
`1D_1phase/`），也可以走 `output/simulator/` 落盘文件。

## 扩展约定

- **唯一的扩展轴是 `Well` Protocol**。求解器只通过 `cell_index`、`rhs_term(p_old, dt)`、
  `diag_term(dt)` 与井交互。新井类型只要实现这三件事，求解器和 Config 都不用动。
- 不要在 `simulator/` 里堆 2D / 双相 / PINN / surrogate 的「通用基类」——那些场景会在新文件夹里完全重写。
