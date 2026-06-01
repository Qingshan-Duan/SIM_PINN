# sim_pinn

从零手写的学习项目：用**数值模拟**和**机器学习（PINN / 数据驱动代理）** 求解同一套油藏流动物理，
对比两条路线各自的适用边界。每个物理系统（1D 单相、1D 两相…）一个顶层文件夹，内部再分三条**平行**的线：

- **数值模拟器**（`simulator/`）—— 有限体积 + 隐式时间推进，作为参考"真值"求解器；
- **PINN**（`pinn*/`，Family 1）—— 纯物理损失，网络直接拟合 PDE 解，不读数据；
- **代理模型**（`surrogate/`，Family 2）—— 数据驱动 + 离散物理残差正则。

> 目前整个 **1D 单相** 系统已完成。一个值得记的结论：1D 单相是**线性**问题，一步算子精确仿射，
> 数据本身就够、物理正则可测得的价值集中在**去噪**；物理真正的大舞台在**非线性**（下一站 1D 两相）。

## 当前状态

| 路径 | 内容 | 状态 |
| --- | --- | --- |
| `1D_1phase/simulator/` | 一维单相 FVM 隐式模拟器，多井 + 可配置边界 | ✅ 完成 |
| `1D_1phase/pinn/` | 1D 单相 PINN：单场景纯物理损失求解 PDE。无井 + 定流量井均已驯服（软约束 + 自适应权重，对解析真解 L2≈2e-4） | ✅ 完成 |
| `1D_1phase/pinn_param/` | 参数化 PINN：一个网络 `(x̂,t̂,q̂)→p̂` 覆盖 ±20% 流量范围，纯物理无数据。跨 q R²≈0.9999，追平单流量基线 | ✅ 完成 |
| `1D_1phase/pinn_tv/` | 时变井控 PINN：整条 15 段调度作输入 `(x̂,t̂,q̂₁..q̂₁₅)→p̂`（17 维），纯物理。加 Fourier 时间特征后 R²≈0.999，追平常流量 | ✅ 完成 |
| `1D_1phase/surrogate/` | 1D 单相代理模型：自回归 + 离散 BE 物理正则。三档对比（纯数据/数据+物理/物理-only）。**结论：线性问题数据本身就够，物理价值在去噪；大舞台是非线性两相** | ✅ 完成 |
| `1D_2phase/` | 1D 两相（水驱）物理系统：模拟器 + PINN + 代理（同样内部结构）。非线性，物理正则该显威 | 🕐 下一步 |

## 运行方式

> ⚠️ **每个子项目都是独立的 Python 包**。`simulator` / `pinn` 这些 import 用的是包名而不是绝对路径，
> 所以**必须先 `cd` 到对应的物理系统目录**再跑命令，不能从仓库根目录跑。

依赖：Python 3.10+，`numpy` / `scipy` / `matplotlib` / `pytorch`（PINN 与代理用）。下面的 `python` / `pytest`
指你所在环境的解释器。

### `1D_1phase`

```powershell
cd 1D_1phase

# 运行模拟器，产物落到 1D_1phase/output/simulator/
python -m simulator.main

# 跑测试
pytest

# 跑某一个测试
pytest simulator/tests/test_solver.py
```

详细配置/物理模型说明：[`1D_1phase/README.md`](1D_1phase/README.md)。

## 路线图

| 系统 | 状态 |
| --- | --- |
| `1D_1phase/`（1D 单相） | ✅ 完成：模拟器 + 三个 PINN（Family 1）+ 代理（Family 2） |
| `1D_2phase/`（1D 两相水驱） | 🕐 下一站：IMPES 模拟器（自定义/调和渗透率、Corey 相渗、注水定流量 + 采油定流压、双产量）+ PINN + 代理。非线性 ⇒ 物理正则与（将来）参数反演才显威 |

> **2D 暂不做**——单相线性对代理太简单，先把两相非线性这块更有分辨度的做透。
> 新子项目同样遵循"先 `cd` 进目录再跑"，具体命令以各自子目录的 README 为准。

> 子项目之间没有共享框架/基类，但**同一物理系统目录下共享 Python 包根**——`pinn/` 和 `surrogate/`
> 里写脚本时可以直接 `from simulator.config import Config; from simulator.core import run`
> 在内存里调用模拟器，也可以走 `output/simulator/` 落盘文件。每个子项目仍然独立改、独立跑、独立失败。

## 约定速查

- **SI 单位**贯穿所有数值代码；展示层（plot/print）才换算成 MPa、day。
- **测试纪律**：数学/物理意义复杂的函数写测试，trivial 拼装层（IO、画图、dataclass 字段）跳过。
- **Git**：默认分支 `main`，开发分支 `feat/...`。
- 生成产物 `output/`、设计文档 `docs/` 均已 gitignore。
