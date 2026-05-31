# sim_pinn

学习项目：油藏数值模拟 + PINN 求同一物理。每个物理系统（1D 单相、2D、双相…）一个顶层文件夹，
里面再分**数值模拟器**、**PINN**、**代理模型（surrogate）** 三条平行的线。

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

Python 环境在 conda 的 `tor`，但**统一用绝对路径直接调用**，不要套 `conda run`：

- Python：`D:\WorkSoftware\anaconnda\envs\tor\python.exe`
- pytest：`D:\WorkSoftware\anaconnda\envs\tor\Scripts\pytest.exe`

### `1D_1phase`

```powershell
cd 1D_1phase

# 运行模拟器，产物落到 1D_1phase/output/simulator/
D:\WorkSoftware\anaconnda\envs\tor\python.exe -m simulator.main

# 跑测试
D:\WorkSoftware\anaconnda\envs\tor\Scripts\pytest.exe

# 跑某一个测试
D:\WorkSoftware\anaconnda\envs\tor\Scripts\pytest.exe simulator/tests/test_solver.py
```

详细配置/物理模型说明：[`1D_1phase/README.md`](1D_1phase/README.md)。

### 未来子项目

每个新加的子项目都遵循同样的"先 cd 再跑"规则，比如：

```powershell
cd 1D_1phase            # 跑 1D 的 simulator / pinn / surrogate
cd 2D                   # 跑 2D 子项目
```

具体命令以各自子目录的 README 为准。

## 未来计划

按当前顺序：

1. **`1D_1phase/pinn/`** — 单场景纯 PINN 求解 1D 单相方程（只用物理残差 + IC + BC，不读模拟器数据）。
   单目录迭代，不分 stage。无井 + 中心定流量井已走通（关键招数：高斯正则化点源、软约束 IC、
   梯度范数自适应权重；误差对**解析真解**算、模拟器留作交叉校验）。待续：时变井控等。
2. **`1D_1phase/surrogate/`** —— 与 PINN **平行**的第二条线：自回归代理 + 离散 BE 物理正则。
   ✅ 已完成；结论见 `surrogate/NOTES.md`（线性问题数据本身就够，物理价值集中在去噪/反演）。
3. **`1D_2phase/`** — **下一站**：1D 两相水驱。重写模拟器（IMPES、自定义/调和渗透率、Corey 相渗、
   注水定流量 + 采油定流压、双产量），再做 PINN / 代理。非线性 ⇒ 物理正则与（将来）参数反演才显威。
   **2D 暂时不做。**

> 子项目之间没有共享框架/基类，但**同一物理系统目录下共享 Python 包根**——`pinn/` 和 `surrogate/`
> 里写脚本时可以直接 `from simulator.config import Config; from simulator.core import run`
> 在内存里调用模拟器，也可以走 `output/simulator/` 落盘文件。每个子项目仍然独立改、独立跑、独立失败。

## 约定速查

- **SI 单位**贯穿所有数值代码；展示层（plot/print）才换算成 MPa、day。
- **测试纪律**：数学/物理意义复杂的函数写测试，trivial 拼装层（IO、画图、dataclass 字段）跳过。
- **Git**：默认分支 `main`，开发分支 `feat/...`。远端 `origin → github.com/Qingshan-Duan/SIM_PINN.git`。
- 生成产物 `1D_1phase/output/`、设计文档 `/docs/`、`CLAUDE.md` 都已 gitignore。
