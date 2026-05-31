"""模拟器入口。从 1D_1phase/ 运行：python -m simulator.main"""
from pathlib import Path

from simulator.config import Config
from simulator.core import run
from simulator.io import save_results
from simulator.plot import plot_heatmap, plot_profiles, plot_well_rates
from simulator.well import BHPWellSpec


def main() -> None:
    cfg = Config()
    history = run(cfg)

    here = Path(__file__).resolve().parent          # .../1D_1phase/simulator
    out_dir = here.parent / "output" / "simulator"  # .../1D_1phase/output/simulator
    out_dir.mkdir(parents=True, exist_ok=True)

    save_results(history, cfg, out_dir)
    plot_profiles(history, cfg, out_dir / "profiles.png")
    plot_heatmap(history, cfg, out_dir / "heatmap.png")
    plot_well_rates(history, cfg, out_dir / "well_rates.png")

    print(f"Done. Output written to: {out_dir}")
    for i, spec in enumerate(cfg.wells):
        idx = spec.cell_index
        print(
            f"  cell {idx}: P_init = {history.p[0, idx] / 1e6:.3f} MPa, "
            f"P_final = {history.p[-1, idx] / 1e6:.3f} MPa"
        )
        if isinstance(spec, BHPWellSpec):
            q_series = history.q[:, i]
            q_mean = q_series.mean()
            q_total_vol = q_series.sum() * cfg.dt  # m^3 (累计体积，符号同 q)
            label = "采出" if spec.kind == "producer" else "注入"
            print(
                f"    BHP {label}井: avg q = {q_mean:+.3e} m^3/s, "
                f"累计 = {abs(q_total_vol):.3e} m^3 (符号 {'-' if q_total_vol < 0 else '+'})"
            )


if __name__ == "__main__":
    main()
