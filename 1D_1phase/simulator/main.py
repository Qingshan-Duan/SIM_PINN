"""模拟器入口。从 1D_1phase/ 运行：python -m simulator.main"""
from pathlib import Path

from simulator.config import Config
from simulator.well import ConstantRateWell
from simulator.core import run
from simulator.io import save_results
from simulator.plot import plot_profiles, plot_heatmap


def main() -> None:
    cfg = Config()
    wells = [ConstantRateWell(cell_index=cfg.well_index, rate=cfg.well_rate)]

    history = run(cfg, wells)

    # 输出目录：1D_1phase/output/simulator/
    here = Path(__file__).resolve().parent          # .../1D_1phase/simulator
    out_dir = here.parent / "output" / "simulator"  # .../1D_1phase/output/simulator
    out_dir.mkdir(parents=True, exist_ok=True)

    save_results(history, cfg, out_dir)
    plot_profiles(history, cfg, out_dir / "profiles.png")
    plot_heatmap(history, cfg, out_dir / "heatmap.png")

    print(f"Done. Output written to: {out_dir}")
    print(f"  Initial pressure (cell {cfg.well_index}): {history.p[0, cfg.well_index] / 1e6:.3f} MPa")
    print(f"  Final pressure   (cell {cfg.well_index}): {history.p[-1, cfg.well_index] / 1e6:.3f} MPa")


if __name__ == "__main__":
    main()
