"""生成一张GCML离散路径与连续曲线的对照图。"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import train
from evaluation import Tester, config
from .smooth import simplify_path, smooth_path


# ======================== 全部曲线参数 ========================
SEED, CASES, COLUMNS = 0, 12, 4
SAMPLES, TENSION = 40, .8
FIGURE = Path("outputs/figures/10_curve_path.png")
# =============================================================


def main():
    tester = Tester()
    m = tester.m
    free = lambda xy: bool(np.all(train.World.clearance(m["sdf"], xy) >= float(m["radius"])+.08))

    with plt.rc_context(config.CHINESE_STYLE):
        rows = (CASES+COLUMNS-1)//COLUMNS
        fig, axes = plt.subplots(rows, COLUMNS, figsize=(4*COLUMNS, 4.2*rows), squeeze=False)
        for i, ax in enumerate(axes.flat[:CASES]):
            case = tester.case(SEED+i)
            route, success, _ = tester.gcml(case, seed=SEED+400+i)
            if not success:
                raise RuntimeError(f"案例{i+1}导航失败，没有完整路径可曲线化")
            points = np.vstack((case["start"], m["points"][route], case["goal"]))
            reduced = simplify_path(points, free)
            curve = smooth_path(reduced, SAMPLES, TENSION, free)
            ax.imshow(m["grid"], origin="lower", cmap=ListedColormap(["#ffffff", "#8495a7"]))
            ax.plot(*points.T, "o--", color="#2878b5", lw=1, ms=3)
            ax.plot(*curve.T, color="#e56a19", lw=2.2)
            ax.scatter(*case["start"], color="#172b4d", s=30, zorder=4)
            ax.scatter(*case["goal"], color="#f9a825", marker="*", s=75, zorder=4)
            ax.set(title=f"案例{i+1:02d}｜路径点 {len(points)}→{len(reduced)}", xticks=[], yticks=[])
        for ax in axes.flat[CASES:]:
            ax.set_visible(False)
        handles = [Line2D([], [], color="#2878b5", marker="o", ls="--", label="GCML原始离散线段"),
                   Line2D([], [], color="#e56a19", lw=2.2, label="安全三次多项式曲线")]
        fig.suptitle(f"GCML逐点对曲线化｜{CASES}组案例｜先删冗余折点，再平滑且检查墙体", fontsize=18)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=2, frameon=False)
        fig.subplots_adjust(left=.025, right=.985, bottom=.025, top=.925, wspace=.12, hspace=.22)
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("已保存：", FIGURE)


if __name__ == "__main__":
    main()
