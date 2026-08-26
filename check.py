"""Generate a compact 10-case robustness sheet."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt

from train import draw_result, run


def main():
    parser = argparse.ArgumentParser(
        description="Run random GCML tasks with exactly one blocked passage"
    )
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output", default="check_10_blocked.png")
    args = parser.parse_args()

    results = []
    for case in range(args.cases):
        settings = SimpleNamespace(
            width=100,
            height=100,
            seed=args.seed + case,
            latent_dim=24,
            epochs=80,
            coverage_radius=8.0,
            max_landmarks=100,
            noise=0.18,
            graph_rollouts=256,
            spline_rollouts=64,
            random_task=True,
            block_passage=True,
        )
        result = run(settings)
        results.append(result)
        print(
            f"case {case + 1:02d}: "
            f"graph={result['high_level']['success']}, "
            f"curve={result['curve_metrics']['safe']}, "
            f"fallback={result['high_level']['fallback_used']}"
        )

    columns = 5 if args.cases > 4 else args.cases
    rows = (args.cases + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
    axes = [axes] if args.cases == 1 else list(axes.flat)
    for case, (ax, result) in enumerate(zip(axes, results), start=1):
        draw_result(ax, result, compact=True)
        ax.set_title(
            f"#{case:02d}  blocked=1  safe={result['curve_metrics']['safe']}",
            fontsize=9,
        )
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(
        f"{args.cases} random tasks: exactly one passage blocked per map",
        fontsize=13,
    )
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print("saved:", output)


if __name__ == "__main__":
    main()
