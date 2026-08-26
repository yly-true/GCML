import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcml import LandmarkGCML
from landmarks import LandmarkGraph
from navigation_map import LargeNavigationMap
from spline_planner import SplineRolloutPlanner


def run(args):
    navigation_map = LargeNavigationMap(
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    blocked_passage = None
    if args.block_passage:
        blocked_passage = navigation_map.block_random_passage(
            np.random.default_rng(args.seed + 100)
        )

    graph = LandmarkGraph(
        navigation_map,
        coverage_radius=args.coverage_radius,
        max_landmarks=args.max_landmarks,
        seed=args.seed,
    )
    model = LandmarkGCML(
        graph,
        latent_dim=args.latent_dim,
        noise_std=args.noise,
        seed=args.seed,
    )
    model.train(epochs=args.epochs)

    if args.random_task:
        start, goal = navigation_map.random_task(
            np.random.default_rng(args.seed + 200)
        )
    else:
        start = np.array([5.0, 5.0])
        goal = np.array([args.width - 6.0, args.height - 6.0])
    if not navigation_map.is_free(start) or not navigation_map.is_free(goal):
        raise RuntimeError("Selected start or goal is not collision-free")

    if navigation_map.line_is_free(start, goal, margin=0.05):
        node_path = []
        high_level = {
            "success": True,
            "candidate_count": 0,
            "fallback_used": False,
            "score": float(np.linalg.norm(goal - start)),
        }
    else:
        start_candidates = graph.visible_nodes(start, count=8)
        goal_candidates = graph.visible_nodes(goal, count=8)
        start_node, goal_node = min(
            (
                (source, target)
                for source in start_candidates
                for target in goal_candidates
            ),
            key=lambda pair: (
                np.linalg.norm(start - graph.positions[pair[0]])
                + graph.shortest_distances[pair[0], pair[1]]
                + np.linalg.norm(goal - graph.positions[pair[1]])
            ),
        )
        high_level = model.rollout(
            start_node,
            goal_node,
            candidates=args.graph_rollouts,
        )
        node_path = high_level["nodes"]

    waypoints = graph.waypoints(start, goal, node_path)
    spline_planner = SplineRolloutPlanner(
        navigation_map,
        candidates=args.spline_rollouts,
        seed=args.seed + 300,
    )
    curve, curve_metrics = spline_planner.plan(waypoints)

    return {
        "map": navigation_map,
        "graph": graph,
        "model": model,
        "start": start,
        "goal": goal,
        "blocked_passage": blocked_passage,
        "high_level": high_level,
        "waypoints": waypoints,
        "curve": curve,
        "curve_metrics": curve_metrics,
    }


def draw_result(ax, result, compact=False):
    navigation_map = result["map"]
    graph = result["graph"]
    curve = result["curve"]
    waypoints = result["waypoints"]

    ax.imshow(
        navigation_map.occupancy,
        origin="lower",
        extent=(-0.5, navigation_map.width - 0.5, -0.5, navigation_map.height - 0.5),
        cmap="Blues",
        alpha=0.75,
        interpolation="nearest",
    )
    for source in range(graph.num_nodes):
        for target in np.flatnonzero(graph.adjacency[source]):
            if target <= source:
                continue
            points = graph.positions[[source, target]]
            ax.plot(points[:, 0], points[:, 1], color="#b0bec5", linewidth=0.45)
    ax.scatter(
        graph.positions[:, 0],
        graph.positions[:, 1],
        s=5 if compact else 12,
        color="#546e7a",
        label=f"landmarks ({graph.num_nodes})",
        zorder=3,
    )
    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "--",
        color="#7e57c2",
        linewidth=0.8 if compact else 1.2,
        label="GCML landmark route",
        zorder=4,
    )
    ax.plot(
        curve[:, 0],
        curve[:, 1],
        color="#ef6c00",
        linewidth=1.7 if compact else 2.6,
        label="selected B-spline rollout",
        zorder=5,
    )
    ax.scatter(
        *result["start"],
        s=35 if compact else 90,
        color="#1565c0",
        label="start",
        zorder=6,
    )
    ax.scatter(
        *result["goal"],
        s=65 if compact else 150,
        marker="*",
        color="#f9a825",
        label="goal",
        zorder=6,
    )
    if result["blocked_passage"] is not None:
        ax.scatter(
            *result["blocked_passage"],
            s=38 if compact else 90,
            marker="s",
            color="#d32f2f",
            label="blocked passage",
            zorder=6,
        )
    ax.set_xlim(-0.5, navigation_map.width - 0.5)
    ax.set_ylim(-0.5, navigation_map.height - 0.5)
    ax.set_aspect("equal")
    if not compact:
        ax.set_xlabel("continuous x")
        ax.set_ylabel("continuous y")
        ax.set_title("Sparse Landmark GCML + Smooth B-spline Rollout")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_result(result, output_path, show=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_result(ax, result)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(result, output_path):
    graph = result["graph"]
    model = result["model"]
    diagnostics = model.diagnostics()
    high_level = result["high_level"]
    curve = result["curve_metrics"]
    grid_states = result["map"].width * result["map"].height

    print("map cells:", grid_states)
    print("landmarks:", graph.num_nodes)
    print("directed landmark actions:", graph.num_actions)
    print("latent dimension:", model.latent_dim)
    print("Q/V/W parameter count:", diagnostics["parameter_count"])
    print("Q change:", f"{diagnostics['Q_change']:.6f}")
    print("V change:", f"{diagnostics['V_change']:.6f}")
    print("W change:", f"{diagnostics['W_change']:.6f}")
    print("transition RMSE:", f"{diagnostics['transition_rmse']:.6f}")
    print("finite parameters:", diagnostics["finite"])
    print("graph rollout success:", high_level["success"])
    print("graph rollout candidates:", high_level["candidate_count"])
    print("shortest-path fallback used:", high_level["fallback_used"])
    print("landmark waypoints:", len(result["waypoints"]))
    print("spline candidates:", curve["candidate_count"])
    print("curve collision-free:", curve["safe"])
    print("curve length:", f"{curve['path_length']:.3f}")
    print("curve curvature cost:", f"{curve['curvature']:.3f}")
    print("minimum wall clearance:", f"{curve['minimum_clearance']:.3f}")
    print("saved:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sparse landmark GCML with smooth B-spline rollout"
    )
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--coverage-radius", type=float, default=8.0)
    parser.add_argument("--max-landmarks", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.18)
    parser.add_argument("--graph-rollouts", type=int, default=384)
    parser.add_argument("--spline-rollouts", type=int, default=96)
    parser.add_argument("--random-task", action="store_true")
    parser.add_argument("--block-passage", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="landmark_gcml.png")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run(args)
    plot_result(result, args.output, show=args.show)
    print_summary(result, args.output)


if __name__ == "__main__":
    main()
