"""加载已训练模型进行离散地标导航；不训练、无fallback、无连续曲线。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import train


# ======================== 全部推理参数 ========================
MODEL_FILE = "gcml_model.npz"
SEED = 100
TEST_CASES = 15
ROLLOUTS = 100              # 每个真实决策点生成的候选轨迹数
ROLLOUT_HORIZON = 32        # 每条候选最多在潜空间想象32步
EXECUTION_HORIZON = 64      # 最多真实执行并重规划64次
LATENT_GOAL_TOLERANCE = .8
NOISE = .18

BENCHMARK_CASES = 30
ROLLOUT_BUDGETS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100)
HORIZON_BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
CASE_FIGURE = "test_15_cases.png"
REPORT_FIGURE = "test_report.png"
# =============================================================


def load_model():
    """读取npz，并预计算固定图的出边表和最短距离；不会修改Q/V/W/G。"""
    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError("请先运行 python train.py")
    with np.load(MODEL_FILE) as data:
        model = {key: data[key].copy() for key in data.files}

    points, edges = model["points"], model["edges"]
    model["sdf"] = train.distance_field(model["grid"])
    model["available"] = [np.flatnonzero(edges[:, 0] == node)
                          for node in range(len(points))]
    pair = np.linalg.norm(points[:, None]-points[None], axis=2)
    distance = np.where(model["adjacency"], pair, np.inf)
    np.fill_diagonal(distance, 0)
    for k in range(len(points)):
        distance = np.minimum(distance, distance[:, k, None]+distance[None, k, :])
    model["distance"] = distance
    return model


def random_point(model, rng):
    """从固定地图自由空间采样一个连续坐标。"""
    cells = np.argwhere(~model["grid"])
    while True:
        y, x = cells[rng.integers(len(cells))]
        point = np.array([x, y])+rng.uniform(-.3, .3, 2)
        if train.clearance(model["sdf"], point)[0] >= float(model["radius"])+.5:
            return point


def make_case(model, seed):
    """只随机连续起终点；固定地图、节点、动作和模型参数完全不变。"""
    rng, points = np.random.default_rng(seed+200), model["points"]
    while True:
        start, goal = random_point(model, rng), random_point(model, rng)
        if np.linalg.norm(start-goal) <= .55*min(model["grid"].shape):
            continue
        visible = lambda p: [int(i) for i in np.argsort(np.linalg.norm(points-p, axis=1))
                             if train.line_free(model["sdf"], p, points[i])][:6]
        pairs = [(i, j) for i in visible(start) for j in visible(goal)
                 if np.isfinite(model["distance"][i, j])]
        if pairs:
            # 图距离只用于连续坐标挂接地标，不参与后续GCML rollout。
            start_node, goal_node = min(
                pairs, key=lambda z: np.linalg.norm(start-points[z[0]])+
                model["distance"][z]+np.linalg.norm(goal-points[z[1]])
            )
            return dict(start=start, goal=goal, start_node=start_node, goal_node=goal_node)


def imagine_first_action(model, node, goal, real_actions, samples, horizon, rng):
    """并行生成samples条高维候选，评分后返回最佳候选的第一个动作。"""
    q, v, w, g = (model[name] for name in "QVWG")
    action_count = len(model["edges"])
    target = q[:, goal, None]                         # shape=(潜维数,1)
    state = np.repeat(q[:, node, None], samples, axis=1)
    paths = [[] for _ in range(samples)]              # 每条候选的动作编号序列
    arrived = np.zeros(samples, bool)
    residual = np.full(samples, np.inf)
    noise = rng.normal(0, NOISE, (samples, horizon, action_count))

    for step in range(horizon):
        active = np.flatnonzero(~arrived)
        if not len(active):
            break
        utility = w@(target-state)                    # 论文公式(16)
        utility /= np.maximum(np.linalg.norm(utility, axis=0), 1e-9)
        gate = np.zeros_like(utility) if step == 0 else g@state  # 公式(17)
        if step == 0:
            gate[real_actions, :] = 1                 # 第一想象步采用真实affordance
        eligibility = gate*(utility+noise[:, step].T) # 公式(18)
        if step == 0:
            eligibility[gate == 0] = -np.inf
        chosen = np.argmax(eligibility, axis=0)       # 公式(19)

        for sample in active:
            action = int(chosen[sample])
            if not np.isfinite(eligibility[action, sample]):
                continue
            paths[sample].append(action)
            state[:, sample] += v[:, action]          # 公式(20)，不查询真实下一节点
            residual[sample] = np.linalg.norm(state[:, sample]-target[:, 0])
            arrived[sample] = residual[sample] <= LATENT_GOAL_TOLERANCE

    # 二级排序：(失败标志, 成功时步数 / 失败时残差)。成功永远优先。
    ranked = [((not arrived[i], len(path) if arrived[i] else residual[i]), i)
              for i, path in enumerate(paths) if path]
    if not ranked:
        return None, (False, 0, 0, 0, 0)
    best = min(ranked, key=lambda item: item[0])[1]
    valid = [tuple(path) for path in paths if path]
    stats = (bool(arrived[best]), len(paths[best]), int(arrived.sum()),
             len({path[0] for path in valid}), len(set(valid)))
    return int(paths[best][0]), stats


def gcml_rollout(model, case, samples=ROLLOUTS, horizon=ROLLOUT_HORIZON, seed=0):
    """类似MPC：选最佳想象轨迹，但每轮只真实执行它的第一个动作。"""
    edges, available = model["edges"], model["available"]
    node, goal = case["start_node"], case["goal_node"]
    route, trace = [node], []

    for step in range(EXECUTION_HORIZON):
        if node == goal:
            break
        real_actions = available[node]
        if not len(real_actions):
            break
        action, stats = imagine_first_action(
            model, node, goal, real_actions, samples, horizon,
            np.random.default_rng(seed+step)  # 各决策独立，使不同samples共享噪声前缀
        )
        if action is None:
            break
        trace.append(stats)
        node = int(edges[action, 1])                  # 这里只执行最佳候选第一步
        route.append(node)
    return route, node == goal, trace


def cml_rollout(model, case):
    """CML基线：无噪声、无G、无多步想象，每次真实状态只贪心走一步。"""
    q, w, edges = model["Q"], model["W"], model["edges"]
    node, goal = case["start_node"], case["goal_node"]
    route = [node]                                  # 每次一步决策，累计后仍形成完整路线
    for _ in range(EXECUTION_HORIZON):
        if node == goal:
            return route, True
        actions = model["available"][node]
        if not len(actions):
            return route, False
        utility = w@(q[:, goal]-q[:, node])
        action = int(actions[np.argmax(utility[actions])])
        node = int(edges[action, 1])
        route.append(node)
    return route, False


def model_metrics(model):
    """只读检查训练误差、G恢复率以及buffer覆盖率。"""
    q, v, g, edges = model["Q"], model["V"], model["G"], model["edges"]
    assert v.shape == (len(q), len(edges))
    assert model["W"].shape == g.shape == (len(edges), len(q))
    assert all(np.isfinite(model[name]).all() for name in "QVWG")
    assert np.isfinite(model["distance"]).all(), "地标图不连通"
    assert np.all(model["adjacency"][:len(model["passages"])].sum(1) >= 2)
    actions = model["buffer_a"]
    assert np.all(edges[actions, 0] == model["buffer_o"])
    assert np.all(edges[actions, 1] == model["buffer_next"])
    delta = np.column_stack([q[:, j]-q[:, i] for i, j in edges])
    true_g = np.zeros((len(edges), q.shape[1]))
    true_g[np.arange(len(edges)), edges[:, 0]] = 1
    prediction, recall = g@q, []
    for node in range(q.shape[1]):
        actual = np.flatnonzero(edges[:, 0] == node)
        predicted = np.argsort(prediction[:, node])[-len(actual):]
        recall.append(len(set(actual)&set(predicted))/len(actual))
    visited_nodes = np.unique(np.concatenate((model["buffer_o"].ravel(),
                                               model["buffer_next"].ravel())))
    visited_actions = np.unique(model["buffer_a"])
    return dict(
        transition=float(np.sqrt(np.mean((delta-v)**2))),
        affordance=float(np.sqrt(np.mean((true_g-prediction)**2))),
        recall=float(np.mean(recall)),
        node_coverage=len(visited_nodes)/q.shape[1],
        action_coverage=len(visited_actions)/len(edges),
    )


def benchmark(model):
    """分别量化候选数量、想象horizon和潜状态维数。"""
    cases = [make_case(model, SEED+1000+i) for i in range(BENCHMARK_CASES)]
    untrained = dict(model)
    for name in "QVWG":
        untrained[name] = model[name+"0"]

    def evaluate(current, samples=ROLLOUTS, horizon=ROLLOUT_HORIZON, seed=0):
        runs = [gcml_rollout(current, case, samples, horizon, seed+i)
                for i, case in enumerate(cases)]
        trace = [hit for _, _, items in runs for hit, *_ in items]
        return 100*np.mean([run[1] for run in runs]), 100*np.mean(trace) if trace else 100.

    cml_rate = 100*np.mean([cml_rollout(model, case)[1] for case in cases])
    trained = [evaluate(model, budget, seed=SEED+4000) for budget in ROLLOUT_BUDGETS]
    random = [evaluate(untrained, budget, seed=SEED+5000) for budget in ROLLOUT_BUDGETS]
    count = {"GCML": [x[0] for x in trained], "untrained GCML": [x[0] for x in random]}
    horizon_stats = [evaluate(model, horizon=value, seed=SEED+6000)
                     for value in HORIZON_BUDGETS]

    dimensions, dim_gcml, dim_cml = model["ablation_dims"].tolist(), [], []
    for dim in dimensions:
        current = dict(model)
        if dim != model["Q"].shape[0]:
            for name in "QVWG":
                current[name] = model[f"{name}_d{dim}"]
        dim_gcml.append(evaluate(current, seed=SEED+7000)[0])
        dim_cml.append(100*np.mean([cml_rollout(current, case)[1] for case in cases]))
    return dict(count=count, count_reach=[x[1] for x in trained], cml=cml_rate,
                horizon=[x[0] for x in horizon_stats],
                horizon_reach=[x[1] for x in horizon_stats],
                dimensions=dimensions, dim_gcml=dim_gcml, dim_cml=dim_cml)


def draw_case(ax, model, result, index):
    """橙色为GCML；蓝色虚线是CML重复执行一步决策后形成的真实路线。"""
    points, adjacency = model["points"], model["adjacency"]
    ax.imshow(model["grid"], origin="lower", cmap="Blues", alpha=.75)
    for i, j in np.argwhere(np.triu(adjacency, 1)):
        ax.plot(points[[i, j], 0], points[[i, j], 1], color="#b0bec5", lw=.45)
    passage_count = len(model["passages"])
    ax.scatter(*points[passage_count:].T, s=8, color="#546e7a")
    ax.scatter(*points[:passage_count].T, s=16, marker="s", color="#6a1b9a",
               label="passage landmark")
    def physical(route, success):
        path = np.vstack(([result["start"]], points[route]))
        return np.vstack((path, result["goal"])) if success else path
    ax.plot(*physical(result["gcml_route"], result["gcml_success"]).T,
            color="#ef6c00", lw=2, label="GCML")
    decisions = points[result["gcml_route"][:-1]]
    reached = np.asarray([hit for hit, *_ in result["rollout_trace"]], bool)
    if len(decisions):
        ax.scatter(*decisions[reached].T, facecolors="none", edgecolors="#2e7d32",
                   s=35, lw=1.5, zorder=4, label="rollout reached")
        ax.scatter(*decisions[~reached].T, color="#c62828", marker="x",
                   s=35, lw=1.5, zorder=4, label="rollout missed")
    cml = physical(result["cml_route"], result["cml_success"])
    seen = set()  # 重复走同一边时只画一次，避免多层虚线叠成实线
    for a, b in zip(cml[:-1], cml[1:]):
        key = tuple(sorted((tuple(a), tuple(b))))
        if key not in seen:
            ax.plot(*np.vstack((a, b)).T, color="#1565c0", ls=(0, (3, 3)),
                    lw=1.4, label="CML repeated 1-step" if not seen else None)
            seen.add(key)
    ax.scatter(*cml[1:-1].T, color="#1565c0", s=7, zorder=3)
    ax.scatter(*result["start"], color="#1565c0", s=30)
    ax.scatter(*result["goal"], color="#f9a825", marker="*", s=65)
    hits = int(reached.sum())
    ax.set(title=f"#{index:02d} G={result['gcml_success']} C={result['cml_success']} "
                 f"Rhit={hits}/{len(result['rollout_trace'])}",
           aspect="equal")
    if index == 2:
        ax.legend(fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])


def save_figures(model, results, scores, metrics):
    fig, axes = plt.subplots(3, 5, figsize=(16, 9.5))
    for index, (ax, result) in enumerate(zip(axes.flat, results), 1):
        draw_case(ax, model, result, index)
    fig.tight_layout(); fig.savefig(CASE_FIGURE, dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for name, rate in scores["count"].items():
        ax.plot(ROLLOUT_BUDGETS, rate, "o-", label=name)
    ax.plot(ROLLOUT_BUDGETS, scores["count_reach"], "o--", label="rollout reached")
    ax.scatter([1], [scores["cml"]], marker="D", s=65, label="CML one-step")
    ax.set(xscale="log", xticks=ROLLOUT_BUDGETS, ylim=(-2, 102),
           xlabel="trajectory count", ylabel="success rate (%)", title="Effect of trajectory count")
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()); ax.grid(alpha=.2); ax.legend()

    ax = axes[0, 1]
    ax.plot(HORIZON_BUDGETS, scores["horizon"], "o-", label="navigation success")
    ax.plot(HORIZON_BUDGETS, scores["horizon_reach"], "o--", label="rollout reached")
    ax.set(ylim=(-2, 102), xlabel="rollout horizon", ylabel="success rate (%)",
           title=f"Effect of horizon (trajectories={ROLLOUTS})")
    ax.grid(alpha=.2); ax.legend()

    ax = axes[1, 0]
    ax.plot(scores["dimensions"], scores["dim_gcml"], "o-", label="GCML")
    ax.plot(scores["dimensions"], scores["dim_cml"], "o--", label="CML one-step")
    ax.set(ylim=(-2, 102), xlabel="latent dimension", ylabel="success rate (%)",
           title="Effect of high-dimensional state")
    ax.grid(alpha=.2); ax.legend()

    trace = [item for result in results for item in result["rollout_trace"]]
    hit_depths = [depth for hit, depth, *_ in trace if hit]
    def route_stats(name):
        good = [r for r in results if r[name+"_success"]]
        steps = [len(r[name+"_route"])-1 for r in good]
        return np.mean(steps) if steps else 0
    info = axes[1, 1]
    info.axis("off")
    info.text(.05, .95,
              f"nodes / actions: {len(model['points'])} / {len(model['edges'])}\n"
              f"latent dimension: {model['Q'].shape[0]}\n"
              f"buffer: {model['buffer_o'].shape[0]} x {model['buffer_o'].shape[1]}\n"
              f"node coverage: {metrics['node_coverage']:.1%}\n"
              f"action coverage: {metrics['action_coverage']:.1%}\n"
              f"transition RMSE: {metrics['transition']:.6f}\n"
              f"G RMSE: {metrics['affordance']:.6f}\n"
              f"G top-k recall: {metrics['recall']:.1%}\n\n"
              f"GCML success: {sum(r['gcml_success'] for r in results)}/{len(results)}\n"
              f"CML success: {sum(r['cml_success'] for r in results)}/{len(results)}\n"
              f"mean steps GCML / CML: {route_stats('gcml'):.1f} / {route_stats('cml'):.1f}\n"
              f"selected rollout reached: {len(hit_depths)}/{len(trace)} "
              f"({len(hit_depths)/max(len(trace), 1):.1%})\n"
              f"mean hit depth: {np.mean(hit_depths) if hit_depths else 0:.1f}\n"
              f"mean reached candidates: {np.mean([x[2] for x in trace]):.1f}/{ROLLOUTS}\n"
              f"mean unique first / paths: {np.mean([x[3] for x in trace]):.1f} / "
              f"{np.mean([x[4] for x in trace]):.1f}\n"
              f"No fallback; failures remain visible.",
              va="top", family="monospace", fontsize=11)
    fig.tight_layout(); fig.savefig(REPORT_FIGURE, dpi=180); plt.close(fig)


def main():
    model = load_model()
    frozen = {name: model[name].copy() for name in "QVWG"}
    metrics = model_metrics(model)
    results = []
    for i in range(TEST_CASES):
        case = make_case(model, SEED+i)
        gcml_route, gcml_success, rollout_trace = gcml_rollout(
            model, case, ROLLOUTS, ROLLOUT_HORIZON, SEED+400+i
        )
        cml_route, cml_success = cml_rollout(model, case)
        result = dict(**case, gcml_route=gcml_route, gcml_success=gcml_success,
                      rollout_trace=rollout_trace,
                      cml_route=cml_route, cml_success=cml_success)
        # 每个真实动作都必须属于固定图中的有效边。
        routes = (gcml_route, cml_route)
        if any(not model["adjacency"][a, b] for route in routes
               for a, b in zip(route[:-1], route[1:])):
            raise AssertionError("GCML执行了无效动作")
        results.append(result)
        reach_text = "".join("T" if hit else "F" for hit, *_ in rollout_trace) or "-"
        print(f"case {i+1:02d}: GCML={gcml_success}({len(gcml_route)-1} steps), "
              f"CML={cml_success}({len(cml_route)-1} steps), "
              f"rollout_reached={reach_text}")
    if any(not np.array_equal(model[name], frozen[name]) for name in "QVWG"):
        raise AssertionError("推理阶段修改了Q/V/W/G")

    scores = benchmark(model)
    save_figures(model, results, scores, metrics)
    print(f"buffer coverage: nodes={metrics['node_coverage']:.1%}, "
          f"actions={metrics['action_coverage']:.1%}")
    print(f"transition_RMSE={metrics['transition']:.6f}, G_RMSE={metrics['affordance']:.6f}")
    print(f"benchmark: GCML@{ROLLOUTS}={scores['count']['GCML'][-1]:.1f}%, "
          f"CML={scores['cml']:.1f}%")
    print("saved:", CASE_FIGURE, REPORT_FIGURE)


if __name__ == "__main__":
    main()
