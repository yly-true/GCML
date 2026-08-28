"""加载已训练模型进行离散地标导航；不训练、无fallback、无连续曲线。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import train


# ======================== 全部推理参数 ========================
MODEL_FILE = "gcml_model.npz"
SEED = 100
TEST_CASES = 10
ROLLOUTS = 60               # 每个真实决策点生成的候选轨迹数
ROLLOUT_HORIZON = 32        # 每条候选最多在潜空间想象32步
EXECUTION_HORIZON = 64      # 最多真实执行并重规划64次
LATENT_GOAL_TOLERANCE = .8
NOISE = .18

RUN_BENCHMARK = True
BENCHMARK_CASES = 20
ROLLOUT_BUDGETS = (1, 2, 5, 10, 20, 60)
HORIZON_BUDGETS = (2, 4, 8, 16, 32, 64)
CASE_FIGURE = "test_10_cases.png"
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


def imagine_first_action(model, node, goal, real_actions, used_actions, samples, horizon, rng):
    """并行生成samples条高维候选，评分后返回最佳候选的第一个动作。"""
    q, v, w, g = (model[name] for name in "QVWG")
    action_count = len(model["edges"])
    target = q[:, goal, None]                         # shape=(潜维数,1)
    state = np.repeat(q[:, node, None], samples, axis=1)
    used = np.zeros((action_count, samples), bool)    # 每列对应一条候选轨迹
    used[used_actions, :] = True
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
        eligibility[used] = -np.inf
        if step == 0:
            eligibility[gate == 0] = -np.inf
        chosen = np.argmax(eligibility, axis=0)       # 公式(19)

        for sample in active:
            action = int(chosen[sample])
            if not np.isfinite(eligibility[action, sample]):
                continue
            paths[sample].append(action)
            used[action, sample] = True
            state[:, sample] += v[:, action]          # 公式(20)，不查询真实下一节点
            residual[sample] = np.linalg.norm(state[:, sample]-target[:, 0])
            arrived[sample] = residual[sample] <= LATENT_GOAL_TOLERANCE

    # 成功候选按想象步数排序；全部失败时只比较潜空间目标残差。
    ranked = [((not arrived[i], len(path) if arrived[i] else residual[i]), path)
              for i, path in enumerate(paths) if path]
    return None if not ranked else int(min(ranked, key=lambda item: item[0])[1][0])


def gcml_rollout(model, case, samples=ROLLOUTS, horizon=ROLLOUT_HORIZON, seed=0):
    """类似MPC：选最佳想象轨迹，但每轮只真实执行它的第一个动作。"""
    edges, available = model["edges"], model["available"]
    node, goal = case["start_node"], case["goal_node"]
    route, used_actions = [node], []                 # route存真实执行后到达的节点
    rng = np.random.default_rng(seed)

    for _ in range(EXECUTION_HORIZON):
        if node == goal:
            break
        real_actions = np.asarray([a for a in available[node] if a not in used_actions])
        if not len(real_actions):
            break
        action = imagine_first_action(
            model, node, goal, real_actions, used_actions, samples, horizon, rng
        )
        if action is None:
            break
        used_actions.append(action)
        node = int(edges[action, 1])                  # 这里只执行最佳候选第一步
        route.append(node)
    return route, node == goal


def cml_rollout(model, case):
    """CML基线：无噪声、无G、无多步想象，每次真实状态只贪心走一步。"""
    q, w, edges = model["Q"], model["W"], model["edges"]
    node, goal, used = case["start_node"], case["goal_node"], []
    route = [node]                                  # 每次一步决策，累计后仍形成完整路线
    for _ in range(EXECUTION_HORIZON):
        if node == goal:
            return route, True
        actions = np.asarray([a for a in model["available"][node] if a not in used])
        if not len(actions):
            return route, False
        utility = w@(q[:, goal]-q[:, node])
        action = int(actions[np.argmax(utility[actions])])
        used.append(action)
        node = int(edges[action, 1])
        route.append(node)
    return route, False


def model_metrics(model):
    """只读检查训练误差、G恢复率以及buffer覆盖率。"""
    q, v, g, edges = model["Q"], model["V"], model["G"], model["edges"]
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
    cml_rate = 100*np.mean([cml_rollout(model, case)[1] for case in cases])
    count = {name: [] for name in ("GCML", "untrained GCML")}
    for budget in ROLLOUT_BUDGETS:
        count["GCML"].append(100*np.mean([
            gcml_rollout(model, case, budget, ROLLOUT_HORIZON, SEED+4000+i)[1]
            for i, case in enumerate(cases)
        ]))
        count["untrained GCML"].append(100*np.mean([
            gcml_rollout(untrained, case, budget, ROLLOUT_HORIZON, SEED+5000+i)[1]
            for i, case in enumerate(cases)
        ]))

    horizon = [100*np.mean([
        gcml_rollout(model, case, ROLLOUTS, value, SEED+6000+i)[1]
        for i, case in enumerate(cases)
    ]) for value in HORIZON_BUDGETS]

    dimensions, dim_gcml, dim_cml = model["ablation_dims"].tolist(), [], []
    for dim in dimensions:
        current = dict(model)
        if dim != model["Q"].shape[0]:
            for name in "QVWG":
                current[name] = model[f"{name}_d{dim}"]
        dim_gcml.append(100*np.mean([
            gcml_rollout(current, case, ROLLOUTS, ROLLOUT_HORIZON, SEED+7000+i)[1]
            for i, case in enumerate(cases)
        ]))
        dim_cml.append(100*np.mean([cml_rollout(current, case)[1] for case in cases]))
    return dict(count=count, cml=cml_rate, horizon=horizon,
                dimensions=dimensions, dim_gcml=dim_gcml, dim_cml=dim_cml)


def draw_case(ax, model, result, index):
    """橙色为GCML；蓝色虚线是CML重复执行一步决策后形成的真实路线。"""
    points, adjacency = model["points"], model["adjacency"]
    ax.imshow(model["grid"], origin="lower", cmap="Blues", alpha=.75)
    for i, j in np.argwhere(np.triu(adjacency, 1)):
        ax.plot(points[[i, j], 0], points[[i, j], 1], color="#b0bec5", lw=.45)
    ax.scatter(*points.T, s=8, color="#546e7a")
    def physical(route, success):
        path = np.vstack(([result["start"]], points[route]))
        return np.vstack((path, result["goal"])) if success else path
    ax.plot(*physical(result["gcml_route"], result["gcml_success"]).T,
            color="#ef6c00", lw=2, label="GCML")
    ax.plot(*physical(result["cml_route"], result["cml_success"]).T,
            color="#1565c0", ls="--", lw=1.5, label="CML repeated 1-step")
    ax.scatter(*result["start"], color="#1565c0", s=30)
    ax.scatter(*result["goal"], color="#f9a825", marker="*", s=65)
    ax.set(title=f"#{index:02d} G={result['gcml_success']} C={result['cml_success']}",
           aspect="equal")
    if index == 1:
        ax.legend(fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])


def save_figures(model, results, scores, metrics):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for index, (ax, result) in enumerate(zip(axes.flat, results), 1):
        draw_case(ax, model, result, index)
    fig.tight_layout(); fig.savefig(CASE_FIGURE, dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for name, rate in scores["count"].items():
        ax.plot(ROLLOUT_BUDGETS, rate, "o-", label=name)
    ax.scatter([1], [scores["cml"]], marker="D", s=65, label="CML one-step")
    ax.set(xscale="log", xticks=ROLLOUT_BUDGETS, ylim=(-2, 102),
           xlabel="trajectory count", ylabel="success rate (%)", title="Effect of trajectory count")
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()); ax.grid(alpha=.2); ax.legend()

    ax = axes[0, 1]
    ax.plot(HORIZON_BUDGETS, scores["horizon"], "o-", color="#7e57c2")
    ax.set(ylim=(-2, 102), xlabel="rollout horizon", ylabel="success rate (%)",
           title=f"Effect of horizon (trajectories={ROLLOUTS})")
    ax.grid(alpha=.2)

    ax = axes[1, 0]
    ax.plot(scores["dimensions"], scores["dim_gcml"], "o-", label="GCML")
    ax.plot(scores["dimensions"], scores["dim_cml"], "o--", label="CML one-step")
    ax.set(ylim=(-2, 102), xlabel="latent dimension", ylabel="success rate (%)",
           title="Effect of high-dimensional state")
    ax.grid(alpha=.2); ax.legend()

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
        gcml_route, gcml_success = gcml_rollout(
            model, case, ROLLOUTS, ROLLOUT_HORIZON, SEED+400+i
        )
        cml_route, cml_success = cml_rollout(model, case)
        result = dict(**case, gcml_route=gcml_route, gcml_success=gcml_success,
                      cml_route=cml_route, cml_success=cml_success)
        # 每个真实动作都必须属于固定图中的有效边。
        routes = (gcml_route, cml_route)
        if any(not model["adjacency"][a, b] for route in routes
               for a, b in zip(route[:-1], route[1:])):
            raise AssertionError("GCML执行了无效动作")
        results.append(result)
        print(f"case {i+1:02d}: GCML={gcml_success}({len(gcml_route)-1} steps), "
              f"CML={cml_success}({len(cml_route)-1} steps)")
    if any(not np.array_equal(model[name], frozen[name]) for name in "QVWG"):
        raise AssertionError("推理阶段修改了Q/V/W/G")

    scores = benchmark(model) if RUN_BENCHMARK else {}
    save_figures(model, results, scores, metrics)
    print(f"buffer coverage: nodes={metrics['node_coverage']:.1%}, "
          f"actions={metrics['action_coverage']:.1%}")
    print(f"transition_RMSE={metrics['transition']:.6f}, G_RMSE={metrics['affordance']:.6f}")
    print("trajectory count:", scores["count"], "CML point:", scores["cml"])
    print("horizon:", dict(zip(HORIZON_BUDGETS, scores["horizon"])))
    print("dimension GCML:", dict(zip(scores["dimensions"], scores["dim_gcml"])))
    print("dimension CML:", dict(zip(scores["dimensions"], scores["dim_cml"])))
    print("saved:", CASE_FIGURE, REPORT_FIGURE)


if __name__ == "__main__":
    main()
