"""生成随机游走buffer，训练Q/V/W/G并保存；本文件不负责推理。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ======================== 全部训练参数 ========================
WIDTH = HEIGHT = 100
SEED = 0
AGENT_RADIUS = .35
LINE_SAMPLE_SPACING = .30

LANDMARKS_PER_ROOM = 5       # 每个房间5个内部节点
ROOM_NEIGHBORS = 2           # 每点补充的房间内最近可见邻居数
LATENT_DIM = 128
ABLATION_DIMS = (8, 16, 32, 64)  # 仅用于潜空间维数消融

BUFFER_TRAJECTORIES = 200    # 至少采集多少条随机游走轨迹
MAX_BUFFER_TRAJECTORIES = 500
TRAJECTORY_LENGTH = 100      # 每条训练轨迹包含100次状态转移
BATCH_TRAJECTORIES = 16      # 每轮从buffer随机抽取的轨迹数
EPOCHS = 1000

# 论文给出的抽象图默认初始化与学习率
ETA_Q = .1
ETA_V = ETA_W = ETA_G = .01

MODEL_FILE = "gcml_model.npz"
LOSS_FIGURE = "train_loss.png"
# =============================================================


def make_map():
    """建立固定4x4房间地图，并记录18个房间通道中心。"""
    grid = np.zeros((HEIGHT, WIDTH), bool)
    grid[[0, -1], :] = True
    grid[:, [0, -1]] = True
    passages = []                 # 每行数据为[x坐标, y坐标, 墙方向]

    def gap(axis, fixed, centre):
        centre = int(round(centre))
        if axis == 0:             # 0表示竖墙上的通道
            grid[centre-3:centre+4, fixed] = False
            passages.append((fixed, centre, axis))
        else:                     # 1表示横墙上的通道
            grid[fixed, centre-3:centre+4] = False
            passages.append((centre, fixed, axis))

    xs, ys = [25, 50, 75], [25, 50, 75]
    for x in xs:
        grid[1:-1, x] = True
    for y in ys:
        grid[y, 1:-1] = True
    for x, fractions in zip(xs, ((.14, .62, .87), (.10, .38, .84), (.16, .58, .90))):
        for fraction in fractions:
            gap(0, x, fraction*(HEIGHT-1))
    for y, fractions in zip(ys, ((.12, .43, .88), (.17, .66, .91), (.09, .47, .82))):
        for fraction in fractions:
            gap(1, y, fraction*(WIDTH-1))
    for x0, y0, x1, y1 in (
        (8, 32, 17, 38), (33, 8, 40, 17), (58, 31, 69, 37),
        (80, 57, 90, 64), (31, 80, 41, 88), (56, 57, 64, 68),
    ):
        grid[y0:y1, x0:x1] = True
    return grid, np.asarray(passages, int)


def distance_field(grid):
    """计算每个栅格到墙面的近似距离；用于连续碰撞检查。"""
    h, w = grid.shape
    d, root2 = np.where(grid, 0., 1e6), np.sqrt(2.)
    for y in range(h):
        for x in range(w):
            for dy, dx, cost in ((-1, 0, 1), (0, -1, 1), (-1, -1, root2), (-1, 1, root2)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < h and 0 <= xx < w:
                    d[y, x] = min(d[y, x], d[yy, xx]+cost)
    for y in range(h-1, -1, -1):
        for x in range(w-1, -1, -1):
            for dy, dx, cost in ((1, 0, 1), (0, 1, 1), (1, 1, root2), (1, -1, root2)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < h and 0 <= xx < w:
                    d[y, x] = min(d[y, x], d[yy, xx]+cost)
    return d-.5


def clearance(sdf, points):
    """双线性插值：输入连续坐标(...,2)，输出对应墙面距离。"""
    p = np.atleast_2d(np.asarray(points, float))
    h, w = sdf.shape
    x, y = np.clip(p[:, 0], 0, w-1), np.clip(p[:, 1], 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    dx, dy = x-x0, y-y0
    return ((1-dx)*(1-dy)*sdf[y0, x0]+dx*(1-dy)*sdf[y0, x1]+
            (1-dx)*dy*sdf[y1, x0]+dx*dy*sdf[y1, x1])


def line_free(sdf, a, b, margin=.08):
    """线段上所有采样点都满足智能体半径，才认为两个节点可直达。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    count = max(2, int(np.linalg.norm(b-a)/LINE_SAMPLE_SPACING)+2)
    return bool(np.all(clearance(sdf, np.linspace(a, b, count)) >= AGENT_RADIUS+margin))


def build_graph(grid, passages):
    """房间内局部连边；门口节点同时属于相邻两个房间。"""
    sdf, bounds = distance_field(grid), (1, 25, 50, 75, 99)
    points = [p[:2].astype(float) for p in passages]
    rooms = [[] for _ in range(16)]

    def room_id(point):
        col, row = np.searchsorted(bounds[1:-1], point, side="right")
        return 4*row+col

    # 每个门口加入墙两侧房间，保证所有相邻房间一定通过它连接。
    for i, (x, y, axis) in enumerate(passages):
        offsets = ((-1, 0), (1, 0)) if axis == 0 else ((0, -1), (0, 1))
        for dx, dy in offsets:
            rooms[room_id((x+dx, y+dy))].append(i)

    # 最远点采样使房间内节点分布均匀，并主动远离已有门口节点。
    for row in range(4):
        for col in range(4):
            room, (x0, x1), (y0, y1) = 4*row+col, bounds[col:col+2], bounds[row:row+2]
            candidates = np.array([[x, y] for y in range(y0+2, y1-1, 2)
                                   for x in range(x0+2, x1-1, 2)
                                   if clearance(sdf, (x, y))[0] >= AGENT_RADIUS+.6], float)
            chosen = []
            for _ in range(LANDMARKS_PER_ROOM):
                anchors = np.asarray([points[i] for i in rooms[room]]+chosen)
                distance = np.linalg.norm(candidates[:, None]-anchors[None], axis=2).min(1)
                chosen.append(candidates[int(np.argmax(distance))])
            rooms[room] += list(range(len(points), len(points)+len(chosen)))
            points += chosen

    points = np.asarray(points)
    adjacency = np.zeros((len(points), len(points)), bool)
    for members in rooms:
        visible = sorted((np.linalg.norm(points[i]-points[j]), i, j)
                         for n, i in enumerate(members) for j in members[n+1:]
                         if line_free(sdf, points[i], points[j]))
        # 最小生成树先保证房间内所有节点和门口连通。
        connected = {members[0]}
        while len(connected) < len(members):
            bridge = next((edge for edge in visible
                           if (edge[1] in connected) != (edge[2] in connected)), None)
            if bridge is None:
                raise RuntimeError("房间内地标无法连通")
            _, i, j = bridge
            adjacency[i, j] = adjacency[j, i] = True
            connected.update((i, j))
        # 每点再保留最近的少量可见邻居，提供冗余但避免蜘蛛网。
        for i in members:
            local = [(d, j if a == i else a) for d, a, j in visible if i in (a, j)]
            for _, j in local[:ROOM_NEIGHBORS]:
                adjacency[i, j] = adjacency[j, i] = True

    # 删除无冗余的内部叶节点；门口节点始终保留且必须连接墙两侧。
    keep = (adjacency.sum(1) > 1) | (np.arange(len(points)) < len(passages))
    points, adjacency = points[keep], adjacency[np.ix_(keep, keep)]
    assert np.all(adjacency[:len(passages)].sum(1) >= 2)
    edges = np.argwhere(adjacency).astype(int)
    return points, adjacency, edges


def collect_buffer(edges, node_count):
    """随机起点+随机动作，直到轨迹数足够且所有节点/动作都被访问。"""
    rng = np.random.default_rng(SEED+1)
    outgoing = [np.flatnonzero(edges[:, 0] == node) for node in range(node_count)]
    observations, actions, next_observations = [], [], []
    seen_nodes, seen_actions = np.zeros(node_count, bool), np.zeros(len(edges), bool)

    while (len(observations) < BUFFER_TRAJECTORIES or
           not (seen_nodes.all() and seen_actions.all())):
        if len(observations) >= MAX_BUFFER_TRAJECTORIES:
            raise RuntimeError("随机游走buffer没有覆盖全部节点和动作")
        node = int(rng.integers(node_count))
        o, a, o_next = [], [], []
        seen_nodes[node] = True
        for _ in range(TRAJECTORY_LENGTH):
            action = int(rng.choice(outgoing[node]))
            target = int(edges[action, 1])
            o.append(node); a.append(action); o_next.append(target)
            seen_actions[action] = seen_nodes[target] = True
            node = target
        observations.append(o); actions.append(a); next_observations.append(o_next)

    # 三个二维数组的shape均为(轨迹数, 每条轨迹长度)。
    return (np.asarray(observations, np.int16), np.asarray(actions, np.int16),
            np.asarray(next_observations, np.int16), seen_nodes, seen_actions)


def train_model(edges, node_count, buffer, latent_dim=LATENT_DIM):
    """每轮随机抽取若干完整轨迹，再用论文局部规则训练Q/V/W/G。"""
    rng, action_count = np.random.default_rng(SEED), len(edges)
    dim = latent_dim

    # 论文明确给出的随机初始化，不使用图拉普拉斯初始化或Q白化。
    q = rng.normal(0, 1, (dim, node_count))
    v = rng.normal(0, .1, (dim, action_count))
    w = rng.normal(0, .1, (action_count, dim))
    g = rng.normal(0, .1, (action_count, dim))
    initial = tuple(x.copy() for x in (q, v, w, g))

    true_g = np.zeros((action_count, node_count))
    true_g[np.arange(action_count), edges[:, 0]] = 1
    history = {"transition": [], "w_alignment": [], "affordance": []}

    def record():
        delta = np.column_stack([q[:, j]-q[:, i] for i, j in edges])
        dot = np.sum(w*delta.T, axis=1)
        norm = np.linalg.norm(w, axis=1)*np.linalg.norm(delta.T, axis=1)
        history["transition"].append(float(np.sqrt(np.mean((delta-v)**2))))
        history["w_alignment"].append(float(1-np.mean(dot/np.maximum(norm, 1e-9))))
        history["affordance"].append(float(np.sqrt(np.mean((true_g-g@q)**2))))

    obs_buffer, action_buffer, next_buffer = buffer[:3]
    record()
    for _ in range(EPOCHS):
        ids = rng.choice(len(obs_buffer), min(BATCH_TRAJECTORIES, len(obs_buffer)), replace=False)
        source, action, target = (array[ids].reshape(-1)
                                  for array in (obs_buffer, action_buffer, next_buffer))
        state, next_state = q[:, source], q[:, target]
        error = next_state-state-v[:, action]                 # s(t+1)-预测状态
        state_difference = next_state-state

        # 同一节点/动作在batch中可能出现多次；先求平均再做一次局部更新。
        for index in np.unique(target):
            # 作者代码更新下一观测列，使真实下一状态向当前预测靠拢。
            q[:, index] -= ETA_Q*error[:, target == index].mean(1)       # 公式(13)
        for index in np.unique(action):
            mask = action == index
            v[:, index] += ETA_V*error[:, mask].mean(1)                  # 公式(12)
            w[index] += ETA_W*state_difference[:, mask].mean(1)         # 公式(14)
        for index in np.unique(source):
            state_i = q[:, index]
            g += ETA_G*np.outer(true_g[:, index]-g@state_i, state_i)     # 公式(17)
        record()
    return (q, v, w, g), initial, true_g, history


def main():
    grid, passages = make_map()
    points, adjacency, edges = build_graph(grid, passages)
    buffer = collect_buffer(edges, len(points))
    (q, v, w, g), initial, true_g, history = train_model(edges, len(points), buffer)

    # 使用相同buffer额外训练较低维模型，只用于维数消融，不参与主模型推理。
    ablation = {}
    for dim in ABLATION_DIMS:
        print(f"training latent-dimension ablation: d={dim}")
        matrices, _, _, _ = train_model(edges, len(points), buffer, dim)
        for name, value in zip("QVWG", matrices):
            ablation[f"{name}_d{dim}"] = value

    payload = dict(
        grid=grid, passages=passages, points=points,
        adjacency=adjacency, edges=edges, Q=q, V=v, W=w, G=g,
        Q0=initial[0], V0=initial[1], W0=initial[2], G0=initial[3],
        buffer_o=buffer[0], buffer_a=buffer[1], buffer_next=buffer[2],
        loss_transition=history["transition"], loss_w=history["w_alignment"],
        loss_g=history["affordance"], radius=AGENT_RADIUS,
        ablation_dims=np.asarray((*ABLATION_DIMS, LATENT_DIM)),
    )
    payload.update(ablation)
    np.savez_compressed(MODEL_FILE, **payload)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["transition"], label="Q/V transition RMSE")
    ax.plot(history["w_alignment"], label="W alignment loss")
    ax.plot(history["affordance"], label="G affordance RMSE")
    ax.set(xlabel="epoch", ylabel="loss", title="Trajectory-buffer GCML training")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(LOSS_FIGURE, dpi=180); plt.close(fig)

    delta = np.column_stack([q[:, j]-q[:, i] for i, j in edges])
    changes = [np.linalg.norm(x-y) for x, y in zip((q, v, w, g), initial)]
    print(f"rooms=16, interior/room={LANDMARKS_PER_ROOM}, passage_nodes={len(passages)}")
    print(f"nodes={len(points)}, actions={len(edges)}, latent_dim={q.shape[0]}")
    print(f"buffer={len(buffer[0])} trajectories x {TRAJECTORY_LENGTH} steps, "
          f"node_coverage={buffer[3].mean():.1%}, action_coverage={buffer[4].mean():.1%}")
    print("Q/V/W/G changes="+str(tuple(round(float(x), 3) for x in changes)))
    print(f"transition_RMSE={np.sqrt(np.mean((delta-v)**2)):.6f}, "
          f"G_RMSE={np.sqrt(np.mean((true_g-g@q)**2)):.6f}")
    print("saved:", Path(MODEL_FILE).resolve(), Path(LOSS_FIGURE).resolve())


if __name__ == "__main__":
    main()
