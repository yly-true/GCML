"""极简稀疏地标 GCML：修改下方参数后直接运行本文件。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ======================== 全部可调参数 ========================
# 地图与随机种子
WIDTH = 100
HEIGHT = 100
SEED = 0
AGENT_RADIUS = 0.35

# GCML 训练与地标图
LATENT_DIM = 24
EPOCHS = 120
ETA_Q = 0.002
ETA_V = 0.04
ETA_W = 0.01
COVERAGE_RADIUS = 8.0
RETRY_COVERAGE_RADIUS = 6.0
MAX_LANDMARKS = 140
CONNECTION_RADIUS = 19.0
MAX_NEIGHBORS = 8
NOISE = 0.18
GRAPH_ROLLOUTS = 384
CURVE_ROLLOUTS = 96

# 连续碰撞检测与曲线评分
LINE_SAMPLE_SPACING = 0.30
CONTROL_SPACING = 3.0
CURVE_SAMPLE_SPACING = 0.20
CURVE_CLEARANCE_MARGIN = 0.12
COLLISION_WEIGHT = 1e5
CURVATURE_WEIGHT = 0.30
WALL_WEIGHT = 2.0

# 任务模式：普通训练或十组随机封堵检查
RANDOM_TASK = False
BLOCK_ONE_PASSAGE = False
RUN_CHECK = False          # True 时生成 CHECK_CASES 组随机封堵实验
CHECK_CASES = 10

OUTPUT = "landmark_gcml.png"
CHECK_OUTPUT = "check_10_blocked.png"
SHOW = False
# ===============================================================


class World:
    """自建连续地图：栅格只用于墙体和距离场，智能体位置仍是连续坐标。"""

    def __init__(self, width=WIDTH, height=HEIGHT, seed=SEED, radius=AGENT_RADIUS):
        if min(width, height) < 40:
            raise ValueError("map width and height must be >= 40")
        self.width, self.height, self.radius = int(width), int(height), radius
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((self.height, self.width), bool)
        self.passages = []
        self._build()
        self._distance_field()

    def _gap(self, axis, fixed, centre, half=3):
        """在水平或竖直隔墙上打开一个通道，并记录通道中心。"""
        centre = int(round(centre))
        if axis == "v":
            self.grid[centre - half : centre + half + 1, fixed] = False
            self.passages.append((fixed, centre, axis))
        else:
            self.grid[fixed, centre - half : centre + half + 1] = False
            self.passages.append((centre, fixed, axis))

    def _build(self):
        """生成外墙、三横三竖隔墙、通道以及房间内部障碍物。"""
        g = self.grid
        g[[0, -1], :] = True
        g[:, [0, -1]] = True
        xs = [self.width // 4, self.width // 2, 3 * self.width // 4]
        ys = [self.height // 4, self.height // 2, 3 * self.height // 4]
        for x in xs:
            g[1:-1, x] = True
        for y in ys:
            g[y, 1:-1] = True
        vf = ((.14, .62, .87), (.10, .38, .84), (.16, .58, .90))
        hf = ((.12, .43, .88), (.17, .66, .91), (.09, .47, .82))
        for x, fractions in zip(xs, vf):
            for f in fractions:
                self._gap("v", x, f * (self.height - 1))
        for y, fractions in zip(ys, hf):
            for f in fractions:
                self._gap("h", y, f * (self.width - 1))
        for x0, y0, x1, y1 in (
            (8, 32, 17, 38), (33, 8, 40, 17), (58, 31, 69, 37),
            (80, 57, 90, 64), (31, 80, 41, 88), (56, 57, 64, 68),
        ):
            x0, x1 = round(x0 * self.width / 100), round(x1 * self.width / 100)
            y0, y1 = round(y0 * self.height / 100), round(y1 * self.height / 100)
            g[y0:y1, x0:x1] = True

    def _distance_field(self):
        """两遍 chamfer transform 近似墙面距离，从而不依赖 SciPy。"""
        d = np.where(self.grid, 0.0, 1e6)
        root2 = np.sqrt(2.0)
        for y in range(self.height):
            for x in range(self.width):
                if y:
                    d[y, x] = min(d[y, x], d[y - 1, x] + 1)
                    if x:
                        d[y, x] = min(d[y, x], d[y - 1, x - 1] + root2)
                    if x + 1 < self.width:
                        d[y, x] = min(d[y, x], d[y - 1, x + 1] + root2)
                if x:
                    d[y, x] = min(d[y, x], d[y, x - 1] + 1)
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width - 1, -1, -1):
                if y + 1 < self.height:
                    d[y, x] = min(d[y, x], d[y + 1, x] + 1)
                    if x:
                        d[y, x] = min(d[y, x], d[y + 1, x - 1] + root2)
                    if x + 1 < self.width:
                        d[y, x] = min(d[y, x], d[y + 1, x + 1] + root2)
                if x + 1 < self.width:
                    d[y, x] = min(d[y, x], d[y, x + 1] + 1)
        self.sdf = d - .5

    def clearance(self, points):
        """双线性插值得到一个或多个连续坐标到墙面的近似距离。"""
        p = np.asarray(points, float)
        one = p.ndim == 1
        p = p[None] if one else p
        inside = ((p[:, 0] >= 0) & (p[:, 0] <= self.width - 1) &
                  (p[:, 1] >= 0) & (p[:, 1] <= self.height - 1))
        x = np.clip(p[:, 0], 0, self.width - 1)
        y = np.clip(p[:, 1], 0, self.height - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, self.width - 1)
        y1 = np.minimum(y0 + 1, self.height - 1)
        dx, dy = x - x0, y - y0
        value = ((1 - dx) * (1 - dy) * self.sdf[y0, x0] +
                 dx * (1 - dy) * self.sdf[y0, x1] +
                 (1 - dx) * dy * self.sdf[y1, x0] +
                 dx * dy * self.sdf[y1, x1])
        value[~inside] = -1
        return float(value[0]) if one else value

    def free(self, point, margin=0):
        return self.clearance(point) >= self.radius + margin

    def line_free(self, a, b, margin=.05):
        """沿线段密集采样；所有点满足安全半径时才视为可直达。"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / LINE_SAMPLE_SPACING) + 2)
        return bool(np.all(self.clearance(np.linspace(a, b, n)) >= self.radius + margin))

    def random_point(self, rng):
        cells = np.argwhere(~self.grid)
        for _ in range(1000):
            y, x = cells[rng.integers(len(cells))]
            p = np.array([x, y]) + rng.uniform(-.3, .3, 2)
            if self.free(p, .5):
                return p
        raise RuntimeError("could not sample a free point")

    def task(self, rng):
        """随机生成距离足够远的连续起点和终点。"""
        for _ in range(1000):
            a, b = self.random_point(rng), self.random_point(rng)
            if np.linalg.norm(a - b) > .55 * min(self.width, self.height):
                return a, b
        raise RuntimeError("could not sample a separated task")

    def block_one(self, rng):
        """随机且只封堵一个已生成通道，然后刷新距离场。"""
        x, y, axis = self.passages[rng.integers(len(self.passages))]
        if axis == "v":
            self.grid[y - 3 : y + 4, x] = True
        else:
            self.grid[y, x - 3 : x + 4] = True
        self._distance_field()
        return np.array([x, y])


class Graph:
    """从连续自由空间自动构建稀疏地标可视图。"""

    def __init__(self, world, coverage=8, limit=100):
        self.world = world
        selected = [np.array(p[:2], float) for p in world.passages if world.free(p[:2])]
        candidates = np.array([[x, y] for y in range(2, world.height - 2, 3)
                               for x in range(2, world.width - 2, 3)
                               if world.free((x, y), .6)], float)

        # 通道中心必须保留；其余地标采用最远点采样补足空间覆盖。
        def fill(radius):
            while len(selected) < limit:
                nearest = np.linalg.norm(
                    candidates[:, None] - np.array(selected)[None], axis=2
                ).min(1)
                i = int(np.argmax(nearest))
                if nearest[i] <= radius:
                    break
                selected.append(candidates[i])

        fill(coverage)
        # 极少数封堵会让 8 单位覆盖过稀，此时才用备用半径补点。
        while True:
            self.p = np.array(selected)
            pair = np.linalg.norm(self.p[:, None] - self.p[None], axis=2)
            try:
                self.adj = self._connect(pair)
                break
            except RuntimeError:
                old_count = len(selected)
                fill(RETRY_COVERAGE_RADIUS)
                if len(selected) == old_count:
                    raise
        self.edges = np.array(np.argwhere(self.adj), int)
        # 每条有向边就是一个离散高层动作。
        self.out = [np.flatnonzero(self.edges[:, 0] == i) for i in range(len(self.p))]
        self.dist = np.where(self.adj, pair, np.inf)
        np.fill_diagonal(self.dist, 0)
        # Floyd-Warshall 只用于候选评分和失败兜底，不参与 GCML 动作选择。
        for k in range(len(self.p)):
            self.dist = np.minimum(self.dist, self.dist[:, k, None] + self.dist[None, k, :])

    def _connect(self, pair):
        """连接距离近且视线无碰撞的地标，并确保整张图连通。"""
        n = len(self.p)
        for attempt, scale in enumerate((1, 1.3, 1.7, 2.2, 3)):
            adj = np.zeros((n, n), bool)
            neighbour_limit = MAX_NEIGHBORS + 2 * attempt
            for i in range(n):
                linked = 0
                for j in np.argsort(pair[i]):
                    if (i != j and pair[i, j] <= CONNECTION_RADIUS * scale and
                            self.world.line_free(self.p[i], self.p[j], .08)):
                        adj[i, j] = adj[j, i] = True
                        linked += 1
                        if linked == neighbour_limit:
                            break
            seen, stack = {0}, [0]
            while stack:
                for j in np.flatnonzero(adj[stack.pop()]):
                    if int(j) not in seen:
                        seen.add(int(j))
                        stack.append(int(j))
            if len(seen) == n:
                return adj

        # 近邻截断可能把真实连通空间误分成多块；只补最少的无碰撞桥边。
        while True:
            labels = np.full(n, -1, int)
            component = 0
            for root in range(n):
                if labels[root] >= 0:
                    continue
                labels[root] = component
                stack = [root]
                while stack:
                    for j in np.flatnonzero(adj[stack.pop()]):
                        if labels[j] < 0:
                            labels[j] = component
                            stack.append(int(j))
                component += 1
            if component == 1:
                return adj

            bridge = None
            for i, j in np.argwhere(np.triu(labels[:, None] != labels[None], 1)):
                if ((bridge is None or pair[i, j] < bridge[0]) and
                        self.world.line_free(self.p[i], self.p[j], .08)):
                    bridge = (pair[i, j], int(i), int(j))
            if bridge is None:
                raise RuntimeError("landmarks cannot represent the connected free space")
            _, i, j = bridge
            adj[i, j] = adj[j, i] = True

    def visible(self, point, count=8):
        return self.visible_in(self.world, point, count)

    def visible_in(self, world, point, count=8):
        """在当前障碍层中寻找与连续位置直达的固定地标。"""
        answer = []
        for i in np.argsort(np.linalg.norm(self.p - point, axis=1)):
            if world.line_free(point, self.p[i]):
                answer.append(int(i))
                if len(answer) == count:
                    return answer
        if answer:
            return answer
        raise RuntimeError("no visible landmark")

    def affordance(self, world):
        """不改动作编号，只屏蔽在当前障碍层中已不可执行的固定边。"""
        active = np.zeros_like(self.adj)
        for i, j in np.argwhere(np.triu(self.adj, 1)):
            if world.line_free(self.p[i], self.p[j], .08):
                active[i, j] = active[j, i] = True

        available = []
        for node in range(len(self.p)):
            actions = self.out[node]
            available.append(
                actions[active[node, self.edges[actions, 1]]]
            )

        pair = np.linalg.norm(self.p[:, None] - self.p[None], axis=2)
        distance = np.where(active, pair, np.inf)
        np.fill_diagonal(distance, 0)
        for k in range(len(self.p)):
            distance = np.minimum(
                distance,
                distance[:, k, None] + distance[None, k, :],
            )
        return available, distance, active

    def route_length(self, route):
        if len(route) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.p[route], axis=0), axis=1).sum())


class GCML:
    """论文中的 Q/V/W 学习与基于想象状态的 stochastic rollout。"""

    def __init__(self, graph, dim=24, noise=.18, seed=0):
        self.g, self.rng, self.noise = graph, np.random.default_rng(seed), noise
        dim = min(dim, len(graph.p) - 1)
        # 图拉普拉斯特征向量只提供一个不坍缩的 Q 初值，之后 Q 仍会训练。
        degree = graph.adj.sum(1)
        lap = np.eye(len(degree)) - graph.adj / np.sqrt(np.outer(degree, degree))
        self.Q = np.linalg.eigh(lap)[1][:, 1 : dim + 1].T
        self.Q += self.rng.normal(0, .01, self.Q.shape)
        self._white()
        self.V = self.rng.normal(0, .05, (dim, len(graph.edges)))
        self.W = self.rng.normal(0, .05, (len(graph.edges), dim))
        self.initial = tuple(x.copy() for x in (self.Q, self.V, self.W))

    def _white(self):
        """中心化并白化 Q，防止所有地标嵌入坍缩到同一点。"""
        self.Q -= self.Q.mean(1, keepdims=True)
        value, vector = np.linalg.eigh(self.Q @ self.Q.T / self.Q.shape[1])
        self.Q = vector @ np.diag(1 / np.sqrt(value + 1e-6)) @ vector.T @ self.Q

    def train(self, epochs=120):
        """逐边执行论文公式 (12)、(13)、(14)，因此 Q/V/W 都会更新。"""
        for _ in range(epochs):
            for a in self.rng.permutation(len(self.g.edges)):
                i, j = self.g.edges[a]
                delta = self.Q[:, j] - self.Q[:, i]
                error = delta - self.V[:, a]
                self.V[:, a] += ETA_V * error               # 论文公式 (12)
                self.Q[:, i] -= ETA_Q * error               # 论文公式 (13)
                self.W[a] += ETA_W * delta                  # 论文公式 (14)
            self._white()
        # 固定最终 Q，再让 V 追上最终状态位移；这里不再改变 Q 和 W。
        for _ in range(30):
            for a, (i, j) in enumerate(self.g.edges):
                self.V[:, a] += ETA_V * (self.Q[:, j] - self.Q[:, i] - self.V[:, a])

    @staticmethod
    def _erase(route):
        """删除随机 rollout 中的回环，避免输出重复地标。"""
        clean = []
        for node in route:
            clean = clean[: clean.index(node) + 1] if node in clean else clean + [node]
        return clean

    def _sample(self, start, goal, horizon, scale, available):
        """生成一条 WTA 轨迹；imagined 始终按 V 自举，不读取下一观测。"""
        node = start
        route = [start]
        imagined, target = self.Q[:, start].copy(), self.Q[:, goal]
        for _ in range(horizon):
            if node == goal:
                break
            actions = available[node]  # 当前障碍层给出的动态 affordance
            if len(actions) == 0:
                break
            utility = self.W @ (target - imagined)
            utility /= max(np.linalg.norm(utility), 1e-9)
            noisy = utility[actions] + self.rng.normal(0, self.noise * scale, len(actions))
            a = int(actions[np.argmax(noisy)])
            imagined += self.V[:, a]                        # 论文公式 (20)
            node = int(self.g.edges[a, 1])
            route.append(node)
        return self._erase(route)

    def rollout(self, start, goal, available, distance, samples=384):
        """采样多条想象轨迹，优先成功，再比较路线长度。"""
        best = None
        for k in range(samples):
            scale = .55 + 1.15 * k / max(samples - 1, 1)
            route = self._sample(
                start, goal, 2 * len(self.g.p), scale, available
            )
            length = self.g.route_length(route)
            key = (
                route[-1] != goal,
                length if route[-1] == goal
                else distance[route[-1], goal] + .08 * length,
            )
            if best is None or key < best[0]:
                best = key, route
        fallback = best[1][-1] != goal
        if fallback:
            # 只有全部随机候选失败时才沿预计算图距离恢复一条安全路径。
            if not np.isfinite(distance[start, goal]):
                raise RuntimeError("当前动态 affordance 下起点与终点不连通")
            route = [start]
            while route[-1] != goal:
                i = route[-1]
                actions = available[i]
                neighbours = self.g.edges[actions, 1]
                edge = np.linalg.norm(self.g.p[neighbours] - self.g.p[i], axis=1)
                route.append(int(neighbours[np.argmin(edge + distance[neighbours, goal])]))
        else:
            route = best[1]
        return route, fallback

    def stats(self):
        transition = np.column_stack([self.Q[:, j] - self.Q[:, i] for i, j in self.g.edges])
        changes = [np.linalg.norm(x - y) for x, y in zip((self.Q, self.V, self.W), self.initial)]
        return changes + [float(np.sqrt(np.mean((transition - self.V) ** 2)))]


def smooth_rollout(world, waypoints, samples=96, seed=0):
    """把高层地标序列变成多条 Catmull-Rom 曲线并选出最低分者。"""
    rng = np.random.default_rng(seed)
    controls = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / CONTROL_SPACING)))
        controls.extend((1 - t) * a + t * b for t in np.linspace(0, 1, n + 1)[1:])
    controls = np.array(controls)

    def catmull(c):
        """显式计算三次 Catmull-Rom，多段拼接后经过全部控制点。"""
        pieces = []
        for i in range(len(c) - 1):
            p0, p1 = c[max(0, i - 1)], c[i]
            p2, p3 = c[i + 1], c[min(len(c) - 1, i + 2)]
            n = max(4, int(np.linalg.norm(p2 - p1) / CURVE_SAMPLE_SPACING) + 2)
            t = np.linspace(0, 1, n, endpoint=i == len(c) - 2)[:, None]
            pieces.append(.5 * (2*p1 + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t**2 +
                                 (-p0+3*p1-3*p2+p3)*t**3))
        return np.vstack(pieces)

    best = None
    for k in range(samples):
        c = controls.copy()
        if k:
            # 只扰动远离墙面的内部控制点，通道附近的关键点保持不动。
            movable = world.clearance(c) > world.radius + 2
            movable[[0, -1]] = False
            proposed = c + rng.normal(0, rng.uniform(0, .8), c.shape) * movable[:, None]
            for i in np.flatnonzero(movable):
                if world.free(proposed[i], .2):
                    c[i] = proposed[i]
        curve = catmull(c)
        clearance = world.clearance(curve)
        segment = np.diff(curve, axis=0)
        length = np.linalg.norm(segment, axis=1)
        direction = segment / np.maximum(length[:, None], 1e-9)
        collision = np.maximum(world.radius + CURVE_CLEARANCE_MARGIN - clearance, 0)
        curvature = np.linalg.norm(np.diff(direction, axis=0), axis=1).sum()
        proximity = np.exp(-np.maximum(clearance - world.radius, 0) / .8).mean()
        # 碰撞是硬优先级，其余三项依次偏好短、平滑、远离墙面的曲线。
        score = (COLLISION_WEIGHT * collision.dot(collision) + length.sum() +
                 CURVATURE_WEIGHT * curvature + WALL_WEIGHT * proximity)
        key = (np.any(clearance < world.radius), score)
        if best is None or key < best[0]:
            best = key, curve, (float(length.sum()), float(curvature), float(clearance.min()))
    return best[1], {"safe": not best[0][0], "length": best[2][0],
                     "curvature": best[2][1], "clearance": best[2][2]}


def build_base():
    """只执行一次：建立基础地图、固定地标/动作集合并训练 Q/V/W。"""
    world = World(WIDTH, HEIGHT, SEED)
    graph = Graph(world, COVERAGE_RADIUS, MAX_LANDMARKS)
    model = GCML(graph, LATENT_DIM, NOISE, SEED)
    model.train(EPOCHS)
    return world, graph, model


def run(base=None, seed=SEED, random_task=RANDOM_TASK,
        block_passage=BLOCK_ONE_PASSAGE):
    """规划一个 case；基础地标、动作和 Q/V/W 始终保持不变。"""
    if base is None:
        base = build_base()
    base_world, graph, model = base
    # 每个 case 只重置采样噪声；Q/V/W 数值保持完全不变。
    model.rng = np.random.default_rng(seed + 400)

    # 每个 case 从同一基础地图复制障碍层；只允许改变一个通道。
    world = World(base_world.width, base_world.height, SEED)
    blocked = (
        world.block_one(np.random.default_rng(seed + 100))
        if block_passage else None
    )
    available, active_distance, active_adj = graph.affordance(world)
    if random_task:
        start, goal = world.task(np.random.default_rng(seed + 200))
    else:
        start, goal = np.array([5., 5.]), np.array([WIDTH - 6., HEIGHT - 6.])
    fallback = False
    if world.line_free(start, goal):
        # 起终点能直达时不必人为经过地标，但 Q/V/W 仍已完成训练。
        route = []
    else:
        # 起终点不是训练状态：运行时临时挂接到若干可见地标。
        starts = graph.visible_in(world, start)
        goals = graph.visible_in(world, goal)
        sn, gn = min(((i, j) for i in starts for j in goals),
                     key=lambda z: np.linalg.norm(start-graph.p[z[0]]) + active_distance[z] +
                     np.linalg.norm(goal-graph.p[z[1]]))
        if not np.isfinite(active_distance[sn, gn]):
            raise RuntimeError("连续起终点无法通过当前动态 affordance 连接")
        route, fallback = model.rollout(
            sn, gn, available, active_distance, GRAPH_ROLLOUTS
        )
    middle = graph.p[route] if route else np.empty((0, 2))
    waypoints = np.vstack(([start], middle, [goal]))
    curve, metrics = smooth_rollout(world, waypoints, CURVE_ROLLOUTS, seed + 300)
    return dict(world=world, graph=graph, model=model, start=start, goal=goal,
                blocked=blocked, waypoints=waypoints, curve=curve,
                metrics=metrics, fallback=fallback, active_adj=active_adj)


def draw(ax, result, compact=False):
    """绘制墙体、地标图、高层虚线与最终连续曲线。"""
    world, graph = result["world"], result["graph"]
    ax.imshow(world.grid, origin="lower",
              extent=(-.5, world.width-.5, -.5, world.height-.5),
              cmap="Blues", alpha=.75)
    for i, j in np.argwhere(np.triu(graph.adj, 1)):
        enabled = result["active_adj"][i, j]
        ax.plot(
            graph.p[[i, j], 0], graph.p[[i, j], 1],
            color="#b0bec5" if enabled else "#d32f2f",
            ls="-" if enabled else ":",
            lw=.4 if enabled else 1.0,
        )
    ax.scatter(*graph.p.T, s=5 if compact else 12, color="#546e7a",
               label=f"landmarks ({len(graph.p)})")
    ax.plot(*result["waypoints"].T, "--", color="#7e57c2", lw=1, label="GCML rollout")
    ax.plot(*result["curve"].T, color="#ef6c00", lw=1.7 if compact else 2.6,
            label="smooth rollout")
    ax.scatter(*result["start"], s=35 if compact else 90, color="#1565c0", label="start", zorder=5)
    ax.scatter(*result["goal"], s=65 if compact else 150, marker="*",
               color="#f9a825", label="goal", zorder=5)
    if result["blocked"] is not None:
        ax.scatter(*result["blocked"], s=38 if compact else 90, marker="s",
                   color="#d32f2f", label="blocked", zorder=5)
    ax.set(xlim=(-.5, world.width-.5), ylim=(-.5, world.height-.5), aspect="equal")
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set(xlabel="continuous x", ylabel="continuous y",
               title="Minimal Landmark GCML + Smooth Rollout")
        ax.legend(fontsize=8)


def save_one(result, output, show=False):
    fig, ax = plt.subplots(figsize=(9, 9))
    draw(ax, result)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def check():
    """使用同一套顶部参数运行多组随机起终点和单通道封堵实验。"""
    base = build_base()
    _, fixed_graph, fixed_model = base
    fixed_parameters = tuple(x.copy() for x in (fixed_model.Q, fixed_model.V, fixed_model.W))
    print(
        f"fixed base: landmarks={len(fixed_graph.p)}, "
        f"actions={len(fixed_graph.edges)}, train_count=1"
    )
    results = []
    for i in range(CHECK_CASES):
        result = run(
            base,
            SEED + i,
            random_task=True,
            block_passage=True,
        )
        results.append(result)
        print(f"case {i+1:02d}: curve={result['metrics']['safe']}, fallback={result['fallback']}")
    if not all(
        np.array_equal(now, before)
        for now, before in zip(
            (fixed_model.Q, fixed_model.V, fixed_model.W), fixed_parameters
        )
    ):
        raise RuntimeError("检查期间 Q/V/W 被意外修改")
    cols = min(5, CHECK_CASES)
    rows = int(np.ceil(CHECK_CASES / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < len(results):
            draw(ax, results[i], True)
            ax.set_title(f"#{i+1:02d} blocked=1 safe={results[i]['metrics']['safe']}", fontsize=9)
        else:
            ax.axis("off")
    fig.suptitle(f"{CHECK_CASES} random tasks: one blocked passage each")
    fig.tight_layout()
    fig.savefig(CHECK_OUTPUT, dpi=170)
    plt.close(fig)
    print("saved:", CHECK_OUTPUT)


def main():
    """唯一入口：由顶部 RUN_CHECK 决定普通训练还是批量检查。"""
    if RUN_CHECK:
        check()
        return
    result = run()
    save_one(result, OUTPUT, SHOW)
    q, v, w, rmse = result["model"].stats()
    count = result["model"].Q.size + result["model"].V.size + result["model"].W.size
    print(f"cells={result['world'].width*result['world'].height}, "
          f"landmarks={len(result['graph'].p)}, actions={len(result['graph'].edges)}")
    print(f"Q/V/W parameters={count}, changes=({q:.3f}, {v:.3f}, {w:.3f})")
    print(f"transition RMSE={rmse:.6f}, fallback={result['fallback']}, "
          f"curve safe={result['metrics']['safe']}")
    print(f"curve length={result['metrics']['length']:.3f}, "
          f"minimum clearance={result['metrics']['clearance']:.3f}")
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
