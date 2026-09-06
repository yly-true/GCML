"""创建地图、采集轨迹并训练Q/V/W/G；不负责推理。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ======================== 全部训练参数 ========================
WIDTH = HEIGHT = 100
SEED = 0
AGENT_RADIUS = .35
LINE_SAMPLE_SPACING = .30
LANDMARKS_PER_ROOM = 4
BLOCKED_PASSAGES = ((25, 37), (50, 62), (62, 25), (75, 87))  # 训练前固定关闭的门中心；()表示全开
LATENT_DIM = 128
ABLATION_DIMS = (8, 16, 32, 64)
REPEAT_DIMS, REPEAT_SEEDS = (64, 128), (1, 2)
OUTPUT_DIR = Path("outputs")
MODEL_DIR, FIGURE_DIR = OUTPUT_DIR/"models", OUTPUT_DIR/"figures"
REPEAT_FILE = MODEL_DIR/"gcml_repeats.npz"
BUFFER_TRAJECTORIES, MAX_BUFFER_TRAJECTORIES = 200, 500
TRAJECTORY_LENGTH, BATCH_TRAJECTORIES, EPOCHS = 100, 16, 1000
ETA_Q, ETA_V, ETA_W, ETA_G = .1, .01, .01, .01
MODEL_FILE, LOSS_FIGURE = MODEL_DIR/"gcml_model.npz", FIGURE_DIR/"01_training_loss.png"
# =============================================================


class World:
    """规则地图及其离散地标图。"""

    def __init__(self):
        if (HEIGHT, WIDTH, LANDMARKS_PER_ROOM) != (100, 100, 4):
            raise ValueError("当前简洁地图固定为100×100、每房间4节点；请勿只改尺寸参数")
        self.grid, self.passages = self._map()  #self.grid这个保存了地图的信息，bool类型，是否是空地还是障碍物      self.passages是list，(x, y, direction)，direction为1表示是竖墙的 否则是横墙的  xy是坐标  是房间与房间联通的

        # _map仍生成24个门；此处封墙并删除对应门节点，所有测试共享这张地图。
        closed = np.array([tuple(p[:2]) in BLOCKED_PASSAGES for p in self.passages])
        self.blocked = self.passages[closed].copy()
        if len(self.blocked) != len(BLOCKED_PASSAGES):
            raise ValueError("BLOCKED_PASSAGES中存在重复或不存在的门中心")
        for x, y, axis in self.blocked:
            if axis == 0:
                self.grid[y-3:y+4, x] = True
            else:
                self.grid[y, x-3:x+4] = True
        self.passages = self.passages[~closed]
        self.points, self.adjacency, self.edges = self._graph()
        reached = np.arange(len(self.points)) == 0
        for _ in self.points:
            reached |= self.adjacency @ reached
        if not reached.all():
            raise ValueError("封闭这些门会使地图断连，请减少BLOCKED_PASSAGES")

    @staticmethod
    def _map():
        """创建100×100规则地图；False为空地，True为墙。"""
        # np.zeros生成全为False的布尔矩阵，shape=(100,100)。
        grid = np.zeros((HEIGHT, WIDTH), bool)

        # grid[y, x]是第y行、第x列；:表示该维度全部元素。
        # [0, -1]同时选中第一项和最后一项，-1表示倒数第一项。
        grid[[0, -1], :] = True                # 上下边界
        grid[:, [0, -1]] = True                # 左右边界

        # 三条内部竖墙/横墙，把地图分成4×4共16个房间。
        walls = (25, 50, 75)
        bounds = (1, 25, 50, 75, 99)            # 房间边界坐标
        # bounds[:-1]去掉最后一个元素：(1,25,50,75)。
        # bounds[1:]去掉第一个元素：(25,50,75,99)。
        # zip按位置配对：(1,25)、(25,50)、(50,75)、(75,99)。
        # (a+b)//2计算每段的整数中点；//是整除并向下取整。
        # 外层[]是列表推导式，等价于循环后逐个append。
        centres = [(a+b)//2 for a, b in zip(bounds[:-1], bounds[1:])]
        # centres=[13,37,62,87]，即4个房间段的中心位置。
        passages = []                           # 每项为(x, y, 墙方向)

        # 在竖墙上开门：方向0表示墙沿y轴延伸。
        for x in walls:                         # 依次处理x=25,50,75
            grid[1:-1, x] = True
            # 1:-1从第1项取到最后一项之前，避开外边界。
            for y in centres:
                # 切片左闭右开：y-3,...,y+3，共7格，设为False打开门。
                grid[y-3:y+4, x] = False
                # passages记录(x坐标,y坐标,方向)；0表示竖墙门。
                passages.append((x, y, 0))

        # 在横墙上开门：方向1表示墙沿x轴延伸。
        for y in walls:                         # 依次处理y=25,50,75
            grid[y, 1:-1] = True
            for x in centres:
                # 固定行y，在x方向取连续7格作为门。
                grid[y, x-3:x+4] = False
                passages.append((x, y, 1))      # 1表示横墙门

        # grid用于碰撞/绘图，passages用于后续创建地标图。
        # np.asarray把Python列表转换为整数NumPy数组，shape=(24,3)。
        return grid, np.asarray(passages, int)

    def _graph(self):
        """每房间放置2x2节点；每个门连接墙两侧最近节点。"""
        bounds, passages = (1, 25, 50, 75, 99), self.passages
        points, rooms, links = [p[:2].astype(float) for p in passages], [[] for _ in range(16)], []

        def room_id(point):
            # searchsorted返回坐标属于哪个区间；行号×4+列号把二维房间编号变成0~15。
            col, row = np.searchsorted(bounds[1:-1], point, side="right")
            return 4*row+col

        for i, (x, y, axis) in enumerate(passages):
            offsets = ((-1, 0), (1, 0)) if axis == 0 else ((0, -1), (0, 1))
            for dx, dy in offsets:
                rooms[room_id((x+dx, y+dy))].append(i)

        for row in range(4):
            for col in range(4):
                room, (x0, x1), (y0, y1) = 4*row+col, bounds[col:col+2], bounds[row:row+2]
                xs = x0+(x1-x0)*np.array((1, 2))/3
                ys = y0+(y1-y0)*np.array((1, 2))/3
                inside = list(range(len(points), len(points)+LANDMARKS_PER_ROOM))
                points += [np.array((x, y)) for y in ys for x in xs]
                links += [(inside[a], inside[b]) for a, b in ((0, 1), (0, 2), (1, 3), (2, 3))]
                for door in rooms[room]:
                    links.append((door, min(inside, key=lambda i: np.linalg.norm(points[i]-points[door]))))

        points = np.asarray(points)
        # adjacency[i,j]表示i能否走到j；argwhere把所有True的位置变成[A,2]有向边表。
        adjacency = np.zeros((len(points), len(points)), bool)
        for i, j in links:
            adjacency[i, j] = adjacency[j, i] = True
        assert np.all(adjacency[:len(passages)].sum(1) >= 2)
        return points, adjacency, np.argwhere(adjacency).astype(int)

    @staticmethod
    def distance_field(grid):
        """两遍扫描近似计算每个栅格到墙的距离。"""
        h, w, root2 = *grid.shape, np.sqrt(2.)
        d = np.where(grid, 0., 1e6)
        passes = ((range(h), ((-1, 0, 1), (0, -1, 1), (-1, -1, root2), (-1, 1, root2))),
                  (range(h-1, -1, -1), ((1, 0, 1), (0, 1, 1), (1, 1, root2), (1, -1, root2))))
        for scan, (ys, moves) in enumerate(passes):
            for y in ys:
                # 反向扫描必须同时反转列顺序，才能传播右侧邻居的距离。
                for x in (range(w) if scan == 0 else range(w-1, -1, -1)):
                    for dy, dx, cost in moves:
                        yy, xx = y+dy, x+dx
                        if 0 <= yy < h and 0 <= xx < w:
                            d[y, x] = min(d[y, x], d[yy, xx]+cost)
        return d-.5

    @staticmethod
    def clearance(sdf, points):
        """连续坐标处的墙面距离（双线性插值）。"""
        p, (h, w) = np.atleast_2d(np.asarray(points, float)), sdf.shape
        x, y = np.clip(p[:, 0], 0, w-1), np.clip(p[:, 1], 0, h-1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
        dx, dy = x-x0, y-y0
        return ((1-dx)*(1-dy)*sdf[y0, x0]+dx*(1-dy)*sdf[y0, x1]+
                (1-dx)*dy*sdf[y1, x0]+dx*dy*sdf[y1, x1])

    @staticmethod
    def line_free(sdf, a, b, margin=.08):
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(2, int(np.linalg.norm(b-a)/LINE_SAMPLE_SPACING)+2)
        return bool(np.all(World.clearance(sdf, np.linspace(a, b, count)) >= AGENT_RADIUS+margin))


class GCML:
    """随机轨迹buffer和Q/V/W/G局部学习。"""

    def __init__(self, world):
        self.world, self.edges, self.node_count = world, world.edges, len(world.points)

    def collect(self):
        # outgoing[n]存节点n的可行动作编号；每条轨迹分别保存起点、动作、下一节点。
        rng, edges = np.random.default_rng(SEED+1), self.edges
        outgoing = [np.flatnonzero(edges[:, 0] == n) for n in range(self.node_count)]
        obs, actions, nxt = [], [], []
        seen_n, seen_a = np.zeros(self.node_count, bool), np.zeros(len(edges), bool)
        while len(obs) < BUFFER_TRAJECTORIES or not (seen_n.all() and seen_a.all()):
            if len(obs) >= MAX_BUFFER_TRAJECTORIES:
                raise RuntimeError("随机游走buffer没有覆盖全部节点和动作")
            node, o, a, o2 = int(rng.integers(self.node_count)), [], [], []
            seen_n[node] = True
            for _ in range(TRAJECTORY_LENGTH):
                action = int(rng.choice(outgoing[node]))
                target = int(edges[action, 1])
                o.append(node); a.append(action); o2.append(target)
                seen_a[action] = seen_n[target] = True
                node = target
            obs.append(o); actions.append(a); nxt.append(o2)
        # 前三个数组均为[轨迹数,轨迹长度]；最后两个布尔数组检查节点/动作是否全部见过。
        return (np.asarray(obs, np.int16), np.asarray(actions, np.int16),
                np.asarray(nxt, np.int16), seen_n, seen_a)

    def fit(self, buffer, dim=LATENT_DIM, seed=SEED):
        # Q[d,N]、V[d,A]、W/G[A,d]；normal第二个参数是标准差，不是方差。
        rng, edges, action_count = np.random.default_rng(seed), self.edges, len(self.edges)
        q = rng.normal(0, 1, (dim, self.node_count))
        v = rng.normal(0, .1, (dim, action_count))
        w = rng.normal(0, .1, (action_count, dim))
        g = rng.normal(0, .1, (action_count, dim))
        initial = tuple(x.copy() for x in (q, v, w, g))
        true_g = np.zeros((action_count, self.node_count))
        true_g[np.arange(action_count), edges[:, 0]] = 1
        history = {"transition": [], "w": [], "g": []}

        def record():
            delta = q[:, edges[:, 1]]-q[:, edges[:, 0]]
            cosine = np.sum(w*delta.T, 1)/np.maximum(np.linalg.norm(w, axis=1)*np.linalg.norm(delta.T, axis=1), 1e-9)
            history["transition"].append(float(np.sqrt(np.mean((delta-v)**2))))
            history["w"].append(float(1-cosine.mean()))
            history["g"].append(float(np.sqrt(np.mean((true_g-g@q)**2))))

        record()
        for _ in range(EPOCHS):
            # ids抽取整条轨迹；reshape(-1)把batch展平，列索引一次取出全部转移。
            ids = rng.choice(len(buffer[0]), min(BATCH_TRAJECTORIES, len(buffer[0])), replace=False)
            source, action, target = (x[ids].reshape(-1) for x in buffer[:3])
            state, next_state = q[:, source], q[:, target]
            error, difference = next_state-state-v[:, action], next_state-state
            # 布尔mask选同一节点/动作的样本；mean(1)沿样本维求均值，避免重复索引覆盖。
            for index in np.unique(target):
                q[:, index] -= ETA_Q*error[:, target == index].mean(1)
            for index in np.unique(action):
                mask = action == index
                v[:, index] += ETA_V*error[:, mask].mean(1)
                w[index] += ETA_W*difference[:, mask].mean(1)
            for index in np.unique(source):
                s = q[:, index]
                # outer([A],[d])生成[A,d]外积；用更新后的Q列纠正G，不使用目标节点信息。
                g += ETA_G*np.outer(true_g[:, index]-g@s, s)
            record()
        return (q, v, w, g), initial, true_g, history

    def repeats(self):
        """固定原buffer，改变初始化和batch随机种子；不覆盖主模型。"""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with np.load(MODEL_FILE) as data:
            assert np.array_equal(data["edges"], self.edges) and np.array_equal(data["grid"], self.world.grid)
            buffer = tuple(data[k] for k in ("buffer_o", "buffer_a", "buffer_next"))
        payload = dict(edges=self.edges, buffer_o=buffer[0], buffer_a=buffer[1], buffer_next=buffer[2],
                       seeds=REPEAT_SEEDS, dimensions=REPEAT_DIMS, epochs=EPOCHS)
        for dim in REPEAT_DIMS:
            for seed in REPEAT_SEEDS:
                for name, value in zip("QVWG", self.fit(buffer, dim, seed)[0]):
                    payload[f"{name}_d{dim}_s{seed}"] = value
                print(f"repeat trained: d={dim}, seed={seed}", flush=True)
        np.savez_compressed(REPEAT_FILE, **payload)

    def run(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True); FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        buffer = self.collect()
        (q, v, w, g), initial, true_g, history = self.fit(buffer)
        ablation = {}
        for dim in ABLATION_DIMS:
            print(f"training latent-dimension ablation: d={dim}")
            for name, value in zip("QVWG", self.fit(buffer, dim)[0]):
                ablation[f"{name}_d{dim}"] = value
        payload = dict(grid=self.world.grid, passages=self.world.passages, blocked=self.world.blocked, points=self.world.points,
                       adjacency=self.world.adjacency, edges=self.edges, Q=q, V=v, W=w, G=g,
                       Q0=initial[0], V0=initial[1], W0=initial[2], G0=initial[3],
                       buffer_o=buffer[0], buffer_a=buffer[1], buffer_next=buffer[2],
                       loss_transition=history["transition"], loss_w=history["w"], loss_g=history["g"],
                       radius=AGENT_RADIUS, ablation_dims=np.asarray((*ABLATION_DIMS, LATENT_DIM)))
        payload.update(ablation)
        np.savez_compressed(MODEL_FILE, **payload)

        plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for values, label in ((history["transition"], "Q/V转移误差"),
                              (history["w"], "W方向误差"), (history["g"], "G可执行性误差")):
            ax.plot(values, label=label)
        ax.set(xlabel="训练轮次", ylabel="误差（越低越好）", title="基于轨迹缓存的GCML训练")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(LOSS_FIGURE, dpi=180); plt.close(fig)

        delta = q[:, self.edges[:, 1]]-q[:, self.edges[:, 0]]
        changes = tuple(round(float(np.linalg.norm(x-y)), 3) for x, y in zip((q, v, w, g), initial))
        print(f"rooms=16, interior/room={LANDMARKS_PER_ROOM}, passage_nodes={len(self.world.passages)}")
        print(f"nodes={self.node_count}, actions={len(self.edges)}, latent_dim={len(q)}")
        print(f"buffer={len(buffer[0])} trajectories x {TRAJECTORY_LENGTH} steps, "
              f"node_coverage={buffer[3].mean():.1%}, action_coverage={buffer[4].mean():.1%}")
        print(f"Q/V/W/G changes={changes}")
        print(f"transition_RMSE={np.sqrt(np.mean((delta-v)**2)):.6f}, "
              f"G_RMSE={np.sqrt(np.mean((true_g-g@q)**2)):.6f}")
        print("saved:", Path(MODEL_FILE).resolve(), Path(LOSS_FIGURE).resolve())
        self.repeats()


if __name__ == "__main__":
    world = World()
    model = GCML(world)
    model.run()
