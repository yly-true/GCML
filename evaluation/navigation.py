"""模型加载与GCML/CML推理。"""

from pathlib import Path
import numpy as np
import train
from . import config as cfg


class Navigator:
    def __init__(self):
        cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True); cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not Path(cfg.MODEL_FILE).exists():
            raise FileNotFoundError("请先运行 python train.py")
        with np.load(cfg.MODEL_FILE) as data:
            self.m = {key: data[key].copy() for key in data.files}
        m, edges, points = self.m, self.m["edges"], self.m["points"]
        m["sdf"] = train.World.distance_field(m["grid"])
        m["available"] = [np.flatnonzero(edges[:, 0] == n) for n in range(len(points))]
        pair = np.linalg.norm(points[:, None]-points[None], axis=2)
        distance = np.where(m["adjacency"], pair, np.inf)
        np.fill_diagonal(distance, 0)
        for k in range(len(points)):
            distance = np.minimum(distance, distance[:, k, None]+distance[None, k, :])
        m["distance"] = distance

    def random_point(self, rng):
        m, cells = self.m, np.argwhere(~self.m["grid"])
        while True:
            y, x = cells[rng.integers(len(cells))]
            point = np.array((x, y))+rng.uniform(-.3, .3, 2)
            if train.World.clearance(m["sdf"], point)[0] >= float(m["radius"])+.5:
                return point

    def case(self, seed):
        """连续起终点只用于挂接地标，不参与潜空间rollout。"""
        rng, m, points = np.random.default_rng(seed+200), self.m, self.m["points"]
        while True:
            start, goal = self.random_point(rng), self.random_point(rng)
            if np.linalg.norm(start-goal) <= .55*min(m["grid"].shape):
                continue
            visible = lambda p: [int(i) for i in np.argsort(np.linalg.norm(points-p, axis=1))
                                 if train.World.line_free(m["sdf"], p, points[i])][:6]
            pairs = [(i, j) for i in visible(start) for j in visible(goal) if np.isfinite(m["distance"][i, j])]
            if pairs:
                i, j = min(pairs, key=lambda z: np.linalg.norm(start-points[z[0]])+
                           m["distance"][z]+np.linalg.norm(goal-points[z[1]]))
                return dict(start=start, goal=goal, start_node=i, goal_node=j)

    @staticmethod
    def imagine(m, node, goal, real_actions, samples, horizon, rng, noise_scale=cfg.NOISE,
                tolerance=cfg.LATENT_GOAL_TOLERANCE, audit=None, details=None):
        """在潜空间并行生成候选，返回最佳候选第一动作与统计。"""
        # 无首步动作就不能开始想象，不能跳到后续G门控而“凭空走出”死路。
        if not len(real_actions) or samples == 0 or horizon == 0:
            return None, (False, 0, 0, 0, 0)
        q, v, w, g = (m[name] for name in "QVWG")
        action_count, target = len(m["edges"]), q[:, goal, None]
        # None保留列维度；state[d,K]每列是一条候选，repeat复制同一真实起始编码。
        state = np.repeat(q[:, node, None], samples, axis=1)
        paths, arrived = [[] for _ in range(samples)], np.zeros(samples, bool)
        snapshots = []
        residual, noise = np.full(samples, np.inf), rng.normal(0, noise_scale, (samples, horizon, action_count))
        # noise[K,H,A]按候选优先生成，固定种子下增加K会保留原候选的噪声前缀。
        for step in range(horizon):
            active = np.flatnonzero(~arrived)
            if not len(active):
                break
            utility = w@(target-state)
            utility /= np.maximum(np.linalg.norm(utility, axis=0), 1e-9)
            gate = np.zeros_like(utility) if step == 0 else g@state
            # 首步知道真实可行动作；后续只读G@state，绝不查真实下一节点。
            if step == 0:
                gate[real_actions] = 1
            score = gate*(utility+noise[:, step].T)
            if step == 0:
                score[gate == 0] = -np.inf
            chosen = np.argmax(score, axis=0)
            if details is not None and details.get("explain") and step < 2:
                snapshots.append([x.copy() for x in (utility, gate, noise[:, step].T, score)])
            for sample in active:
                action = int(chosen[sample])
                if not np.isfinite(score[action, sample]):
                    continue
                paths[sample].append(action)
                state[:, sample] += v[:, action]
                residual[sample] = np.linalg.norm(state[:, sample]-target[:, 0])
                arrived[sample] = residual[sample] <= tolerance

        ranked = [((not arrived[i], len(path) if arrived[i] else residual[i]), i)
                  for i, path in enumerate(paths) if path]
        if not ranked:
            return None, (False, 0, 0, 0, 0)
        best = min(ranked)[1]
        if details is not None:
            details.update(paths=paths, best=best, arrived=arrived.tolist(), residual=residual.tolist())
            if snapshots:
                details["steps"] = [dict(zip(("utility", "gate", "noise", "score"),
                    [x[:, best].tolist() for x in values])) for values in snapshots[:len(paths[best])]]
        # 元组按从左到右比较：到达优先；到达者选短的，否则选最终残差小的。
        valid = [tuple(path) for path in paths if path]
        stats = (bool(arrived[best]), len(paths[best]), int(arrived.sum()),
                 len({path[0] for path in valid}), len(set(valid)))
        if audit is not None:
            # 排名结束后才读真实边；仅验证，不改变评分、候选或执行动作。
            valid_path, actual_hit = [], []
            for path in paths:
                current, executable = node, True
                for action in path:
                    source, target_node = m["edges"][action]
                    if source != current:
                        executable = False
                        break
                    current = int(target_node)
                valid_path.append(executable)
                actual_hit.append(executable and current == goal)
            valid_path, actual_hit = np.array(valid_path), np.array(actual_hit)
            audit.append(dict(candidates=samples, latent_hits=int(arrived.sum()),
                valid=int(valid_path.sum()), actual_hits=int(actual_hit.sum()),
                false_hits=int(np.sum(arrived & ~actual_hit)),
                invalid_hits=int(np.sum(arrived & ~valid_path)),
                selected_hit=int(arrived[best]), selected_false=int(arrived[best] and not actual_hit[best])))
        return int(paths[best][0]), stats

    def gcml(self, case, samples=cfg.ROLLOUTS, horizon=cfg.ROLLOUT_HORIZON, seed=0, model=None,
             noise_scale=cfg.NOISE, tolerance=cfg.LATENT_GOAL_TOLERANCE, audit=None, one_step=False):
        """MPC式GCML：想象多步，只真实执行最佳候选第一步。"""
        m, node, goal = self.m if model is None else model, case["start_node"], case["goal_node"]
        route, trace = [node], []
        for step in range(cfg.EXECUTION_HORIZON):
            if node == goal:
                break
            actions = m["available"][node]
            if not len(actions):
                break
            if one_step:
                utility = m["W"]@(m["Q"][:, goal]-m["Q"][:, node])
                action = int(actions[np.argmax(utility[actions])])
            else:
                action, stats = self.imagine(m, node, goal, actions, samples, horizon,
                                             np.random.default_rng(seed+step), noise_scale, tolerance, audit)
                if action is not None:
                    trace.append(stats)
            if action is None:
                break
            # 只执行候选的第一个动作；下一轮重新读取Q[:,node]，不沿用预测状态。
            node = int(m["edges"][action, 1])
            route.append(node)
        return route, node == goal, trace

    def cml(self, case, model=None):
        """确定性一步CML：无噪声、G和潜空间多步想象。"""
        return self.gcml(case, model=model, one_step=True)[:2]
