"""数学自检与矩阵诊断。"""

import json
import matplotlib.pyplot as plt
import numpy as np
import train
from . import config as cfg


class MathTests:
    def metrics(self):
        m, q, v, g, edges = self.m, self.m["Q"], self.m["V"], self.m["G"], self.m["edges"]
        assert v.shape == (len(q), len(edges)) and m["W"].shape == g.shape == (len(edges), len(q))
        assert all(np.isfinite(m[name]).all() for name in "QVWG") and np.isfinite(m["distance"]).all()
        assert np.all(m["adjacency"][:len(m["passages"])].sum(1) >= 2)
        actions = m["buffer_a"]
        assert np.all(edges[actions, 0] == m["buffer_o"]) and np.all(edges[actions, 1] == m["buffer_next"])
        delta = q[:, edges[:, 1]]-q[:, edges[:, 0]]
        true_g = np.zeros((len(edges), q.shape[1])); true_g[np.arange(len(edges)), edges[:, 0]] = 1
        prediction, recall = g@q, []
        for node in range(q.shape[1]):
            actual = np.flatnonzero(edges[:, 0] == node)
            predicted = np.argsort(prediction[:, node])[-len(actual):]
            recall.append(len(set(actual)&set(predicted))/len(actual))
        nodes = np.unique(np.concatenate((m["buffer_o"].ravel(), m["buffer_next"].ravel())))
        return dict(transition=float(np.sqrt(np.mean((delta-v)**2))),
                    affordance=float(np.sqrt(np.mean((true_g-prediction)**2))), recall=float(np.mean(recall)),
                    node_coverage=len(nodes)/q.shape[1], action_coverage=len(np.unique(actions))/len(edges))

    def math_checks(self):
        """快速性质测试：代数恒等式必须成立，学习误差只报告而不冒充定理。"""
        from types import SimpleNamespace
        from unittest.mock import patch
        m, rng, results = self.m, np.random.default_rng(391), {}
        frozen = {k: m[k].copy() for k in "QVWG"}
        def check(name, condition):
            if not condition:
                raise AssertionError(name)
            results[name] = True
        self.metrics()
        q, v, w, g, edges = (m[k] for k in ("Q", "V", "W", "G", "edges"))
        source = np.eye(q.shape[1])[:, edges[:, 0]]
        b = np.eye(q.shape[1])[:, edges[:, 1]]-source
        r = q@b-v
        check("incidence_rank", np.linalg.matrix_rank(b) == q.shape[1]-1)
        check("buffer_continuity", np.array_equal(m["buffer_next"][:, :-1], m["buffer_o"][:, 1:]))
        check("graph_edges", np.array_equal(np.argwhere(m["adjacency"]), edges) and
              np.array_equal(m["adjacency"], m["adjacency"].T) and not np.diag(m["adjacency"]).any())
        check("collision_free_edges", all(train.World.line_free(m["sdf"], m["points"][i], m["points"][j]) for i, j in edges))
        # c[A]记录动作次数；Bc消掉中间节点，误差由-Rc精确给出，不要求已经训练好。
        worst, bound, path_errors = 0., 0., []
        for _ in range(100):
            start = node = int(rng.integers(q.shape[1]))
            c = np.zeros(len(edges))
            for _ in range(32):
                a = int(rng.choice(m["available"][node])); c[a] += 1; node = int(edges[a, 1])
            error = q[:, start]+v@c-q[:, node]
            check("path_identity", np.allclose(error, -r@c, atol=1e-10))
            limit = 32*np.linalg.norm(r, axis=0).max()
            check("path_error_bound", np.linalg.norm(error) <= limit+1e-10)
            worst, bound = max(worst, float(np.linalg.norm(error))), float(limit)
            path_errors.append(float(np.linalg.norm(error)))
        _, _, vh = np.linalg.svd(b, full_matrices=True)
        z = vh[np.linalg.matrix_rank(b):].T
        check("cycle_identity", np.allclose(b@z, 0, atol=1e-10) and np.allclose(v@z, -r@z, atol=1e-10))
        reverse = {tuple(edge): a for a, edge in enumerate(edges)}
        check("reverse_edge_identity", all(np.allclose(v[:, a]+v[:, reverse[j, i]],
              -r[:, a]-r[:, reverse[j, i]], atol=1e-10) for a, (i, j) in enumerate(edges)))
        # G单样本外积更新后，同一状态处的残差恰好乘(1-eta*||s||²)。不承诺全图误差单调。
        s, target = q[:, 0], source.T[:, 0]
        updated = g+train.ETA_G*np.outer(target-g@s, s)
        check("g_update_identity", np.allclose(target-updated@s, (1-train.ETA_G*(s@s))*(target-g@s)))
        # rank(GQ)<=d：真实affordance矩阵的奇异值尾部给出任何线性GQ的最佳可能误差下界。
        singular, capacity = np.linalg.svd(source.T, compute_uv=False), []
        for dim in m["ablation_dims"]:
            dq, dg = (m[k] if dim == len(q) else m[f"{k}_d{dim}"] for k in ("Q", "G"))
            lower = float(np.sqrt(np.sum(singular[int(dim):]**2)/source.size))
            actual = float(np.sqrt(np.mean((dg@dq-source.T)**2)))
            check("g_rank_lower_bound", actual+1e-10 >= lower)
            capacity.append(dict(dim=int(dim), lower_rmse=lower, actual_rmse=actual))
        d, displacement = q[:, -1]-q[:, 0], v[:, 0]
        check("distance_identity", np.allclose(np.sum((d-displacement)**2)-d@d,
                                               -2*d@displacement+displacement@displacement))
        # 正交旋转只更换潜空间坐标轴，应保持效用、门控、距离和规划结果。
        rotation = np.linalg.qr(rng.normal(size=(len(q), len(q))))[0]
        rotated = dict(m, Q=rotation@q, V=rotation@v, W=w@rotation.T, G=g@rotation.T)
        args = (0, q.shape[1]-1, m["available"][0], 12, 8)
        base = self.imagine(m, *args, np.random.default_rng(7))
        check("rotation_invariance", base == self.imagine(rotated, *args, np.random.default_rng(7)))
        check("audit_read_only", base == self.imagine(m, *args, np.random.default_rng(7), audit=[]))
        explanation = {"explain": True}
        check("explanation_read_only", base == self.imagine(m, *args, np.random.default_rng(7), details=explanation))
        for i, values in enumerate(explanation["steps"]):
            scores = np.asarray(values["score"])
            expected = np.asarray(values["gate"])*(np.asarray(values["utility"])+values["noise"])
            check("explanation_matches_scores", np.allclose(scores[np.isfinite(scores)], expected[np.isfinite(scores)]) and
                  int(np.argmax(scores)) == explanation["paths"][explanation["best"]][i])
        check("zero_budget", self.imagine(m, *args[:3], 0, 8, rng)[0] is None and
              self.imagine(m, *args[:3], 12, 0, rng)[0] is None)
        same = dict(start_node=0, goal_node=0)
        check("already_at_goal", self.gcml(same) == ([0], True, []) and self.cml(same) == ([0], True))
        # 手工3节点模型：候选0走两步，候选1一步到达；固定噪声避免随机测试碰巧通过。
        tq, te = np.eye(3), np.array(((0, 1), (0, 2), (1, 2), (2, 1)))
        tv = tq[:, te[:, 1]]-tq[:, te[:, 0]]
        tiny = dict(Q=tq, V=tv, W=tv.T, G=tq[:, te[:, 0]].T, edges=te)
        noise = np.zeros((2, 2, 4)); noise[0, 0, 0] = noise[0, 1, 2] = noise[1, 0, 1] = 10
        fixed = SimpleNamespace(normal=lambda *args: noise.copy())
        choice = self.imagine(tiny, 0, 2, [0, 1], 2, 2, fixed, tolerance=.01)
        check("shortest_reached_selected", choice[0] == 1 and choice[1][:3] == (True, 1, 2))
        tiny["V"] = tv.copy(); tiny["V"][:, 0] = .8*(tq[:, 2]-tq[:, 0]); tiny["V"][:, 1] *= .2
        choice = self.imagine(tiny, 0, 2, [0, 1], 2, 1, fixed, tolerance=.01)
        check("nearest_failed_selected", choice[0] == 0 and not choice[1][0])
        tiny["V"][:, 0] = tq[:, 2]-tq[:, 0]
        check("reached_before_failed", self.imagine(tiny, 0, 2, [0, 1], 2, 1, fixed, tolerance=.01)[0] == 0)
        check("no_available_action", self.imagine(tiny, 0, 2, [], 2, 1, fixed)[0] is None)
        # 直接调用真实推理函数，确认最后允许的一步到达算成功、失败不被补齐。
        tiny.update(V=tv, available=[np.flatnonzero(te[:, 0] == i) for i in range(3)])
        case = dict(start_node=0, goal_node=2)
        with patch.object(cfg, "EXECUTION_HORIZON", 1):
            check("cml_last_step_success", self.cml(case, tiny) == ([0, 2], True))
            check("gcml_last_step_success", self.gcml(case, model=tiny, samples=2, horizon=1,
                                                      noise_scale=0)[0:2] == ([0, 2], True))
            blocked = dict(tiny, available=[np.array([], int) for _ in range(3)])
            check("dead_end_stays_failed", self.cml(case, blocked) == ([0], False) and
                  self.gcml(case, model=blocked)[0:2] == ([0], False))
        with patch.object(cfg, "EXECUTION_HORIZON", 0):
            check("zero_execution_budget", self.cml(case, tiny) == ([0], False) and
                  self.gcml(case, model=tiny)[0:2] == ([0], False))
        # 强制第二步合法动作得负分：零门控的非法动作胜出，audit必须如实识别。
        scripted = np.zeros((1, 2, 4)); scripted[0, 0, 0] = 10; scripted[0, 1, 2] = -10
        audit = []
        self.imagine(tiny, 0, 2, [0, 1], 1, 2, SimpleNamespace(normal=lambda *args: scripted), audit=audit)
        check("audit_detects_invalid_path", audit[0]["valid"] == 0 and audit[0]["actual_hits"] == 0)
        # 相同随机流重复推理应逐项一致；前缀比较直接检查实际候选的到达统计。
        one = self.imagine(tiny, 0, 2, [0, 1], 1, 2,
                           SimpleNamespace(normal=lambda *args: noise[:1].copy()), tolerance=.01)
        check("single_candidate_prefix", one[0] == 0 and one[1][:3] == (True, 2, 1))
        check("deterministic_replay", self.gcml(case, model=tiny, seed=91) == self.gcml(case, model=tiny, seed=91))
        # 相同候选复制4份，不应凭复制提高路线质量；只应改变候选计数。
        copies = self.imagine(tiny, 0, 2, [0, 1], 4, 2,
                              SimpleNamespace(normal=lambda *args: np.repeat(noise[:1], 4, axis=0)), tolerance=.01)
        check("duplicate_candidates", copies == (0, (True, 2, 4, 1, 1)))
        # 重编号同时重排行为矩阵与噪声；无平局时物理动作应相同。
        order = np.array((2, 0, 3, 1))
        renamed = dict(tiny, V=tv[:, order], W=tv.T[order], G=tiny["G"][order], edges=te[order])
        selected = self.imagine(renamed, 0, 2, np.flatnonzero(te[order, 0] == 0), 2, 2,
                                SimpleNamespace(normal=lambda *args: noise[:, :, order]), tolerance=.01)
        check("action_relabeling", order[selected[0]] == 1 and selected[1][:3] == (True, 1, 2))
        scaled = dict(tiny, Q=3*tq, V=3*tv, W=tv.T/3, G=tiny["G"]/3)
        check("consistent_scale", self.imagine(scaled, 0, 2, [0, 1], 2, 2, fixed, tolerance=.03) ==
              self.imagine(tiny, 0, 2, [0, 1], 2, 2, fixed, tolerance=.01))
        threshold = float(np.linalg.norm(tq[:, 1]-tq[:, 2]))
        first = SimpleNamespace(normal=lambda *args: noise[:1, :1].copy())
        check("threshold_inclusive", self.imagine(tiny, 0, 2, [0, 1], 1, 1, first, tolerance=threshold)[1][0])
        check("threshold_below", not self.imagine(tiny, 0, 2, [0, 1], 1, 1, first,
                                                   tolerance=np.nextafter(threshold, 0))[1][0])
        audit = []
        self.imagine(tiny, 0, 2, [0, 1], 2, 2, fixed, tolerance=.01, audit=audit)
        check("audit_exact_counts", audit == [dict(candidates=2, latent_hits=2, valid=2, actual_hits=2,
              false_hits=0, invalid_hits=0, selected_hit=1, selected_false=0)])
        # 目标存在但不可达，两个动作迫使0/1循环；到执行上限必须保留失败。
        loop_v = tq[:, [1, 0]]-tq[:, [0, 1]]
        loop = dict(Q=tq, V=loop_v, W=loop_v.T, G=tq[:, [0, 1]].T,
                    edges=np.array(((0, 1), (1, 0))), available=[[0], [1], []])
        with patch.object(cfg, "EXECUTION_HORIZON", 4):
            check("cml_cycle_failure", self.cml(case, loop) == ([0, 1, 0, 1, 0], False))
            check("gcml_cycle_failure", self.gcml(case, model=loop, samples=1, horizon=1,
                                                  noise_scale=0)[:2] == ([0, 1, 0, 1, 0], False))
        # G的谱范数是潜状态误差的最坏放大倍数；小QV误差不一定意味着小门控误差。
        perturbation = rng.normal(0, .01, (len(q), 20))
        gain = float(np.linalg.norm(g, 2))
        check("g_error_amplification_bound", np.all(np.linalg.norm(g@perturbation, axis=0) <=
              gain*np.linalg.norm(perturbation, axis=0)+1e-10))
        # 反例：动作2再动作0顺序非法，但位移和仍到目标；零门控也可能胜过负的合法得分。
        check("order_counterexample", te[2, 0] != 0 and np.allclose(tq[:, 0]+tv[:, 2]+tv[:, 0], tq[:, 2]))
        check("soft_gate_counterexample", np.argmax(np.array((0., 1.))*np.array((1., -1.))) == 0)
        grid = np.zeros((9, 9), bool); grid[4, 4] = True
        sdf = train.World.distance_field(grid)
        check("distance_field_symmetry", np.allclose(sdf, sdf[:, ::-1]) and np.allclose(sdf, sdf[::-1]))
        check("interpolation_at_grid_points", np.allclose(train.World.clearance(sdf, np.array(((2, 3), (4, 4)))),
                                                           sdf[[3, 4], [2, 4]]))
        check("matrices_unchanged", all(np.array_equal(m[k], value) for k, value in frozen.items()))
        report = dict(checks=results, path_samples=100, path_horizon=32, max_path_error=worst,
                      path_error_bound=bound, g_capacity=capacity, g_spectral_norm=gain, path_errors=path_errors)
        # rc_context只在这张图内使用中文字体，不改变已有英文报告的样式。
        with plt.rc_context(dict(cfg.CHINESE_STYLE, **{"font.size": 11})):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained", facecolor="#f4f7fb")
            ax = axes[0, 0]
            ax.plot(range(1, 101), path_errors, color="#2678b8", lw=1.5, label="实测终点误差")
            ax.axhline(bound, color="#d65f35", ls="--", label=f"理论上界 {bound:.5f}")
            ax.set(title="① 连走32步，预测误差有没有超过上界？", xlabel="随机有效路径编号（共100条）",
                   ylabel="潜空间终点误差（越低越好）", ylim=(0, bound*1.25))
            ax.legend(loc="upper right")
            ax = axes[0, 1]
            x = np.arange(len(capacity))
            ax.bar(x-.18, [c["lower_rmse"] for c in capacity], .36, color="#a5b6c9", label="秩约束下的理论下界")
            ax.bar(x+.18, [c["actual_rmse"] for c in capacity], .36, color="#2678b8", label="已训练模型的实测误差")
            ax.set(title="② 低维下，G的误差能降到零吗？", xlabel="潜空间维数",
                   ylabel="可执行动作预测的均方根误差", xticks=x, xticklabels=[c["dim"] for c in capacity])
            ax.legend(loc="upper right")
            ax = axes[1, 0]
            upper = gain*np.linalg.norm(perturbation, axis=0)
            ax.scatter(np.linalg.norm(perturbation, axis=0), np.linalg.norm(g@perturbation, axis=0),
                       color="#2678b8", s=40, label="20组随机扰动的实测值")
            xmax = np.linalg.norm(perturbation, axis=0).max()*1.1
            ax.plot([0, xmax], [0, gain*xmax], "--", color="#d65f35", label=f"最坏放大界（倍数 {gain:.2f}）")
            ax.set(title="③ 想象状态有误差，G会放大多少？", xlabel="潜状态扰动大小",
                   ylabel="G输出的变化大小", xlim=(0, xmax), ylim=(0, gain*xmax*1.1))
            ax.legend(loc="upper left")
            for ax in axes.flat[:3]:
                ax.set_facecolor("white"); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
            ax = axes[1, 1]; ax.axis("off")
            ax.text(.03, .96, f"④ 自检通过 {len(results)} / {len(results)} 组", va="top", fontsize=18, weight="bold", color="#237b58")
            ax.text(.03, .81, "如何读这张图\n\n① 蓝线都在上界以下：有效路径满足误差界。\n"
                    "② 蓝柱不能低于灰柱：低维存在表示能力限制。\n"
                    "③ 蓝点在虚线以下：验证G的误差放大上界。\n\n"
                    "测试范围\n路径与闭环、候选排序、编号与尺度变换、\n阈值边界、死路与循环、诊断只读、几何插值。\n\n"
                    "不代表：所有任务必达，或想象动作必定可执行。\n图中上界与下界不是导航成功率。",
                    va="top", linespacing=1.65, color="#334155")
            fig.suptitle("GCML 数学与边界测试｜中文解读", fontsize=21, weight="bold", color="#20334d")
            fig.savefig(cfg.MATH_FIGURE, dpi=170); plt.close(fig)
        cfg.MATH_DATA.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"math checks: {len(results)} groups passed; max path error={worst:.6g}, bound={bound:.6g}", flush=True)
        return report

    def cognition(self):
        """只读分析：关联矩阵B表达图结构，对比训练前后的代数一致性。"""
        plt.rcParams.update(cfg.CHINESE_STYLE)
        m, edges = self.m, self.m["edges"]
        n, a = len(m["points"]), len(edges)
        source = np.eye(n)[:, edges[:, 0]]
        target = np.eye(n)[:, edges[:, 1]]
        b = target-source                       # shape=(节点数,动作数)，每列为终点one-hot减起点
        _, _, vh = np.linalg.svd(b, full_matrices=True)
        cycles = vh[np.linalg.matrix_rank(b):].T  # B的零空间；包含所有闭合路径的动作计数向量
        hop = np.where(m["adjacency"], 1., np.inf)
        np.fill_diagonal(hop, 0)
        for k in range(n):
            hop = np.minimum(hop, hop[:, k, None]+hop[None, k, :])
        pairs, report = np.triu_indices(n, 1), {}
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        for suffix, label in (("0", "训练前"), ("", "训练后")):
            q, v, w, g = (m[key+suffix] for key in "QVWG")
            delta, error = q@b, q@b-v
            latent = np.linalg.norm(q[:, :, None]-q[:, None, :], axis=0)
            alignment = np.sum(w*delta.T, 1)/np.maximum(np.linalg.norm(w, axis=1)*np.linalg.norm(delta, axis=0), 1e-12)
            predicted = q[:, edges[:, 0]]+v
            decoded = np.argmin(np.linalg.norm(predicted[:, None, :]-q[:, :, None], axis=0), axis=0)
            gate, truth = g@q, source.T.astype(bool)
            mask = gate >= .5
            report[label] = dict(q_rank=int(np.linalg.matrix_rank(q)),
                min_state_distance=float(latent[pairs].min()),
                hop_distance_correlation=float(np.corrcoef(hop[pairs], latent[pairs])[0, 1]),
                transition_relative=float(np.linalg.norm(error)/max(np.linalg.norm(delta), 1e-12)),
                max_edge_error=float(np.linalg.norm(error, axis=0).max()),
                cycle_relative=float(np.linalg.norm(v@cycles)/max(np.linalg.norm(v), 1e-12)),
                transition_decode=float(np.mean(decoded == edges[:, 1])),
                w_cosine=float(alignment.mean()), g_rmse=float(np.sqrt(np.mean((gate-truth)**2))),
                g_f1=float(2*np.sum(mask & truth)/max(mask.sum()+truth.sum(), 1)))
            axes[0, 0].scatter(hop[pairs], latent[pairs], s=4, alpha=.15, label=label)
            axes[0, 1].plot(np.linalg.norm(error, axis=0), lw=1, label=label)
            axes[1, 0].plot(np.sort(alignment), lw=1, label=label)
        im = axes[1, 1].imshow(m["G"]@m["Q"], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[1, 1], label="预测可执行性（仅显示范围裁剪至[0,1]）")
        for ax, title, x, y in zip(axes.flat,
                ("Q：潜空间几何与图距离", "Q/V：逐边转移一致性", "W：动作方向对齐", "GQ：恢复各节点出边"),
                ("图上最短跳数", "动作编号", "动作排序位置", "节点编号"),
                ("潜空间距离", "转移误差（L2）", "余弦相似度", "动作编号")):
            ax.set(title=title, xlabel=x, ylabel=y)
        axes[0, 1].set_yscale("log")
        for ax in list(axes.flat)[:3]:
            ax.legend(); ax.grid(alpha=.15)
        fig.suptitle("固定训练图上的矩阵证据（不是跨地图泛化测试）")
        fig.savefig(cfg.MATRIX_FIGURE, dpi=180); plt.close(fig)
        cfg.MATRIX_DATA.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("matrix analysis:", json.dumps(report), flush=True)
        return report
