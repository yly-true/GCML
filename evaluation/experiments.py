"""量化消融与多种子复验。"""

import json
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import train
from . import config as cfg


class Experiments:
    def model_for(self, dim, seed=0, repeats=None):
        data = repeats if seed else self.m
        suffix = f"_d{dim}_s{seed}" if seed else (f"_d{dim}" if dim != len(self.m["Q"]) else "")
        return dict(self.m, **{k: data[k+suffix] for k in "QVWG"})

    def trial(self, case, model, seed, noise, tolerance, inspect=False):
        """所有消融共用一次推理、合法性检查和SPL统计。"""
        audit, started = [] if inspect else None, perf_counter()
        route, success, trace = self.gcml(case, seed=seed, model=model, noise_scale=noise,
                                         tolerance=tolerance, audit=audit)
        elapsed = perf_counter()-started
        assert all(model["adjacency"][a, b] for a, b in zip(route[:-1], route[1:]))
        length = np.linalg.norm(np.diff(model["points"][route], axis=0), axis=1).sum()
        shortest = float(model["distance"][case["start_node"], case["goal_node"]])
        totals = {key: sum(a[key] for a in audit) for key in audit[0]} if audit else {}
        result = dict(success=bool(success), steps=len(route)-1,
                      spl=float(success)*shortest/max(shortest, float(length), 1e-12))
        result.update(totals if inspect else dict(seconds=elapsed, hits=sum(t[0] for t in trace), decisions=len(trace)))
        return result

    def benchmark(self):
        cases = [self.case(cfg.SEED+1000+i) for i in range(cfg.BENCHMARK_CASES)]
        untrained = dict(self.m)
        for name in "QVWG":
            untrained[name] = self.m[name+"0"]

        def evaluate(model, samples=cfg.ROLLOUTS, horizon=cfg.ROLLOUT_HORIZON, seed=0):
            runs = [self.gcml(case, samples, horizon, seed+i, model) for i, case in enumerate(cases)]
            hits = [hit for _, _, trace in runs for hit, *_ in trace]
            return 100*np.mean([run[1] for run in runs]), 100*np.mean(hits) if hits else 100.

        trained = [evaluate(self.m, n, seed=cfg.SEED+4000) for n in cfg.ROLLOUT_BUDGETS]
        raw = [evaluate(untrained, n, seed=cfg.SEED+5000) for n in cfg.ROLLOUT_BUDGETS]
        horizons = [evaluate(self.m, horizon=h, seed=cfg.SEED+6000) for h in cfg.HORIZON_BUDGETS]
        scores = dict(count={"GCML": [x[0] for x in trained], "未训练GCML": [x[0] for x in raw]},
                      count_reach=[x[1] for x in trained],
                      cml=100*np.mean([self.cml(case)[1] for case in cases]),
                      horizon=[x[0] for x in horizons], horizon_reach=[x[1] for x in horizons])
        cfg.ROLLOUT_DATA.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        return scores

    def ablations(self):
        """单因素配对实验；最短路只用于SPL评分，不参与想象或执行。"""
        plt.rcParams.update(cfg.CHINESE_STYLE)
        cases = [self.case(cfg.SEED+1000+i) for i in range(cfg.BENCHMARK_CASES)]
        groups = {"dimension": self.m["ablation_dims"].tolist(),
                  "noise": cfg.NOISE_LEVELS, "tolerance": cfg.GOAL_TOLERANCES}
        report, cache = {}, {}
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for column, (name, values) in enumerate(groups.items()):
            rows = []
            for value in values:
                dim = int(value) if name == "dimension" else len(self.m["Q"])
                noise = float(value) if name == "noise" else cfg.NOISE
                tolerance = float(value) if name == "tolerance" else cfg.LATENT_GOAL_TOLERANCE
                key, model = (dim, noise, tolerance), self.model_for(dim)
                if key not in cache:
                    trials = [dict(case=i, start=case["start_node"], goal=case["goal_node"],
                                   **self.trial(case, model, cfg.SEED+9000+i, noise, tolerance))
                              for i, case in enumerate(cases)]
                    q, g, edges = model["Q"], model["G"], model["edges"]
                    truth = edges[:, 0, None] == np.arange(q.shape[1])
                    mask = g@q >= .5
                    p, n = np.mean([t["success"] for t in trials]), len(trials)
                    centre, half = (p+1.96**2/(2*n))/(1+1.96**2/n), 1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/(1+1.96**2/n)
                    cache[key] = dict(success=float(p), ci95=[centre-half, centre+half],
                        spl=float(np.mean([t["spl"] for t in trials])),
                        decision_ms=1000*sum(t["seconds"] for t in trials)/max(sum(t["decisions"] for t in trials), 1),
                        rollout_hit=sum(t["hits"] for t in trials)/max(sum(t["decisions"] for t in trials), 1),
                        g_f1=float(2*np.sum(mask & truth)/max(mask.sum()+truth.sum(), 1)), trials=trials)
                rows.append(dict(value=value, **cache[key]))
                print(f"ablation {name}={value}: success={rows[-1]['success']:.1%}, SPL={rows[-1]['spl']:.3f}", flush=True)
            report[name] = rows
            rate = np.array([r["success"] for r in rows])*100
            interval = np.array([r["ci95"] for r in rows]).T*100
            axes[0, column].errorbar(values, rate, yerr=np.maximum([rate-interval[0], interval[1]-rate], 0),
                                     fmt="o-", capsize=4, color="#e87924", label="成功率与95% Wilson区间")
            axes[1, column].plot(values, [r["spl"] for r in rows], "o-", color="#2878b5", label="SPL（失败计0）")
            for row in range(2):
                axes[row, column].set(xlabel={"dimension":"潜空间维数", "noise":"噪声标准差", "tolerance":"到达阈值"}[name],
                                      ylim=(-2, 102) if row == 0 else (-.02, 1.02),
                                      ylabel="成功率（%）" if row == 0 else "路径效率")
                axes[row, column].grid(alpha=.15); axes[row, column].legend(fontsize=8)
        fig.suptitle(f"单因素配对实验｜{len(cases)}组任务｜{cfg.ROLLOUTS}条候选｜想象长度{cfg.ROLLOUT_HORIZON}\n"
                     "每个维数仅一个训练种子；区间表示任务抽样误差，不表示训练随机性")
        fig.savefig(cfg.ABLATION_FIGURE, dpi=180); plt.close(fig)
        cfg.ABLATION_DATA.write_text(json.dumps(dict(seed=cfg.SEED, candidates=cfg.ROLLOUTS,
            horizon=cfg.ROLLOUT_HORIZON, noise=cfg.NOISE, tolerance=cfg.LATENT_GOAL_TOLERANCE,
            cases=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in c.items()} for c in cases],
            results=report), indent=2), encoding="utf-8")
        return report

    def reliability(self):
        """新节点对上的诊断与三训练种子复验；测试不训练、不修正想象轨迹。"""
        plt.rcParams.update(cfg.CHINESE_STYLE)
        with np.load(train.REPEAT_FILE) as data:
            repeats = dict(data)
        for key in ("edges", "buffer_o", "buffer_a", "buffer_next"):
            assert np.array_equal(repeats[key], self.m[key]), "重复模型与主模型的图或buffer不一致，请重新训练"
        rng = np.random.default_rng(cfg.RELIABILITY_SEED)
        cases = [dict(zip(("start_node", "goal_node"), map(int, rng.choice(len(self.m["Q"].T), 2, replace=False))))
                 for _ in range(cfg.RELIABILITY_CASES)]
        configs = [(d, 0, cfg.NOISE, mode) for d in self.m["ablation_dims"].tolist() for mode in ("absolute", "relative")]
        configs += [(int(d), int(s), noise, "absolute") for d in repeats["dimensions"]
                    for s in (0, *repeats["seeds"]) for noise in cfg.RELIABILITY_NOISES]
        rows = []
        for dim, seed, noise, mode in dict.fromkeys(configs):
            model = self.model_for(dim, seed, repeats)
            q, edges = model["Q"], model["edges"]
            scale = float(np.median(np.linalg.norm(q[:, edges[:, 1]]-q[:, edges[:, 0]], axis=0)))
            tolerance = cfg.RELATIVE_TOLERANCE*scale if mode == "relative" else cfg.LATENT_GOAL_TOLERANCE
            trials = [dict(**case, **self.trial(case, model, cfg.RELIABILITY_SEED+1000+i, noise, tolerance, True))
                      for i, case in enumerate(cases)]
            total = lambda key: sum(t.get(key, 0) for t in trials)
            row = dict(dim=dim, seed=seed, noise=noise, mode=mode, tolerance=tolerance, edge_scale=scale,
                success=float(np.mean([t["success"] for t in trials])), spl=float(np.mean([t["spl"] for t in trials])),
                false_hit=total("false_hits")/max(total("latent_hits"), 1),
                invalid_hit=total("invalid_hits")/max(total("latent_hits"), 1),
                valid=total("valid")/max(total("candidates"), 1),
                selected_false=total("selected_false")/max(total("selected_hit"), 1), trials=trials)
            rows.append(row)
            print(f"reliability d={dim} seed={seed} noise={noise} {mode}: success={row['success']:.1%}, "
                  f"SPL={row['spl']:.3f}, false-hit={row['false_hit']:.1%}", flush=True)
            cfg.RELIABILITY_DATA.write_text(json.dumps(dict(task_seed=cfg.RELIABILITY_SEED,
                candidates=cfg.ROLLOUTS, horizon=cfg.ROLLOUT_HORIZON, relative_factor=cfg.RELATIVE_TOLERANCE,
                results=rows), indent=2), encoding="utf-8")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for mode in ("absolute", "relative"):
            selected = [r for r in rows if r["seed"] == 0 and r["noise"] == cfg.NOISE and r["mode"] == mode]
            for ax, key in zip(axes[0], ("success", "false_hit")):
                ax.plot([r["dim"] for r in selected], [100*r[key] for r in selected], "o-",
                        label={"absolute":"固定阈值", "relative":"归一化阈值"}[mode])
        for noise in cfg.RELIABILITY_NOISES:
            for ax, key in zip(axes[1], ("success", "spl")):
                values = [[r[key]*100 for r in rows if r["dim"] == d and r["noise"] == noise and r["mode"] == "absolute"]
                          for d in repeats["dimensions"]]
                ax.errorbar(repeats["dimensions"], np.mean(values, axis=1),
                            yerr=np.std(values, axis=1, ddof=1), fmt="o-", capsize=5, label=f"噪声={noise}")
        for ax, title in zip(axes.flat, ("真实导航成功率（%）", "潜空间到达中的虚假比例（%）",
                                       "成功率：训练种子均值±标准差（%）", "SPL：训练种子均值±标准差（×100）")):
            ax.set(title=title, xlabel="潜空间维数", ylim=(-5, 105)); ax.grid(alpha=.15); ax.legend()
        fig.suptitle(f"可靠性实验｜{len(cases)}组新随机节点对｜真实图只用于离线诊断\n"
                     "上排：种子0；下排：相同buffer的3个训练种子（误差棒不是置信区间）")
        fig.savefig(cfg.RELIABILITY_FIGURE, dpi=180); plt.close(fig)
        return rows
