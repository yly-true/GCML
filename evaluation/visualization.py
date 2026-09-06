"""导航与想象轨迹可视化。"""

import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from . import config as cfg


class Plots:
    def decision_figure(self):
        """解释真实记录的最佳候选前两步；不是重新计算一套解释用策略。"""
        m, detail = self.m, {"explain": True}
        start, goal = map(int, np.random.default_rng(53000).choice(len(m["points"]), 2, replace=False))
        action, _ = self.imagine(m, start, goal, m["available"][start], cfg.ROLLOUTS, cfg.ROLLOUT_HORIZON,
                                 np.random.default_rng(54000), details=detail)
        best, paths = detail["best"], detail["paths"]
        rank = sorted(range(len(paths)), key=lambda i: (not detail["arrived"][i],
                      len(paths[i]) if detail["arrived"][i] else detail["residual"][i], i))
        with plt.rc_context(cfg.CHINESE_STYLE):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
            def table(ax, title, headers, rows):
                ax.axis("off"); ax.set_title(title, fontsize=13, pad=15)
                cells = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
                cells.auto_set_font_size(False); cells.set_fontsize(10); cells.scale(1, 1.9)
                for (r, c), cell in cells.get_celld().items():
                    cell.set_edgecolor("white")
                    cell.set_facecolor("#dfeaf5" if r == 0 else "#f0f5f9")
            for step, ax in enumerate(axes[0]):
                if step >= len(detail.get("steps", [])):
                    ax.axis("off"); ax.text(.1, .5, "候选已在首步想象到达，无第二步。")
                    continue
                values, chosen = detail["steps"][step], paths[best][step]
                ids = m["available"][start] if step == 0 else np.argsort(values["score"])[-6:][::-1]
                rows = [[f"{'★' if a == chosen else ''}{a}: {m['edges'][a, 0]}→{m['edges'][a, 1]}",
                         *[f"{values[k][a]:.4f}" for k in ("utility", "noise", "gate", "score")]] for a in ids]
                table(ax, "① 首步：仅显示真实可行动作" if step == 0 else "② 第二步：显示得分最高的6个动作",
                      ["动作编号：源→终", "W归一化效用", "噪声ε", "门控g", "最终得分"], rows)
            table(axes[1, 0], "③ 为什么选这条候选？（完整候选排名前6）",
                  ["候选编号", "想象到达", "想象步数", "终点残差", "首动作"],
                  [[f"{'★' if i == best else ''}{i}", "是" if detail["arrived"][i] else "否", len(paths[i]),
                    f"{detail['residual'][i]:.5f}", paths[i][0]] for i in rank[:6]])
            ax = axes[1, 1]; ax.axis("off")
            ax.text(.02, .95, "④ 从矩阵到真实执行", fontsize=17, weight="bold", va="top", color="#237b58")
            ax.text(.02, .81, "Q：把当前节点与目标节点编码成高维向量。\n"
                    "W：目标方向 → 动作效用，再归一化。\n"
                    "G：第二想象步起预测门控；首步用真实可行性。\n"
                    "动作得分 = 门控 ×（归一化效用 + 噪声）。\n"
                    "V：选出动作后，用其列向量推进想象状态。\n\n"
                    "候选排序：到达优先 → 到达者选短的；\n"
                    "未到达者选终点残差小的；同分取先生成者。\n\n"
                    f"本次选择候选 {best}，仅执行首动作 {action}。\n"
                    "★为被选项；首步非法动作被硬屏蔽，不在表中。\n"
                    "表中数值有四舍五入；G不是概率，也未截断。\n"
                    "这是计算过程解释，不证明想象路径真实有效。",
                    va="top", fontsize=11, linespacing=1.7)
            fig.suptitle(f"GCML 为什么这样选？｜节点 {start} → {goal}｜真实推理数值", fontsize=19, weight="bold")
            fig.savefig(cfg.DECISION_FIGURE, dpi=170); plt.close(fig)
        # 非法首动作得分为负无穷，JSON用null表示屏蔽，避免非标准Infinity。
        for step in detail.get("steps", []):
            step["score"] = [x if np.isfinite(x) else None for x in step["score"]]
        cfg.DECISION_DATA.write_text(json.dumps(dict(start=start, goal=goal, **detail), indent=2), encoding="utf-8")
        return detail

    def imagination_figure(self):
        """复用真实rollout；地图只画有效前缀，PCA仅用于显示而非规划。"""
        m, rng, records = self.m, np.random.default_rng(53000), []
        q, v, points = m["Q"], m["V"], m["points"]
        centre = q.mean(1, keepdims=True)
        basis = np.linalg.svd(q-centre, full_matrices=False)[0][:, :2]
        project = lambda state: (basis.T@(state-centre)).T
        with plt.rc_context(cfg.CHINESE_STYLE):
            fig, axes = plt.subplots(cfg.IMAGINATION_CASES, 3, figsize=(16, 4.5*cfg.IMAGINATION_CASES),
                                     squeeze=False, layout="constrained")
            for row, (map_ax, latent_ax, error_ax) in enumerate(axes):
                start, goal = map(int, rng.choice(q.shape[1], 2, replace=False))
                detail, audit = {}, []
                action, _ = self.imagine(m, start, goal, m["available"][start], cfg.ROLLOUTS, cfg.ROLLOUT_HORIZON,
                    np.random.default_rng(54000+row), audit=audit, details=detail)
                shown = list(dict.fromkeys([*range(min(cfg.SHOWN_ROLLOUTS, cfg.ROLLOUTS)), detail["best"]]))
                map_ax.imshow(m["grid"], origin="lower", cmap=ListedColormap(["#ffffff", "#8495a7"]))
                for i, j in np.argwhere(np.triu(m["adjacency"], 1)):
                    map_ax.plot(*points[[i, j]].T, color="#dce3e9", lw=.6)
                latent_ax.scatter(*project(q).T, s=8, color="#b5bec8", label="真实节点的投影")
                for index in sorted(shown, key=lambda i: i == detail["best"]):
                    path, best = detail["paths"][index], index == detail["best"]
                    color, width, alpha = ("#e56a19", 2.5, 1.) if best else ("#438fba", 1., .3)
                    label = "被选候选" if best else ("其他候选" if index == next(i for i in shown if i != detail["best"]) else None)
                    states = q[:, start, None]+np.column_stack((np.zeros(len(q)), np.cumsum(v[:, path], axis=1)))
                    latent_ax.plot(*project(states).T, color=color, lw=width, alpha=alpha, label=label)
                    error_ax.plot(np.linalg.norm(states-q[:, goal, None], axis=0), color=color, lw=width,
                                  alpha=alpha, label=label)
                    route = [start]
                    for a in path:
                        source, target = m["edges"][a]
                        if source != route[-1]:
                            map_ax.scatter(*points[route[-1]], marker="x", color="#c62828", s=45, zorder=5)
                            break
                        route.append(int(target))
                    map_ax.plot(*points[route].T, color=color, lw=width, alpha=alpha)
                for ax, locations in ((map_ax, points), (latent_ax, project(q))):
                    ax.scatter(*locations[start], s=50, color="#182e4a", zorder=6, label="起点")
                    ax.scatter(*locations[goal], s=110, marker="*", color="#e9ad16", zorder=6, label="目标")
                map_ax.annotate("", xy=points[m["edges"][action, 1]], xytext=points[start],
                                arrowprops=dict(arrowstyle="->", color="#21855b", lw=3), zorder=7)
                map_ax.set(title=f"案例{row+1}：节点{start} → {goal}｜可执行前缀",
                           xlabel="红叉：下一动作非法，停止画线；绿箭头：将执行的首步", xticks=[], yticks=[])
                latent_ax.set(title="高维想象的二维PCA投影（不是实际地图）", xlabel="第一主成分", ylabel="第二主成分")
                error_ax.axhline(cfg.LATENT_GOAL_TOLERANCE, ls="--", color="#c62828", label="到达阈值")
                error_ax.set(title=f"全部{cfg.ROLLOUTS}条：想象到达{audit[0]['latent_hits']}，真实有效到达{audit[0]['actual_hits']}",
                             xlabel="想象步数（不是实际执行步数）", ylabel="到目标的高维距离")
                latent_ax.legend(fontsize=8); error_ax.legend(fontsize=8); error_ax.grid(alpha=.15)
                records.append(dict(start=start, goal=goal, shown=shown, first_action=action, audit=audit[0], **detail))
            fig.suptitle(f"想象路线是什么样？｜每次生成{cfg.ROLLOUTS}条，展示前{cfg.SHOWN_ROLLOUTS}条及最佳候选\n"
                         "每行是一次独立决策；橙线只是被选想象，不会整条执行；二维重叠不代表高维到达", fontsize=15)
            fig.savefig(cfg.IMAGINATION_FIGURE, dpi=170); plt.close(fig)
        cfg.IMAGINATION_DATA.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return records

    def draw_case(self, ax, result, index):
        m, points, adjacency = self.m, self.m["points"], self.m["adjacency"]
        ax.imshow(m["grid"], origin="lower", cmap=ListedColormap(["#ffffff", "#8495a7"]),
                  interpolation="nearest", vmin=0, vmax=1)
        for i, j in np.argwhere(np.triu(adjacency, 1)):
            ax.plot(points[[i, j], 0], points[[i, j], 1], color="#d1dbe3", lw=.55)
        p = len(m["passages"])
        if len(m.get("blocked", [])):
            ax.scatter(*m["blocked"][:, :2].T, marker="s", color="#bd4e56", s=30, zorder=3)
        ax.scatter(*points[p:].T, s=7, color="#a3b0bc")
        ax.scatter(*points[:p].T, s=14, marker="s", color="#8a79ae")

        def physical(route, success):
            path = np.vstack(([result["start"]], points[route]))
            return np.vstack((path, result["goal"])) if success else path

        ax.plot(*physical(result["gcml_route"], result["gcml_success"]).T, color="#ef6c00", lw=2, label="GCML")
        decisions = points[result["gcml_route"][:-1]]
        reached = np.asarray([hit for hit, *_ in result["trace"]], bool)
        if len(decisions):
            ax.scatter(*decisions[reached].T, facecolors="none", edgecolors="#2e7d32", s=35, lw=1.5,
                       zorder=4, label="想象到达")
            ax.scatter(*decisions[~reached].T, color="#c62828", marker="x", s=35, lw=1.5,
                       zorder=4, label="想象未到达")
        cml, seen = physical(result["cml_route"], result["cml_success"]), set()
        for a, b in zip(cml[:-1], cml[1:]):
            key = tuple(sorted((tuple(a), tuple(b))))
            if key not in seen:
                ax.plot(*np.vstack((a, b)).T, color="#1565c0", ls=(0, (3, 3)), lw=1.4,
                        label="CML逐步执行" if not seen else None)
                seen.add(key)
        ax.scatter(*result["start"], color="#172b4d", edgecolors="white", s=45, zorder=6)
        ax.scatter(*result["goal"], color="#f9a825", edgecolors="#6b4e00", linewidths=.5,
                   marker="*", s=100, zorder=6)
        for name, color in (("gcml", "#ef6c00"), ("cml", "#1565c0")):
            if not result[name+"_success"]:
                ax.scatter(*points[result[name+"_route"][-1]], marker="X", color=color,
                           edgecolors="white", linewidths=.7, s=65, zorder=5)
        status = lambda name: "成功" if result[name+"_success"] else "失败"
        ax.set_title(f"{index:02d}   GCML {status('gcml')}  |  CML {status('cml')}",
                     loc="left", fontsize=10, fontweight="bold", pad=9)
        ax.set_xlabel(f"实际步数 G/C：{len(result['gcml_route'])-1}/{len(result['cml_route'])-1}"
                      f"    被选想象到达：{reached.sum()}/{len(reached)}", fontsize=8, labelpad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

    def rollout_figure(self, scores):
        """只画rollout自身的两个变量，避免与其他消融图重复。"""
        with plt.rc_context(cfg.CHINESE_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", facecolor="#f3f6fa")
            fig.suptitle("GCML 想象参数的影响", fontsize=21, weight="bold")
            for name, rate in scores["count"].items():
                axes[0].plot(cfg.ROLLOUT_BUDGETS, rate, "o-", label=name)
            axes[0].plot(cfg.ROLLOUT_BUDGETS, scores["count_reach"], "o--", label="被选想象到达率")
            axes[0].scatter([1], [scores["cml"]], marker="D", s=65, label="CML一步基线")
            axes[0].set(xscale="log", xticks=cfg.ROLLOUT_BUDGETS, xlabel="候选轨迹数量",
                        ylabel="比例（%）", title="候选数量")
            axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
            axes[1].plot(cfg.HORIZON_BUDGETS, scores["horizon"], "o-", label="真实导航成功率")
            axes[1].plot(cfg.HORIZON_BUDGETS, scores["horizon_reach"], "o--", label="被选想象到达率")
            axes[1].set(xlabel="想象轨迹最大长度", ylabel="比例（%）", title="想象长度")
            for ax in axes:
                ax.set_ylim(-2, 102); ax.set_facecolor("white")
                ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.2); ax.legend()
            fig.savefig(cfg.REPORT_FIGURE, dpi=180); plt.close(fig)

    def figures(self, results, scores):
        # 共享图例放在地图外；多余子图隐藏，避免增加case后被zip静默丢掉。
        plt.rcParams.update({**cfg.CHINESE_STYLE, "font.size": 10,
                             "axes.labelcolor": "#334155", "text.color": "#172b4d",
                             "axes.prop_cycle": plt.cycler(color=["#e87924", "#2878b5", "#279578"])})
        rows = (len(results)+cfg.CASE_COLUMNS-1)//cfg.CASE_COLUMNS
        fig, axes = plt.subplots(rows, cfg.CASE_COLUMNS, figsize=(3.6*cfg.CASE_COLUMNS, 3.9*rows+1.5),
                                 squeeze=False, facecolor="#f3f6fa")
        for i, (ax, result) in enumerate(zip(axes.flat, results), 1):
            self.draw_case(ax, result, i)
        for ax in axes.flat[len(results):]:
            ax.set_visible(False)
        counts = [sum(r[name+"_success"] for r in results) for name in ("gcml", "cml")]
        fig.suptitle("GCML 导航案例", x=.035, y=.987, ha="left", fontsize=23, weight="bold")
        fig.text(.035, .962, f"{len(results)}组随机任务｜GCML {counts[0]}/{len(results)}｜"
                 f"CML {counts[1]}/{len(results)}｜潜维数={self.m['Q'].shape[0]}｜"
                 f"候选数={cfg.ROLLOUTS}｜想象长度={cfg.ROLLOUT_HORIZON}", fontsize=11)
        handles = [Line2D([], [], color=c, linestyle=ls, label=label, lw=2)
                   for label, c, ls in (("GCML", "#ef6c00", "-"), ("CML", "#1565c0", "--"))]
        handles += [Line2D([], [], color=c, marker=marker, linestyle="none", label=label,
                          markerfacecolor="none" if marker == "o" and label == "想象到达" else c)
                    for label, c, marker in (("起点", "#172b4d", "o"), ("终点", "#f9a825", "*"),
                                            ("开放通道", "#8a79ae", "s"), ("封闭通道", "#bd4e56", "s"), ("想象到达", "#2e7d32", "o"),
                                            ("想象未到达", "#c62828", "x"), ("失败停点", "#64748b", "X"))]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.03, .953), ncol=9,
                   frameon=False, fontsize=9)
        fig.subplots_adjust(left=.03, right=.985, top=.917, bottom=.025, hspace=.30, wspace=.15)
        fig.savefig(cfg.CASE_FIGURE, dpi=180); plt.close(fig)

        # 同一case的真实步数与失败标记；失败步数不当作成功效率。
        self.rollout_figure(scores)
