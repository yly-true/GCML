"""测试入口：组合推理、数学检查、实验与绘图，不在此训练。"""

import numpy as np
from . import config as cfg
from .navigation import Navigator
from .mathematics import MathTests
from .experiments import Experiments
from .visualization import Plots


class Tester(Navigator, MathTests, Experiments, Plots):
    """各部分共享self.m，保留原Tester的调用方式。"""

    def run(self):
        self.math_checks()
        self.imagination_figure()
        self.decision_figure()
        if cfg.MATH_ONLY:
            return
        frozen, metrics, results = {name: self.m[name].copy() for name in "QVWG"}, self.metrics(), []
        self.cognition()
        for i in range(cfg.TEST_CASES):
            case = self.case(cfg.SEED+i)
            route, success, trace = self.gcml(case, seed=cfg.SEED+400+i)
            cml_route, cml_success = self.cml(case)
            result = dict(**case, gcml_route=route, gcml_success=success, trace=trace,
                          cml_route=cml_route, cml_success=cml_success)
            if any(not self.m["adjacency"][a, b] for path in (route, cml_route) for a, b in zip(path[:-1], path[1:])):
                raise AssertionError("执行了无效动作")
            results.append(result)
            print(f"case {i+1:02d}: GCML={success}({len(route)-1}), CML={cml_success}({len(cml_route)-1}), "
                  f"rollout_reached={''.join('T' if x[0] else 'F' for x in trace) or '-'}")
        if any(not np.array_equal(self.m[name], frozen[name]) for name in "QVWG"):
            raise AssertionError("推理阶段修改了Q/V/W/G")
        scores = self.benchmark()
        self.ablations()
        self.reliability()
        if any(not np.array_equal(self.m[name], frozen[name]) for name in "QVWG"):
            raise AssertionError("完整评测修改了Q/V/W/G")
        self.figures(results, scores)
        print(f"buffer coverage: nodes={metrics['node_coverage']:.1%}, actions={metrics['action_coverage']:.1%}")
        print(f"transition_RMSE={metrics['transition']:.6f}, G_RMSE={metrics['affordance']:.6f}")
        print(f"benchmark: GCML@{cfg.ROLLOUTS}={scores['count']['GCML'][-1]:.1f}%, CML={scores['cml']:.1f}%")
        print("saved:", cfg.CASE_FIGURE, cfg.REPORT_FIGURE)
