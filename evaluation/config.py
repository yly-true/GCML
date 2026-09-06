"""测试参数集中放在这里；不使用命令行参数。"""

from pathlib import Path

# ======================== 全部推理参数 ========================
OUTPUT_DIR = Path("outputs")
MODEL_DIR, FIGURE_DIR, DATA_DIR = (OUTPUT_DIR/name for name in ("models", "figures", "data"))
MODEL_FILE = MODEL_DIR/"gcml_model.npz"
SEED, TEST_CASES = 100, 30   # 可视化案例数量；布局随数量自动调整
ROLLOUTS, ROLLOUT_HORIZON, EXECUTION_HORIZON = 100, 32, 64
LATENT_GOAL_TOLERANCE, NOISE = .8, .18
BENCHMARK_CASES = 30
ROLLOUT_BUDGETS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100)
HORIZON_BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
CASE_COLUMNS = 5
CASE_FIGURE, REPORT_FIGURE = FIGURE_DIR/"02_navigation_cases.png", FIGURE_DIR/"03_rollout_effects.png"
ROLLOUT_DATA = DATA_DIR/"03_rollout_effects.json"
MATRIX_FIGURE = FIGURE_DIR/"04_matrix_analysis.png"
NOISE_LEVELS = (0., .05, .10, .18, .30, .50)
GOAL_TOLERANCES = (.05, .2, .5, .8, 1.2, 2.)
ABLATION_FIGURE = FIGURE_DIR/"05_ablation.png"
RELIABILITY_CASES, RELIABILITY_SEED = 30, 51000
RELATIVE_TOLERANCE = .2
RELIABILITY_NOISES = (.10, .18)
MATH_ONLY = False  # True只跑快速数学/边界自检；False继续原有完整实验。
MATH_FIGURE = FIGURE_DIR/"07_math_checks.png"
RELIABILITY_FIGURE = FIGURE_DIR/"06_reliability.png"
IMAGINATION_FIGURE, DECISION_FIGURE = FIGURE_DIR/"08_imagination.png", FIGURE_DIR/"09_decision.png"
MATH_DATA, MATRIX_DATA = DATA_DIR/"07_math_checks.json", DATA_DIR/"04_matrix_analysis.json"
ABLATION_DATA, RELIABILITY_DATA = DATA_DIR/"05_ablation.json", DATA_DIR/"06_reliability.json"
IMAGINATION_DATA, DECISION_DATA = DATA_DIR/"08_imagination.json", DATA_DIR/"09_decision.json"
IMAGINATION_CASES, SHOWN_ROLLOUTS = 3, 12
CHINESE_STYLE = {"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False}
# =============================================================
