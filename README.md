# Sparse Landmark GCML

这是一个轻量的连续导航实验：不再把大地图中的每个网格都定义成状态，也不再使用 FourRooms。程序自己生成一张 `100 x 100` 的连续二维地图，只在关键通道和开阔区域放置少量地标；GCML 在地标图上训练 `Q/V/W`，最后用随机 B 样条 rollout 把地标折线变成可执行的平滑曲线。

默认实验中，10,000 个地图单元只产生约 90 个地标和约 400 个有向动作。代码只依赖 NumPy、SciPy 和 Matplotlib，不需要 TensorFlow。

## 运行

```bash
conda activate tf215_gpu
pip install -r requirements.txt
python train.py
```

随机起终点，并且随机堵住**一个**通道：

```bash
python train.py --random-task --block-passage --seed 3 --output landmark_gcml_blocked.png
```

常用参数：

```text
--width / --height       地图尺寸，最小为 40
--coverage-radius        地标覆盖半径；越大，地标越少
--max-landmarks          地标数量上限
--latent-dim             GCML 状态维数 d
--epochs                 Q/V/W 训练轮数
--graph-rollouts         高层随机 rollout 数量
--spline-rollouts        连续曲线候选数量
--random-task            随机起点和终点
--block-passage          随机且只封堵一个通道
```

运行后会保存导航图。默认示例为 `landmark_gcml.png`。

生成 10 组随机检查图；每一组独立随机起终点，并且恰好堵住一个通道：

```bash
python check.py
```

输出为 `check_10_blocked.png`。检查脚本复用同一套实现，只把训练轮数和候选数调低以缩短批量验证时间。

## 1. 为什么使用稀疏地标

若地图包含 `N` 个离散状态和 `A` 个离散动作，原始表格式模型需要

\[
Q\in\mathbb R^{d\times N},\qquad
V\in\mathbb R^{d\times A},\qquad
W\in\mathbb R^{A\times d}.
\]

地图变大后，`N` 和逐点动作数会快速增加；规划结果也天然是一段段直线。

本实现只选择 `M` 个地标，且 `M\ll N`。通道中心被保留为必要地标，房间内部使用最远点采样（farthest-point sampling）补足覆盖。只有视线无碰撞、距离足够近的地标才连边。若每个地标至多连接 `k` 个邻居，则有向边数 `E=O(kM)`，模型变为

\[
Q\in\mathbb R^{d\times M},\qquad
V\in\mathbb R^{d\times E},\qquad
W\in\mathbb R^{E\times d}.
\]

因此模型规模主要由地图的拓扑复杂度决定，而不是由地图面积或网格精度直接决定。起点和终点是连续坐标，只在规划时临时连接到可见地标，并不会扩大 `Q/V/W`。

## 2. 状态、动作与观测

- 连续物理状态：位置 \(p=(x,y)\in\mathbb R^2\)。它用于碰撞检测和曲线执行。
- 高层状态：当前地标 \(i\)，其 one-hot 观测为 \(o_i\in\mathbb R^M\)。
- GCML 状态：

\[
s_i=Qo_i=Q_{:i}\in\mathbb R^d.
\]

- 高层动作：沿一条可见图边 \(a=(i\rightarrow j)\) 前往相邻地标。它不是“每个像素走一步”，而是一段局部可达运动。
- 低层动作：B 样条上的连续位置/速度指令；本示例生成轨迹，不绑定具体机器人动力学。

这里即使物理位置只有二维，GCML 观测也不是直接把 `(x,y)` 送入一个二维点积。`Q` 学到的是稀疏拓扑图上的 `d` 维状态表示，所以隔墙但欧氏距离很近的两个位置可以在表示中相距很远。

## 3. Q/V/W 的训练

每条有向地标边给出一个转移样本 \((i,a,j)\)。按照 GCML 的线性状态转移假设：

\[
s_i=Qo_i,
\qquad
\hat s_j=s_i+Va,
\]

其中 \(a\in\mathbb R^E\) 是动作 one-hot。转移误差为

\[
\delta_{iaj}=Q_{:j}-Q_{:i}-V_{:a}.
\]

程序按论文公式 (12) 和 (13) 对每条边执行局部更新：

\[
V_{:a}\leftarrow V_{:a}+\eta_V\delta_{iaj},
\]

\[
Q_{:i}\leftarrow Q_{:i}-\eta_Q\delta_{iaj},\qquad
Q_{:j}\text{ 不在该样本中更新}.
\]

也就是把“预测下一状态减真实下一状态”写入当前观测列。每轮之后对 `Q` 做中心化和白化，防止所有状态坍缩到同一点。`W` 按论文公式 (14) 进行 Hebbian 学习；由于 \(a\) 是 one-hot，只有当前动作行被更新：

\[
\Delta s_{ij}=Q_{:j}-Q_{:i},
\]

\[
W_{a:}\leftarrow W_{a:}
+\eta_W\Delta s_{ij}^{\mathsf T}.
\]

所以 `Q`、`V` 和 `W` 都会被实际更新。训练结束时程序打印三者相对初始化的变化量，以及

\[
\operatorname{RMSE}
=\sqrt{\frac1E\sum_{(i,a,j)}
\|Q_{:j}-Q_{:i}-V_{:a}\|^2}.
\]

## 4. 高层 stochastic rollout

目标地标为 \(g\)，当前地标为 \(i\)。核心驱动仍然是

\[
u_i=W(s_g-s_i),
\]

程序先把完整 utility 向量归一化，再用当前地标真实出边作为 affordance 掩码。这里不额外训练 `G`；稀疏图已经精确给出了哪些动作当前可执行。对可行动作加入高斯噪声：

\[
e_{i,a}=\hat g_{i,a}\left(u_{i,a}+\epsilon_a\right),
\qquad
\epsilon_a\sim\mathcal N(0,\sigma^2).
\]

然后按照论文公式 (19) 进行 winner-take-all：

\[
a_t=\operatorname{WTA}(e_i).
\]

想象状态不会在每一步重新读取地标观测，而是严格使用论文公式 (20) 自举：

\[
\hat s_{t+1}=\hat s_t+Va_t.
\]

一次规划采样多条候选。到达目标的轨迹优先，再按几何长度评分；未到达者还会受到剩余图距离惩罚。轨迹中的回环会被消除。如果所有随机候选都失败，才使用图最短路作为安全兜底，并在输出中明确打印 `shortest-path fallback used`。

## 5. 从地标到平滑曲线

高层只负责决定“依次经过哪些拓扑区域”。对地标序列加密为控制点 \(c_i\)，构造三次 B 样条：

\[
p(t)=\sum_i B_{i,3}(t)c_i,\qquad t\in[0,1].
\]

程序随机扰动位于开阔区的内部控制点，并改变平滑量，生成多条曲线候选。每条曲线使用地图的距离场计算

\[
J=10^5J_{\text{collision}}
+L(p)+0.35J_{\text{curvature}}
+2J_{\text{wall}}.
\]

其中 `collision` 是安全距离违反量，\(L(p)\) 是曲线长度，`curvature` 抑制急转弯，`wall` 偏好更大的墙面间距。评分最低的无碰撞曲线被选中。这样，拓扑搜索仍由 Q/V/W 完成，而平滑性不会迫使状态或动作表随轨迹分辨率膨胀。

## 6. 代码结构

- `navigation_map.py`：自建连续地图、距离场、碰撞检测、随机封堵一个通道。
- `landmarks.py`：自动地标选择、稀疏可视图、连续起终点连接。
- `gcml.py`：Q/V/W 训练和高层 stochastic rollout。
- `spline_planner.py`：B 样条候选生成、碰撞/长度/曲率评分。
- `train.py`：训练、规划、日志和可视化入口。
- `check.py`：一次生成 10 组“随机起终点 + 单通道封堵”的检查图。

默认 `100 x 100` 实验约使用 2.2 万个 Q/V/W 参数，而不是为 10,000 个网格状态和所有逐点动作建表。调大地图时可以先增大 `--coverage-radius` 保持地标稀疏；只有地图拓扑变复杂时再提高 `--max-landmarks`。
