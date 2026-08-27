# Sparse Landmark GCML

这是一个极简的连续导航实验：不再把大地图中的每个网格都定义成状态，也不再使用 FourRooms。程序自己生成一张 `100 x 100` 的连续二维地图，只在关键通道和开阔区域放置少量地标；GCML 在地标图上训练 `Q/V/W`，最后用随机三次样条 rollout 把地标折线变成平滑曲线。

默认实验中，10,000 个地图单元只产生约 90 个地标和约 400 个有向动作。全部规划算法只使用 NumPy，Matplotlib 仅用于保存图片；不需要 SciPy、TensorFlow、Gym 或 MiniGrid。

## 运行

```bash
conda activate tf215_gpu
pip install -r requirements.txt
python train.py
```

不使用任何命令行参数。所有配置都集中在 `train.py` 最顶部。例如随机起终点并且随机堵住**一个**通道：

```python
SEED = 3
RANDOM_TASK = True
BLOCK_ONE_PASSAGE = True
OUTPUT = "landmark_gcml_blocked.png"
```

训练、地图和曲线参数也都在同一区域：

```text
WIDTH / HEIGHT           地图尺寸
LATENT_DIM / EPOCHS      状态维数和训练轮数
ETA_Q / ETA_V / ETA_W    三个矩阵的学习率
COVERAGE_RADIUS          地标覆盖半径；越大，地标越少
MAX_LANDMARKS            地标数量上限
GRAPH_ROLLOUTS           高层随机 rollout 数量
CURVE_ROLLOUTS           连续曲线候选数量
RUN_CHECK                是否直接运行批量检查
```

运行后会保存导航图。默认示例为 `landmark_gcml.png`。

生成检查图时，先在 `train.py` 顶部设置 `CHECK_CASES`、`CHECK_OUTPUT` 等参数，然后运行：

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
- 低层动作：三次样条上的连续位置/速度指令；本示例生成轨迹，不绑定具体机器人动力学。

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

程序先把完整 utility 向量归一化，再用 `Graph.affordance()` 根据当前障碍层生成动作掩码。这里不额外训练 `G`：动作全集保持不变，穿过新障碍的动作临时置为不可执行。对其余动作加入高斯噪声：

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

一次规划采样多条候选。到达目标的轨迹优先，再按几何长度评分；未到达者还会受到剩余图距离惩罚。轨迹中的回环会被消除。如果所有随机候选都失败，才使用图最短路作为安全兜底，并在输出中打印 `fallback=True`。

## 5. 从地标到平滑曲线

高层只负责决定“依次经过哪些拓扑区域”。对地标序列加密为控制点，并用纯 NumPy 计算 Catmull-Rom 三次样条。对相邻控制点 \(p_1,p_2\)，令 \(t\in[0,1]\)：

\[
p(t)=\frac12\left[2p_1+(-p_0+p_2)t
+(2p_0-5p_1+4p_2-p_3)t^2
+(-p_0+3p_1-3p_2+p_3)t^3\right].
\]

程序随机扰动位于开阔区的内部控制点，生成多条曲线候选。每条曲线使用地图的距离场计算

\[
J=10^5J_{\text{collision}}
+L(p)+0.3J_{\text{curvature}}
+2J_{\text{wall}}.
\]

其中 `collision` 是安全距离违反量，\(L(p)\) 是曲线长度，`curvature` 抑制急转弯，`wall` 偏好更大的墙面间距。评分最低的无碰撞曲线被选中。这样，拓扑搜索仍由 Q/V/W 完成，而平滑性不会迫使状态或动作表随轨迹分辨率膨胀。

## 6. 代码结构

- `train.py`：唯一核心文件，包含地图、距离场、地标图、Q/V/W、stochastic rollout、三次样条、评分、绘图和批量检查。
- `check.py`：2 行兼容入口，直接使用 `train.py` 顶部的 `CHECK_CASES` 等配置。

为了去掉 SciPy，距离场使用两遍 chamfer transform，图最短距离使用 NumPy Floyd-Warshall，平滑曲线使用显式三次多项式。它们足以覆盖当前约 100 个地标的小型实验，同时避免额外框架和隐藏逻辑。

默认 `100 x 100` 实验约使用 2.2 万个 Q/V/W 参数，而不是为 10,000 个网格状态和所有逐点动作建表。调大地图时可以先增大 `COVERAGE_RADIUS` 保持地标稀疏；只有地图拓扑变复杂时再提高 `MAX_LANDMARKS`。

## 7. 按执行顺序导读代码

整个项目真正需要阅读的只有 `train.py`。建议不要从第一个类逐行往下背，而是先从文件底部的 `main()` 看调用关系：

```text
main()
├── build_base()：只执行一次
│   ├── World：建立无封堵基础地图
│   ├── Graph：固定地标和动作全集
│   └── GCML.train()：训练一次 Q/V/W
├── RUN_CHECK=True  -> check() -> 多次 run(base, case_seed)
└── RUN_CHECK=False -> run(base)
                      ├── 复制基础障碍层并可选封堵一个通道
                      ├── affordance()：屏蔽失效动作
                      ├── GCML.rollout()：生成高层地标路线
                      └── smooth_rollout()：生成并评分连续曲线
                    -> save_one() -> draw() -> 保存图片
```

### 第一步：顶部参数

文件最上方是唯一配置区。首次阅读时重点关注：

- `WIDTH, HEIGHT`：地图规模。
- `COVERAGE_RADIUS`：地标稀疏程度。它越大，地标越少。
- `RETRY_COVERAGE_RADIUS`：基础地标图本身过稀时使用的备用覆盖半径。
- `LATENT_DIM`：每个地标在 Q 中的状态向量维数。
- `EPOCHS, ETA_Q, ETA_V, ETA_W`：Q/V/W 的训练设置。
- `GRAPH_ROLLOUTS`：采样多少条高层想象路线。
- `CURVE_ROLLOUTS`：对选中的地标路线生成多少条连续曲线。
- `RANDOM_TASK, BLOCK_ONE_PASSAGE, RUN_CHECK`：控制实验模式。

代码中没有命令行参数。修改这里以后运行 `python train.py` 即可。

### 第二步：`World` 生成连续环境

`World.__init__()` 依次调用 `_build()` 和 `_distance_field()`：

1. `_build()` 在布尔数组 `grid` 中生成外墙、内部隔墙、通道和矩形障碍物。
2. `_distance_field()` 用两遍 chamfer transform 计算每个栅格中心到墙面的近似距离。
3. `clearance(points)` 对距离场做双线性插值，因此输入可以是任意连续坐标，而不必落在整数格点。
4. `line_free(a,b)` 沿连续线段采样，用于判断两个位置是否能够无碰撞直达。
5. `block_one()` 只选择一个通道封堵，并马上刷新距离场。

这里的 `grid` 只是地图表示和碰撞查询工具，不是 GCML 的逐格状态表。

### 第三步：`Graph` 把大地图压缩成少量状态

`Graph.__init__()` 在无封堵基础地图上保留全部通道中心，然后用最远点采样覆盖房间内部。正常情况使用 `COVERAGE_RADIUS`；只有基础地标图本身无法连通时，才使用 `RETRY_COVERAGE_RADIUS` 自动补点。测试期间不会再次选择地标。

`_connect()` 只连接满足两个条件的地标：

1. 欧氏距离没有超过连接半径；
2. `line_free()` 确认两点之间没有墙。

这里产生四个核心数组：

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `graph.p` | `(M, 2)` | M 个连续地标坐标 |
| `graph.adj` | `(M, M)` | 地标间是否可直达 |
| `graph.edges` | `(E, 2)` | E 个有向高层动作 `(source,target)` |
| `graph.dist` | `(M, M)` | 地标图上的最短几何距离 |

因此，一个高层动作不是“向上移动一个像素”，而是“沿可见边从地标 i 前往地标 j”。

新障碍出现后，`Graph.affordance(world)` 逐条检查这些固定边，返回：

- `available[i]`：当前地标 i 仍可执行的固定动作编号。
- `active_adj`：当前哪些基础边有效，仅用于检查和绘图。
- `active_distance`：在当前有效边上的最短距离，用于轨迹评分和失败兜底。

它不会修改 `graph.p`、`graph.edges` 或任何 Q/V/W 参数。

### 第四步：`GCML` 初始化 Q/V/W

`GCML.__init__()` 创建：

| 矩阵 | 形状 | 作用 |
| --- | --- | --- |
| `Q` | `(d, M)` | 地标 one-hot 观测到高维状态的映射 |
| `V` | `(d, E)` | 每条动作边在状态空间中的预测位移 |
| `W` | `(E, d)` | 从目标状态差得到各动作 utility |

`Q` 用图拉普拉斯特征向量初始化，使相邻地标一开始就具有基本拓扑结构；这不是冻结的手工地图，后续 `train()` 仍会更新 Q。`_white()` 只负责避免 Q 整体坍缩。

### 第五步：`GCML.train()` 训练三个矩阵

训练循环随机遍历每条有向边 `(i,a,j)`：

```python
delta = Q[:, j] - Q[:, i]
error = delta - V[:, a]
V[:, a] += ETA_V * error
Q[:, i] -= ETA_Q * error
W[a] += ETA_W * delta
```

这三行分别对应论文公式 (12)、(13)、(14)。训练末尾只固定 Q 继续微调 V，使 `V[:,a]` 更接近最终的 `Q[:,j]-Q[:,i]`。日志中的 `Q/V/W changes` 可以直接确认三个矩阵都不是未训练状态。

### 第六步：`build_base()` 与 `run()` 分离训练和测试

`build_base()` 只建立一次无封堵地图、固定地标图和 Q/V/W。普通实验与批量检查都会把返回的 `(base_world, graph, model)` 传给 `run()`。

`run()` 为当前 case 创建相同地图的障碍层，最多调用一次 `block_one()`，再计算动态 affordance。它不会重新建立 Graph，也不会重新训练模型。

接着，起点和终点作为连续坐标接入固定地标图。

起点和终点是连续坐标，不会成为 Q 的新列。`graph.visible()` 分别找出起点、终点附近能够直达的地标，再选择总连接距离最短的一对作为高层 rollout 的开始和结束地标。

如果起点到终点本身可以直达，程序会直接进入连续曲线阶段；Q/V/W 仍然已经完成训练。

### 第七步：`GCML.rollout()` 进行想象

`_sample()` 同时维护两个量：

- `node`：当前处于哪个真实地标，只用于查询该 case 的 `available[node]`。
- `imagined`：当前想象状态，初始化为 `Q[:,start]`，之后不再读取下一观测。

每一步执行：

```python
utility = W @ (target - imagined)
available = dynamic_affordance[node]
a = WTA(utility[available] + noise)
imagined += V[:, a]
node = graph.edges[a, 1]
```

这正是 `W(s*-ŝ)`、可行动作掩码、Gaussian noise、WTA 和 `ŝ←ŝ+Va`。`rollout()` 重复采样多条路线，先选择成功到达目标的候选，再比较长度。只有所有候选都失败时才触发图距离兜底，并输出 `fallback=True`。

### 第八步：`smooth_rollout()` 产生连续曲线

GCML 输出的是少量地标序列。`smooth_rollout()` 将其加密为控制点，然后：

1. 用 Catmull-Rom 三次多项式生成经过控制点的连续曲线；
2. 只扰动远离墙面的内部控制点，生成多个候选；
3. 从距离场读取整条曲线的墙面间距；
4. 按碰撞、长度、曲率和贴墙程度评分；
5. 选择无碰撞且分数最低的曲线。

紫色虚线是 GCML 高层地标路线，橙色实线才是最终连续轨迹。

### 第九步：绘图与批量检查

`draw()` 只负责把已有结果画到 Matplotlib 坐标轴；它不参与规划。所有线段的端点都来自同一组基础动作边：当前有效边画成灰色，当前被新障碍屏蔽的边画成红色虚线。`save_one()` 保存单次实验。

`check()` 的执行顺序是：

1. 在循环外调用一次 `build_base()`。
2. 固定同一组地标、动作编号和 Q/V/W。
3. 对每个 `SEED+i` 随机起终点。
4. 每个 case 只封堵一个通道并更新 affordance。
5. 使用同一个模型完成 rollout 和曲线规划。

因此，10 张图之间只有起点、终点、红色封堵、失效边和最终路线不同；地标位置、基础连边和模型参数完全一致。

## 8. 建议你的阅读和实验顺序

1. 先保持默认参数运行一次，观察日志中的地标数、动作数、Q/V/W 变化和 RMSE。
2. 设置 `RANDOM_TASK=True`，确认起终点可以是任意连续坐标。
3. 再设置 `BLOCK_ONE_PASSAGE=True`，观察地标图和路线如何绕开新障碍。
4. 把 `GRAPH_ROLLOUTS` 暂时降到 20，多换几个 `SEED`，观察 stochastic rollout 的失败概率。
5. 调节 `NOISE`，比较路线多样性；噪声太低容易局部贪心，太高则方向性下降。
6. 调节 `COVERAGE_RADIUS`，观察地标数量、Q/V/W 参数量与规划可靠性的权衡。
7. 最后设置 `RUN_CHECK=True`，一次验证多组任务。

如果要加真实机器人执行层，建议保持当前高层代码不变，只把最终曲线转换成速度或转向控制指令；不要重新把每个曲线采样点扩展成 Q 的离散状态。
