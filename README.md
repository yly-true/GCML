# GCML + MiniGrid FourRooms

这是一个把 2026 年 GCML（Generative Cognitive Map Learner）通用机制应用到
官方 `MiniGrid-FourRooms-v0` 地图的紧凑研究框架。模型通过随机探索学习认知地图，
随后仅在隐空间中生成到目标的 stochastic rollout，再把想象出的动作序列放回
FourRooms 转移图执行。

当前版本用于验证一个核心问题：`Q` 和 `V` 是否能学出满足动作位移关系的
FourRooms 认知地图。它不是论文全部实验的逐行复现。

## 安装与运行

```bash
pip install -r requirements.txt
python train.py
```

运行结束后会生成 `fourrooms_gcml.png`。一次固定种子运行的典型输出为：

```text
states: 260
start : (3, 15)
goal  : (13, 12)
transition RMSE: 0.1334
learned G accuracy (diagnostic only): 87.69%
planning affordance: official FourRooms wall mask
finite parameters: True
imagined action count: 13
execution success: True
```

## 为什么需要 FourRooms wrapper？

MiniGrid 原生动作空间包含：

```text
left / right / forward / pickup / drop / toggle / done
```

其中 `forward` 的空间效果依赖智能体朝向。同一个动作可能向上、下、左或右移动，
不符合当前 GCML 线性 forward model 的固定 action displacement 假设：

$$
\hat{\mathbf s}_{t+1}\approx
\mathbf s_t+\mathbf V\mathbf a_t.
$$

因此 `FourRoomsDiscrete` 保留官方 MiniGrid 生成的地图、墙体、门洞、起点和目标，
但将控制接口转换为四个绝对动作：

$$
\mathcal A=\{\text{up},\text{down},\text{left},\text{right}\}.
$$

状态和动作均使用 one-hot 表示：

$$
\mathbf o_t\in\{0,1\}^{N},\qquad
\mathbf a_t\in\{0,1\}^{4},
$$

其中 $N$ 是当前地图中的可通行格子数。若动作会撞墙，wrapper 的状态保持不变；
训练随机探索只采样当前可执行的动作。

## 参数与维度

设隐空间维度为 $d$、观察数为 $N$、动作数为 $A=4$：

| 参数 | 维度 | 作用 |
| --- | --- | --- |
| $\mathbf Q$ | $d\times N$ | 把 one-hot 观察编码为隐状态 |
| $\mathbf V$ | $d\times A$ | 把动作编码为隐空间位移 |
| $\mathbf W$ | $A\times d$ | inverse model，把状态差映射为动作 utility |
| $\mathbf G$ | $A\times d$ | 预测当前状态下的动作 affordance |

初始化遵循论文的量级：

$$
\mathbf Q\sim\mathcal N(0,1),\qquad
\mathbf V,\mathbf W,\mathbf G\sim\mathcal N(0,0.1).
$$

默认学习率为：

$$
\eta_q=0.1,\qquad
\eta_v=\eta_w=\eta_g=0.01.
$$

## 1. 状态编码与 forward model

观察首先由 $\mathbf Q$ 编码：

$$
\boxed{\mathbf s_t=\mathbf Q\mathbf o_t}.
$$

给定动作 $\mathbf a_t$，模型预测下一个隐状态：

$$
\boxed{
\hat{\mathbf s}_{t+1}
=\mathbf s_t+\mathbf V\mathbf a_t
}.
$$

真实下一观察的当前隐空间表示为：

$$
\mathbf s_{t+1}=\mathbf Q\mathbf o_{t+1}.
$$

因此 transition prediction error 是：

$$
\boldsymbol\delta_t
=\mathbf s_{t+1}-\hat{\mathbf s}_{t+1}.
$$

## 2. 学习 $Q$ 和 $V$

动作 embedding 使用 delta rule：

$$
\boxed{
\mathbf V\leftarrow\mathbf V
+\eta_v\boldsymbol\delta_t\mathbf a_t^\top
}.
$$

当前实现对 $Q$ 使用式 (11) 对应的 semi-gradient，把
$\mathbf Q\mathbf o_t$ 朝 $\mathbf Q\mathbf o_{t+1}-\mathbf V\mathbf a_t$
移动：

$$
\boxed{
\mathbf Q\leftarrow\mathbf Q
+\eta_q\boldsymbol\delta_t\mathbf o_t^\top
}.
$$

这里不能直接把论文式 (13) 中的
$(\hat{\mathbf s}_{t+1}-\mathbf s_{t+1})\mathbf o_t^\top$
通过 `Q += ...` 写入当前列，否则会成为反梯度，使误差、参数范数和 `G` 依次发散。

训练后的 transition 诊断遍历所有可执行边，计算：

$$
\operatorname{RMSE}_{\text{transition}}
=\sqrt{\frac{1}{|\mathcal E|}
\sum_{(s,a,s')\in\mathcal E}
\left\|
\mathbf Q\mathbf o_{s'}-
(\mathbf Q\mathbf o_s+\mathbf V\mathbf a)
\right\|_2^2}.
$$

## 3. 学习 inverse model $W$

一次真实转移产生局部状态差：

$$
\Delta\mathbf s_t=\mathbf s_{t+1}-\mathbf s_t.
$$

$\mathbf W$ 应从这个状态差恢复导致它的动作：

$$
\hat{\mathbf a}_t=\mathbf W\Delta\mathbf s_t,
\qquad
\mathbf r_t=\mathbf a_t-\hat{\mathbf a}_t.
$$

当前实现使用官方代码采用的 error-corrected inverse learning，并针对逐样本在线
更新使用 normalized LMS 步长：

$$
\boxed{
\mathbf W\leftarrow\mathbf W+
\eta_w
\frac{\mathbf r_t\Delta\mathbf s_t^\top}
{\max(\|\Delta\mathbf s_t\|_2^2,\varepsilon_0)}
}.
$$

归一化只调节有效步长，不改变 $\mathbf W\Delta\mathbf s_t=\mathbf a_t$
这一固定点，同时避免高维隐向量让单步更新爆炸。

## 4. 学习 affordance model $G$

wrapper 可为每个真实状态提供二值可行动作向量：

$$
\mathbf g_t\in\{0,1\}^{4}.
$$

线性 affordance prediction 为：

$$
\hat{\mathbf g}_t=\mathbf G\mathbf s_t.
$$

其 normalized LMS 更新为：

$$
\boxed{
\mathbf G\leftarrow\mathbf G+
\eta_g
\frac{(\mathbf g_t-\mathbf G\mathbf s_t)\mathbf s_t^\top}
{\max(\|\mathbf s_t\|_2^2,\varepsilon_0)}
}.
$$

但是，当前 FourRooms 默认规划不使用学习到的 $G$。原因是这里只有 4 个全局共享
动作，而不连续墙段和门洞对应的 affordance 在近似空间坐标中通常不是线性可分的。
论文的通用图实验为每条有向边分配独立 action，因此不存在完全相同的问题。

本项目仍训练和报告 $G$ 的 accuracy，便于研究后续非线性 affordance、place-cell
features 或其他障碍编码，但默认把官方 FourRooms 地图的二值墙体 mask 作为空间
实验的外部障碍约束。这类似论文空间实验中单独提供的 object/barrier constraint。

## 5. Goal-directed stochastic rollout

给定起点观察 $\mathbf o_0$ 和目标观察 $\mathbf o^*$：

$$
\hat{\mathbf s}_0=\mathbf Q\mathbf o_0,
\qquad
\mathbf s^*=\mathbf Q\mathbf o^*.
$$

每个想象步首先用 inverse model 计算动作 utility：

$$
\boxed{
\mathbf u_t=\mathbf W(\mathbf s^*-\hat{\mathbf s}_t)
}.
$$

按照论文 Methods，在加入噪声前先将非零 utility 归一化：

$$
\bar{\mathbf u}_t=
\frac{\mathbf u_t}{\|\mathbf u_t\|_2}.
$$

采样高斯噪声：

$$
\boldsymbol\epsilon_t\sim
\mathcal N(\mathbf 0,\sigma^2\mathbf I).
$$

用 affordance mask $\mathbf g_t$ 得到 eligibility：

$$
\boxed{
\mathbf e_t=\mathbf g_t\odot
(\bar{\mathbf u}_t+\boldsymbol\epsilon_t)
}.
$$

不可执行动作还会被硬 mask 为 $-\infty$，避免 utility 为负时，零 eligibility 的
撞墙动作反而赢过合法动作。随后执行 winner-take-all：

$$
\boxed{
\mathbf a_t=\operatorname{WTA}(\mathbf e_t)
}.
$$

最后不读取下一时刻环境观察，而是使用 $V$ 进行内部 bootstrapping：

$$
\boxed{
\hat{\mathbf s}_{t+1}
=\hat{\mathbf s}_t+\mathbf V\mathbf a_t
}.
$$

循环持续到规划状态到达目标或超过 `horizon`。改变随机种子或 `noise_std` 会得到
不同但仍受目标方向约束的动作序列。

## 6. 当前有没有 play 过程？

有“生成后执行”，但目前没有交互式或逐帧 MiniGrid GUI play。

当前 `train.py` 包含三个阶段：

1. `train(...)`：在 FourRooms 中随机探索，学习 `Q/V/W/G`。
2. `imagined_path(...)`：从起点到目标进行隐空间 stochastic rollout，产生动作序列。
3. `execute_actions(...)`：从真实起点重新开始，在 wrapper 的真实转移图上依次 replay
   想象动作，检测是否到达目标，并把 executed path 交给 `plot(...)`。

因此当前的 play 是无界面的离散 replay：

```text
random exploration -> imagined rollout -> execute/replay actions -> plot result
```

`fourrooms_gcml.png` 同时绘制：

- `imagined`：每个内部状态通过最近 `Q` embedding 解码得到的路径；
- `executed`：同一动作序列在真实 FourRooms 转移图上的路径；
- `start` 和 `goal`；
- 官方地图的墙体和门洞。

如果需要人眼可见的实时 play，还需要增加一个独立入口，将绝对动作转换为
MiniGrid 原生的转向/前进动作，逐步调用 `env.step(...)`，并通过 `render_mode="human"`
显示动画。这个实时动画层目前尚未实现，也不参与现有训练和成功率验证。

## 代码结构

- `env_wrapper.py`：官方 FourRooms 地图、one-hot 状态/动作、绝对动作转移和墙体 mask。
- `gcml.py`：`Q/V/W/G` 学习、utility、noise、WTA 与 imagined bootstrap。
- `train.py`：随机探索、诊断、rollout、动作 replay 和结果绘图。
- `requirements.txt`：Python 依赖。

## 已验证行为

当前固定地图和训练种子下：

- 所有参数保持有限，不再出现 overflow 或 `NaN`；
- 主程序可以生成成功到达目标的动作序列；
- 对同一个已训练模型测试 50 个 rollout 噪声种子，成功率为 `50/50`；
- 动作序列长度为 13 到 21 步；
- Python 语法检查通过。
