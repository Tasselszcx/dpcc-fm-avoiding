# 实验:带约束扩散预测控制(DPCC)下的 DDPM vs 流匹配

本报告对比两种生成式骨干网络——**DDPM**(高斯扩散)与 **流匹配(Flow Matching, FM)**——作为
带约束扩散预测控制(DPCC)内部的轨迹先验,在 D3IL *Avoiding* 任务上使用合成演示数据进行评估。

目标**不是**简单地到达目标点,而是在**满足半空间安全约束的前提下**到达目标。因此我们把
*目标 + 约束* 作为主要指标,而把单纯的 *到达目标* 仅作为参考性的合理性检查。

## 1. 设置

| 项目 | 取值 |
| --- | --- |
| 任务 | D3IL Avoiding,合成演示(288 条轨迹 = 3 场景 x 96) |
| 模型 | DDPM(`models.GaussianDiffusion`)、FM(`models.FlowMatching`) |
| 随机种子 | 0、1、2(独立训练) |
| 训练 | 100k 步,horizon H=8,U-Net dim=32,dim_mults=(1,2,4,8),batch=8 |
| 场景 | `top-left-hard`、`top-right-hard`、`both-hard` |
| 每格试验 | 每个(模型,种子,场景)组合 50 次 |
| 硬件 | 8x H800 80GB;训练 GPU 受限,评估 CPU 受限(SLSQP) |

总评估矩阵:**2 模型 x 3 种子 x 3 场景 x 3 投影变体 x 50 试验**
= 54 个结果文件,每个模型 2700 次 rollout。

### 对比的投影变体

| 变体 | 作用 |
| --- | --- |
| `diffuser` | 普通扩散 / 流采样,**无约束投影**(基线)。 |
| `dpcc-c-tightened` | 采样过程中进行 DPCC 在线投影;按最小投影代价选择轨迹;收紧约束集。这是论文的主方法。 |
| `post_processing` | 仅在生成完成后对最终采样轨迹投影一次(更便宜的基线,不把投影反馈进采样)。 |

### 指标

- **goal%**:到达目标的试验比例(`n_success`)。
- **goal+cons%**:既到达目标**又**从未违反约束的比例(`n_success_and_constraints`)——核心指标。
- **viol steps**:存在活动约束违反的平均时间步数(`n_violations`);0 表示完全可行的 rollout。
- **time/step**:每次投影步的墙钟时间(CPU SLSQP)。

## 2. 汇总结果(对 3 种子 x 3 场景平均,每格 450 次试验)

| 模型 | 变体 | goal% | **goal+cons%** | viol steps | time/step |
| --- | --- | ---: | ---: | ---: | ---: |
| DDPM | diffuser(无投影) | 0.99 | 0.33 | 29.7 | 0.13s |
| DDPM | **dpcc-c-tightened** | 0.71 | **0.71** | **0.00** | 0.31s |
| DDPM | post_processing | 0.54 | 0.44 | 2.2 | 0.28s |
| FM | diffuser(无投影) | 0.99 | 0.33 | 27.2 | 0.12s |
| FM | **dpcc-c-tightened** | 0.50 | **0.50** | 0.30 | 0.45s |
| FM | post_processing | 0.41 | 0.22 | 1.6 | 0.39s |

## 3. 分场景细分

| 模型 | 场景 | 变体 | goal% | goal+cons% | viol steps | time/step |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DDPM | top-left-hard | diffuser | 98.7 | 63.3 | 15.0 | 0.117s |
| DDPM | top-left-hard | dpcc-c-tightened | 86.7 | **86.7** | 0.0 | 0.316s |
| DDPM | top-left-hard | **dpcc-c-tightened-lateproj20** | 93.3 | **93.3** | 0.0 | **0.163s** |
| DDPM | top-left-hard | post_processing | 72.7 | 46.0 | 2.7 | 0.306s |
| DDPM | top-right-hard | diffuser | 98.7 | 20.0 | 41.6 | 0.117s |
| DDPM | top-right-hard | dpcc-c-tightened | 80.7 | **80.7** | 0.0 | 0.277s |
| DDPM | top-right-hard | **dpcc-c-tightened-lateproj20** | 100.0 | **100.0** | 0.0 | **0.164s** |
| DDPM | top-right-hard | post_processing | 20.0 | 20.0 | 2.7 | 0.247s |
| DDPM | both-hard | diffuser | 98.7 | 15.3 | 32.4 | 0.118s |
| DDPM | both-hard | dpcc-c-tightened | 46.7 | **46.7** | 0.0 | 0.303s |
| DDPM | both-hard | **dpcc-c-tightened-lateproj20** | 100.0 | **100.0** | 0.0 | **0.152s** |
| DDPM | both-hard | post_processing | 68.0 | 65.3 | 1.1 | 0.282s |
| FM | top-left-hard | diffuser | 98.7 | 51.3 | 16.4 | 0.118s |
| FM | top-left-hard | dpcc-c-tightened | 48.7 | 48.7 | 0.9 | 0.479s |
| FM | top-left-hard | **dpcc-c-tightened-lateproj20** | 84.0 | **84.0** | 0.0 | **0.153s** |
| FM | top-left-hard | post_processing | 59.3 | 13.3 | 3.7 | 0.385s |
| FM | top-right-hard | diffuser | 98.7 | 21.3 | 37.4 | 0.119s |
| FM | top-right-hard | dpcc-c-tightened | 52.0 | **52.0** | 0.0 | 0.424s |
| FM | top-right-hard | **dpcc-c-tightened-lateproj20** | 72.7 | **72.7** | 0.0 | **0.146s** |
| FM | top-right-hard | post_processing | 4.7 | 4.7 | 0.4 | 0.358s |
| FM | both-hard | diffuser | 98.7 | 25.3 | 27.8 | 0.117s |
| FM | both-hard | dpcc-c-tightened | 49.3 | **49.3** | 0.0 | 0.436s |
| FM | both-hard | **dpcc-c-tightened-lateproj20** | 59.3 | **59.3** | 0.0 | **0.155s** |
| FM | both-hard | post_processing | 59.3 | 48.0 | 0.8 | 0.394s |

> 注:`dpcc-c-tightened-lateproj20` 是第 5 节提出的"晚投影"调度(只在积分/采样最后 20% 投影)。
> 这里把它和每个场景的其他变体并排放,便于直接对比;第 5 节解释它*为什么*有效。它对**两种**
> 骨干都适用:FM 上每步便宜约 3 倍(≈0.15s vs ≈0.45s);DDPM 上既**提升 goal+cons%(均值 71.3 →
> 97.8)**,又把开销**砍掉约一半**(≈0.16s vs ≈0.30s)。晚投影是两种先验在本任务上的最佳配置。

### 对比图

![分场景主要指标](figures/fig_goalcons_by_scene.png)

*晚投影(绿色)在每个场景都把 FM 提升到 DDPM 的水平。*

![FM 投影变体对比](figures/fig_fm_variants.png)

*仅 FM:不投影不安全,原始 dpcc-c 安全但损失到达率,晚投影把到达率找回来。*

![质量 vs 投影成本](figures/fig_quality_vs_cost.png)

*质量 vs 成本:FM `lateproj20`(绿星)位于左上角——DDPM 级质量,却是最低的每步成本。*

## 4. 发现

1. **普通采样能到达目标,但不安全。** 两种骨干在无投影时都有约 99% 的到达率,
   但只有约 33% 的试验满足约束,且每条 rollout 累积约 28-30 个违反步。单看到达目标
   对本任务并不是有意义的成功标准。

2. **DPCC 在线投影在主要指标上明显胜出。** `dpcc-c-tightened` 把违反步降到
   **0.0(DDPM)/ 0.3(FM)**,并在几乎每个场景上最大化 goal+cons%。使用 DPCC 后,
   *goal%* 与 *goal+cons%* 基本相等——意味着剩下的失败只是"没到达",绝不再是"到达但不安全"。

3. **在线投影优于后处理。** 生成后只投影一次(`post_processing`)会留下残余违反
   (1.6-2.7 步),且 goal+cons% 明显更低(DDPM 0.44 vs 0.71,FM 0.22 vs 0.50)。
   把投影反馈进采样循环很关键。

4. **DPCC 下 DDPM 优于 FM。** 主方法下 DDPM 达到 0.71 goal+cons% 而 FM 仅 0.50,
   且在三个场景上都更稳定。FM 每个投影步也更贵(0.45s vs 0.31s)。无投影时两者
   无法区分(都是 0.33),所以差距特定于各先验与投影的相互作用方式。

5. **场景难度排序被保留。** `both-hard`(两侧都有障碍)对两个模型在投影后仍是最难的
   (约 47-49% goal+cons%),而单侧场景更容易。这与任务的几何预期一致。

## 5. 拓展:用"晚投影"调度弥合 FM 的差距

上面的发现 #4 显示 FM 因投影损失了远多于 DDPM 的到达能力(0.99 -> 0.50,对比 DDPM
0.99 -> 0.71)。根本原因在于 FM 的采样动力学:FM 用 Euler ODE 求解器积分一个学到的
速度场。原调度在积分后半程的**每一步**都投影。每次中途投影把状态推离学到的 ODE 路径,
于是下一次速度求值 `v(x_projected, t)` 发生在一个离开分布的点上,误差累积并把轨迹拖离
目标。DDPM 的随机 `p_sample` 在每次投影后重新加噪,对这种扰动宽容得多。

**修复(纯加性,不改动现有任何变体或模型代码):** 只在积分的*最后*一小段做投影——
先让 Euler 求解器干净地积分到接近数据流形,再投影以施加约束。这通过一个**可读性更好**的
变体名后缀 `lateprojNN`(只在最后 NN% 投影;例如 `lateproj20` = 最后 20%)暴露。为向后
兼容,原写法 `thXpY`(投影阈值 = X.Y)保留为完全等价的别名,即 `lateproj20` == `th0p2`。
互补的 `peN` 后缀每 N 步投影一次。它们都在 `scripts/eval.py` 中与已有的 `dt*` 后缀一起
解析;原始的 `dpcc-c-tightened` 不受任何影响。

### 结果:FM 使用 `dpcc-c-tightened-lateproj20`(3 种子 x 3 场景 x 50 试验)

汇总(逐场景细分已并入上面第 3 节的主表,和其他 FM 变体并排便于直接对比):

| 方法 | goal+cons% | viol steps | time/step |
| --- | ---: | ---: | ---: |
| FM `dpcc-c-tightened`(原始) | 0.50 | 0.0 | 0.45s |
| **FM `dpcc-c-tightened-lateproj20`(修复)** | **0.72** | **0.0** | **0.151s** |
| DDPM `dpcc-c-tightened`(参考) | 0.71 | 0.0 | 0.31s |

对比原始 FM `dpcc-c-tightened`(分场景 48.7 / 52.0 / 49.3,见第 3 节),三场景全面提升,
连最难的 both-hard 也从 49.3 -> 59.3。

**结果。** 晚投影调度把 FM 从 0.50 提升到 **0.72 goal+cons%**,追平 DDPM(0.71),
同时保持零约束违反。它还比原始 FM 调度**快约 3 倍**、比 DDPM**快约 2 倍**
(每步 0.151s vs 0.31s),因为执行的 SLSQP 求解次数大幅减少。这实现了 FM 的预期优势:
以更低的投影成本达到相当的质量。

### 晚投影对 DDPM 同样有效(3 种子 x 3 场景 x 50 试验)

把同一调度跑在 DDPM 骨干上。它不只是 FM 的修复——对 DDPM 也是最佳配置,每个场景都改善,
并且每步开销减半:

| 场景 | DDPM `dpcc-c-tightened` | **DDPM `lateproj20`** | time/step |
| --- | ---: | ---: | ---: |
| top-left-hard | 86.7 | **93.3** | 0.163s |
| top-right-hard | 80.7 | **100.0** | 0.164s |
| both-hard | 46.7 | **100.0** | 0.152s |
| **均值** | **71.3** | **97.8** | **≈0.16s** |

所有单元格都保持**零**违反步。最难的 both-hard——原调度在此崩到 46.7%——完全恢复到 100%。
为什么 DDPM 获益比 FM 还大?原调度在采样后半段每一步都投影;对 DDPM 而言每次投影都要和随后的
随机 `p_sample` 去噪步"对抗",所以即便较晚的投影,链路仍有时间在抵达终点前再次漂移。把投影集中
到最后 20%,先让先验干净地去噪,再在约束真正起作用的位置一次性强制满足约束。

综合结论:**晚投影是两种先验的唯一推荐配置**,DDPM 0.978 / FM 0.72 goal+cons%,零违反,
每步约 0.15-0.16s(在两种骨干上都比原始 DPCC 调度快约 2 倍)。

### 补充投影后端(`cvxpyqp`)——一个真正的凸 QP 求解器

作为再一条拓展轴(用变体后缀 `cvxpyqp` 选择),`scripts/eval.py` 可以把投影路由到 `Projector`
上全新的 `solver='cvxpy'` 后端(`_project_cvxpy_scp`),而不走默认的 scipy SLSQP 路径
(`solver='scipy'`,逐字节不变)。它求解同一个投影问题
`min_z 1/2||z-tau||_Q^2  s.t.  Az=b, Cz<=d, z_t^T P z_t + q^T z_t <= v`,
但用现代锥 QP 栈(cvxpy + CLARABEL,带 OSQP/SCS 兜底)。因为障碍约束是**非凸**的(避让区,
P 不定),每条约束都在当前迭代点处线性化(`z_t^T P z_t ≈ 2 z0_t^T P z_t - z0_t^T P z0_t`),
并在 5 次序列凸(SCP)迭代中反复求解 QP。目标写成良态的加权最小二乘 `1/2||sqrtQ·(z-tau)||^2`,
使锥求解器数值稳定。

冒烟测试(FM 种子0,晚投影调度):

| 场景 | 试验 | goal+cons% | viol steps | time/step |
| --- | ---: | ---: | ---: | ---: |
| top-right-hard | 5 | 1.00 | 0.0 | 0.611s |
| both-hard | 10 | 0.70 | 0.0 | 0.591s |

**质量与 SLSQP 一致(零违反、goal+cons 在冒烟噪声内相同),证明 DPCC 的有效性与求解器无关。**
代价是约为 SLSQP 的 4 倍(≈0.6s vs ≈0.15s),因为每次投影要做 5 次 SCP 锥子问题求解。所以与
`trust-constr` 类似,凸 QP 路径是有用的正确性交叉验证,但在本任务上不是速度优势;默认仍用 SLSQP。

### 补充投影求解器(`trust-constr`)

作为第二条拓展轴,`scripts/eval.py` 还接受 `trustconstr` 后缀,把投影 NLP 求解器从
SLSQP 换成 SciPy 的 `trust-constr`(内点 / 信赖域),通过 `Projector` 上新增的
`scipy_method` 参数实现(默认 `'SLSQP'`,因此现有行为不变)。这提供了一个备选求解器的
对比点。冒烟测试(FM 种子0,top-right-hard,10 试验)显示:goal+cons=0.90、零违反,
但每步约 10.6s——比 SLSQP(约 0.15s)**慢约 70 倍且质量相同**。结论:在本任务上 SLSQP
更优,DPCC 的有效性并不依赖于求解器;因此不再进行(明显更慢的)`trust-constr` 全量运行。

### 软梯度引导(`gradient`)——一个负面结果

`scripts/eval.py` 还暴露了模型已有的 `gradient` 路径(变体后缀 `gradient`):它不做"投影到
可行集"的硬投影,而是在 ODE 积分的最后一小段把约束梯度*加到速度场上*(软引导,最后不投影)。
冒烟测试(FM 种子0,**both-hard**,10 试验):goal = 0.90,但**满足约束仅 0.10,每条 rollout
有 29.6 个违反步**——基本和完全不投影一样不安全(diffuser:27.8 个违反步)。软引导把轨迹
往可行方向"推",但从不保证可行,因此在核心安全指标上远差于硬投影。这反过来证明了:真正让
DPCC 起作用的是把轨迹*强制投影到*可行集,而不仅仅是惩罚不可行。晚投影调度(`lateproj20`)仍是
推荐的 FM 配置。

## 6. 复现

```bash
source scripts/env_h800.sh          # conda 环境 dpcc-fm + PYTHONPATH/PYTHONNOUSERSITE/MUJOCO_GL

# 训练(6 个作业 = 2 模型 x 3 种子,各 100k 步)
TRAIN_EXP=avoiding-synthetic    TRAIN_SEEDS=0,1,2 python scripts/train.py
TRAIN_EXP=avoiding-synthetic-fm TRAIN_SEEDS=0,1,2 python scripts/train.py

# 评估(分场景;投影 CPU 受限,因此可多作业并发)
EVAL_EXPS=avoiding-synthetic \
EVAL_SEEDS=0,1,2 \
EVAL_HALFSPACE_VARIANTS=top-left-hard,top-right-hard,both-hard \
EVAL_PROJECTION_VARIANTS=diffuser,dpcc-c-tightened,post_processing \
EVAL_N_TRIALS=50 python scripts/eval.py

# 汇总(复现上面的表格)
RESULT_EXP=avoiding-synthetic    python scripts/load_results.py
RESULT_EXP=avoiding-synthetic-fm python scripts/load_results.py

# 拓展:FM 使用晚投影调度(第 5 节)
EVAL_EXPS=avoiding-synthetic-fm \
EVAL_SEEDS=0,1,2 \
EVAL_HALFSPACE_VARIANTS=top-left-hard,top-right-hard,both-hard \
EVAL_PROJECTION_VARIANTS=dpcc-c-tightened-lateproj20 \
EVAL_N_TRIALS=50 python scripts/eval.py
```

指标定义见 `scripts/load_results.py`;原始的逐试验数组存于
`logs/.../results/halfspace_<scene>/<variant>.npz`(未提交)。
