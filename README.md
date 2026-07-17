# DPCC Flow Matching — Avoiding

带**约束的扩散预测控制**（Diffusion Predictive Control with Constraints, **DPCC**）在 D3IL
*Avoiding* 任务上的实验代码库。核心问题：把生成式轨迹先验换成 **Flow Matching (FM)** 后，
能否在**更少的采样步数、更低的投影成本**下，达到与 **DDPM**（高斯扩散）相当的「到达目标 +
满足安全约束」质量。

> 目标**不是**单纯到达目标点，而是在**满足半空间安全约束的前提下**到达。因此主指标是
> *goal + constraints*（到达且全程不违反约束），单纯 *goal reached* 仅作合理性参考。

## 1. 支持的两套骨干 × 两种数据

| 维度 | 取值 |
| --- | --- |
| 生成骨干 | DDPM (`models.GaussianDiffusion`)、FM (`models.FlowMatching`) |
| 数据 | 真实 D3IL 演示（`avoiding-d3il*`，**主**）、合成三场景数据（`avoiding-synthetic*`，快速上手） |
| 场景 | `top-left-hard`、`top-right-hard`、`both-hard` |
| 横轴变量 | horizon `H`、采样/去噪步数 `K`、投影调度、投影求解器 |

config 文件一一对应：`config/avoiding-d3il.py`（DDPM 真实）、`config/avoiding-d3il-fm.py`（FM
真实）、`config/avoiding-synthetic.py`、`config/avoiding-synthetic-fm.py`。

## 2. 环境

```bash
source scripts/env_h800.sh    # 激活 conda dpcc-fm，钉好 PYTHONPATH/PYTHONNOUSERSITE/MUJOCO_GL
```

`env_h800.sh` 做三件关键事（详见脚本注释）：
- `PYTHONNOUSERSITE=1` 隔离 `~/.local` 里的 verl editable 包，避免其顶层 `scripts/` 覆盖本项目；
- `PYTHONPATH=<root>:<root>/d3il`，让 d3il 内部的 `from environments.d3il...` 可导入；
- `MUJOCO_GL=egl` 走无显示渲染。

首次在新机器上从零搭环境见 `setup_wsl_cuda.sh`（CUDA PyTorch + 本地 `diffuser`/`d3il` +
`pinocchio==2.7.0` 走 pip）。集群细节见 `README_H800.md`。

## 3. 训练与评估的环境变量开关

训练和评估都通过**环境变量**参数化，无需改代码即可扫 H / K / 投影变体。

### 训练 `scripts/train.py`

| 变量 | 含义 |
| --- | --- |
| `TRAIN_EXP` | config 名，如 `avoiding-d3il`（DDPM）/ `avoiding-d3il-fm`（FM） |
| `TRAIN_HORIZON` | 规划 horizon H（如 8 / 16） |
| `TRAIN_NDIFF` | 去噪步数 K（仅 DDPM；FM 步数在推理期设定） |
| `TRAIN_SEEDS` | 随机种子，逗号分隔（如 `0,1,2`） |

### 评估 `scripts/eval.py`

| 变量 | 含义 |
| --- | --- |
| `EVAL_EXPS` | config 名 |
| `EVAL_HORIZON` / `EVAL_NDIFF` | 覆盖 H / K，自动加载对应 `H<H>_K<K>_` 权重 |
| `EVAL_FM_NTIMESTEPS` | FM 推理步数（FM 单模型，步数在推理期可调） |
| `EVAL_DDIM_STEPS` | DDPM 上启用 DDIM 确定性子采样（免训练 few-step） |
| `EVAL_HALFSPACE_VARIANTS` | 场景，逗号分隔 |
| `EVAL_PROJECTION_VARIANTS` | 投影变体，逗号分隔（见下） |
| `EVAL_N_TRIALS` / `EVAL_SEEDS` | 每格试验数 / 种子 |
| `EVAL_SAVE_TAG` | 给结果目录追加 `_<tag>` 后缀，隔离不同扫参批次 |
| `EVAL_DEVICE` | `cuda` / `cpu` |

### 投影变体命名（变体名里的子串触发对应行为）

| 变体名子串 | 作用 |
| --- | --- |
| `diffuser` | 普通采样，**无投影**（基线） |
| `dpcc-c-tightened` | DPCC 在线投影 + 收紧约束集（**论文主方法**，默认 SLSQP） |
| `post_processing` | 仅生成后投影一次（更便宜的基线） |
| `lateprojNN` / `thXpY` | 只在积分**最后 NN%**（= 阈值 X.Y）投影；等价别名，如 `lateproj20`==`th0p2` |
| `peN` | 每 N 步投影一次 |
| `gradient` | 软梯度引导（把约束梯度加进速度场，最后不硬投影）——负面对照 |
| `cvxpyqp` | 投影求解器换成凸 QP（cvxpy + CLARABEL，SCP 线性化非凸避让约束）替代默认 scipy SLSQP |

示例——真实数据 FM，H=16，扫 K∈{2,4,8,16}，晚投影调度，3 场景 × 3 种子 × 100 试验：

```bash
source scripts/env_h800.sh
for K in 2 4 8 16; do
  EVAL_EXPS=avoiding-d3il-fm EVAL_HORIZON=16 EVAL_FM_NTIMESTEPS=$K \
  EVAL_HALFSPACE_VARIANTS=top-left-hard,top-right-hard,both-hard \
  EVAL_PROJECTION_VARIANTS=dpcc-c-tightened-lateproj20 \
  EVAL_N_TRIALS=100 EVAL_SEEDS=0,1,2 EVAL_DEVICE=cuda \
  EVAL_SAVE_TAG=d3il_h16_fm_k${K} python scripts/eval.py
done
```

## 4. 指标

结果以逐试验数组存于 `logs/<exp>/.../results/halfspace_<scene>/<variant>.npz`（未提交）：

- **goal%** = `mean(n_success)`：到达目标比例；
- **goal+cons%** = `mean(n_success_and_constraints)`：到达**且**全程满足约束——**核心指标**；
- **viol steps** = `mean(total_violations)`：平均违反步数，0 = 完全可行；
- **time/step** = `mean(avg_time)`：每投影步墙钟时间（CPU 求解器）。

汇总脚本见 `scripts/load_results.py` / `scripts/aggregate_*.py`。

## 5. 仓库结构

```
dpcc-fm-avoiding/
├── config/                     # 4 个实验 config（d3il/synthetic × ddpm/fm）+ projection_eval.yaml
├── diffuser/                   # 本地 diffuser 包（GaussianDiffusion / FlowMatching / Projector）
├── d3il/                       # 本地 D3IL 环境（Avoiding 任务、数据集加载）
├── scripts/
│   ├── env_h800.sh             # 统一环境设置（source 后即可跑）
│   ├── train.py / eval.py      # 训练 / 评估入口（env 变量参数化，见 §3）
│   ├── load_results.py         # 结果汇总（复现报告表格）
│   ├── aggregate_*.py          # n=100 / sweep / threshold 汇总
│   ├── generate_synthetic_data.py, audit_synthetic_data.py   # 合成数据生成与审计
│   ├── visualize_data_constraints.py, plot_*.py              # 可视化
│   ├── compare_speed.py, speed_benchmark.py, run_speed_single.sh  # 计时
│   └── run_*.sh                # 各批次实验启动器（见下）
├── figures/                    # 报告图（quality-vs-cost、by-scene、fm-variants）
├── reports/                    # 数据质量审计、真实 vs 合成分布图
├── experiment.md / experiment.zh.md   # 合成数据阶段完整实验报告（英/中）
├── README.md / README_H800.md         # 本文件 / 集群运行说明
└── CLAUDE.md                          # 提交身份与推送约定
```

### 实验启动器 `scripts/run_*.sh` 分组

| 类别 | 脚本 |
| --- | --- |
| 端到端 / 集群 | `run_d3il_fm_experiment.sh`、`run_h800_parallel.sh`、`run_pilot_4090.sh` |
| n=100 评测 | `run_eval_n100.sh`、`run_baselines_n100.sh` |
| FM 扫参 | `run_fm_ksweep.sh`、`run_fm_thresholdsweep.sh` |
| DDPM 扫参 / 重训 | `run_ddpm_ksweep.sh`、`run_ddpm_thresholdsweep.sh`、`run_ddpm_retrain.sh` |
| DDIM few-step | `run_ddim_sweep.sh` |
| H=16 系列 | `run_h16_matrix.sh`（训练矩阵）、`run_h16_eval.sh`、`run_h16_fill.sh`、`run_h16_extra.sh`（DDIM sweep + DDPM K1/2 重训）、`run_h16_projfreq.sh`（投影次数 trade-off） |

## 6. 结果报告

- 合成数据阶段（H=8, n=50）的完整叙事、表格与图：`experiment.md` / `experiment.zh.md`。
- 真实 D3IL 数据（H=8 / H=16, n=100）的 K-sweep / DDIM / 投影方法结果目前维护在硕论
  Overleaf 稿件中，逐试验 npz 存于 `logs/`（未提交，通过上面的汇总脚本重算）。

## 7. 备注

- 评估中的投影是 CPU 受限（SLSQP / cvxpy），并行多作业时需设
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` 防线程超订；
  单进程干净计时则去掉这些限制。
- 运行输出、日志、checkpoint 均被 git 忽略（`logs/`、`run_logs/`、`*.pt`、`*.log` 等）。
- 提交身份与推送代理约定见 `CLAUDE.md`。
