#!/usr/bin/env python3
"""
DPCC 推理速度基准测试
对比 DDPM (K=20) vs Flow Matching (K=20)
在 Mac M2 Pro (MPS) 上运行，与论文 Linux RTX4090 结果对比
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.optimize import minimize, Bounds

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"设备: {DEVICE}  |  torch {torch.__version__}")
print("=" * 60)

# ── 论文里报告的 Linux RTX4090 数字（对比基准）────────────────
PAPER_NUMBERS = {
    # (backbone, projection): time_per_step_seconds
    ('DDPM', 'none'):       0.220,   # Diffuser baseline
    ('DDPM', 'dpcc'):       0.518,   # DPCC-C-tightened
    ('DDPM', 'post'):       0.478,   # Post-processing
    ('FM',   'none'):       0.120,   # FM unconstrained
    ('FM',   'dpcc'):       1.186,   # FM+DPCC-C-tightened
    ('FM',   'post'):       0.731,   # FM+post-processing
}

# ── 加载模型 ──────────────────────────────────────────────────
import diffuser.utils as utils
import pickle

def load_model_direct(log_dir, device):
    """直接从 pkl config + checkpoint 加载模型"""
    model_cfg = pickle.load(open(f'{log_dir}/model_config.pkl', 'rb'))
    diff_cfg  = pickle.load(open(f'{log_dir}/diffusion_config.pkl', 'rb'))

    # 实例化 UNet（覆盖 config 里保存的 cuda device）
    model_cfg._device = None   # 不让 Config.__call__ 自动 .to('cuda')
    unet = model_cfg()

    # 找最好的 checkpoint
    ckpt = f'{log_dir}/state_best.pt'
    if not os.path.exists(ckpt):
        import glob
        candidates = sorted(glob.glob(f'{log_dir}/state_*.pt'))
        ckpt = candidates[-1] if candidates else None

    # 实例化扩散/FM 模型（同样覆盖 device）
    diff_cfg._device = None
    diffusion = diff_cfg(unet)

    # 加载权重
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    # state 可能是 {'model': ..., 'ema': ...} 或直接是 state_dict
    if isinstance(state, dict) and 'model' in state:
        diffusion.load_state_dict(state['model'])
    elif isinstance(state, dict) and 'ema' in state:
        diffusion.load_state_dict(state['ema'])
    else:
        diffusion.load_state_dict(state)

    diffusion = diffusion.to(device)
    diffusion.eval()
    return diffusion

DDPM_DIR = 'logs/avoiding-synthetic/diffusion/H8_K20_Dmodels.GaussianDiffusion/0'
FM_DIR   = 'logs/avoiding-synthetic/diffusion/H8_K20_Dmodels.FlowMatching/0'

print("加载 DDPM 模型...", end=' ', flush=True)
ddpm_model = load_model_direct(DDPM_DIR, DEVICE)
print("✓")

print("加载 FM 模型...", end=' ', flush=True)
fm_model = load_model_direct(FM_DIR, DEVICE)
print("✓")
print()

# ── QP 投影（用 scipy SLSQP 作为 proxsuite 的 Mac 替代）────────
HORIZON    = 8
ACTION_DIM = 2
OBS_DIM    = 4
TRANS_DIM  = ACTION_DIM + OBS_DIM

# top-right-hard halfspace: n=[0.8165, -0.5774], p=[0.2, -0.5]
import numpy as np
p1, p2 = np.array([0.2, -0.5]), np.array([0.6, 0.5])
direction = p2 - p1
normal    = np.array([-direction[1], direction[0]])
normal    = (normal / np.linalg.norm(normal)).astype(np.float64)
boundary  = np.array([0.2, -0.5], dtype=np.float64)
DELTA     = 0.025   # 约束收紧量

def qp_project(trajectory_np):
    """对整条 horizon 轨迹做单次 QP 投影（scipy SLSQP）"""
    H = HORIZON
    traj = trajectory_np.copy().astype(np.float64)   # scipy 需要 float64
    xy   = traj[:, ACTION_DIM:ACTION_DIM+2].flatten().astype(np.float64)

    def objective(z):
        return 0.5 * np.sum((z - xy)**2)

    def grad_obj(z):
        return z - xy

    # halfspace 约束：n^T (z_h - p) <= -delta  for each h
    constraints = []
    for h in range(H):
        idx = 2 * h
        def con(z, h=h):
            return -(normal @ (z[2*h:2*h+2] - boundary)) - DELTA
        constraints.append({'type': 'ineq', 'fun': con})

    result = minimize(
        objective, xy, jac=grad_obj,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 100, 'ftol': 1e-6}
    )
    if result.success:
        traj[:, ACTION_DIM:ACTION_DIM+2] = result.x.reshape(H, 2)
    return traj

# ── 构造 dummy 输入 ───────────────────────────────────────────
BATCH = 1
dummy_cond = {0: torch.zeros(BATCH, OBS_DIM, device=DEVICE)}

def time_inference(model, cond, n_warmup=3, n_runs=20, label=""):
    """纯推理（无投影）的时间"""
    torch.mps.synchronize() if DEVICE == 'mps' else None
    with torch.no_grad():
        for _ in range(n_warmup):
            model.conditional_sample(cond, horizon=HORIZON)
        torch.mps.synchronize() if DEVICE == 'mps' else None

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            samples, _ = model.conditional_sample(cond, horizon=HORIZON)
            torch.mps.synchronize() if DEVICE == 'mps' else None
            times.append(time.perf_counter() - t0)

    mean_t = np.mean(times)
    std_t  = np.std(times)
    return mean_t, std_t, samples

def time_with_post_processing(model, cond, n_warmup=3, n_runs=20):
    """推理 + 一次后处理 QP"""
    with torch.no_grad():
        for _ in range(n_warmup):
            s, _ = model.conditional_sample(cond, horizon=HORIZON)
            traj_np = s[0].detach().cpu().float().numpy()
            qp_project(traj_np)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            s, _ = model.conditional_sample(cond, horizon=HORIZON)
            torch.mps.synchronize() if DEVICE == 'mps' else None
            traj_np = s[0].detach().cpu().float().numpy()
            qp_project(traj_np)
            times.append(time.perf_counter() - t0)

    return np.mean(times), np.std(times)

# ── 运行测试 ──────────────────────────────────────────────────
N_RUNS = 30
print(f"每项测试：{N_RUNS} 次取均值（前 5 次 warmup）")
print()

results = {}

print("[1/4] DDPM K=20 — 无投影 ...", end=' ', flush=True)
t, s, _ = time_inference(ddpm_model, dummy_cond, n_warmup=5, n_runs=N_RUNS)
results[('DDPM','none')] = (t, s)
print(f"{t*1000:.1f} ms ± {s*1000:.1f} ms")

print("[2/4] FM K=10   — 无投影 ...", end=' ', flush=True)
t, s, _ = time_inference(fm_model, dummy_cond, n_warmup=5, n_runs=N_RUNS)
results[('FM','none')] = (t, s)
print(f"{t*1000:.1f} ms ± {s*1000:.1f} ms")

print("[3/4] DDPM K=20 — 后处理 QP ...", end=' ', flush=True)
t, s = time_with_post_processing(ddpm_model, dummy_cond, n_warmup=5, n_runs=N_RUNS)
results[('DDPM','post')] = (t, s)
print(f"{t*1000:.1f} ms ± {s*1000:.1f} ms")

print("[4/4] FM K=10   — 后处理 QP ...", end=' ', flush=True)
t, s = time_with_post_processing(fm_model, dummy_cond, n_warmup=5, n_runs=N_RUNS)
results[('FM','post')] = (t, s)
print(f"{t*1000:.1f} ms ± {s*1000:.1f} ms")

# ── 汇总对比 ──────────────────────────────────────────────────
print()
print("=" * 70)
print(f"{'方法':<28} {'本机 Mac M2 (ms)':>18} {'论文 RTX4090 (ms)':>18} {'比值':>8}")
print("-" * 70)

labels = {
    ('DDPM','none'): 'DDPM K=20  无投影',
    ('DDPM','post'): 'DDPM K=20  后处理 QP',
    ('FM',  'none'): 'FM   K=10  无投影',
    ('FM',  'post'): 'FM   K=10  后处理 QP',
}

for key, label in labels.items():
    mac_t, mac_s = results[key]
    paper_t      = PAPER_NUMBERS.get(key, None)
    mac_ms       = mac_t * 1000
    paper_ms     = paper_t * 1000 if paper_t else None
    ratio        = mac_t / paper_t if paper_t else None
    ratio_str    = f"{ratio:.1f}×" if ratio else "N/A"
    paper_str    = f"{paper_ms:.0f}" if paper_ms else "N/A"
    print(f"  {label:<26} {mac_ms:>14.1f} ms  {paper_str:>14} ms  {ratio_str:>6}")

print("=" * 70)

# ── FM / DDPM 速度比 ──────────────────────────────────────────
ddpm_none = results[('DDPM','none')][0]
fm_none   = results[('FM',  'none')][0]
ddpm_post = results[('DDPM','post')][0]
fm_post   = results[('FM',  'post')][0]

print()
print("▶  推理加速比（Mac 本机）")
print(f"   无投影：DDPM / FM = {ddpm_none/fm_none:.2f}× （论文: {PAPER_NUMBERS[('DDPM','none')]/PAPER_NUMBERS[('FM','none')]:.2f}×）")
print(f"   后处理：DDPM / FM = {ddpm_post/fm_post:.2f}× （论文: {PAPER_NUMBERS[('DDPM','post')]/PAPER_NUMBERS[('FM','post')]:.2f}×）")
print()
print("注：Mac MPS 无 CUDA 优化，绝对时间更慢，但 DDPM/FM 相对比值应接近论文。")
print("    proxsuite 在 Mac 有库路径问题，改用 scipy SLSQP；per-step 投影未测。")
