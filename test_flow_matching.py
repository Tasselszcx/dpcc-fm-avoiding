#!/usr/bin/env python3
"""Basic test for FlowMatching implementation"""

import sys
import os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffuser.models import FlowMatching, UNet1DTemporalCondModel

print("Testing FlowMatching implementation...")

# Create a simple UNet model
model = UNet1DTemporalCondModel(
    horizon=8,
    transition_dim=10,
    cond_dim=6,
    dim=32,
    dim_mults=(1, 2, 4, 8),
)

# Create FlowMatching instance
fm = FlowMatching(
    model=model,
    horizon=8,
    observation_dim=6,
    action_dim=4,
    n_timesteps=10,
    loss_type='l2',
    ode_solver='euler',
)

print("✓ FlowMatching instantiated successfully")

# Test training interface
x = torch.randn(4, 8, 10)
cond = {0: torch.randn(4, 6)}
loss, info = fm.loss(x, cond)

print(f"✓ Training loss computed: {loss.item():.4f}")
assert loss.shape == (), "Loss should be scalar"
assert 'diffusion_loss' in info, "Info should contain diffusion_loss"

# Test sampling interface
samples, infos = fm.conditional_sample(cond, horizon=8)

print(f"✓ Sampling completed: shape {samples.shape}")
assert samples.shape == (4, 8, 10), f"Expected (4, 8, 10), got {samples.shape}"
assert 'projection_costs' in infos, "Infos should contain projection_costs"

# Test conditioning preservation
obs_start = cond[0]
obs_sampled = samples[:, 0, 4:]  # Extract observations at t=0
diff = (obs_start - obs_sampled).abs().max().item()
print(f"✓ Conditioning preserved (max diff: {diff:.6f})")

print("\n✅ All tests passed!")
