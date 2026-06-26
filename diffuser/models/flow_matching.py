import torch
import torch.nn as nn
from .helpers import (
    apply_conditioning,
    Losses,
)

class FlowMatching(nn.Module):
    """
    Flow Matching for trajectory generation.
    Uses ODE-based sampling with learned velocity fields.

    Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
    """

    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        goal_dim=0,
        n_timesteps=10,
        loss_type='l2',
        clip_denoised=False,
        predict_epsilon=False,  # 兼容性参数，Flow Matching 不使用
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        returns_condition=False,
        condition_guidance_w=0.1,
        ode_solver='euler',
        device='cuda',
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.transition_dim = observation_dim + action_dim

        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.ode_solver = ode_solver

        # Loss function
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.register_buffer('loss_weights', loss_weights)
        self.loss_fn = Losses[loss_type](self.loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        Compute loss weights for trajectory.
        Copied from GaussianDiffusion for compatibility.
        """
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # Apply per-dimension weights
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # Exponential decay over horizon
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float32)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # Override first action weight
        if action_weight is not None:
            loss_weights[0, :self.action_dim] = action_weight

        return loss_weights

    def compute_loss(self, x_start, cond, t, returns=None):
        """
        Compute Flow Matching loss at time t.

        Args:
            x_start: [batch, horizon, transition_dim] - clean trajectories
            cond: dict - conditioning information
            t: [batch] - time in [0, 1]
            returns: [batch, 1] - optional return values

        Returns:
            loss: scalar
            info: dict with loss components
        """
        batch_size = len(x_start)

        # Sample noise at t=0 and learn the probability path to data at t=1.
        x_0 = torch.randn_like(x_start)

        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1, where x_1 is data.
        t_expanded = t.view(batch_size, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_start
        x_t = apply_conditioning(x_t, cond, self.action_dim, goal_dim=self.goal_dim)

        # Target velocity (constant for linear paths): noise -> data.
        v_target = x_start - x_0
        v_target = apply_conditioning(v_target, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        # Predict velocity (scale t to [0, 1000] for time embedding)
        t_scaled = (t * 1000).long()
        v_pred = self.model(x_t, cond, t_scaled, returns)

        # Compute weighted loss
        loss = (self.loss_weights * (v_pred - v_target) ** 2).mean()

        return loss, {'diffusion_loss': loss}

    def loss(self, x, cond, returns=None):
        """
        Training loss interface (matches GaussianDiffusion API).

        Args:
            x: [batch, horizon, transition_dim] - trajectories
            cond: dict - conditioning
            returns: [batch, 1] - optional returns

        Returns:
            loss: scalar
            info: dict
        """
        batch_size = len(x)
        t = torch.rand(batch_size, device=x.device)  # Sample t ~ Uniform(0, 1)
        return self.compute_loss(x, cond, t, returns)

    def predict_velocity(self, x, cond, t, returns=None):
        """
        Predict velocity field with optional classifier-free guidance.

        Args:
            x: [batch, horizon, transition_dim]
            cond: dict - conditioning
            t: [batch] - time scaled to [0, 1000]
            returns: [batch, 1] - optional returns

        Returns:
            v: [batch, horizon, transition_dim] - predicted velocity
        """
        if self.returns_condition:
            v_cond = self.model(x, cond, t, returns, use_dropout=False)
            v_uncond = self.model(x, cond, t, returns, force_dropout=True)
            v = v_uncond + self.condition_guidance_w * (v_cond - v_uncond)
        else:
            v = self.model(x, cond, t)

        if self.clip_denoised:
            v = v.clamp(-10., 10.)

        return v

    def euler_step(self, x, cond, t, dt, returns=None, projector=None, constraints=None):
        """
        Single Euler integration step: x_{t+dt} = x_t + dt * v(x_t, t)

        Args:
            x: [batch, horizon, transition_dim]
            cond: dict - conditioning
            t: [batch] - time in [0, 1]
            dt: float - time step
            returns: [batch, 1] - optional returns
            projector: Projector instance or None
            constraints: constraint list or None

        Returns:
            x_next: [batch, horizon, transition_dim]
        """
        # Scale t to [0, 1000] for model
        t_scaled = (t * 1000).long()

        # Predict velocity
        v = self.predict_velocity(x, cond, t_scaled, returns)

        # Apply gradient-based projection if needed
        if projector is not None and projector.gradient:
            if t[0].item() >= (1 - projector.diffusion_timestep_threshold):
                if self.goal_dim > 0:
                    grad = projector.compute_gradient(x[:,:,:-self.goal_dim], constraints)
                    v[:,:,:-self.goal_dim] = v[:,:,:-self.goal_dim] + grad
                else:
                    grad = projector.compute_gradient(x, constraints)
                    v = v + grad

        # Euler update
        return x + dt * v

    def rk4_step(self, x, cond, t, dt, returns=None, projector=None, constraints=None):
        """
        Single RK4 integration step (4th order accuracy).

        Args:
            x: [batch, horizon, transition_dim]
            cond: dict - conditioning
            t: [batch] - time in [0, 1]
            dt: float - time step
            returns: [batch, 1] - optional returns
            projector: Projector instance (gradient mode only)
            constraints: constraint list or None

        Returns:
            x_next: [batch, horizon, transition_dim]
        """
        # RK4 coefficients
        k1 = self.predict_velocity(x, cond, (t * 1000).long(), returns)
        k2 = self.predict_velocity(x + 0.5*dt*k1, cond, ((t + 0.5*dt) * 1000).long(), returns)
        k3 = self.predict_velocity(x + 0.5*dt*k2, cond, ((t + 0.5*dt) * 1000).long(), returns)
        k4 = self.predict_velocity(x + dt*k3, cond, ((t + dt) * 1000).long(), returns)

        # RK4 update
        return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    @torch.no_grad()
    def ode_sample_loop(self, shape, cond, returns=None, return_diffusion=False,
                        projector=None, constraints=None):
        """
        ODE integration loop from t=0 (noise) to t=1 (data).

        Args:
            shape: (batch_size, horizon, transition_dim)
            cond: dict - conditioning
            returns: [batch, 1] - optional returns
            return_diffusion: bool - whether to return full trajectory
            projector: Projector instance or None
            constraints: constraint list or None

        Returns:
            x: [batch, horizon, transition_dim] - final sample
            infos: dict with 'diffusion' and 'projection_costs'
        """
        device = self.loss_weights.device
        batch_size = shape[0]

        # Start from noise (t=0)
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion:
            diffusion = [x]
        costs = {}

        # ODE integration from t=0 to t=1
        dt = 1.0 / self.n_timesteps

        for step in range(self.n_timesteps):
            t_current = step * dt
            t_tensor = torch.full((batch_size,), t_current, device=device)

            # ODE step
            if self.ode_solver == 'euler':
                x = self.euler_step(x, cond, t_tensor, dt, returns, projector, constraints)
            elif self.ode_solver == 'rk4':
                x = self.rk4_step(x, cond, t_tensor, dt, returns, projector, constraints)
            else:
                raise ValueError(f"Unknown ODE solver: {self.ode_solver}")

            # Apply conditioning
            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            # Apply projection-based constraints (late timesteps, near data)
            if projector is not None and not projector.gradient:
                should_project = t_current >= (1 - projector.diffusion_timestep_threshold)
                should_project = should_project and ((step + 1) % projector.project_every == 0 or step == self.n_timesteps - 1)
                if should_project:
                    if self.goal_dim > 0:
                        x[:,:,:-self.goal_dim], proj_costs = projector.project(x[:,:,:-self.goal_dim], constraints)
                    else:
                        x, proj_costs = projector.project(x, constraints)
                    costs[step] = proj_costs

            # Re-apply conditioning after projection
            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion:
                diffusion.append(x)

        infos = {}
        if return_diffusion:
            infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs

        return x, infos

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        """
        Conditional sampling interface (matches GaussianDiffusion API).

        Args:
            cond: dict - conditioning with format {t: values}
            returns: [batch, 1] - optional return values
            horizon: int - trajectory length (defaults to self.horizon)
            *args, **kwargs: passed to ode_sample_loop

        Returns:
            samples: [batch, horizon, transition_dim]
            infos: dict
        """
        device = self.loss_weights.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.ode_sample_loop(shape, cond, returns, *args, **kwargs)

    def forward(self, cond, *args, **kwargs):
        """
        Forward pass calls conditional_sample (matches GaussianDiffusion API).
        """
        return self.conditional_sample(cond=cond, *args, **kwargs)
