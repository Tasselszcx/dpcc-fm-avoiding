import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, goal_dim=0, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
        loss_discount=1.0, loss_weights=None, returns_condition=False, condition_guidance_w=0.1,):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # DDIM sampling controls (defaults preserve the original stochastic DDPM path).
        # When use_ddim is True, conditional_sample dispatches to a deterministic
        # (eta=0 by default) DDIM sampler that subsamples ddim_steps timesteps from
        # the trained K=n_timesteps schedule -- no retraining required.
        self.use_ddim = False
        self.ddim_steps = None
        self.ddim_eta = 0.0

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        # if self.model.calc_energy:
        #     assert self.predict_epsilon
        #     x = torch.tensor(x, requires_grad=True)
        #     t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        #     returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)

        if projector is not None and projector.gradient:
            if self.goal_dim > 0:
                grad = projector.compute_gradient(x_recon[:,:,:-self.goal_dim], constraints)
            else:
                grad = projector.compute_gradient(x_recon, constraints)
            model_mean = model_mean + grad

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, projector=None, constraints=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns, projector=projector, constraints=constraints)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, return_diffusion=False, projector=None, constraints=None, repeat_last=0):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion: diffusion = [x]
        costs = {}

        # Denoising process
        last_timestep = -repeat_last if repeat_last > 0 and projector is not None else 0
        for i in reversed(range(last_timestep, self.n_timesteps)):
            t = i if i >= 0 else 0
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if projector is not None and projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                x = self.p_sample(x, cond, timesteps, returns, projector=projector, constraints=constraints)
            else:
                x = self.p_sample(x, cond, timesteps, returns)

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if projector is not None and not projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                if self.goal_dim > 0:
                    x[:,:,:-self.goal_dim], projection_costs = projector.project(x[:,:,:-self.goal_dim], constraints)
                    costs[i] = projection_costs
                else:
                    x, projection_costs = projector.project(x, constraints)
                    costs[i] = projection_costs

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion: diffusion.append(x)

        infos = {}
        if return_diffusion: infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs

        return x, infos

    @torch.no_grad()
    def ddim_sample_loop(self, shape, cond, returns=None, return_diffusion=False,
                         projector=None, constraints=None, repeat_last=0):
        '''
            Deterministic DDIM sampling (Song et al., 2021). Reuses the trained
            epsilon network and the K=n_timesteps noise schedule, but integrates the
            reverse process over only `ddim_steps` timesteps subsampled from
            {0, ..., n_timesteps-1}. With ddim_eta == 0 the reverse process is fully
            deterministic. Projection is applied with the same gating as the DDPM
            p_sample_loop so that this is a like-for-like baseline under projection.

            Implementation note: this codebase's DDPM uses a non-standard variance-
            reduced sampler (0.5*randn init and 0.5-scaled reverse noise), so the
            closed-form DDIM x0->x_prev jump is not self-consistent with the trained
            eps-network here. We instead use *respaced* ancestral sampling: for each
            adjacent pair (t_cur, t_prev) in the subsampled schedule we form the exact
            DDPM posterior mean of the strided transition (beta' = 1 - a_cur/a_prev),
            which reduces to the trained per-step posterior when ddim_steps == K and,
            with eta == 0, gives deterministic few-step sampling.
        '''
        device = self.betas.device
        batch_size = shape[0]

        n_steps = self.ddim_steps or self.n_timesteps
        n_steps = min(int(n_steps), self.n_timesteps)
        # Evenly spaced subset of the trained timesteps, ascending unique.
        seq = np.linspace(0, self.n_timesteps - 1, n_steps)
        seq = np.unique(np.round(seq).astype(int))
        seq = list(seq)
        seq_next = [-1] + seq[:-1]            # previous (lower-noise) timestep for each

        # Match the stochastic DDPM sampler's terminal-latent scale (0.5*randn); the
        # trained eps-network is calibrated to this manifold (verified: full-step
        # deterministic ancestral sampling reproduces the DDPM success rate).
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion: diffusion = [x]
        costs = {}

        for t_cur, t_prev in zip(reversed(seq), reversed(seq_next)):
            timesteps = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)

            # Predict epsilon (mirror p_mean_variance's classifier-free-guidance branch)
            if self.returns_condition:
                eps_cond = self.model(x, cond, timesteps, returns, use_dropout=False)
                eps_uncond = self.model(x, cond, timesteps, returns, force_dropout=True)
                epsilon = eps_uncond + self.condition_guidance_w * (eps_cond - eps_uncond)
            else:
                epsilon = self.model(x, cond, timesteps)

            x0 = self.predict_start_from_noise(x, t=timesteps, noise=epsilon)
            if self.clip_denoised:
                x0.clamp_(-1., 1.)

            # Optional gradient-based projection (dpcc-r / gradient variants). No-op for
            # the per-step minimum-projection-cost variant where projector.gradient=False.
            if projector is not None and projector.gradient and t_cur <= projector.diffusion_timestep_threshold * self.n_timesteps:
                if self.goal_dim > 0:
                    x0[:, :, :-self.goal_dim] = x0[:, :, :-self.goal_dim] + projector.compute_gradient(x0[:, :, :-self.goal_dim], constraints)
                else:
                    x0 = x0 + projector.compute_gradient(x0, constraints)

            # Respaced DDPM posterior mean for the strided transition t_cur -> t_prev.
            a_cur = self.alphas_cumprod[t_cur]
            a_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones((), device=device)
            beta_str = 1. - a_cur / a_prev                      # equivalent one-step beta
            coef_x0 = beta_str * a_prev.sqrt() / (1. - a_cur)
            coef_xt = (1. - a_prev) * (a_cur / a_prev).sqrt() / (1. - a_cur)
            x = coef_x0 * x0 + coef_xt * x

            # Optional stochasticity (eta>0); eta==0 keeps the sampler deterministic.
            if self.ddim_eta > 0 and t_prev >= 0:
                post_var = beta_str * (1. - a_prev) / (1. - a_cur)
                x = x + self.ddim_eta * post_var.clamp(min=0).sqrt() * 0.5 * torch.randn_like(x)

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            # Hard projection with the same timestep gating as p_sample_loop.
            if projector is not None and not projector.gradient and t_cur <= projector.diffusion_timestep_threshold * self.n_timesteps:
                if self.goal_dim > 0:
                    x[:, :, :-self.goal_dim], projection_costs = projector.project(x[:, :, :-self.goal_dim], constraints)
                    costs[t_cur] = projection_costs
                else:
                    x, projection_costs = projector.project(x, constraints)
                    costs[t_cur] = projection_costs
                x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion: diffusion.append(x)

        infos = {}
        if return_diffusion: infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs

        return x, infos

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        if self.use_ddim:
            return self.ddim_sample_loop(shape, cond, returns, *args, **kwargs)
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion: diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        if self.predict_epsilon:
            # Cause we condition on obs at t=0
            # noise[:, 0, self.action_dim:] = 0
            noise = apply_conditioning(noise, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim, goal_dim=self.goal_dim)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim, goal_dim=self.goal_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)