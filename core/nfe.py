
import torch
import torch.nn as nn
import numpy as np
import os
from .nfe_critic import HiguchiFDCritic


# Global cache for the critic weights to avoid reloading from disk for every layer
_CRITIC_WEIGHTS_CACHE = None

class NeuralFractalEstimator(nn.Module):
    """
    Neural Fractal Estimator (NFE)

    Uses a pre-trained Neural Critic to estimate the Higuchi Fractal Dimension (D_H)
    of a trajectory.

    If weights are not found, falls back to the heuristic Correlation Dimension.
    """

    def __init__(self, hidden_dim=None, weights_path="models/checkpoints/nfe_critic.pt"):
        super().__init__()
        global _CRITIC_WEIGHTS_CACHE

        self.use_critic = False
        self.projector = None # Lazy init

        # 1. Critic (Higuchi FD)
        self.critic = HiguchiFDCritic()
        # Look for weights in repo
        repo_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), weights_path)

        if os.path.exists(repo_weights):
            # Use cached weights if available
            if _CRITIC_WEIGHTS_CACHE is None:
                print(f"[*] Loading NFE Critic from {repo_weights}")
                try:
                    _CRITIC_WEIGHTS_CACHE = torch.load(repo_weights, map_location='cpu')
                except Exception as e:
                    print(f"[!] Failed to load NFE Critic: {e}")
                    _CRITIC_WEIGHTS_CACHE = False

            if _CRITIC_WEIGHTS_CACHE:
                self.critic.load_state_dict(_CRITIC_WEIGHTS_CACHE)
                self.critic.eval()
                for p in self.critic.parameters():
                    p.requires_grad = False
                self.use_critic = True

                # Pre-initialize if hidden_dim is known
                if hidden_dim is not None:
                     self.projector = nn.Linear(hidden_dim, 1)
                     nn.init.orthogonal_(self.projector.weight)
            else:
                 self.use_critic = False

        else:
            if _CRITIC_WEIGHTS_CACHE is None: # Only warn once
                print(f"[!] NFE Critic weights not found at {repo_weights}. Using Fallback Correlation Dim.")
                _CRITIC_WEIGHTS_CACHE = False # Mark as checked and failed

            self.use_critic = False

    def forward(self, x):
        # x: (batch, seq, hidden_dim)

        if self.use_critic:
            # Lazy Init Projector if needed
            if self.projector is None:
                input_dim = x.size(-1)
                self.projector = nn.Linear(input_dim, 1).to(x.device)
                nn.init.orthogonal_(self.projector.weight)


        if self.use_critic:
            # Neural Higuchi Estimation
            # 1. Project to 1D signal
            signal = self.projector(x).squeeze(-1) # (batch, seq)

            # 2. Normalize signal (Critic expects mean=0, std=1)
            mean = signal.mean(dim=1, keepdim=True)
            std = signal.std(dim=1, keepdim=True)
            signal = (signal - mean) / (std + 1e-8)

            # 3. Estimate D_H
            # Critic output is unbounded, but D_H is typically [1.0, 2.0]
            d_h = self.critic(signal)

            # Clamp to valid range for stability
            return torch.clamp(d_h, 1.0, 2.5)

        else:
            # Fallback: Correlation Dimension (D_2)
            # (Original implementation logic)
            if x.dim() == 3:
                activations = x[0] # Take first batch item approximation
            else:
                activations = x

            N = activations.size(0)
            if N < 5: return torch.tensor(2.0, device=x.device)

            activations = (activations - activations.mean(0)) / (activations.std(0) + 1e-6)
            dists = torch.cdist(activations, activations)
            max_d = torch.max(dists)
            epsilons = torch.logspace(np.log10(max_d.item()*0.01), np.log10(max_d.item()*0.5), 5).to(x.device)
            log_eps = torch.log(epsilons)
            log_counts = []
            for eps in epsilons:
                count = (dists < eps).float().sum() - N
                log_counts.append(torch.log(count / (N * (N - 1)) + 1e-8))
            log_counts = torch.stack(log_counts)
            A = torch.stack([log_eps, torch.ones_like(log_eps)], dim=1)
            sol = torch.linalg.lstsq(A, log_counts.unsqueeze(1)).solution
            return torch.clamp(sol[0][0], 1.0, 3.5)

# Alias for backward compatibility
StableDifferentiableNFE = NeuralFractalEstimator

