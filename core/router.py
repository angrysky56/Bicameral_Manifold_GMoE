
import torch
import torch.nn as nn
from .nfe import StableDifferentiableNFE


class LearnableGeometricRouter(nn.Module):
    """
    The original Learnable Geometric Router.
    Uses a learnable polarization layer and NFE for ID-based routing.
    """
    def __init__(self, dim, threshold=1.9):
        super().__init__()
        # A simple linear projection to 'polarize' the latent space for ID estimation
        self.polarizer = nn.Linear(dim, dim, bias=False)
        # Initialize as identity to start with the raw manifold
        with torch.no_grad():
            self.polarizer.weight.copy_(torch.eye(dim))

        self.nfe = StableDifferentiableNFE()
        self.threshold = threshold
        self.dim = dim

    def forward(self, x):
        # x: (batch, seq, dim)
        # Apply polarization before ID estimation
        x_polarized = self.polarizer(x)

        # We estimate ID on the polarized sequence manifold
        current_id = self.nfe(x_polarized)

        # Hard routing for the forward pass, but ID is differentiable
        is_logic = current_id < self.threshold
        return is_logic, current_id


class ACC(nn.Module):
    """
    Artificial Corpus Callosum (ACC)

    An advanced router inspired by the biological Corpus Callosum.
    Implements:
    - Geometric Routing via Intrinsic Dimension (ID) estimation
    - Excitatory-Inhibitory (E-I) Balance for homeostatic regulation
    - Soft Callosal Gating for smooth transitions between Logic/Creative

    Reference: docs/CallosalNet: Artificial Corpus Callosum.md
    Reference: docs/Designing-a-Lifelong-Learning-Architecture.md
    """
    def __init__(self, dim, threshold=1.8, target_rate=0.5, ei_lr=0.01):
        super().__init__()
        # Polarization Layer (analogous to callosal fiber topology)
        self.polarizer = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.polarizer.weight.copy_(torch.eye(dim))

        # Intrinsic Dimension Estimator
        self.nfe = StableDifferentiableNFE()

        # Learnable Threshold (the "critical point" of the callosal gate)
        # Initialized at 1.8 (the Fractal Bottleneck target)
        self.threshold = nn.Parameter(torch.tensor(threshold))

        self.dim = dim

        # --- E-I Balance Parameters ---
        # Inhibitory weight: dynamically adjusted to maintain target_rate
        # Positive inhibitory_weight increases the threshold (more logic)
        # Negative inhibitory_weight decreases the threshold (more creative)
        self.inhibitory_weight = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.target_rate = target_rate  # Target proportion of "Logic" routing
        self.ei_lr = ei_lr  # Learning rate for E-I homeostatic updates

        # Exponential Moving Average of the Logic routing rate
        self.register_buffer('ema_rate', torch.tensor(target_rate))

        # --- Soft Callosal Gate (for smooth routing) ---
        # Gate temperature: lower = harder routing, higher = softer routing
        self.gate_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        # x: (batch, seq, dim)
        x_polarized = self.polarizer(x)
        current_id = self.nfe(x_polarized)

        # Effective threshold after E-I modulation
        effective_threshold = self.threshold + self.inhibitory_weight

        # Hard routing (for discrete expert selection)
        is_logic = current_id < effective_threshold

        # Soft gate (for interpolated routing if needed)
        # gate_value = 1.0 -> Full Logic, 0.0 -> Full Creative
        gate_value = torch.sigmoid(
            (effective_threshold - current_id) / self.gate_temperature
        )

        # E-I Balance Update (during training only)
        if self.training:
            # Update EMA of Logic rate
            current_rate = is_logic.float().mean()
            self.ema_rate = 0.99 * self.ema_rate + 0.01 * current_rate.detach()

            # Homeostatic Inhibitory Plasticity Rule:
            # If we're routing to Logic too often (ema_rate > target_rate),
            # increase inhibitory weight (raise threshold) to suppress Logic routing.
            # If we're routing to Creative too often, decrease inhibitory weight.
            with torch.no_grad():
                deviation = self.ema_rate - self.target_rate
                self.inhibitory_weight.add_(self.ei_lr * deviation)

        return is_logic, current_id, gate_value

    def get_routing_stats(self):
        """Return current routing statistics for monitoring."""
        return {
            'threshold': self.threshold.item(),
            'inhibitory_weight': self.inhibitory_weight.item(),
            'effective_threshold': (self.threshold + self.inhibitory_weight).item(),
            'ema_logic_rate': self.ema_rate.item(),
            'target_rate': self.target_rate
        }
