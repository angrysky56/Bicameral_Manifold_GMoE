
import torch
import torch.nn as nn
from .router import LearnableGeometricRouter, ACC

MANIFOLD_TELEMETRY = {"mode": "INIT", "id": 2.0, "gate": 0.5}


class BicameralMLP(nn.Module):
    """
    Bicameral Multi-Layer Perceptron (MLP)

    Replaces standard MLP with dual Logic/Creative experts.
    Routing is performed by a Geometric Router (ID-based).

    Modes:
    - Hard Routing: Discrete selection of Logic or Creative expert
    - Soft Routing (use_soft_routing=True): Blends outputs based on gate value
    - Forced Mode: Training mode where expert is pre-selected

    Reference: docs/Dual-Manifold-Architecture-Design.md
    """
    def __init__(self, original_mlp, use_acc=False, use_soft_routing=False):
        super().__init__()
        in_features = original_mlp.gate_proj.in_features
        out_features = original_mlp.gate_proj.out_features

        # Logic Expert (Fractal-Compressed Manifold)
        self.logic_gate = nn.Linear(in_features, out_features, bias=False)
        self.logic_up = nn.Linear(in_features, out_features, bias=False)
        self.logic_down = nn.Linear(out_features, in_features, bias=False)

        # Creative Expert (High-Entropy Expansive Manifold)
        self.creative_gate = nn.Linear(in_features, out_features, bias=False)
        self.creative_up = nn.Linear(in_features, out_features, bias=False)
        self.creative_down = nn.Linear(out_features, in_features, bias=False)

        # Router Selection: ACC (advanced) or LearnableGeometricRouter (simple)
        if use_acc:
            self.router = ACC(in_features)
            self.acc_mode = True
        else:
            self.router = LearnableGeometricRouter(in_features)
            self.acc_mode = False

        self.use_soft_routing = use_soft_routing
        self.forced_mode = None
        self.cached_id = None
        self.cached_mode = None
        self.cached_gate = None

        # Initialize experts with original MLP weights (both start identical)
        with torch.no_grad():
            self.logic_gate.weight.copy_(original_mlp.gate_proj.weight)
            self.logic_up.weight.copy_(original_mlp.up_proj.weight)
            self.logic_down.weight.copy_(original_mlp.down_proj.weight)
            self.creative_gate.weight.copy_(original_mlp.gate_proj.weight)
            self.creative_up.weight.copy_(original_mlp.up_proj.weight)
            self.creative_down.weight.copy_(original_mlp.down_proj.weight)

        self.act_fn = nn.SiLU()

    def _compute_logic(self, x):
        """Logic Expert forward pass."""
        return self.logic_down(self.act_fn(self.logic_gate(x)) * self.logic_up(x))

    def _compute_creative(self, x):
        """Creative Expert forward pass."""
        return self.creative_down(self.act_fn(self.creative_gate(x)) * self.creative_up(x))

    def forward(self, x):
        seq_len = x.size(1)
        gate_val = 0.5  # Default gate value

        # 1. Routing Logic
        if self.forced_mode:
            is_logic = (self.forced_mode == "LOGIC")
            # We still calculate ID for the loss, but force the choice
            if self.acc_mode:
                _, id_val, gate_val = self.router(x)
            else:
                _, id_val = self.router(x)
        elif seq_len > 1:
            # Full routing on prefill (prompt processing)
            if self.acc_mode:
                is_logic, id_val, gate_val = self.router(x)
            else:
                is_logic, id_val = self.router(x)
            self.cached_id = id_val
            self.cached_mode = is_logic
            self.cached_gate = gate_val
        elif self.cached_mode is not None:
            # Use cached routing for token-by-token generation
            id_val = self.cached_id
            is_logic = self.cached_mode
            gate_val = self.cached_gate if self.cached_gate is not None else 0.5
        else:
            # Fallback: default to Creative
            id_val = torch.tensor(2.0, device=x.device)
            is_logic = False
            gate_val = 0.5

        # Update global telemetry
        global MANIFOLD_TELEMETRY
        MANIFOLD_TELEMETRY["mode"] = "LOGIC" if is_logic else "CREATIVE"
        MANIFOLD_TELEMETRY["id"] = id_val.item() if torch.is_tensor(id_val) else id_val
        MANIFOLD_TELEMETRY["gate"] = gate_val.item() if torch.is_tensor(gate_val) else gate_val

        # 2. Expert Execution
        if self.use_soft_routing and self.acc_mode:
            # Soft Routing: Blend outputs based on gate value
            # gate_val=1.0 -> Full Logic, gate_val=0.0 -> Full Creative
            logic_out = self._compute_logic(x)
            creative_out = self._compute_creative(x)
            return gate_val * logic_out + (1 - gate_val) * creative_out
        else:
            # Hard Routing: Discrete expert selection
            if is_logic:
                return self._compute_logic(x)
            else:
                return self._compute_creative(x)


def convert_to_bicameral(model, use_acc=False, use_soft_routing=False):
    """
    Perform Bicameral Surgery on a transformer model.

    Replaces each MLP layer with a BicameralMLP (dual Logic/Creative experts).

    Args:
        model: HuggingFace transformer model
        use_acc: If True, use ACC (Artificial Corpus Callosum) for routing
        use_soft_routing: If True, blend Logic/Creative outputs (requires use_acc=True)

    Returns:
        Modified model with Bicameral MLPs
    """
    router_name = "ACC" if use_acc else "LearnableGeometricRouter"
    routing_mode = "Soft" if use_soft_routing else "Hard"
    print("[*] Performing Learnable Bicameral Surgery...")
    print(f"    Router: {router_name} | Routing Mode: {routing_mode}")

    for layer in model.model.layers:
        layer.mlp = BicameralMLP(layer.mlp, use_acc=use_acc, use_soft_routing=use_soft_routing)

    return model

