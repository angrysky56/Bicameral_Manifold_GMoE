
import torch
import os
import sys
from safetensors.torch import load_file, save_file

# Add repo to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def build_bicameral_weights():
    base_path = "/home/ty/Repositories/ai_workspace/Bicameral_Manifold_GMoE/weights/pytorch"
    output_path = "/home/ty/Repositories/ai_workspace/Bicameral_Manifold_GMoE/weights/bicameral"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("[*] Loading base BitNet weights...")
    state_dict = load_file(os.path.join(base_path, "model.safetensors"))

    new_state_dict = {}

    print("[*] Transforming weights into Bicameral Manifold structure...")
    for key, value in state_dict.items():
        if "mlp" in key:
            # Duplicate weights for Logic and Creative experts
            # key example: model.layers.0.mlp.gate_proj.weight
            logic_key = key.replace("mlp.", "mlp.logic_") # Placeholder logic
            # Correct mapping based on our BicameralMLP class:
            # gate_proj -> gate_proj_logic and gate_proj_creative

            if "gate_proj" in key:
                new_state_dict[key.replace("gate_proj", "gate_proj_logic")] = value.clone()
                new_state_dict[key.replace("gate_proj", "gate_proj_creative")] = value.clone()
            elif "up_proj" in key:
                new_state_dict[key.replace("up_proj", "up_proj_logic")] = value.clone()
                new_state_dict[key.replace("up_proj", "up_proj_creative")] = value.clone()
            elif "down_proj" in key:
                new_state_dict[key.replace("down_proj", "down_proj_logic")] = value.clone()
                new_state_dict[key.replace("down_proj", "down_proj_creative")] = value.clone()
        else:
            # Keep other weights (attention, embeddings, etc.) as is
            new_state_dict[key] = value

    print(f"[*] Saving Bicameral weights to {output_path}...")
    save_file(new_state_dict, os.path.join(output_path, "model.safetensors"))
    print("[SUCCESS] Bicameral weights built.")

if __name__ == "__main__":
    build_bicameral_weights()
