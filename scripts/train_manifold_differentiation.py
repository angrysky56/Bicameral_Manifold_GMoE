
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
from tqdm import tqdm

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
REPO_PATH = "/home/ty/Repositories/ai_workspace/Bicameral_Manifold_GMoE"
sys.path.append(REPO_PATH)

from core.modeling_bicameral import convert_to_bicameral, MANIFOLD_TELEMETRY

def load_local_data(filename):
    path = os.path.join(REPO_PATH, "data/processed", filename)
    with open(path, "r") as f:
        return f.read().split("\n---SAMPLE_SEP---\n")

def set_model_forced_mode(model, mode):
    for layer in model.model.layers:
        layer.mlp.forced_mode = mode

def differentiate_manifolds():
    output_path = os.path.join(REPO_PATH, "models/checkpoints/smollm3_bicameral_diff")
    if not os.path.exists(output_path): os.makedirs(output_path)

    print(f"[*] Loading {MODEL_ID} and Data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = convert_to_bicameral(model)

    logic_samples = load_local_data("logic_manifold.txt")
    creative_samples = load_local_data("creative_manifold.txt")

    # --- PHASE 1: LOGIC + ROUTER POLARIZATION ---
    print("[*] Phase 1: Specializing Logic Experts + Training Router...")
    set_model_forced_mode(model, "LOGIC")

    # We train logic experts AND the router polarizers
    trainable_params = []
    for n, p in model.named_parameters():
        if "logic_" in n or "router.polarizer" in n:
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False

    optimizer = optim.AdamW(trainable_params, lr=1e-5)

    model.train()
    pbar = tqdm(logic_samples[:100], desc="Logic Training")
    for text in pbar:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model(**inputs, labels=inputs["input_ids"])

        # 1. Language Modeling Loss
        lm_loss = outputs.loss

        # 2. Topological Loss: Force ID towards 1.5 (Fractal Bottleneck for Logic)
        # Target D*_Logic = 1.5 (sparser than critical point of 1.8)
        ids = [layer.mlp.cached_id for layer in model.model.layers if layer.mlp.cached_id is not None]
        if ids:
            avg_id = torch.stack(ids).mean()
            topo_loss = (avg_id - 1.5)**2
        else:
            avg_id = torch.tensor(0.0)
            topo_loss = torch.tensor(0.0, device=lm_loss.device)

        total_loss = lm_loss + 0.1 * topo_loss  # Lambda=0.1 weighting

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log ID for monitoring manifold polarization
        pbar.set_postfix(loss=f"{lm_loss.item():.3f}", ID=f"{avg_id.item():.2f}")

    # --- PHASE 2: CREATIVE + ROUTER POLARIZATION ---
    print("[*] Phase 2: Specializing Creative Experts + Training Router...")
    set_model_forced_mode(model, "CREATIVE")

    trainable_params = []
    for n, p in model.named_parameters():
        if "creative_" in n or "router.polarizer" in n:
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False

    optimizer = optim.AdamW(trainable_params, lr=1e-5)

    pbar = tqdm(creative_samples[:100], desc="Creative Training")
    for text in pbar:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model(**inputs, labels=inputs["input_ids"])

        lm_loss = outputs.loss

        # Topological Loss: Force ID towards 3.0 (Isotropic/Expansive Manifold for Creative)
        # Target D*_Creative = 3.0 (more expansive than critical point of 1.8)
        ids = [layer.mlp.cached_id for layer in model.model.layers if layer.mlp.cached_id is not None]
        if ids:
            avg_id = torch.stack(ids).mean()
            topo_loss = (avg_id - 3.0)**2
        else:
            avg_id = torch.tensor(0.0)
            topo_loss = torch.tensor(0.0, device=lm_loss.device)

        total_loss = lm_loss + 0.1 * topo_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log ID for monitoring manifold polarization
        pbar.set_postfix(loss=f"{lm_loss.item():.3f}", ID=f"{avg_id.item():.2f}")

    set_model_forced_mode(model, None)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[SUCCESS] Differentiated SmolLM with Learnable Router saved to {output_path}")

if __name__ == "__main__":
    differentiate_manifolds()
