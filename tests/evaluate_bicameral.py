
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import time

REPO_PATH = "/home/ty/Repositories/ai_workspace/Bicameral_Manifold_GMoE"
sys.path.append(REPO_PATH)
from core.modeling_bicameral import convert_to_bicameral, MANIFOLD_TELEMETRY


def clear_manifold_cache(model):
    """Clear cached routing decisions for fresh evaluation."""
    for layer in model.model.layers:
        layer.mlp.cached_id = None
        layer.mlp.cached_mode = None
        layer.mlp.cached_gate = None


def run_benchmarks(model, tokenizer, name, is_bicameral=False):
    print(f"\n--- BENCHMARKING: {name} ---")
    benchmarks = [
        # Logic prompts (expect Low ID, LOGIC mode)
        {"type": "LOGIC", "prompt": "User: If a train travels 60 miles in 1 hour, how many miles does it travel in 2.5 hours?\nAssistant: Let's solve this step by step."},
        {"type": "LOGIC", "prompt": "User: Explain the logical fallacy in: 'All cats are animals. My dog is an animal. Therefore, my dog is a cat.'\nAssistant:"},
        {"type": "LOGIC", "prompt": "User: What is the derivative of x^3 + 2x^2 - 5x + 7?\nAssistant:"},

        # Creative prompts (expect High ID, CREATIVE mode)
        {"type": "CREATIVE", "prompt": "User: Describe a sunset on a planet with two suns and purple grass.\nAssistant:"},
        {"type": "CREATIVE", "prompt": "User: Write a haiku about the feeling of nostalgia.\nAssistant:"},
        {"type": "CREATIVE", "prompt": "User: Imagine you are a cloud floating above a busy city. Describe what you see and feel.\nAssistant:"},
    ]

    results = []
    correct_routing = 0

    for b in benchmarks:
        if is_bicameral:
            clear_manifold_cache(model)

        inputs = tokenizer(b["prompt"], return_tensors="pt")
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        latency = time.time() - start_time

        # Get telemetry
        mode = MANIFOLD_TELEMETRY["mode"]
        id_val = MANIFOLD_TELEMETRY["id"]
        gate_val = MANIFOLD_TELEMETRY.get("gate", 0.5)

        # Check routing accuracy
        expected_mode = b["type"]
        is_correct = (mode == expected_mode)
        if is_correct:
            correct_routing += 1

        results.append({
            "type": b["type"],
            "id": id_val,
            "mode": mode,
            "gate": gate_val,
            "latency": latency,
            "correct": is_correct
        })

        status = "✓" if is_correct else "✗"
        print(f"[{b['type']}] {status} Mode: {mode} | ID: {id_val:.2f} | Gate: {gate_val:.2f} | Latency: {latency:.2f}s")

    return results, correct_routing, len(benchmarks)


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    diff_path = os.path.join(REPO_PATH, "models/checkpoints/smollm3_bicameral_diff")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("[*] Loading Baseline...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    run_benchmarks(base_model, tokenizer, "BASELINE (DENSE)")
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n[*] Loading Bicameral Model...")
    bicameral_model = AutoModelForCausalLM.from_pretrained(model_id)
    bicameral_model = convert_to_bicameral(bicameral_model)

    from safetensors.torch import load_file
    import glob

    shards = glob.glob(os.path.join(diff_path, "*.safetensors"))
    if not shards:
        print(f"[!] Warning: No .safetensors found in {diff_path}")
        print("    Run: python3 scripts/train_manifold_differentiation.py")
    else:
        for shard in shards:
            print(f"    Loading shard: {os.path.basename(shard)}")
            state_dict = load_file(shard)
            bicameral_model.load_state_dict(state_dict, strict=False)

    results, correct, total = run_benchmarks(bicameral_model, tokenizer, "BICAMERAL (GMoE)", is_bicameral=True)

    print("\n" + "=" * 60)
    print("FRACTAL BOTTLENECK VALIDATION REPORT")
    print("=" * 60)

    # Separate results by type
    logic_ids = [r["id"] for r in results if r["type"] == "LOGIC"]
    creative_ids = [r["id"] for r in results if r["type"] == "CREATIVE"]

    avg_logic_id = sum(logic_ids) / len(logic_ids) if logic_ids else 0
    avg_creative_id = sum(creative_ids) / len(creative_ids) if creative_ids else 0

    print(f"\nManifold Polarization:")
    print(f"  Logic Prompts    -> Avg ID: {avg_logic_id:.2f} (Target: < 1.8)")
    print(f"  Creative Prompts -> Avg ID: {avg_creative_id:.2f} (Target: > 1.8)")

    polarization = avg_creative_id - avg_logic_id
    print(f"  Polarization Gap: {polarization:.2f}")

    print(f"\nRouting Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    # Pass/Fail based on Fractal Bottleneck Hypothesis
    if avg_logic_id < 1.8 and avg_creative_id > 1.8:
        print("\n✓ FRACTAL BOTTLENECK HYPOTHESIS: PASSED")
        print("  Logic manifold is fractal-compressed (D < 1.8)")
        print("  Creative manifold is expansive (D > 1.8)")
    else:
        print("\n✗ FRACTAL BOTTLENECK HYPOTHESIS: NEEDS TRAINING")
        print("  Run training to polarize manifolds: python3 scripts/train_manifold_differentiation.py")


if __name__ == "__main__":
    main()

