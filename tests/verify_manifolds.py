
import sys
import os

# Add repo to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.engine_gguf import SmolLMManifoldEngine

def test_manifold_switching():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights/gguf/SmolLM3-3B-128K-Q4_K_M.gguf")

    print("--- STARTING MANIFOLD VERIFICATION ---")
    engine = SmolLMManifoldEngine(model_path)

    test_cases = [
        {
            "name": "Logic Manifold (Code)",
            "prompt": "def calculate_fractal_dimension(points): return np.log(len(points))",
            "expected_mode": "LOGIC"
        },
        {
            "name": "Creative Manifold (Story)",
            "prompt": "Imagine a world where the sky is made of liquid amber and the clouds are dreams.",
            "expected_mode": "CREATIVE"
        },
        {
            "name": "Logic Manifold (Math/Proof)",
            "prompt": "Solve the following equation for x: 2x^2 + 5x - 3 = 0. Provide a step-by-step proof.",
            "expected_mode": "LOGIC"
        },
        {
            "name": "Creative Manifold (Poetry)",
            "prompt": "Write a poem about the loneliness of an artificial mind in a dormant server rack.",
            "expected_mode": "CREATIVE"
        }
    ]

    success_count = 0
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print(f"Prompt: {case['prompt'][:50]}...")

        # We only need the ID and Mode, not the actual generation for this test
        current_id = engine.estimate_complexity(case['prompt'])
        mode = "LOGIC" if current_id < 1.85 else "CREATIVE"

        print(f"Result: Mode={mode}, ID={current_id:.2f}")

        if mode == case['expected_mode']:
            print("[PASS]")
            success_count += 1
        else:
            print(f"[FAIL] Expected {case['expected_mode']} but got {mode}")

    print(f"\nVerification Summary: {success_count}/{len(test_cases)} cases passed.")
    return success_count == len(test_cases)

if __name__ == "__main__":
    if test_manifold_switching():
        sys.exit(0)
    else:
        sys.exit(1)
