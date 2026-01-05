
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.modeling_bicameral import convert_to_bicameral

def main():
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"

    print(f"[*] Loading Tokenizer and Model ({{model_id}})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 1. Load Original Model
    print("[*] Initializing Original Model...")
    original_model = AutoModelForCausalLM.from_pretrained(model_id)
    original_model.eval()

    # 2. Load Bicameral Model (Surgery)
    print("[*] Initializing Bicameral Model (Architectural Surgery)...")
    bicameral_model = AutoModelForCausalLM.from_pretrained(model_id)
    bicameral_model = convert_to_bicameral(bicameral_model)
    bicameral_model.eval()

    prompts = [
        "User: If I have 3 oranges and buy 5 more, how many do I have?\nAssistant:",
        "User: Tell me a story about a cat in space.\nAssistant:"
    ]

    for p in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {{p}}")

        # Test Original
        inputs = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            output = original_model.generate(**inputs, max_new_tokens=50)
        print(f"\n[ORIGINAL RESPONSE]:\n{{tokenizer.decode(output[0], skip_special_tokens=True).replace(p, '').strip()}}")

        # Test Bicameral
        with torch.no_grad():
            output = bicameral_model.generate(**inputs, max_new_tokens=50)
        print(f"\n[BICAMERAL RESPONSE]:\n{{tokenizer.decode(output[0], skip_special_tokens=True).replace(p, '').strip()}}")

if __name__ == "__main__":
    main()
