import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .modeling_bicameral import convert_to_bicameral, MANIFOLD_TELEMETRY
from safetensors.torch import load_file
import os
import glob

class VirtualBicameralEngine:
    def __init__(self, model_id="HuggingFaceTB/SmolLM3-3B", weights_path="models/checkpoints/smollm3_bicameral_diff"):
        print(f"[*] Initializing Virtual Bicameral Mind ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load Base
        print("    Loading Base Model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

        # Surgery
        print("    Performing Bicameral Surgery...")
        self.model = convert_to_bicameral(self.model)

        # Load Experts
        if os.path.exists(weights_path):
            print(f"    Loading Specialized Manifolds from {weights_path}...")
            shards = glob.glob(os.path.join(weights_path, "*.safetensors"))
            if shards:
                for shard in shards:
                    state_dict = load_file(shard)
                    self.model.load_state_dict(state_dict, strict=False)
            else:
                 print(f"[!] Warning: No .safetensors found in {weights_path}")
        else:
             print(f"[!] Warning: weights path {weights_path} not found. Using untrained experts.")

        self.logic_sys = "You are a formal logic engine. Solve problems step-by-step using first-principles reasoning."
        self.creative_sys = "You are a creative storyteller. Use vivid imagery, metaphors, and fluid narrative styles."

    def route(self, prompt):
        # 1. Forward Pass to trigger Internal Routing
        # The BicameralMLP router will update MANIFOLD_TELEMETRY
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
             # Run forward pass (just embeddings + first layer would be enough but we run all)
             # We interpret the result of the Prefill
             self.model(inputs.input_ids)

        mode = MANIFOLD_TELEMETRY["mode"]
        id_val = MANIFOLD_TELEMETRY["id"]

        return mode, id_val

    def generate(self, prompt):
        mode, id_val = self.route(prompt)

        system_prompt = self.logic_sys if mode == "LOGIC" else self.creative_sys
        temp = 0.0 if mode == "LOGIC" else 0.8

        # Format for SmolLM Instruct
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temp,
                do_sample=(temp > 0),
                repetition_penalty=1.2
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract the assistant part
        return response.split("assistant")[-1].strip(), mode, id_val
