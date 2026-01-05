import time
import os
try:
    from llama_cpp import Llama
except ImportError:
    print("CRITICAL: Run setup_gguf.py first to install llama-cpp-python")

class SmolLMManifoldEngine:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        print("[*] Loading SmolLM3-3B (GGUF) into RAM...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=131072,
            n_threads=4,
            verbose=False
        )

    def estimate_complexity(self, prompt):
        # Heuristic router
        logic_triggers = ['def ', 'return', '{', '}', 'if ', '=', 'import', 'solve', 'proof']
        logic_score = sum(prompt.count(t) for t in logic_triggers) * 0.5

        creative_triggers = ['story', 'imagine', 'poem', 'describe', 'feel', 'narrative']
        creative_score = sum(prompt.count(t) for t in creative_triggers) * 0.5

        base_id = 2.0
        estimated_id = base_id - logic_score + creative_score
        return max(1.1, min(estimated_id, 3.5))

    def generate(self, prompt):
        current_id = self.estimate_complexity(prompt)
        is_logic = current_id < 1.85

        if is_logic:
            # Logic Mode: High precision, deterministic, reasoning enabled
            params = {
                "temperature": 0.1,
                "top_k": 40,
                "repeat_penalty": 1.2,
                "max_tokens": 2048
            }
            mode = "LOGIC"
            system_instruction = "You are a logic engine. Think step-by-step. /think"
        else:
            # Creative Mode: High entropy, fluid, reasoning disabled
            params = {
                "temperature": 0.8,
                "top_p": 0.95,
                "repeat_penalty": 1.05,
                "max_tokens": 2048
            }
            mode = "CREATIVE"
            system_instruction = "You are a creative storyteller. /no_think"

        # ChatML Template with System Instruction
        formatted_prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        output_stream = self.llm(
            formatted_prompt,
            stream=True,
            stop=["<|im_end|>", "<|endoftext|>"],
            **params
        )

        return output_stream, mode, current_id
