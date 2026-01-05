from llama_cpp import Llama
import os

model_path = "weights/gguf/SmolLM3-3B-128K-Q4_K_M.gguf"
print(f"Testing load of: {model_path}")

if not os.path.exists(model_path):
    print("File does not exist!")
    exit(1)

try:
    llm = Llama(
        model_path=model_path,
        verbose=True # Enable verbose to see C++ logs
    )
    print("Success!")
except Exception as e:
    print(f"FAILED: {e}")
