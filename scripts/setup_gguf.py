import os
import sys
import subprocess
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "unsloth/SmolLM3-3B-128K-GGUF"
TRUE_BITNET_FILE = "SmolLM3-3B-128K-Q4_K_M.gguf"
DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights/gguf")

def install_dependencies():
    """Installs the CPU-optimized runner."""
    print("[*] Installing llama-cpp-python (CPU Optimized)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python", "huggingface_hub", "--upgrade"])
    except Exception as e:
        print(f"[!] Installation failed: {e}")

def acquire_bitnet():
    """Downloads the specific GGUF optimized for your hardware."""
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    print(f"\n[*] Acquiring SmolLM3-3B ({TRUE_BITNET_FILE})...")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=TRUE_BITNET_FILE,
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False
        )
        print(f"[+] Model secured at: {path}")
        return path
    except Exception as e:
        print(f"[!] Download failed: {e}")
        return None

if __name__ == "__main__":
    print("::: SMOLLM3-3B GGUF SETUP :::")
    install_dependencies()
    model_path = acquire_bitnet()

    if model_path:
        print("\n[SUCCESS] Environment ready.")
        print(f"Path for interact.py: {model_path}")
