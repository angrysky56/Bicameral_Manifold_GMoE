
import torch
import sys
import os
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress

# Add repo and model weights to path
model_id = "/home/ty/Repositories/ai_workspace/Bicameral_Manifold_GMoE/weights/pytorch"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(model_id)

try:
    from tokenization_bitnet import BitnetTokenizer
except ImportError:
    print("[!] Could not import BitnetTokenizer.")
    sys.exit(1)

from core.modeling_bicameral import convert_to_bicameral, MANIFOLD_TELEMETRY

console = Console()

def make_manifold_meter(current_id, mode):
    color = "cyan" if mode == "LOGIC" else "magenta"
    # Map ID 1.1 - 3.5 to 0-100%
    percentage = ((current_id - 1.1) / (3.5 - 1.1)) * 100

    progress = Progress(
        "[bold " + color + "]" + mode + " MANIFOLD",
        BarColumn(bar_width=40, complete_style=color, finished_style=color),
        "[bold]ID: " + f"{current_id:.2f}"
    )
    progress.add_task("id", total=100, completed=percentage)
    return Panel(progress, title="Geometric Telemetry", border_style=color)

def main():
    diff_path = "/home/ty/Repositories/ai_workspace/Bicameral_Manifold_GMoE/weights/bicameral_differentiated"
    load_path = diff_path if os.path.exists(diff_path) else model_id

    console.print(f"[*] Loading Architecture from {model_id}...")
    tokenizer = BitnetTokenizer.from_pretrained(load_path)
    # Load base architecture
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.float32)
    # Inject Bicameral structure
    model = convert_to_bicameral(model)

    if load_path == diff_path:
        console.print("[*] Loading Differentiated Manifold Weights...")
        # Check for safetensors or bin
        st_path = os.path.join(diff_path, "model.safetensors")
        bin_path = os.path.join(diff_path, "pytorch_model.bin")

        if os.path.exists(st_path):
            state_dict = load_file(st_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            console.print("[!] No weight file found in diff_path. Using base weights.")
            state_dict = None

        if state_dict:
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    console.print(Panel("[bold green]Bicameral Manifold GMoE[/]\n[italic]Differentiated Experts Active[/]", border_style="green"))

    while True:
        user_input = console.input("[bold yellow]User:[/] ")
        if user_input.lower() in ['exit', 'quit']: break

        inputs = tokenizer(user_input, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        # Display telemetry
        console.print(make_manifold_meter(MANIFOLD_TELEMETRY["id"], MANIFOLD_TELEMETRY["mode"]))

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # Clean up response to remove the prompt
        clean_response = response.replace(user_input, "").strip()
        console.print(f"[bold blue]Assistant:[/] {clean_response}\n")

if __name__ == "__main__":
    main()
