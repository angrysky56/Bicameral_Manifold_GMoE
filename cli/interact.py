import os
import sys
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import BarColumn, Progress

# Add parent dir to path to import core
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.engine_gguf import SmolLMManifoldEngine

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
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights/gguf/SmolLM3-3B-128K-Q4_K_M.gguf")

    if not os.path.exists(model_path):
        console.print("[bold red]Error:[/] Model file not found. Run scripts/setup_gguf.py first.")
        return

    engine = SmolLMManifoldEngine(model_path)

    history = []

    console.print(Panel("[bold green]Bicameral Manifold GMoE (BitNet Edition)[/]\n[italic]Type 'exit' to quit.[/]", border_style="green"))

    while True:
        user_input = console.input("[bold yellow]User:[/] ")
        if user_input.lower() in ['exit', 'quit']:
            break

        stream, mode, current_id = engine.generate(user_input)

        # Display telemetry
        console.print(make_manifold_meter(current_id, mode))

        console.print("[bold blue]Assistant:[/] ", end="")
        full_response = ""
        for chunk in stream:
            text = chunk['choices'][0]['text']
            full_response += text
            console.print(text, end="")
        console.print("\n")

if __name__ == "__main__":
    main()
