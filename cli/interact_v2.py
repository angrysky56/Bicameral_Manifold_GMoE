
import sys
import os
from rich.console import Console
from rich.panel import Panel

# Add repo to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.engine_bicameral import VirtualBicameralEngine

console = Console()

def main():
    engine = VirtualBicameralEngine()

    console.print(Panel("[bold green]Bicameral Manifold GMoE v2.0[/]\n[italic]Powered by SmolLM3-3B-Instruct[/]", border_style="green"))

    while True:
        user_input = console.input("[bold yellow]User:[/] ")
        if user_input.lower() in ['exit', 'quit']: break

        response, mode, id_val = engine.generate(user_input)

        color = "cyan" if mode == "LOGIC" else "magenta"
        console.print(Panel(f"Mode: [bold]{mode}[/] | ID: [bold]{id_val:.2f}[/]", border_style=color))
        console.print(f"[bold blue]Assistant:[/] {response}\n")

if __name__ == "__main__":
    main()
