import os
import psutil
import sys
import threading
import time
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static, Label, ProgressBar, RadioSet, RadioButton
from textual.message import Message
from textual.worker import Worker
from textual import work
from textual.reactive import reactive

from textual_plotext import PlotextPlot

# Add parent dir to path to import core
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Try to import engine, handle failure gracefully
try:
    from core.engine_gguf import SmolLMManifoldEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("Error importing engine. Make sure dependencies are installed.")

class ManifoldUpdate(Message):
    """Message sent when manifold telemetry updates."""
    def __init__(self, current_id: float, mode: str):
        self.current_id = current_id
        self.mode = mode
        super().__init__()

class TokenReceived(Message):
    """Message sent when a token is received from the LLM."""
    def __init__(self, text: str, mode: str):
        self.text = text
        self.mode = mode
        super().__init__()

class GenerationComplete(Message):
    """Message sent when generation is finished."""
    pass

class TelemetryPane(Static):
    """Left pane: Manifold Telemetry."""

    current_id = reactive(2.0)
    current_mode = reactive("SYSTEM")

    def compose(self) -> ComposeResult:
        yield Label("INTRINSIC DIMENSION (ID)", classes="header-label")
        yield ProgressBar(total=3.5, show_eta=False, id="id_gauge")
        yield Label("2.00", id="id_value")
        yield Label("PHASE SPACE", classes="header-label")
        yield PlotextPlot(id="phase_plot")

    def on_mount(self):
        self.query_one("#id_gauge").update(total=3.5)
        self.plot = self.query_one(PlotextPlot).plt
        self.plot.title("Manifold Trajectory")
        self.plot.xlabel("Time")
        self.plot.ylabel("Local Dim")
        self.plot.theme("dark")


        # Initialize scatter data
        self.time_steps = []
        self.id_history = []

    def update_telemetry(self, current_id: float, mode: str):
        self.current_id = current_id
        self.current_mode = mode

        # Update Gauge
        bar = self.query_one("#id_gauge")
        bar.progress = current_id

        # Update Value Label
        val_label = self.query_one("#id_value")
        val_label.update(f"{current_id:.2f}")

        # Color Semiotics
        if mode == "LOGIC":
            color = "#00FFFF" # Cyan
            status = "FRACTAL COMPRESSION"
        elif mode == "CREATIVE":
            color = "#FF00FF" # Magenta
            status = "ENTROPIC EXPANSION"
        else:
            color = "#00FF00"
            status = "SYSTEM IDLE"

        val_label.styles.color = color

        # Update Plot
        self.time_steps.append(len(self.time_steps))
        self.id_history.append(current_id)
        if len(self.time_steps) > 50:
            self.time_steps.pop(0)
            self.id_history.pop(0)

        self.plot.clear_data()
        self.plot.scatter(self.time_steps, self.id_history, marker="dot", color=color)
        self.query_one(PlotextPlot).refresh()

class InteractionPane(Static):
    """Center pane: Chat Interaction."""

    def compose(self) -> ComposeResult:
        # Chat history for previous turns
        yield RichLog(id="chat_log", markup=True, wrap=True)
        # Active response area for the current streaming text
        yield Label("", id="active_response_header")
        yield Static("", id="active_response", markup=True)
        yield Input(placeholder="Enter prompt...", id="user_input")

    def append_user_message(self, text: str):
        log = self.query_one("#chat_log")
        log.write(f"[bold yellow]User:[/bold yellow] {text}")

    def update_active_response(self, text: str, mode: str):
        # Update the static widget with the accumulated text
        header = self.query_one("#active_response_header")
        content = self.query_one("#active_response")

        style = "cyan" if mode == "LOGIC" else "magenta"
        font_style = "bold" # Textual doesn't easily support font family switching without CSS classes

        header.update(f"[{style}]Stream ({mode})[/]")
        # We assume 'text' here is the FULL accumulated response, or we manage accumulation in App.
        # Let's manage accumulation in App for simplicity.
        content.update(f"[{style}]{text}[/]")

    def commit_active_response(self):
        # Move active response to log and clear
        content = self.query_one("#active_response")
        text = str(content.renderable).strip() # simplified extraction
        # Actually, extracting from renderable is hard.
        # Better: Pass the full text to this method.
        pass

class ControlPane(Static):
    """Right pane: Controls and System Health."""
    def compose(self) -> ComposeResult:
        yield Label("OVERRIDES", classes="header-label")
        yield RadioSet(
            RadioButton("AUTO", value=True, id="mode_auto"),
            RadioButton("FORCE LOGIC", id="mode_logic"),
            RadioButton("FORCE CREATIVE", id="mode_creative"),
            id="mode_selector"
        )
        yield Label("SYSTEM HEALTH", classes="header-label")
        self.cpu_label = Label("CPU: --%", id="cpu_label")
        self.ram_label = Label("RAM: --GB", id="ram_label")
        self.tps_label = Label("TPS: --", id="tps_label")
        yield self.cpu_label
        yield self.ram_label
        yield self.tps_label

    def on_mount(self):
        self.set_interval(1.0, self.update_health)

    def update_health(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / (1024 ** 3)
        self.cpu_label.update(f"CPU: {cpu:.1f}%")
        self.ram_label.update(f"RAM: {ram:.2f}GB")

    def update_tps(self, tps: float):
        self.tps_label.update(f"TPS: {tps:.1f}")

class BicameralApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 3;
        grid-columns: 25% 50% 25%;
    }

    .header-label {
        background: $primary;
        color: $text;
        padding: 1;
        text-align: center;
        width: 100%;
        text-style: bold;
    }

    #id_gauge {
        width: 100%;
        height: 1;
        margin: 1;
    }

    #id_value {
        text-align: center;
        text-style: bold;
        width: 100%;
        margin-bottom: 2;
    }

    TelemetryPane, InteractionPane, ControlPane {
        height: 100%;
        border: solid green;
        padding: 1;
    }

    #chat_log {
        height: 70%;
        border: solid grey;
    }

    #active_response {
        height: 20%;
        border: dashed cyan;
        overflow-y: scroll;
    }

    #user_input {
        dock: bottom;
    }
    """

    TITLE = "Bicameral Manifold Interface"

    # App state for accumulated response
    current_response_text = reactive("")
    current_response_mode = reactive("SYSTEM")

    def compose(self) -> ComposeResult:
        yield TelemetryPane()
        yield InteractionPane()
        yield ControlPane()
        yield Footer()

    def on_mount(self):
        self.engine = None
        self.load_engine()

    @work(exclusive=True, thread=True)
    def load_engine(self):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights/gguf/SmolLM3-3B-128K-Q4_K_M.gguf")

        log = self.query_one("#chat_log")
        if not os.path.exists(model_path):
            log.write("[bold red]Error: Model file not found! Check weights/gguf/.[/]")
            return

        try:
            self.engine = SmolLMManifoldEngine(model_path)
            log.write("[bold green]System Online. Engine Loaded. Waiting for input...[/]")
        except Exception as e:
            log.write(f"[bold red]Critical Failure Loading Engine: {e}[/]")

    async def on_input_submitted(self, message: Input.Submitted):
        if not self.engine:
            self.query_one("#chat_log").write("[bold red]Engine not ready.[/]")
            return

        prompt = message.value
        self.query_one("#user_input").value = ""
        self.query_one(InteractionPane).append_user_message(prompt)

        # Clear active response mechanism
        self.current_response_text = ""
        self.query_one("#active_response").update("")

        self.run_generation(prompt)

    @work(exclusive=True, thread=True)
    def run_generation(self, prompt: str):
        # Override logic
        radio_set = self.query_one(RadioSet)

        # Determine override mode
        override_mode = None
        if radio_set.pressed_button:
            if radio_set.pressed_button.id == "mode_logic":
                override_mode = "LOGIC"
            elif radio_set.pressed_button.id == "mode_creative":
                override_mode = "CREATIVE"

        try:
            start_time = time.time()
            stream, mode, current_id = self.engine.generate(prompt)

            # Use override mode if set (visual only for now)
            # In a real impl, we'd pass this to engine.generate
            final_mode = override_mode if override_mode else mode

            self.post_message(ManifoldUpdate(current_id, final_mode))

            token_count = 0
            for chunk in stream:
                text = chunk['choices'][0]['text']
                token_count += 1
                self.post_message(TokenReceived(text, final_mode))

            elapsed = time.time() - start_time
            tps = token_count / elapsed if elapsed > 0 else 0.0

            # Update TPS on the control pane (via App method or direct query if careful)
            # Since ControlPane is a widget, we can target it.
            self.app.call_from_thread(self.query_one(ControlPane).update_tps, tps)

            self.post_message(GenerationComplete())

        except Exception as e:
            self.app.call_from_thread(self.query_one("#chat_log").write, f"[bold red]Error: {e}[/]")

    def on_manifold_update(self, message: ManifoldUpdate):
        self.query_one(TelemetryPane).update_telemetry(message.current_id, message.mode)

    def on_token_received(self, message: TokenReceived):
        self.current_response_text += message.text
        self.query_one(InteractionPane).update_active_response(self.current_response_text, message.mode)

    def on_generation_complete(self, message: GenerationComplete):
        # Move text to history
        log = self.query_one("#chat_log")
        style = "cyan" if self.query_one(TelemetryPane).current_mode == "LOGIC" else "magenta"
        log.write(f"[{style}]{self.current_response_text}[/]")

        # Clear active
        self.current_response_text = ""
        self.query_one("#active_response").update("")

if __name__ == "__main__":
    app = BicameralApp()
    app.run()