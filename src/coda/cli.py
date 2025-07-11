"""
Command-line interface for Coda.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

from . import __version__
from .core.assistant import CodaAssistant
from .core.config import load_config
from .interfaces.dashboard.server import CodaDashboardServer
from .interfaces.websocket.server import CodaWebSocketServer

console = Console()


def setup_logging(level: str = "INFO", rich: bool = True) -> None:
    """Set up logging configuration."""
    log_level = getattr(logging, level.upper())

    if rich:
        handler = RichHandler(console=console, rich_tracebacks=True)
        format_str = "%(message)s"
    else:
        handler = logging.StreamHandler()
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=log_level, format=format_str, handlers=[handler])


async def _start_websocket_dashboard(assistant: CodaAssistant, host: str, port: int) -> None:
    """Start the WebSocket dashboard server."""
    try:
        # Create WebSocket server for real-time events
        websocket_server = CodaWebSocketServer(host=host, port=port)

        # Create dashboard server (HTTP server for the web interface)
        dashboard_port = port + 1  # Use next port for HTTP dashboard
        dashboard_server = CodaDashboardServer(host=host, port=dashboard_port)

        # Connect assistant to WebSocket server for event broadcasting
        if hasattr(assistant, 'set_websocket_server'):
            assistant.set_websocket_server(websocket_server)

        # Start WebSocket server
        await websocket_server.start()
        console.print(f"✅ WebSocket server started at ws://{host}:{port}")

        # Start dashboard HTTP server
        await dashboard_server.start()
        console.print(f"✅ Dashboard available at http://{host}:{dashboard_port}")
        console.print(f"🌐 Open your browser to view the real-time dashboard")

        # Store servers for cleanup
        if not hasattr(assistant, '_dashboard_servers'):
            assistant._dashboard_servers = []
        assistant._dashboard_servers.extend([websocket_server, dashboard_server])

    except Exception as e:
        console.print(f"❌ Failed to start dashboard: {e}", style="bold red")
        raise


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option("--no-rich", is_flag=True, help="Disable rich console output")
def cli(log_level: str, no_rich: bool) -> None:
    """Coda - Core Operations & Digital Assistant"""
    setup_logging(level=log_level, rich=not no_rich)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/default.yaml",
    help="Configuration file path",
)
@click.option("--dashboard", is_flag=True, help="Start with WebSocket dashboard")
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=8765, type=int, help="Port to bind to")
def run(config: Path, dashboard: bool, host: str, port: int) -> None:
    """Start the Coda assistant."""
    console.print(f"🚀 Starting Coda v{__version__}", style="bold green")

    async def _async_run():
        try:
            # Load configuration
            console.print(f"📝 Loading configuration from {config}")
            app_config = load_config(config)

            # Initialize assistant
            console.print("🧠 Initializing Coda assistant...")
            assistant = CodaAssistant(app_config)

            if dashboard:
                console.print(f"🌐 Starting WebSocket dashboard on {host}:{port}")
                await _start_websocket_dashboard(assistant, host, port)

            # Start the main conversation loop
            console.print("🎤 Coda is ready! Start speaking...")
            await assistant.run()

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
        except Exception as e:
            console.print(f"❌ Error: {e}", style="bold red")
            sys.exit(1)

    # Run the async function
    asyncio.run(_async_run())


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/default.yaml",
    help="Configuration file path",
)
def validate(config: Path) -> None:
    """Validate configuration file."""
    console.print(f"🔍 Validating configuration: {config}")

    try:
        app_config = load_config(config)
        console.print("✅ Configuration is valid!", style="bold green")

        # Display key settings
        console.print("\n📋 Configuration Summary:")

        # Safe configuration access with fallbacks
        try:
            voice_engine = getattr(getattr(app_config.voice, 'stt', None), 'engine', 'Unknown')
            console.print(f"  Voice Engine: {voice_engine}")
        except AttributeError:
            console.print("  Voice Engine: Configuration not available")

        try:
            memory_enabled = getattr(getattr(app_config.memory, 'long_term', None), 'enabled', False)
            console.print(f"  Memory: {'Enabled' if memory_enabled else 'Disabled'}")
        except AttributeError:
            console.print("  Memory: Configuration not available")

        try:
            enabled_tools = getattr(app_config.tools, 'enabled_tools', [])
            console.print(f"  Tools: {len(enabled_tools)} enabled")
        except AttributeError:
            console.print("  Tools: Configuration not available")

    except Exception as e:
        console.print(f"❌ Configuration error: {e}", style="bold red")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Display system information."""
    import platform

    import torch

    console.print("ℹ️  Coda System Information", style="bold blue")
    console.print(f"Version: {__version__}")
    console.print(f"Python: {platform.python_version()}")
    console.print(f"Platform: {platform.system()} {platform.release()}")

    # GPU information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        console.print(f"GPU: {gpu_name} ({gpu_count} available)")
    else:
        console.print("GPU: Not available (CPU only)")


@cli.command()
@click.option("--output", "-o", default="configs/custom.yaml", help="Output configuration file")
def init(output: str) -> None:
    """Initialize a new configuration file."""
    output_path = Path(output)

    if output_path.exists():
        if not click.confirm(f"Configuration file {output} already exists. Overwrite?"):
            return

    # Create default configuration
    default_config = """# Coda Configuration
voice:
  stt:
    engine: "whisper"  # Options: whisper, kyutai
    model: "base"
    device: "cuda"  # Options: cpu, cuda
    language: "en"
  
  tts:
    engine: "elevenlabs"  # Options: elevenlabs, kyutai
    voice_id: "default"

memory:
  short_term:
    max_turns: 20
  
  long_term:
    enabled: true
    vector_db: "chroma"
    embedding_model: "all-MiniLM-L6-v2"

personality:
  base_personality: "helpful"
  adaptation_enabled: true
  learning_rate: 0.1

tools:
  enabled_tools:
    - "get_time"
    - "get_date"
    - "tell_joke"
    - "search_memory"

llm:
  model_name: "llama3"
  temperature: 0.7
  max_tokens: 256

logging:
  level: "INFO"
  file: "logs/coda.log"

dashboard:
  enabled: false
  host: "localhost"
  port: 8765
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(default_config)

    console.print(f"✅ Configuration file created: {output}", style="bold green")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
