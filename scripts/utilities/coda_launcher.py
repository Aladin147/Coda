#!/usr/bin/env python3
"""
Coda Unified System Launcher

Production-ready launcher that orchestrates all Coda components with:
- Proper initialization order and dependency management
- Graceful startup and shutdown procedures
- Health monitoring and automatic recovery
- Configuration validation and environment checks
- Comprehensive logging and error handling
- Signal handling for clean shutdown

Usage:
    python coda_launcher.py [options]
    
Options:
    --config PATH       Configuration file path (default: configs/production.yaml)
    --log-level LEVEL   Logging level (default: INFO)
    --no-dashboard      Disable dashboard server
    --no-websocket      Disable WebSocket server
    --no-voice          Disable voice processing
    --port PORT         Override WebSocket port
    --dashboard-port    Override dashboard port
    --health-check      Run health check and exit
    --dry-run          Validate configuration without starting
"""

import asyncio
import logging
import signal
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import psutil
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging with UTF-8 encoding for Windows compatibility
import sys
import io

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/coda_launcher.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("coda_launcher")

# Import Coda components
try:
    from coda.core.assistant import CodaAssistant
    from coda.core.config import CodaConfig, load_config
    from coda.interfaces.websocket.server import CodaWebSocketServer
    from coda.interfaces.dashboard.server import CodaDashboardServer
    from coda.core.integration import ComponentIntegrationLayer
    from coda.core.websocket_integration import ComponentWebSocketIntegration
    from coda.core.error_management import get_error_manager, handle_error
    from coda.core.component_recovery import get_recovery_manager
    from coda.core.user_error_interface import get_user_error_interface
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    from coda.components.voice.models import VoiceConfig, MoshiConfig, VoiceProcessingMode
    
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    logger.error("Please ensure you're running from the project root with the virtual environment activated")
    sys.exit(1)


class CodaSystemLauncher:
    """
    Unified system launcher for all Coda components.
    
    Manages the complete lifecycle of the Coda system with proper
    initialization order, dependency management, and graceful shutdown.
    """
    
    def __init__(self, config_path: str = "configs/production.yaml"):
        self.config_path = config_path
        self.config: Optional[CodaConfig] = None
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.start_time = time.time()
        
        # Core components
        self.assistant: Optional[CodaAssistant] = None
        self.websocket_server: Optional[CodaWebSocketServer] = None
        self.dashboard_server: Optional[CodaDashboardServer] = None
        self.websocket_integration: Optional[ComponentWebSocketIntegration] = None

        # Error handling and recovery
        self.error_manager = get_error_manager()
        self.recovery_manager = get_recovery_manager()
        self.user_error_interface = get_user_error_interface()
        
        # Component status tracking
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.initialization_order = [
            "config",
            "directories", 
            "assistant",
            "websocket_server",
            "dashboard_server",
            "integrations"
        ]
        
        # Options
        self.enable_dashboard = True
        self.enable_websocket = True
        self.enable_voice = True
        
        logger.info("Coda System Launcher initialized")
        
    def set_options(self, **options):
        """Set launcher options."""
        self.enable_dashboard = options.get("enable_dashboard", True)
        self.enable_websocket = options.get("enable_websocket", True) 
        self.enable_voice = options.get("enable_voice", True)
        
        if hasattr(self, 'config') and self.config:
            # Override config ports if specified
            if "websocket_port" in options:
                self.config.websocket.port = options["websocket_port"]
            if "dashboard_port" in options:
                self.config.dashboard.port = options["dashboard_port"]
                
    async def validate_environment(self) -> bool:
        """Validate system environment and dependencies."""
        logger.info("[VALIDATE] Validating system environment...")
        
        validation_results = {
            "python_version": sys.version_info >= (3, 9),
            "gpu_available": False,
            "disk_space": False,
            "memory_available": False,
            "ollama_accessible": False
        }
        
        try:
            # Check GPU availability
            import torch
            validation_results["gpu_available"] = torch.cuda.is_available()
            if validation_results["gpu_available"]:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"[OK] GPU available: {gpu_name}")
            else:
                logger.warning("[WARN] No GPU available - voice processing will use CPU")
                
        except ImportError:
            logger.warning("[WARN] PyTorch not available - voice processing disabled")
            
        # Check disk space (need at least 5GB free)
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        validation_results["disk_space"] = free_gb >= 5.0
        logger.info(f"[DISK] Disk space: {free_gb:.1f}GB free")

        # Check memory (need at least 8GB total)
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        validation_results["memory_available"] = total_gb >= 8.0
        logger.info(f"[MEMORY] Memory: {total_gb:.1f}GB total, {memory.percent}% used")
        
        # Check Ollama accessibility
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/version", timeout=5) as response:
                    if response.status == 200:
                        validation_results["ollama_accessible"] = True
                        logger.info("[OK] Ollama server accessible")
                    else:
                        logger.warning("[WARN] Ollama server not responding correctly")
        except Exception as e:
            logger.warning(f"[WARN] Ollama server not accessible: {e}")
            
        # Report validation results
        passed = sum(validation_results.values())
        total = len(validation_results)
        logger.info(f"[VALIDATE] Environment validation: {passed}/{total} checks passed")

        if passed < total:
            logger.warning("[WARN] Some environment checks failed - system may not function optimally")
            
        return passed >= 3  # Require at least 3/5 checks to pass
        
    async def load_and_validate_config(self) -> bool:
        """Load and validate configuration."""
        logger.info(f"📝 Loading configuration from {self.config_path}")
        
        try:
            # Check if config file exists
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
                
            # Load configuration
            self.config = load_config(Path(self.config_path))
            
            # Validate required configuration sections
            required_sections = ["llm", "memory"]
            if self.enable_voice:
                required_sections.append("voice")
                
            for section in required_sections:
                if not hasattr(self.config, section):
                    logger.error(f"Missing required configuration section: {section}")
                    return False
                    
            # Validate LLM configuration
            if not hasattr(self.config.llm, 'providers'):
                # Using simple LLM config - check for basic fields
                if not self.config.llm.provider or not self.config.llm.model:
                    logger.error("LLM provider and model must be configured")
                    return False
            else:
                # Using advanced LLM config with providers
                if not self.config.llm.providers:
                    logger.error("No LLM providers configured")
                    return False
                
            # Set defaults for optional sections
            if not hasattr(self.config, "websocket"):
                from coda.core.config import WebSocketConfig
                self.config.websocket = WebSocketConfig()
                
            if not hasattr(self.config, "dashboard"):
                from coda.core.config import DashboardConfig  
                self.config.dashboard = DashboardConfig()
                
            logger.info("✅ Configuration loaded and validated successfully")
            self.component_status["config"] = {"status": "ready", "timestamp": time.time()}
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
            
    async def create_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("📁 Creating necessary directories...")
        
        directories = [
            "logs",
            "data",
            "data/sessions", 
            "data/memory",
            "data/memory/long_term",
            "cache",
            "models"
        ]
        
        try:
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                
            logger.info("✅ Directories created successfully")
            self.component_status["directories"] = {"status": "ready", "timestamp": time.time()}
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
            
    async def initialize_assistant(self) -> bool:
        """Initialize the main Coda Assistant."""
        logger.info("🧠 Initializing Coda Assistant...")
        
        try:
            # Disable components based on options
            if not self.enable_voice:
                self.config.voice = None
                
            if not self.enable_websocket:
                self.config.websocket.enabled = False
                
            # Create and initialize assistant
            self.assistant = CodaAssistant(self.config)
            await self.assistant.initialize()

            logger.info("[OK] Coda Assistant initialized successfully")
            self.component_status["assistant"] = {
                "status": "running",
                "timestamp": time.time(),
                "health": "healthy"
            }
            return True
            
        except Exception as e:
            # Handle error through error management system
            await handle_error(e, "assistant", "initialization")
            logger.error(f"Failed to initialize Coda Assistant: {e}")
            self.component_status["assistant"] = {
                "status": "failed",
                "timestamp": time.time(),
                "error": str(e)
            }
            return False
            
    async def initialize_websocket_server(self) -> bool:
        """Initialize WebSocket server."""
        if not self.enable_websocket:
            logger.info("[SKIP] WebSocket server disabled")
            return True

        # Check if assistant already has a WebSocket server
        if self.assistant and hasattr(self.assistant, 'websocket_server') and self.assistant.websocket_server:
            logger.info("[REUSE] Using existing WebSocket server from assistant")
            self.websocket_server = self.assistant.websocket_server
            self.component_status["websocket_server"] = {
                "status": "running",
                "timestamp": time.time(),
                "url": f"ws://{self.config.websocket.host}:{self.config.websocket.port}",
                "source": "assistant"
            }
            return True

        logger.info("[CREATE] Initializing new WebSocket server...")

        try:
            self.websocket_server = CodaWebSocketServer(
                host=self.config.websocket.host,
                port=self.config.websocket.port
            )

            await self.websocket_server.start()

            logger.info(f"[OK] WebSocket server started at ws://{self.config.websocket.host}:{self.config.websocket.port}")
            self.component_status["websocket_server"] = {
                "status": "running",
                "timestamp": time.time(),
                "url": f"ws://{self.config.websocket.host}:{self.config.websocket.port}",
                "source": "launcher"
            }
            return True

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket server: {e}")
            self.component_status["websocket_server"] = {
                "status": "failed",
                "timestamp": time.time(),
                "error": str(e)
            }
            return False
            
    async def initialize_dashboard_server(self) -> bool:
        """Initialize dashboard server."""
        if not self.enable_dashboard:
            logger.info("[SKIP] Dashboard server disabled")
            return True

        logger.info("[CREATE] Initializing dashboard server...")

        try:
            # Try alternative ports if default is in use
            ports_to_try = [self.config.dashboard.port, 8081, 8082, 8083, 8084]

            for port in ports_to_try:
                try:
                    self.dashboard_server = CodaDashboardServer(
                        host=self.config.dashboard.host,
                        port=port
                    )

                    # Connect dashboard to assistant integration layer
                    if self.assistant:
                        self.dashboard_server.set_integration_layer(self.assistant.integration_layer)

                    await self.dashboard_server.start()

                    dashboard_url = f"http://{self.config.dashboard.host}:{port}"
                    logger.info(f"[OK] Dashboard server started at {dashboard_url}")
                    self.component_status["dashboard_server"] = {
                        "status": "running",
                        "timestamp": time.time(),
                        "url": dashboard_url,
                        "port": port
                    }
                    return True

                except Exception as port_error:
                    if port == ports_to_try[-1]:  # Last port attempt
                        raise port_error
                    logger.warning(f"[WARN] Port {port} unavailable, trying next port...")
                    continue

        except Exception as e:
            logger.error(f"Failed to initialize dashboard server: {e}")
            logger.warning("[WARN] Dashboard server will not be available")
            self.component_status["dashboard_server"] = {
                "status": "failed",
                "timestamp": time.time(),
                "error": str(e)
            }
            # Don't fail the entire system if dashboard fails
            return True

    async def setup_integrations(self) -> bool:
        """Set up component integrations."""
        logger.info("🔗 Setting up component integrations...")

        try:
            # Connect WebSocket integration if both assistant and WebSocket server exist
            if self.assistant and self.websocket_server:
                if self.assistant.websocket_integration:
                    self.assistant.websocket_integration.set_websocket_server(self.websocket_server)
                    await self.assistant.websocket_integration.start()

                    # Connect voice manager if available
                    if self.assistant.voice_manager:
                        self.assistant.websocket_integration.set_voice_manager(self.assistant.voice_manager)

                    logger.info("✅ WebSocket integration connected")

            # Connect dashboard to WebSocket integration
            if self.dashboard_server and self.assistant and self.assistant.websocket_integration:
                self.dashboard_server.set_websocket_integration(self.assistant.websocket_integration)
                logger.info("✅ Dashboard WebSocket integration connected")

            self.component_status["integrations"] = {
                "status": "ready",
                "timestamp": time.time()
            }
            return True

        except Exception as e:
            logger.error(f"Failed to setup integrations: {e}")
            self.component_status["integrations"] = {
                "status": "failed",
                "timestamp": time.time(),
                "error": str(e)
            }
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "overall_health": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "components": {},
            "system_metrics": {}
        }

        # Check component health
        for component, status in self.component_status.items():
            health_status["components"][component] = {
                "status": status.get("status", "unknown"),
                "healthy": status.get("status") in ["ready", "running"],
                "last_update": status.get("timestamp", 0),
                "error": status.get("error")
            }

        # Check system metrics
        try:
            memory = psutil.virtual_memory()
            health_status["system_metrics"] = {
                "memory_percent": memory.percent,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "disk_usage_percent": psutil.disk_usage('.').percent,
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")

        # Determine overall health
        component_health = [comp["healthy"] for comp in health_status["components"].values()]
        if not all(component_health):
            health_status["overall_health"] = "degraded"

        if health_status["system_metrics"].get("memory_percent", 0) > 90:
            health_status["overall_health"] = "warning"

        return health_status

    async def start_system(self) -> bool:
        """Start the complete Coda system with proper initialization order."""
        logger.info("🚀 Starting Coda System...")
        logger.info("=" * 60)

        # Initialize components in proper order
        for step in self.initialization_order:
            logger.info(f"📋 Step: {step}")

            if step == "config":
                if not await self.load_and_validate_config():
                    logger.error("❌ Configuration validation failed")
                    return False

            elif step == "directories":
                if not await self.create_directories():
                    logger.error("❌ Directory creation failed")
                    return False

            elif step == "assistant":
                if not await self.initialize_assistant():
                    logger.error("❌ Assistant initialization failed")
                    return False

            elif step == "websocket_server":
                if not await self.initialize_websocket_server():
                    logger.error("❌ WebSocket server initialization failed")
                    return False

            elif step == "dashboard_server":
                if not await self.initialize_dashboard_server():
                    logger.error("❌ Dashboard server initialization failed")
                    return False

            elif step == "integrations":
                if not await self.setup_integrations():
                    logger.error("❌ Integration setup failed")
                    return False

        self.running = True

        # Print system status
        await self.print_system_status()

        logger.info("🎉 Coda System started successfully!")
        return True

    async def print_system_status(self):
        """Print comprehensive system status."""
        logger.info("=" * 60)
        logger.info("🎉 CODA SYSTEM STATUS")
        logger.info("=" * 60)

        # Core system info
        logger.info(f"🕐 Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
        logger.info(f"⏱️ Uptime: {time.time() - self.start_time:.1f}s")

        # Component status
        for component, status in self.component_status.items():
            status_icon = "✅" if status.get("status") in ["ready", "running"] else "❌"
            logger.info(f"{status_icon} {component.replace('_', ' ').title()}: {status.get('status', 'unknown')}")

            if "url" in status:
                logger.info(f"   🌐 {status['url']}")

        # System capabilities
        logger.info("")
        logger.info("🔧 System Capabilities:")
        logger.info(f"   💬 Text Processing: {'✅' if self.assistant else '❌'}")
        logger.info(f"   🎤 Voice Processing: {'✅' if self.enable_voice and self.assistant and self.assistant.voice_manager else '❌'}")
        logger.info(f"   🌐 WebSocket API: {'✅' if self.websocket_server else '❌'}")
        logger.info(f"   📊 Dashboard: {'✅' if self.dashboard_server else '❌'}")
        logger.info(f"   🧠 Memory System: {'✅' if self.assistant and self.assistant.memory_manager else '❌'}")
        logger.info(f"   🛠️ Tools: {'✅' if self.assistant and self.assistant.tools_manager else '❌'}")

        # Access URLs
        logger.info("")
        logger.info("🌐 Access Points:")
        if self.websocket_server:
            logger.info(f"   WebSocket: ws://{self.config.websocket.host}:{self.config.websocket.port}")
        if self.dashboard_server:
            logger.info(f"   Dashboard: http://{self.config.dashboard.host}:{self.config.dashboard.port}")

        logger.info("=" * 60)

    async def shutdown(self):
        """Gracefully shutdown all components."""
        if not self.running:
            return

        logger.info("🛑 Shutting down Coda System...")
        self.running = False

        # Shutdown in reverse order
        shutdown_order = list(reversed(self.initialization_order))

        for component in shutdown_order:
            try:
                if component == "integrations" and self.assistant and self.assistant.websocket_integration:
                    await self.assistant.websocket_integration.stop()
                    logger.info("✅ WebSocket integration stopped")

                elif component == "dashboard_server" and self.dashboard_server:
                    await self.dashboard_server.stop()
                    logger.info("✅ Dashboard server stopped")

                elif component == "websocket_server" and self.websocket_server:
                    await self.websocket_server.stop()
                    logger.info("✅ WebSocket server stopped")

                elif component == "assistant" and self.assistant:
                    await self.assistant.shutdown()
                    logger.info("✅ Coda Assistant stopped")

            except Exception as e:
                logger.warning(f"Error shutting down {component}: {e}")

        self.shutdown_event.set()
        logger.info("🛑 Coda System shutdown complete")

    async def run(self):
        """Main run loop with health monitoring."""
        try:
            # Validate environment
            if not await self.validate_environment():
                logger.error("❌ Environment validation failed")
                return False

            # Start system
            if not await self.start_system():
                logger.error("❌ System startup failed")
                return False

            # Set up signal handlers
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}")
                asyncio.create_task(self.shutdown())

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Health monitoring loop
            last_health_check = time.time()
            health_check_interval = 30  # seconds

            while self.running:
                try:
                    # Periodic health check
                    if time.time() - last_health_check > health_check_interval:
                        health = await self.health_check()
                        if health["overall_health"] != "healthy":
                            logger.warning(f"⚠️ System health: {health['overall_health']}")
                        last_health_check = time.time()

                    # Wait for shutdown signal or brief sleep
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=5.0)
                        break
                    except asyncio.TimeoutError:
                        continue

                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)

            return True

        except Exception as e:
            logger.error(f"Fatal error in run loop: {e}")
            return False
        finally:
            await self.shutdown()


def create_default_config() -> str:
    """Create a default production configuration file."""
    config_content = """# Coda Production Configuration
# Generated by Coda Launcher

llm:
  providers:
    ollama:
      provider: ollama
      model: "qwen3:30b-a3b"
      host: "http://localhost:11434"
      temperature: 0.7
      max_tokens: 2000
      system_message: "/no_think Respond naturally and helpfully."
  default_provider: "ollama"

memory:
  short_term:
    max_turns: 20
  long_term:
    storage_path: "data/memory/long_term"
    embedding_model: "all-MiniLM-L6-v2"
    device: "cpu"

voice:
  mode: "moshi_only"
  moshi:
    model_path: "kyutai/moshika-pytorch-bf16"
    device: "cuda"
    vram_allocation: "4GB"
    inner_monologue_enabled: true
    enable_streaming: false

websocket:
  enabled: true
  host: "localhost"
  port: 8765

dashboard:
  host: "localhost"
  port: 8080

personality:
  traits:
    helpfulness: 0.9
    creativity: 0.7
    formality: 0.5
    enthusiasm: 0.6
    curiosity: 0.8
"""

    config_path = "configs/production.yaml"
    Path("configs").mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(config_content)

    return config_path


async def main():
    """Main launcher function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Coda Unified System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coda_launcher.py                           # Start with default config
  python coda_launcher.py --config custom.yaml     # Use custom config
  python coda_launcher.py --no-voice --no-dashboard # Start without voice and dashboard
  python coda_launcher.py --health-check           # Run health check only
  python coda_launcher.py --dry-run                # Validate config only
        """
    )

    parser.add_argument("--config", default="configs/production.yaml",
                       help="Configuration file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Disable dashboard server")
    parser.add_argument("--no-websocket", action="store_true",
                       help="Disable WebSocket server")
    parser.add_argument("--no-voice", action="store_true",
                       help="Disable voice processing")
    parser.add_argument("--port", type=int, help="Override WebSocket port")
    parser.add_argument("--dashboard-port", type=int, help="Override dashboard port")
    parser.add_argument("--health-check", action="store_true",
                       help="Run health check and exit")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate configuration without starting")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration file")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Create default config if requested
    if args.create_config:
        config_path = create_default_config()
        print(f"✅ Created default configuration: {config_path}")
        print("Edit the configuration file and run the launcher again.")
        return 0

    # Check if config exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Run with --create-config to create a default configuration")
        return 1

    # Create launcher
    launcher = CodaSystemLauncher(args.config)

    # Set options
    launcher.set_options(
        enable_dashboard=not args.no_dashboard,
        enable_websocket=not args.no_websocket,
        enable_voice=not args.no_voice,
        websocket_port=args.port,
        dashboard_port=args.dashboard_port
    )

    try:
        # Health check mode
        if args.health_check:
            logger.info("🔍 Running health check...")
            if not await launcher.validate_environment():
                logger.error("❌ Environment validation failed")
                return 1
            if not await launcher.load_and_validate_config():
                logger.error("❌ Configuration validation failed")
                return 1
            logger.info("✅ Health check passed")
            return 0

        # Dry run mode
        if args.dry_run:
            logger.info("🧪 Running configuration validation...")
            if not await launcher.validate_environment():
                logger.error("❌ Environment validation failed")
                return 1
            if not await launcher.load_and_validate_config():
                logger.error("❌ Configuration validation failed")
                return 1
            logger.info("✅ Configuration validation passed")
            return 0

        # Normal startup
        logger.info("🚀 Starting Coda System Launcher...")
        success = await launcher.run()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("👋 Launcher interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"💥 Launcher failed: {e}")
        return 1

    def _register_launcher_recovery_handlers(self):
        """Register launcher-specific recovery handlers."""

        # System shutdown handler
        async def system_shutdown():
            logger.critical("Initiating system shutdown due to critical error")
            await self.shutdown()
            return True

        # Register system-level recovery handlers
        self.recovery_manager.register_shutdown_handler("system", system_shutdown)

        # Register error escalation callback
        def error_escalation_callback(user_error):
            logger.critical(f"Error escalated: {user_error.title} - {user_error.message}")
            # In a production system, this could send notifications, emails, etc.

        self.user_error_interface.register_error_callback(error_escalation_callback)

        logger.info("Launcher recovery handlers registered")

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including error information."""
        base_status = {
            "launcher": {
                "running": self.running,
                "uptime": time.time() - self.start_time,
                "components": self.component_status
            }
        }

        if self.assistant:
            try:
                error_status = await self.assistant.get_error_status()
                base_status.update(error_status)
            except Exception as e:
                logger.warning(f"Failed to get error status: {e}")

        return base_status


if __name__ == "__main__":
    exit(asyncio.run(main()))
