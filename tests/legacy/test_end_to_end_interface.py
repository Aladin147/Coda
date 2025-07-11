#!/usr/bin/env python3
"""
End-to-End Testing Interface for Coda System.

This comprehensive testing interface provides:
- Multiple session support and management
- Conversation export/import functionality  
- Performance monitoring and metrics
- Real-world testing scenarios
- Interactive and automated testing capabilities
"""

import asyncio
import json
import logging
import sys
import time
import uuid
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("end_to_end_testing")

# Import components
from coda.core.assistant import CodaAssistant
from coda.core.config import CodaConfig
from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
from coda.components.voice.models import VoiceConfig, MoshiConfig, VoiceProcessingMode


class PerformanceMonitor:
    """Monitor system performance during testing."""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "session_count": 0,
            "message_count": 0,
            "error_count": 0,
            "start_time": None,
            "end_time": None
        }
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.metrics["start_time"] = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        self.metrics["end_time"] = time.time()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("📊 Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Get system metrics
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                
                self.metrics["memory_usage"].append(memory_percent)
                self.metrics["cpu_usage"].append(cpu_percent)
                
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                
    def record_response_time(self, response_time: float):
        """Record a response time measurement."""
        self.metrics["response_times"].append(response_time)
        
    def record_message(self):
        """Record a message processed."""
        self.metrics["message_count"] += 1
        
    def record_error(self):
        """Record an error occurred."""
        self.metrics["error_count"] += 1
        
    def record_session(self):
        """Record a session created."""
        self.metrics["session_count"] += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        duration = (self.metrics["end_time"] or time.time()) - (self.metrics["start_time"] or time.time())
        
        summary = {
            "duration_seconds": duration,
            "total_sessions": self.metrics["session_count"],
            "total_messages": self.metrics["message_count"],
            "total_errors": self.metrics["error_count"],
            "messages_per_second": self.metrics["message_count"] / duration if duration > 0 else 0,
            "error_rate": self.metrics["error_count"] / max(self.metrics["message_count"], 1) * 100
        }
        
        if self.metrics["response_times"]:
            summary["response_times"] = {
                "min": min(self.metrics["response_times"]),
                "max": max(self.metrics["response_times"]),
                "avg": statistics.mean(self.metrics["response_times"]),
                "median": statistics.median(self.metrics["response_times"]),
                "p95": statistics.quantiles(self.metrics["response_times"], n=20)[18] if len(self.metrics["response_times"]) > 20 else max(self.metrics["response_times"])
            }
            
        if self.metrics["memory_usage"]:
            summary["memory_usage"] = {
                "min": min(self.metrics["memory_usage"]),
                "max": max(self.metrics["memory_usage"]),
                "avg": statistics.mean(self.metrics["memory_usage"])
            }
            
        if self.metrics["cpu_usage"]:
            summary["cpu_usage"] = {
                "min": min(self.metrics["cpu_usage"]),
                "max": max(self.metrics["cpu_usage"]),
                "avg": statistics.mean(self.metrics["cpu_usage"])
            }
            
        return summary


class ConversationManager:
    """Manage conversation export/import functionality."""
    
    def __init__(self, export_dir: str = "test_conversations"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
    def export_conversation(self, session_id: str, conversation_data: Dict[str, Any]) -> str:
        """Export a conversation to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{session_id}_{timestamp}.json"
        filepath = self.export_dir / filename
        
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "conversation": conversation_data,
            "metadata": {
                "message_count": len(conversation_data.get("messages", [])),
                "modalities": list(set(msg.get("modality", "text") for msg in conversation_data.get("messages", []))),
                "duration": conversation_data.get("duration", 0)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 Exported conversation {session_id} to {filename}")
        return str(filepath)
        
    def import_conversation(self, filepath: str) -> Dict[str, Any]:
        """Import a conversation from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"📂 Imported conversation {data['session_id']} from {Path(filepath).name}")
        return data
        
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all exported conversations."""
        conversations = []
        for filepath in self.export_dir.glob("conversation_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations.append({
                        "filepath": str(filepath),
                        "session_id": data["session_id"],
                        "exported_at": data["exported_at"],
                        "message_count": data["metadata"]["message_count"],
                        "modalities": data["metadata"]["modalities"]
                    })
            except Exception as e:
                logger.warning(f"Failed to read conversation file {filepath}: {e}")
                
        return sorted(conversations, key=lambda x: x["exported_at"], reverse=True)


class SessionManager:
    """Manage multiple test sessions."""
    
    def __init__(self, assistant: CodaAssistant):
        self.assistant = assistant
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
    async def create_session(self, session_name: str = None) -> str:
        """Create a new test session."""
        session_id = str(uuid.uuid4())
        session_name = session_name or f"test_session_{len(self.sessions) + 1}"
        
        # Start conversation in assistant
        await self.assistant.start_conversation(session_id)
        
        self.sessions[session_id] = {
            "name": session_name,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "status": "active",
            "modalities_used": set(),
            "message_count": 0,
            "last_activity": datetime.now().isoformat()
        }
        
        logger.info(f"🆕 Created session {session_name} ({session_id})")
        return session_id
        
    async def send_text_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Send a text message to a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        start_time = time.time()
        
        try:
            result = await self.assistant.process_text_message(
                message=message,
                session_id=session_id,
                metadata={"test_session": True, "session_name": self.sessions[session_id]["name"]}
            )
            
            response_time = time.time() - start_time
            
            # Record message
            message_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "text",
                "modality": "text",
                "user_input": message,
                "assistant_response": result.get("response", {}).get("content", ""),
                "response_time": response_time,
                "status": result.get("status", "unknown")
            }
            
            self.sessions[session_id]["messages"].append(message_data)
            self.sessions[session_id]["modalities_used"].add("text")
            self.sessions[session_id]["message_count"] += 1
            self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "response": result,
                "response_time": response_time,
                "message_data": message_data
            }
            
        except Exception as e:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "text",
                "modality": "text",
                "user_input": message,
                "error": str(e),
                "response_time": time.time() - start_time,
                "status": "error"
            }
            
            self.sessions[session_id]["messages"].append(error_data)
            
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "message_data": error_data
            }
            
    async def send_voice_message(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Send a voice message to a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        start_time = time.time()
        
        try:
            result = await self.assistant.process_voice_input(
                audio_data=audio_data,
                session_id=session_id,
                metadata={"test_session": True, "session_name": self.sessions[session_id]["name"]}
            )
            
            response_time = time.time() - start_time
            
            # Record message
            message_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "voice",
                "modality": "voice",
                "audio_size": len(audio_data),
                "extracted_text": result.get("response", {}).get("extracted_text", ""),
                "assistant_response": result.get("response", {}).get("content", ""),
                "has_audio_output": result.get("response", {}).get("audio_data") is not None,
                "response_time": response_time,
                "status": result.get("status", "unknown")
            }
            
            self.sessions[session_id]["messages"].append(message_data)
            self.sessions[session_id]["modalities_used"].add("voice")
            self.sessions[session_id]["message_count"] += 1
            self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "response": result,
                "response_time": response_time,
                "message_data": message_data
            }
            
        except Exception as e:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "voice",
                "modality": "voice",
                "audio_size": len(audio_data),
                "error": str(e),
                "response_time": time.time() - start_time,
                "status": "error"
            }
            
            self.sessions[session_id]["messages"].append(error_data)
            
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "message_data": error_data
            }
            
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session."""
        if session_id not in self.sessions:
            return {}
            
        session = self.sessions[session_id]
        messages = session["messages"]
        
        if not messages:
            return {**session, "modalities_used": list(session["modalities_used"])}
            
        response_times = [msg["response_time"] for msg in messages if "response_time" in msg]
        successful_messages = [msg for msg in messages if msg.get("status") != "error"]
        error_messages = [msg for msg in messages if msg.get("status") == "error"]
        
        summary = {
            **session,
            "modalities_used": list(session["modalities_used"]),
            "total_messages": len(messages),
            "successful_messages": len(successful_messages),
            "error_messages": len(error_messages),
            "success_rate": len(successful_messages) / len(messages) * 100 if messages else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "text_messages": len([msg for msg in messages if msg.get("modality") == "text"]),
            "voice_messages": len([msg for msg in messages if msg.get("modality") == "voice"]),
            "is_multimodal": len(session["modalities_used"]) > 1
        }
        
        return summary
        
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with summaries."""
        return [self.get_session_summary(session_id) for session_id in self.sessions.keys()]
        
    async def close_session(self, session_id: str):
        """Close a session."""
        if session_id in self.sessions:
            await self.assistant.end_conversation(session_id)
            self.sessions[session_id]["status"] = "closed"
            self.sessions[session_id]["closed_at"] = datetime.now().isoformat()
            logger.info(f"🔒 Closed session {self.sessions[session_id]['name']} ({session_id})")
            
    async def close_all_sessions(self):
        """Close all active sessions."""
        for session_id in list(self.sessions.keys()):
            if self.sessions[session_id]["status"] == "active":
                await self.close_session(session_id)


class EndToEndTester:
    """Main end-to-end testing interface."""
    
    def __init__(self):
        self.assistant = None
        self.session_manager = None
        self.performance_monitor = PerformanceMonitor()
        self.conversation_manager = ConversationManager()
        
    async def setup(self):
        """Set up the testing environment."""
        logger.info("🔧 Setting up end-to-end testing environment...")
        
        try:
            # Create config
            config = CodaConfig()
            
            # Configure Ollama LLM
            config.llm = LLMConfig(
                providers={
                    "ollama": ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model="qwen3:30b-a3b",
                        host="http://localhost:11434",
                        temperature=0.7,
                        max_tokens=100,
                        system_message="/no_think Respond naturally and briefly."
                    )
                },
                default_provider="ollama"
            )
            
            # Configure Moshi voice processing
            config.voice = VoiceConfig(
                mode=VoiceProcessingMode.MOSHI_ONLY,
                moshi=MoshiConfig(
                    model_path="kyutai/moshika-pytorch-bf16",
                    device="cuda" if True else "cpu",
                    vram_allocation="4GB",
                    inner_monologue_enabled=True,
                    enable_streaming=False
                )
            )
            
            # Create assistant
            self.assistant = CodaAssistant(config)
            await self.assistant.initialize()
            
            # Create session manager
            self.session_manager = SessionManager(self.assistant)
            
            logger.info("✅ End-to-end testing environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup testing environment: {e}")
            return False
            
    async def cleanup(self):
        """Clean up the testing environment."""
        logger.info("🧹 Cleaning up testing environment...")
        
        try:
            if self.session_manager:
                await self.session_manager.close_all_sessions()
            if self.assistant:
                await self.assistant.shutdown()
                
            logger.info("✅ Cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def generate_test_audio(self, duration_seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
        """Generate synthetic test audio data."""
        import numpy as np
        import wave
        import io

        # Generate a simple sine wave as test audio
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume

        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        wav_buffer.seek(0)
        return wav_buffer.read()

    async def run_text_only_test(self, num_messages: int = 5) -> Dict[str, Any]:
        """Run a text-only conversation test."""
        logger.info(f"🧪 Running text-only test with {num_messages} messages...")

        session_id = await self.session_manager.create_session("text_only_test")

        test_messages = [
            "Hello, how are you today?",
            "What's the weather like?",
            "Can you help me with a math problem?",
            "Tell me a short joke.",
            "What's your favorite color?",
            "How do you process information?",
            "What can you help me with?",
            "Explain quantum computing briefly.",
            "What's the capital of France?",
            "Thank you for the conversation!"
        ]

        results = []
        for i in range(min(num_messages, len(test_messages))):
            message = test_messages[i]
            result = await self.session_manager.send_text_message(session_id, message)
            results.append(result)

            self.performance_monitor.record_message()
            if result["success"]:
                self.performance_monitor.record_response_time(result["response_time"])
            else:
                self.performance_monitor.record_error()

            # Small delay between messages
            await asyncio.sleep(0.5)

        summary = self.session_manager.get_session_summary(session_id)
        await self.session_manager.close_session(session_id)

        return {
            "test_type": "text_only",
            "session_summary": summary,
            "results": results,
            "success": summary["success_rate"] > 80
        }

    async def run_voice_only_test(self, num_messages: int = 3) -> Dict[str, Any]:
        """Run a voice-only conversation test."""
        logger.info(f"🧪 Running voice-only test with {num_messages} messages...")

        session_id = await self.session_manager.create_session("voice_only_test")

        results = []
        for i in range(num_messages):
            # Generate test audio
            audio_data = self.generate_test_audio(duration_seconds=0.5 + i * 0.2)
            result = await self.session_manager.send_voice_message(session_id, audio_data)
            results.append(result)

            self.performance_monitor.record_message()
            if result["success"]:
                self.performance_monitor.record_response_time(result["response_time"])
            else:
                self.performance_monitor.record_error()

            # Longer delay for voice processing
            await asyncio.sleep(1.0)

        summary = self.session_manager.get_session_summary(session_id)
        await self.session_manager.close_session(session_id)

        return {
            "test_type": "voice_only",
            "session_summary": summary,
            "results": results,
            "success": summary["success_rate"] > 60  # Lower threshold for voice
        }

    async def run_multimodal_test(self, num_exchanges: int = 3) -> Dict[str, Any]:
        """Run a mixed text/voice conversation test."""
        logger.info(f"🧪 Running multimodal test with {num_exchanges} exchanges...")

        session_id = await self.session_manager.create_session("multimodal_test")

        results = []
        for i in range(num_exchanges):
            # Alternate between text and voice
            if i % 2 == 0:
                # Text message
                message = f"This is text message {i + 1}. How are you?"
                result = await self.session_manager.send_text_message(session_id, message)
            else:
                # Voice message
                audio_data = self.generate_test_audio(duration_seconds=0.5)
                result = await self.session_manager.send_voice_message(session_id, audio_data)

            results.append(result)

            self.performance_monitor.record_message()
            if result["success"]:
                self.performance_monitor.record_response_time(result["response_time"])
            else:
                self.performance_monitor.record_error()

            await asyncio.sleep(0.8)

        summary = self.session_manager.get_session_summary(session_id)
        await self.session_manager.close_session(session_id)

        return {
            "test_type": "multimodal",
            "session_summary": summary,
            "results": results,
            "success": summary["success_rate"] > 70 and summary["is_multimodal"]
        }

    async def run_concurrent_sessions_test(self, num_sessions: int = 3, messages_per_session: int = 3) -> Dict[str, Any]:
        """Run concurrent sessions test."""
        logger.info(f"🧪 Running concurrent sessions test with {num_sessions} sessions...")

        # Create multiple sessions
        session_ids = []
        for i in range(num_sessions):
            session_id = await self.session_manager.create_session(f"concurrent_test_{i + 1}")
            session_ids.append(session_id)
            self.performance_monitor.record_session()

        # Send messages concurrently
        async def send_messages_to_session(session_id: str, session_index: int):
            results = []
            for msg_index in range(messages_per_session):
                message = f"Session {session_index + 1}, message {msg_index + 1}"
                result = await self.session_manager.send_text_message(session_id, message)
                results.append(result)

                self.performance_monitor.record_message()
                if result["success"]:
                    self.performance_monitor.record_response_time(result["response_time"])
                else:
                    self.performance_monitor.record_error()

                await asyncio.sleep(0.3)
            return results

        # Run all sessions concurrently
        tasks = [
            send_messages_to_session(session_id, i)
            for i, session_id in enumerate(session_ids)
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Get summaries and close sessions
        summaries = []
        for session_id in session_ids:
            summary = self.session_manager.get_session_summary(session_id)
            summaries.append(summary)
            await self.session_manager.close_session(session_id)

        # Calculate overall success
        total_success_rate = sum(s["success_rate"] for s in summaries) / len(summaries)

        return {
            "test_type": "concurrent_sessions",
            "num_sessions": num_sessions,
            "session_summaries": summaries,
            "all_results": all_results,
            "overall_success_rate": total_success_rate,
            "success": total_success_rate > 75
        }

    async def run_session_switching_test(self) -> Dict[str, Any]:
        """Test session switching functionality."""
        logger.info("🧪 Running session switching test...")

        # Create two sessions
        session1_id = await self.session_manager.create_session("switch_test_1")
        session2_id = await self.session_manager.create_session("switch_test_2")

        results = []

        # Send message to session 1
        result1 = await self.session_manager.send_text_message(session1_id, "Hello from session 1")
        results.append(("session1", result1))

        # Send message to session 2
        result2 = await self.session_manager.send_text_message(session2_id, "Hello from session 2")
        results.append(("session2", result2))

        # Switch back to session 1
        result3 = await self.session_manager.send_text_message(session1_id, "Back to session 1")
        results.append(("session1", result3))

        # Switch back to session 2
        result4 = await self.session_manager.send_text_message(session2_id, "Back to session 2")
        results.append(("session2", result4))

        # Record metrics
        for _, result in results:
            self.performance_monitor.record_message()
            if result["success"]:
                self.performance_monitor.record_response_time(result["response_time"])
            else:
                self.performance_monitor.record_error()

        # Get summaries
        summary1 = self.session_manager.get_session_summary(session1_id)
        summary2 = self.session_manager.get_session_summary(session2_id)

        await self.session_manager.close_session(session1_id)
        await self.session_manager.close_session(session2_id)

        success = all(result["success"] for _, result in results)

        return {
            "test_type": "session_switching",
            "session1_summary": summary1,
            "session2_summary": summary2,
            "results": results,
            "success": success
        }

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        logger.info("🚀 Starting comprehensive end-to-end test suite...")
        logger.info("=" * 80)

        # Start performance monitoring
        self.performance_monitor.start_monitoring()

        test_results = {}
        overall_start_time = time.time()

        # Test scenarios
        test_scenarios = [
            ("Text-Only Conversation", self.run_text_only_test, {"num_messages": 5}),
            ("Voice-Only Conversation", self.run_voice_only_test, {"num_messages": 3}),
            ("Multimodal Conversation", self.run_multimodal_test, {"num_exchanges": 4}),
            ("Concurrent Sessions", self.run_concurrent_sessions_test, {"num_sessions": 3, "messages_per_session": 3}),
            ("Session Switching", self.run_session_switching_test, {})
        ]

        passed_tests = 0
        total_tests = len(test_scenarios)

        for test_name, test_func, test_kwargs in test_scenarios:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")

            try:
                start_time = time.time()
                result = await test_func(**test_kwargs)
                duration = time.time() - start_time

                result["duration"] = duration
                test_results[test_name] = result

                if result.get("success", False):
                    logger.info(f"✅ PASS: {test_name} ({duration:.2f}s)")
                    passed_tests += 1
                else:
                    logger.error(f"❌ FAIL: {test_name} ({duration:.2f}s)")

            except Exception as e:
                logger.error(f"❌ ERROR: {test_name} - {e}")
                test_results[test_name] = {
                    "test_type": test_name.lower().replace(" ", "_"),
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - start_time
                }

            # Small delay between tests
            await asyncio.sleep(2)

        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()

        # Calculate overall results
        overall_duration = time.time() - overall_start_time
        success_rate = (passed_tests / total_tests) * 100
        performance_summary = self.performance_monitor.get_summary()

        # Create comprehensive report
        comprehensive_report = {
            "test_suite": "End-to-End Comprehensive Testing",
            "timestamp": datetime.now().isoformat(),
            "overall_results": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "overall_duration": overall_duration,
                "overall_success": success_rate >= 80
            },
            "test_results": test_results,
            "performance_metrics": performance_summary,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total / (1024**3)  # GB
            }
        }

        # Export comprehensive report
        report_file = self.conversation_manager.export_dir / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 COMPREHENSIVE TEST SUITE RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Duration: {overall_duration:.2f}s")
        logger.info(f"Messages Processed: {performance_summary.get('total_messages', 0)}")
        logger.info(f"Average Response Time: {performance_summary.get('response_times', {}).get('avg', 0):.3f}s")
        logger.info(f"Error Rate: {performance_summary.get('error_rate', 0):.1f}%")

        if comprehensive_report["overall_results"]["overall_success"]:
            logger.info("🎉 COMPREHENSIVE TEST SUITE PASSED!")
        else:
            logger.error("❌ COMPREHENSIVE TEST SUITE FAILED")

        logger.info(f"📄 Full report saved to: {report_file}")

        return comprehensive_report

    async def export_all_conversations(self) -> List[str]:
        """Export all current conversations."""
        exported_files = []

        for session_id in self.session_manager.sessions:
            session_data = self.session_manager.sessions[session_id]
            conversation_data = {
                "session_id": session_id,
                "session_info": session_data,
                "messages": session_data["messages"],
                "summary": self.session_manager.get_session_summary(session_id)
            }

            filepath = self.conversation_manager.export_conversation(session_id, conversation_data)
            exported_files.append(filepath)

        logger.info(f"💾 Exported {len(exported_files)} conversations")
        return exported_files

    async def interactive_mode(self):
        """Run interactive testing mode."""
        logger.info("🎮 Starting interactive testing mode...")
        logger.info("Available commands:")
        logger.info("  1. run_text_test - Run text-only test")
        logger.info("  2. run_voice_test - Run voice-only test")
        logger.info("  3. run_multimodal_test - Run multimodal test")
        logger.info("  4. run_concurrent_test - Run concurrent sessions test")
        logger.info("  5. run_switching_test - Run session switching test")
        logger.info("  6. run_full_suite - Run comprehensive test suite")
        logger.info("  7. list_sessions - List active sessions")
        logger.info("  8. export_conversations - Export all conversations")
        logger.info("  9. show_performance - Show performance metrics")
        logger.info("  0. exit - Exit interactive mode")

        while True:
            try:
                command = input("\n🎮 Enter command (1-9, 0 to exit): ").strip()

                if command == "0" or command.lower() == "exit":
                    break
                elif command == "1":
                    result = await self.run_text_only_test()
                    print(f"Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
                elif command == "2":
                    result = await self.run_voice_only_test()
                    print(f"Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
                elif command == "3":
                    result = await self.run_multimodal_test()
                    print(f"Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
                elif command == "4":
                    result = await self.run_concurrent_sessions_test()
                    print(f"Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
                elif command == "5":
                    result = await self.run_session_switching_test()
                    print(f"Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
                elif command == "6":
                    await self.run_comprehensive_test_suite()
                elif command == "7":
                    sessions = self.session_manager.list_sessions()
                    print(f"Active sessions: {len(sessions)}")
                    for session in sessions:
                        print(f"  - {session['name']}: {session['message_count']} messages")
                elif command == "8":
                    files = await self.export_all_conversations()
                    print(f"Exported {len(files)} conversations")
                elif command == "9":
                    summary = self.performance_monitor.get_summary()
                    print(f"Performance Summary:")
                    print(f"  Messages: {summary.get('total_messages', 0)}")
                    print(f"  Avg Response Time: {summary.get('response_times', {}).get('avg', 0):.3f}s")
                    print(f"  Error Rate: {summary.get('error_rate', 0):.1f}%")
                else:
                    print("❌ Invalid command. Please enter 1-9 or 0 to exit.")

            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="End-to-End Testing Interface for Coda System")
    parser.add_argument("--mode", choices=["comprehensive", "interactive", "text", "voice", "multimodal"],
                       default="comprehensive", help="Testing mode")
    parser.add_argument("--export", action="store_true", help="Export conversations after testing")

    args = parser.parse_args()

    tester = EndToEndTester()

    try:
        # Setup
        if not await tester.setup():
            return 1

        # Wait for initialization
        await asyncio.sleep(3)

        # Run tests based on mode
        if args.mode == "comprehensive":
            result = await tester.run_comprehensive_test_suite()
            success = result["overall_results"]["overall_success"]
        elif args.mode == "interactive":
            await tester.interactive_mode()
            success = True
        elif args.mode == "text":
            result = await tester.run_text_only_test()
            success = result["success"]
        elif args.mode == "voice":
            result = await tester.run_voice_only_test()
            success = result["success"]
        elif args.mode == "multimodal":
            result = await tester.run_multimodal_test()
            success = result["success"]
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1

        # Export conversations if requested
        if args.export:
            await tester.export_all_conversations()

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("👋 Testing interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        return 1
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    exit(asyncio.run(main()))
