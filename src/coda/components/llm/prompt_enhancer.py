"""
Prompt enhancer for Coda LLM system.

This module provides prompt enhancement functionality including
personality injection, memory context, and dynamic prompt optimization.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .interfaces import PromptEnhancerInterface
from .models import LLMMessage, MessageRole

logger = logging.getLogger("coda.llm.prompt_enhancer")


class PromptEnhancer(PromptEnhancerInterface):
    """
    Enhances prompts with personality, memory, and context.

    Features:
    - Personality-aware prompt enhancement
    - Memory context injection
    - Dynamic prompt optimization
    - Context-aware formatting
    - Conversation history integration
    """

    def __init__(self):
        """Initialize the prompt enhancer."""
        self._memory_manager: Optional[Any] = None
        self._personality_manager: Optional[Any] = None

        # Base system prompts
        self._base_system_prompt = """You are Coda, an intelligent AI assistant with a dynamic personality and long-term memory. You are helpful, knowledgeable, and adapt your communication style based on context and user preferences.

Key characteristics:
- You have a persistent memory and can remember past conversations
- Your personality evolves based on interactions
- You can use various tools to help users
- You provide thoughtful, contextual responses
- You maintain consistency across conversations"""

        logger.info("PromptEnhancer initialized")

    async def enhance_system_prompt(
        self, base_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance system prompt with personality and context.

        Args:
            base_prompt: Base system prompt
            context: Additional context for enhancement

        Returns:
            Enhanced system prompt
        """
        enhanced_prompt = base_prompt or self._base_system_prompt

        # Add personality context
        if self._personality_manager and context:
            personality_context = await self._get_personality_context(context)
            if personality_context:
                enhanced_prompt += f"\n\nPersonality Context:\n{personality_context}"

        # Add memory context
        if self._memory_manager and context:
            memory_context = await self._get_memory_context(context)
            if memory_context:
                enhanced_prompt += f"\n\nRelevant Memories:\n{memory_context}"

        # Add current context
        if context:
            current_context = self._format_current_context(context)
            if current_context:
                enhanced_prompt += f"\n\nCurrent Context:\n{current_context}"

        # Add timestamp
        enhanced_prompt += f"\n\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return enhanced_prompt

    async def enhance_user_prompt(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance user prompt with context.

        Args:
            prompt: Original user prompt
            context: Additional context

        Returns:
            Enhanced user prompt
        """
        enhanced_prompt = prompt

        # Add relevant memories if available
        if self._memory_manager and context:
            conversation_id = context.get("conversation_id")
            if conversation_id:
                relevant_memories = await self._get_relevant_memories(prompt, conversation_id)
                if relevant_memories:
                    enhanced_prompt = f"[Relevant context: {relevant_memories}]\n\n{prompt}"

        return enhanced_prompt

    async def inject_memory_context(
        self, prompt: str, conversation_id: str, max_memories: int = 5
    ) -> str:
        """
        Inject relevant memories into prompt.

        Args:
            prompt: Original prompt
            conversation_id: Conversation ID for context
            max_memories: Maximum number of memories to include

        Returns:
            Prompt with memory context
        """
        if not self._memory_manager:
            return prompt

        try:
            # Search for relevant memories
            memories = await self._memory_manager.search_memories(
                query=prompt, limit=max_memories, min_relevance=0.3
            )

            if not memories:
                return prompt

            # Format memories
            memory_context = []
            for memory in memories:
                memory_context.append(f"- {memory.content}")

            memory_text = "\n".join(memory_context)

            return f"[Relevant memories:\n{memory_text}]\n\n{prompt}"

        except Exception as e:
            logger.warning(f"Failed to inject memory context: {e}")
            return prompt

    async def inject_personality_context(
        self, prompt: str, personality_state: Dict[str, Any]
    ) -> str:
        """
        Inject personality context into prompt.

        Args:
            prompt: Original prompt
            personality_state: Current personality state

        Returns:
            Prompt with personality context
        """
        try:
            # Extract key personality traits
            traits = []

            if "mood" in personality_state:
                traits.append(f"Current mood: {personality_state['mood']}")

            if "energy_level" in personality_state:
                traits.append(f"Energy level: {personality_state['energy_level']}")

            if "communication_style" in personality_state:
                traits.append(f"Communication style: {personality_state['communication_style']}")

            if "interests" in personality_state:
                interests = personality_state["interests"][:3]  # Top 3 interests
                traits.append(f"Current interests: {', '.join(interests)}")

            if traits:
                personality_text = "\n".join(traits)
                return f"[Personality context:\n{personality_text}]\n\n{prompt}"

            return prompt

        except Exception as e:
            logger.warning(f"Failed to inject personality context: {e}")
            return prompt

    def format_conversation_history(
        self, messages: List[LLMMessage], max_length: int = 2000
    ) -> str:
        """
        Format conversation history for context.

        Args:
            messages: List of conversation messages
            max_length: Maximum length of formatted history

        Returns:
            Formatted conversation history
        """
        if not messages:
            return ""

        # Format messages
        formatted_messages = []
        current_length = 0

        # Process messages from most recent backwards
        for message in reversed(messages):
            if message.role == MessageRole.SYSTEM:
                continue  # Skip system messages in history

            role_name = message.role.value.title()
            formatted_msg = f"{role_name}: {message.content}"

            # Check length limit
            if current_length + len(formatted_msg) > max_length:
                break

            formatted_messages.insert(0, formatted_msg)
            current_length += len(formatted_msg)

        return "\n".join(formatted_messages)

    def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for integration."""
        self._memory_manager = memory_manager
        logger.info("Memory manager set for prompt enhancer")

    def set_personality_manager(self, personality_manager: Any) -> None:
        """Set personality manager for integration."""
        self._personality_manager = personality_manager
        logger.info("Personality manager set for prompt enhancer")

    async def _get_personality_context(self, context: Dict[str, Any]) -> str:
        """Get personality context for prompt enhancement."""
        if not self._personality_manager:
            return ""

        try:
            # Get current personality state
            personality_state = await self._personality_manager.get_current_state()

            # Format personality traits
            traits = []

            if "mood" in personality_state:
                traits.append(f"You are currently feeling {personality_state['mood']}")

            if "energy_level" in personality_state:
                energy = personality_state["energy_level"]
                if energy > 0.7:
                    traits.append("You are energetic and enthusiastic")
                elif energy < 0.3:
                    traits.append("You are calm and thoughtful")

            if "communication_style" in personality_state:
                style = personality_state["communication_style"]
                traits.append(f"Your communication style is {style}")

            if "confidence" in personality_state:
                confidence = personality_state["confidence"]
                if confidence > 0.8:
                    traits.append("You are confident and assertive")
                elif confidence < 0.4:
                    traits.append("You are humble and careful")

            return ". ".join(traits) + "." if traits else ""

        except Exception as e:
            logger.warning(f"Failed to get personality context: {e}")
            return ""

    async def _get_memory_context(self, context: Dict[str, Any]) -> str:
        """Get memory context for prompt enhancement."""
        if not self._memory_manager:
            return ""

        try:
            # Get recent important memories
            memories = await self._memory_manager.search_memories(
                category="preference", limit=3, min_relevance=0.5
            )

            if not memories:
                return ""

            memory_items = []
            for memory in memories:
                memory_items.append(f"- {memory.content}")

            return "\n".join(memory_items)

        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}")
            return ""

    async def _get_relevant_memories(self, prompt: str, conversation_id: str) -> str:
        """Get memories relevant to the current prompt."""
        if not self._memory_manager:
            return ""

        try:
            # Search for relevant memories
            memories = await self._memory_manager.search_memories(
                query=prompt, limit=3, min_relevance=0.4
            )

            if not memories:
                return ""

            memory_items = []
            for memory in memories:
                memory_items.append(memory.content)

            return "; ".join(memory_items)

        except Exception as e:
            logger.warning(f"Failed to get relevant memories: {e}")
            return ""

    def _format_current_context(self, context: Dict[str, Any]) -> str:
        """Format current context information."""
        context_items = []

        if "user_name" in context:
            context_items.append(f"User: {context['user_name']}")

        if "conversation_id" in context:
            context_items.append(f"Conversation ID: {context['conversation_id']}")

        if "session_info" in context:
            session = context["session_info"]
            if "duration" in session:
                context_items.append(f"Session duration: {session['duration']} minutes")

        if "tools_available" in context:
            tools = context["tools_available"]
            if tools:
                context_items.append(f"Available tools: {', '.join(tools[:5])}")

        return "\n".join(context_items) if context_items else ""

    def create_function_calling_prompt(self, available_functions: List[Dict[str, Any]]) -> str:
        """Create a prompt section for function calling."""
        if not available_functions:
            return ""

        function_descriptions = []
        for func in available_functions:
            name = func.get("name", "unknown")
            description = func.get("description", "No description")
            function_descriptions.append(f"- {name}: {description}")

        functions_text = "\n".join(function_descriptions)

        return f"""Available Tools:
{functions_text}

You can use these tools to help answer questions and perform tasks. When you need to use a tool, I will call it for you and provide the results."""

    def optimize_prompt_length(self, prompt: str, max_tokens: int = 4000) -> str:
        """Optimize prompt length to fit within token limits."""
        # Simple optimization: truncate if too long
        estimated_tokens = len(prompt) // 4  # Rough approximation

        if estimated_tokens <= max_tokens:
            return prompt

        # Truncate to approximate token limit
        target_chars = max_tokens * 4
        if len(prompt) > target_chars:
            # Try to truncate at sentence boundaries
            sentences = prompt.split(". ")
            truncated = ""

            for sentence in sentences:
                if len(truncated) + len(sentence) + 2 > target_chars:
                    break
                truncated += sentence + ". "

            return truncated.strip()

        return prompt
