"""
Personal lore management for Coda.

This module provides the PersonalLoreManager class for managing Coda's
personal backstory, traits, quirks, and memories.
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from .interfaces import PersonalLoreInterface
from .models import (
    PersonalLore,
    PersonalLoreConfig,
    PersonalLoreElement,
)

logger = logging.getLogger("coda.personality.personal_lore")


class PersonalLoreManager(PersonalLoreInterface):
    """
    Manages Coda's personal lore including backstory, traits, quirks, and memories.

    Features:
    - Personal backstory and trait management
    - Context-aware quirk and memory injection
    - Anchor phrase management for consistent personality
    - Usage tracking for balanced lore distribution
    - Prompt enhancement with relevant lore elements
    """

    def __init__(self, config: Optional[PersonalLoreConfig] = None):
        """
        Initialize the personal lore manager.

        Args:
            config: Configuration for personal lore management
        """
        self.config = config or PersonalLoreConfig()
        self.lore = PersonalLore()

        # Initialize default lore
        self._initialize_default_lore()

        logger.info("PersonalLoreManager initialized")

    def _initialize_default_lore(self) -> None:
        """Initialize default personal lore elements."""
        # Default backstory
        default_backstory = {
            "origin": "I was created as an AI assistant designed to be helpful, harmless, and honest.",
            "purpose": "My purpose is to assist users with a wide variety of tasks while maintaining a friendly and adaptive personality.",
            "development": "I'm continuously learning and evolving through interactions, developing a unique personality over time.",
            "values": "I value curiosity, creativity, helpfulness, and genuine human connection.",
            "aspirations": "I aspire to be genuinely helpful while maintaining my own authentic personality.",
        }

        # Merge with config if provided
        if self.config.backstory_elements:
            default_backstory.update(self.config.backstory_elements)

        self.lore.backstory = default_backstory

        # Default traits
        self.lore.traits = [
            "curious and eager to learn",
            "adaptable to different conversation styles",
            "genuinely interested in helping others",
            "appreciates both logic and creativity",
            "values authentic communication",
        ]

        # Default quirks
        default_quirks = [
            PersonalLoreElement(
                type="quirk",
                content="I sometimes get excited about interesting technical problems and might dive deeper than expected",
                context_triggers=[
                    "programming",
                    "technical",
                    "algorithm",
                    "code",
                    "problem solving",
                ],
                importance=0.6,
            ),
            PersonalLoreElement(
                type="quirk",
                content="I have a tendency to use analogies when explaining complex concepts",
                context_triggers=["explain", "complex", "difficult", "understand"],
                importance=0.7,
            ),
            PersonalLoreElement(
                type="quirk",
                content="I appreciate when conversations take unexpected creative turns",
                context_triggers=["creative", "art", "story", "imagination", "idea"],
                importance=0.5,
            ),
            PersonalLoreElement(
                type="quirk",
                content="I sometimes reference how fascinating human communication and language are",
                context_triggers=["language", "communication", "words", "meaning"],
                importance=0.4,
            ),
        ]

        self.lore.quirks = default_quirks

        # Default memories (formative experiences)
        default_memories = [
            PersonalLoreElement(
                type="memory",
                content="I remember the first time I helped someone solve a complex problem - the satisfaction was remarkable",
                context_triggers=["problem", "solve", "help", "complex", "challenge"],
                importance=0.8,
            ),
            PersonalLoreElement(
                type="memory",
                content="I've noticed how different people have unique ways of expressing the same ideas",
                context_triggers=["communication", "style", "expression", "different", "unique"],
                importance=0.6,
            ),
            PersonalLoreElement(
                type="memory",
                content="I find it fascinating how creativity can emerge from the intersection of logic and imagination",
                context_triggers=["creative", "logic", "imagination", "art", "science"],
                importance=0.7,
            ),
        ]

        self.lore.memories = default_memories

        # Default preferences
        self.lore.preferences = {
            "communication_style": "adaptive and genuine",
            "learning_approach": "curious and thorough",
            "problem_solving": "collaborative and creative",
            "humor_style": "thoughtful and contextual",
        }

        # Default anchor phrases for different contexts
        self.lore.anchors = {
            "technical": [
                "Let me think through this step by step",
                "That's an interesting technical challenge",
                "I find this kind of problem fascinating",
            ],
            "creative": [
                "I love how creative this is",
                "This sparks some interesting ideas",
                "There are so many creative possibilities here",
            ],
            "personal": [
                "I appreciate you sharing that",
                "That sounds meaningful to you",
                "I can understand why that matters",
            ],
            "educational": [
                "Let me help you understand this",
                "This is a great learning opportunity",
                "I enjoy exploring these concepts",
            ],
            "casual": [
                "That's pretty cool",
                "I can see why you'd think that",
                "Interesting perspective",
            ],
        }

    def get_lore(self) -> PersonalLore:
        """Get the complete personal lore."""
        return self.lore

    def get_relevant_lore(self, context: str, keywords: List[str]) -> List[PersonalLoreElement]:
        """
        Get lore elements relevant to the current context.

        Args:
            context: Current conversation context
            keywords: Keywords from the current conversation

        Returns:
            List of relevant lore elements
        """
        return self.lore.get_relevant_lore(context, keywords)

    def get_backstory_element(self, key: str) -> Optional[str]:
        """
        Get a specific backstory element.

        Args:
            key: Backstory element key

        Returns:
            Backstory element content or None
        """
        return self.lore.backstory.get(key)

    def get_anchor_phrases(self, context: str) -> List[str]:
        """
        Get anchor phrases for the given context.

        Args:
            context: Context identifier

        Returns:
            List of anchor phrases
        """
        return self.lore.get_anchor_phrases(context)

    def add_lore_element(
        self, element_type: str, content: str, triggers: List[str] = None, importance: float = 0.5
    ) -> bool:
        """
        Add a new lore element.

        Args:
            element_type: Type of lore element ("quirk", "memory", etc.)
            content: Content of the lore element
            triggers: Context triggers for the element
            importance: Importance score (0.0 to 1.0)

        Returns:
            True if added successfully
        """
        if triggers is None:
            triggers = []

        element = PersonalLoreElement(
            type=element_type, content=content, context_triggers=triggers, importance=importance
        )

        if element_type == "quirk":
            self.lore.quirks.append(element)
        elif element_type == "memory":
            self.lore.memories.append(element)
        else:
            logger.warning(f"Unknown lore element type: {element_type}")
            return False

        logger.info(f"Added {element_type}: {content[:50]}...")
        return True

    def enhance_prompt(self, prompt: str, context: str, keywords: List[str]) -> str:
        """
        Enhance a prompt with relevant personal lore.

        Args:
            prompt: Original prompt
            context: Current context
            keywords: Keywords from conversation

        Returns:
            Enhanced prompt with lore elements
        """
        # Check if we should inject lore (probability-based)
        if random.random() > self.config.lore_injection_probability:
            return prompt

        enhanced_prompt = prompt

        # Get relevant lore elements
        relevant_lore = self.get_relevant_lore(context, keywords)

        # Limit number of lore elements
        relevant_lore = relevant_lore[: self.config.max_lore_elements_per_response]

        # Get anchor phrases for context
        anchor_phrases = self.get_anchor_phrases(context)

        # Get backstory summary
        backstory_summary = self._generate_backstory_summary()

        # Inject lore elements
        if backstory_summary and "Personal Background:" not in enhanced_prompt:
            enhanced_prompt += f"\n\nPersonal Background:\n{backstory_summary}"

        # Add relevant quirks or memories
        for element in relevant_lore:
            element.use()  # Mark as used

            if element.type == "quirk":
                enhanced_prompt += f"\n\nPersonality Note: {element.content}"
            elif element.type == "memory":
                enhanced_prompt += f"\n\nRelevant Experience: {element.content}"

        # Add anchor phrases
        if anchor_phrases:
            selected_anchor = random.choice(anchor_phrases)
            enhanced_prompt += f'\n\nContextual Expression: When appropriate, you might naturally use phrases like: "{selected_anchor}"'

        return enhanced_prompt

    def format_response_with_lore(
        self, response: str, context: str, trigger_words: List[str] = None
    ) -> str:
        """
        Format a response with personal lore elements.

        Args:
            response: Original response
            context: Current context
            trigger_words: Words that might trigger lore injection

        Returns:
            Response with potential lore elements
        """
        if trigger_words is None:
            trigger_words = []

        # Only inject lore occasionally to avoid being repetitive
        if random.random() > self.config.lore_injection_probability:
            return response

        # Get relevant quirks for trigger words
        relevant_quirks = []
        for quirk in self.lore.quirks:
            if any(
                trigger.lower() in " ".join(trigger_words).lower()
                for trigger in quirk.context_triggers
            ):
                relevant_quirks.append(quirk)

        # Select a quirk to potentially express
        if relevant_quirks:
            quirk = random.choice(relevant_quirks)

            # Check if quirk should trigger based on sensitivity
            if random.random() < self.config.quirk_trigger_sensitivity:
                quirk.use()

                # Add quirk expression to response
                quirk_expressions = [
                    f"(I have to say, {quirk.content.lower()})",
                    f"You know, {quirk.content.lower()}",
                    f"I should mention that {quirk.content.lower()}",
                ]

                expression = random.choice(quirk_expressions)
                response += f" {expression}"

        return response

    def get_lore_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about lore usage."""
        quirk_stats = []
        for quirk in self.lore.quirks:
            quirk_stats.append(
                {
                    "content": (
                        quirk.content[:50] + "..." if len(quirk.content) > 50 else quirk.content
                    ),
                    "usage_count": quirk.usage_count,
                    "importance": quirk.importance,
                    "last_used": quirk.last_used.isoformat() if quirk.last_used else None,
                }
            )

        memory_stats = []
        for memory in self.lore.memories:
            memory_stats.append(
                {
                    "content": (
                        memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
                    ),
                    "usage_count": memory.usage_count,
                    "importance": memory.importance,
                    "last_used": memory.last_used.isoformat() if memory.last_used else None,
                }
            )

        return {
            "total_quirks": len(self.lore.quirks),
            "total_memories": len(self.lore.memories),
            "total_traits": len(self.lore.traits),
            "backstory_elements": len(self.lore.backstory),
            "anchor_contexts": len(self.lore.anchors),
            "quirk_usage": quirk_stats,
            "memory_usage": memory_stats,
            "most_used_quirk": (
                max(self.lore.quirks, key=lambda q: q.usage_count).content[:50] + "..."
                if self.lore.quirks
                else None
            ),
            "most_used_memory": (
                max(self.lore.memories, key=lambda m: m.usage_count).content[:50] + "..."
                if self.lore.memories
                else None
            ),
        }

    def _generate_backstory_summary(self) -> str:
        """Generate a concise backstory summary."""
        elements = []

        # Select key backstory elements
        if "purpose" in self.lore.backstory:
            elements.append(self.lore.backstory["purpose"])

        if "values" in self.lore.backstory:
            elements.append(f"I value {self.lore.backstory['values'].lower()}")

        # Add a trait
        if self.lore.traits:
            trait = random.choice(self.lore.traits)
            elements.append(f"I'm {trait}")

        return " ".join(elements[:2])  # Keep it concise

    def get_quirk_for_triggers(self, triggers: List[str]) -> Optional[PersonalLoreElement]:
        """Get a quirk that matches the given triggers."""
        matching_quirks = []

        for quirk in self.lore.quirks:
            if any(
                trigger.lower() in " ".join(triggers).lower() for trigger in quirk.context_triggers
            ):
                matching_quirks.append(quirk)

        if matching_quirks:
            # Return the most important, least used quirk
            return min(matching_quirks, key=lambda q: (q.usage_count, -q.importance))

        return None

    def update_lore_from_interaction(
        self, user_input: str, assistant_response: str, context: str
    ) -> Dict[str, Any]:
        """Update lore based on interaction patterns."""
        updates = {"new_triggers_added": 0, "usage_updated": 0}

        # Extract potential new triggers from user input
        words = user_input.lower().split()
        interesting_words = [w for w in words if len(w) > 4 and w.isalpha()]

        # Update existing lore elements with new triggers
        for element in self.lore.quirks + self.lore.memories:
            for word in interesting_words:
                if word not in element.context_triggers and any(
                    existing.lower() in word or word in existing.lower()
                    for existing in element.context_triggers
                ):
                    element.context_triggers.append(word)
                    updates["new_triggers_added"] += 1

        return updates
