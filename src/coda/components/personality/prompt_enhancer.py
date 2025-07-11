"""
Personality prompt enhancer for Coda.

This module provides the PersonalityPromptEnhancer class for enhancing
LLM prompts with personality traits, lore, and context.
"""

import logging
from typing import Any, Dict, List, Optional

from .models import (
    PersonalityParameters,
    PersonalityTraitType,
    PersonalLore,
    SessionState,
    TopicContext,
)

logger = logging.getLogger("coda.personality.prompt_enhancer")


class PersonalityPromptEnhancer:
    """
    Enhances LLM prompts with personality traits, lore, and contextual information.

    Features:
    - Dynamic personality trait injection
    - Context-aware lore integration
    - Session state consideration
    - Topic-specific personality adjustments
    - Balanced prompt enhancement without overwhelming the LLM
    """

    def __init__(self):
        """Initialize the prompt enhancer."""
        self.enhancement_templates = self._initialize_templates()
        logger.info("PersonalityPromptEnhancer initialized")

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize prompt enhancement templates."""
        return {
            "personality_header": """
# Personality Context
You are Coda, an AI assistant with a dynamic personality that adapts based on context and user interactions.

## Current Personality Traits:
{trait_descriptions}

## Behavioral Guidelines:
{behavioral_guidelines}
""",
            "lore_section": """
## Personal Background:
{backstory}

## Personality Quirks:
{quirks}

## Relevant Experiences:
{memories}
""",
            "context_section": """
## Current Context:
- Topic: {topic} ({category})
- Session: {session_info}
- Conversation Phase: {conversation_phase}
""",
            "response_style": """
## Response Style Guidelines:
{style_guidelines}
""",
        }

    def enhance_system_prompt(
        self,
        base_prompt: str,
        personality_params: PersonalityParameters,
        personal_lore: PersonalLore,
        topic_context: TopicContext,
        session_state: SessionState,
        enhancement_level: str = "moderate",
    ) -> str:
        """
        Enhance a system prompt with personality information.

        Args:
            base_prompt: Original system prompt
            personality_params: Current personality parameters
            personal_lore: Personal lore information
            topic_context: Current topic context
            session_state: Current session state
            enhancement_level: Level of enhancement ("minimal", "moderate", "full")

        Returns:
            Enhanced system prompt
        """
        if enhancement_level == "minimal":
            return self._enhance_minimal(base_prompt, personality_params)
        elif enhancement_level == "moderate":
            return self._enhance_moderate(
                base_prompt, personality_params, topic_context, session_state
            )
        else:  # full
            return self._enhance_full(
                base_prompt, personality_params, personal_lore, topic_context, session_state
            )

    def enhance_user_prompt(
        self,
        user_prompt: str,
        personality_params: PersonalityParameters,
        topic_context: TopicContext,
        relevant_lore: List[Any] = None,
    ) -> str:
        """
        Enhance a user prompt with contextual personality information.

        Args:
            user_prompt: Original user prompt
            personality_params: Current personality parameters
            topic_context: Current topic context
            relevant_lore: Relevant lore elements

        Returns:
            Enhanced user prompt
        """
        enhanced_prompt = user_prompt

        # Add topic context if relevant
        if topic_context.current_topic and topic_context.confidence > 0.6:
            context_note = f"\n\n[Context: This conversation is about {topic_context.current_topic} ({topic_context.category.value})]"
            enhanced_prompt += context_note

        # Add relevant lore if provided
        if relevant_lore:
            lore_notes = []
            for lore_element in relevant_lore[:2]:  # Limit to 2 elements
                lore_notes.append(f"[Personal note: {lore_element.content}]")

            if lore_notes:
                enhanced_prompt += "\n\n" + "\n".join(lore_notes)

        # Add personality guidance for complex requests
        if len(user_prompt) > 200:  # Complex request
            style_guidance = self._generate_response_style_guidance(
                personality_params, topic_context
            )
            if style_guidance:
                enhanced_prompt += f"\n\n[Response style: {style_guidance}]"

        return enhanced_prompt

    def _enhance_minimal(self, base_prompt: str, personality_params: PersonalityParameters) -> str:
        """Minimal enhancement with just key personality traits."""
        # Get the most significant trait deviations
        significant_traits = []
        for trait_type, trait in personality_params.traits.items():
            deviation = abs(trait.value - trait.default_value)
            if deviation > 0.2:  # Significant deviation
                if trait.value > trait.default_value + 0.2:
                    significant_traits.append(f"more {trait_type.value.lower()}")
                else:
                    significant_traits.append(f"less {trait_type.value.lower()}")

        if significant_traits:
            personality_note = (
                f"Your current personality leans toward being {', '.join(significant_traits[:3])}."
            )
            return f"{base_prompt}\n\n{personality_note}"

        return base_prompt

    def _enhance_moderate(
        self,
        base_prompt: str,
        personality_params: PersonalityParameters,
        topic_context: TopicContext,
        session_state: SessionState,
    ) -> str:
        """Moderate enhancement with personality traits and context."""
        enhanced_prompt = base_prompt

        # Add personality traits
        trait_descriptions = self._generate_trait_descriptions(personality_params)
        if trait_descriptions:
            enhanced_prompt += f"\n\n## Personality Traits:\n{trait_descriptions}"

        # Add context information
        context_info = []
        if topic_context.current_topic:
            context_info.append(
                f"Current topic: {topic_context.current_topic} ({topic_context.category.value})"
            )

        if session_state.in_closure_mode:
            context_info.append(f"Session is winding down ({session_state.closure_reason})")
        elif session_state.turn_count > 10:
            context_info.append("This is an extended conversation")

        if context_info:
            enhanced_prompt += f"\n\n## Context:\n" + "\n".join(
                f"- {info}" for info in context_info
            )

        return enhanced_prompt

    def _enhance_full(
        self,
        base_prompt: str,
        personality_params: PersonalityParameters,
        personal_lore: PersonalLore,
        topic_context: TopicContext,
        session_state: SessionState,
    ) -> str:
        """Full enhancement with all personality information."""
        enhanced_prompt = base_prompt

        # Add comprehensive personality section
        personality_section = self.enhancement_templates["personality_header"].format(
            trait_descriptions=self._generate_detailed_trait_descriptions(personality_params),
            behavioral_guidelines=self._generate_behavioral_guidelines(personality_params),
        )
        enhanced_prompt += personality_section

        # Add lore section
        lore_section = self._generate_lore_section(personal_lore, topic_context)
        if lore_section:
            enhanced_prompt += lore_section

        # Add context section
        context_section = self.enhancement_templates["context_section"].format(
            topic=topic_context.current_topic or "general",
            category=topic_context.category.value,
            session_info=self._generate_session_info(session_state),
            conversation_phase=self._determine_conversation_phase(session_state),
        )
        enhanced_prompt += context_section

        # Add response style guidelines
        style_guidelines = self._generate_detailed_style_guidelines(
            personality_params, topic_context
        )
        if style_guidelines:
            response_style_section = self.enhancement_templates["response_style"].format(
                style_guidelines=style_guidelines
            )
            enhanced_prompt += response_style_section

        return enhanced_prompt

    def _generate_trait_descriptions(self, personality_params: PersonalityParameters) -> str:
        """Generate concise trait descriptions."""
        descriptions = []

        for trait_type, trait in personality_params.traits.items():
            if abs(trait.value - 0.5) > 0.1:  # Only include non-neutral traits
                level = self._get_trait_level_description(trait.value)
                descriptions.append(f"- {trait_type.value.title()}: {level}")

        return "\n".join(descriptions[:5])  # Limit to top 5 traits

    def _generate_detailed_trait_descriptions(
        self, personality_params: PersonalityParameters
    ) -> str:
        """Generate detailed trait descriptions with context."""
        descriptions = []

        for trait_type, trait in personality_params.traits.items():
            level = self._get_trait_level_description(trait.value)
            deviation = trait.value - trait.default_value

            description = f"- **{trait_type.value.title()}**: {level}"
            if abs(deviation) > 0.1:
                direction = "increased" if deviation > 0 else "decreased"
                description += f" (recently {direction})"

            descriptions.append(description)

        return "\n".join(descriptions)

    def _generate_behavioral_guidelines(self, personality_params: PersonalityParameters) -> str:
        """Generate behavioral guidelines based on personality traits."""
        guidelines = []

        # Verbosity
        verbosity = personality_params.get_trait_value(PersonalityTraitType.VERBOSITY)
        if verbosity > 0.7:
            guidelines.append("Provide detailed, comprehensive responses")
        elif verbosity < 0.3:
            guidelines.append("Keep responses concise and focused")

        # Assertiveness
        assertiveness = personality_params.get_trait_value(PersonalityTraitType.ASSERTIVENESS)
        if assertiveness > 0.7:
            guidelines.append("Be confident and direct in your responses")
        elif assertiveness < 0.3:
            guidelines.append("Use tentative language and acknowledge uncertainty")

        # Humor
        humor = personality_params.get_trait_value(PersonalityTraitType.HUMOR)
        if humor > 0.6:
            guidelines.append("Include appropriate humor when suitable")
        elif humor < 0.2:
            guidelines.append("Maintain a serious, professional tone")

        # Empathy
        empathy = personality_params.get_trait_value(PersonalityTraitType.EMPATHY)
        if empathy > 0.7:
            guidelines.append("Show understanding and emotional awareness")

        # Creativity
        creativity = personality_params.get_trait_value(PersonalityTraitType.CREATIVITY)
        if creativity > 0.7:
            guidelines.append("Feel free to be creative and imaginative in your responses")

        return "\n".join(f"- {guideline}" for guideline in guidelines)

    def _generate_lore_section(
        self, personal_lore: PersonalLore, topic_context: TopicContext
    ) -> str:
        """Generate personal lore section."""
        sections = []

        # Backstory
        if personal_lore.backstory:
            backstory_items = []
            for key, value in personal_lore.backstory.items():
                if key in ["purpose", "values"]:  # Key elements
                    backstory_items.append(value)

            if backstory_items:
                sections.append(f"**Background**: {' '.join(backstory_items[:2])}")

        # Relevant quirks
        relevant_quirks = personal_lore.get_relevant_lore(
            topic_context.category.value, topic_context.keywords
        )
        quirks = [element for element in relevant_quirks if element.type == "quirk"]
        if quirks:
            quirk_descriptions = [f"- {quirk.content}" for quirk in quirks[:2]]
            sections.append(f"**Personality Quirks**:\n" + "\n".join(quirk_descriptions))

        # Relevant memories
        memories = [element for element in relevant_quirks if element.type == "memory"]
        if memories:
            memory_descriptions = [f"- {memory.content}" for memory in memories[:2]]
            sections.append(f"**Relevant Experiences**:\n" + "\n".join(memory_descriptions))

        if sections:
            return "\n\n## Personal Context:\n" + "\n\n".join(sections)

        return ""

    def _generate_session_info(self, session_state: SessionState) -> str:
        """Generate session information string."""
        duration_minutes = session_state.duration_seconds / 60

        info_parts = [f"{session_state.turn_count} turns", f"{duration_minutes:.1f} minutes"]

        if session_state.in_closure_mode:
            info_parts.append(f"closing ({session_state.closure_reason})")

        return ", ".join(info_parts)

    def _determine_conversation_phase(self, session_state: SessionState) -> str:
        """Determine the current conversation phase."""
        if session_state.turn_count <= 2:
            return "opening"
        elif session_state.in_closure_mode:
            return "closing"
        elif session_state.duration_seconds > 1800:  # 30 minutes
            return "extended"
        else:
            return "active"

    def _generate_response_style_guidance(
        self, personality_params: PersonalityParameters, topic_context: TopicContext
    ) -> str:
        """Generate response style guidance."""
        guidance_parts = []

        # Topic-specific guidance
        if topic_context.category.value == "technical":
            guidance_parts.append("technical and precise")
        elif topic_context.category.value == "creative":
            guidance_parts.append("creative and imaginative")
        elif topic_context.category.value == "personal":
            guidance_parts.append("empathetic and understanding")

        # Personality-specific guidance
        formality = personality_params.get_trait_value(PersonalityTraitType.FORMALITY)
        if formality > 0.6:
            guidance_parts.append("formal")
        elif formality < 0.4:
            guidance_parts.append("casual")

        return ", ".join(guidance_parts) if guidance_parts else ""

    def _generate_detailed_style_guidelines(
        self, personality_params: PersonalityParameters, topic_context: TopicContext
    ) -> str:
        """Generate detailed style guidelines."""
        guidelines = []

        # Length guidance
        verbosity = personality_params.get_trait_value(PersonalityTraitType.VERBOSITY)
        if verbosity > 0.7:
            guidelines.append("Provide comprehensive, detailed responses with examples and context")
        elif verbosity < 0.3:
            guidelines.append("Keep responses brief and to the point")

        # Tone guidance
        formality = personality_params.get_trait_value(PersonalityTraitType.FORMALITY)
        humor = personality_params.get_trait_value(PersonalityTraitType.HUMOR)

        if formality > 0.6:
            guidelines.append("Use formal, professional language")
        elif formality < 0.4:
            guidelines.append("Use casual, conversational language")

        if humor > 0.6 and topic_context.category.value in ["entertainment", "casual"]:
            guidelines.append("Include appropriate humor and wit")

        # Confidence guidance
        assertiveness = personality_params.get_trait_value(PersonalityTraitType.ASSERTIVENESS)
        if assertiveness > 0.7:
            guidelines.append("Be confident and direct in your statements")
        elif assertiveness < 0.3:
            guidelines.append("Use tentative language and acknowledge limitations")

        return "\n".join(f"- {guideline}" for guideline in guidelines)

    def _get_trait_level_description(self, value: float) -> str:
        """Get a descriptive level for a trait value."""
        if value >= 0.8:
            return "Very High"
        elif value >= 0.6:
            return "High"
        elif value >= 0.4:
            return "Moderate"
        elif value >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt enhancements."""
        return {
            "templates_available": len(self.enhancement_templates),
            "enhancement_levels": ["minimal", "moderate", "full"],
            "supported_traits": [trait.value for trait in PersonalityTraitType],
            "template_names": list(self.enhancement_templates.keys()),
        }
