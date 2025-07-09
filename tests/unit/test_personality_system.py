"""
Unit tests for the personality system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from src.coda.components.personality.models import (
    PersonalityTrait,
    PersonalityTraitType,
    PersonalityParameters,
    BehaviorProfile,
    TopicContext,
    TopicCategory,
    PersonalLore,
    SessionState,
    PersonalityConfig,
)
from src.coda.components.personality.parameters import PersonalityParameterManager
from src.coda.components.personality.behavioral_conditioner import BehavioralConditioner
from src.coda.components.personality.topic_awareness import TopicAwareness
from src.coda.components.personality.personal_lore import PersonalLoreManager
from src.coda.components.personality.session_manager import SessionManager
from src.coda.components.personality.manager import PersonalityManager


class TestPersonalityTrait:
    """Test cases for PersonalityTrait."""
    
    def test_trait_creation(self):
        """Test creating a personality trait."""
        trait = PersonalityTrait(
            name=PersonalityTraitType.VERBOSITY,
            value=0.7,
            default_value=0.5,
            description="Controls response length"
        )
        
        assert trait.name == PersonalityTraitType.VERBOSITY
        assert trait.value == 0.7
        assert trait.default_value == 0.5
        assert trait.adjustment_count == 0
    
    def test_trait_adjustment(self):
        """Test adjusting trait values."""
        trait = PersonalityTrait(
            name=PersonalityTraitType.HUMOR,
            value=0.5,
            default_value=0.5,
            description="Controls humor level"
        )
        
        # Test positive adjustment
        delta = trait.adjust(0.2, "Test adjustment")
        assert trait.value == 0.7
        assert delta == 0.2
        assert trait.adjustment_count == 1
        
        # Test adjustment with bounds
        delta = trait.adjust(0.5, "Large adjustment")
        assert trait.value == 1.0  # Capped at max
        assert delta == 0.3
    
    def test_trait_reset(self):
        """Test resetting trait to default."""
        trait = PersonalityTrait(
            name=PersonalityTraitType.ASSERTIVENESS,
            value=0.8,
            default_value=0.5,
            description="Controls assertiveness"
        )
        
        trait.reset_to_default()
        assert trait.value == 0.5


class TestPersonalityParameterManager:
    """Test cases for PersonalityParameterManager."""
    
    @pytest.fixture
    def parameter_manager(self):
        """Create a test parameter manager."""
        return PersonalityParameterManager()
    
    def test_initialization(self, parameter_manager):
        """Test parameter manager initialization."""
        parameters = parameter_manager.get_parameters()
        assert len(parameters.traits) > 0
        assert PersonalityTraitType.VERBOSITY in parameters.traits
        assert PersonalityTraitType.HUMOR in parameters.traits
    
    def test_get_trait_value(self, parameter_manager):
        """Test getting trait values."""
        verbosity = parameter_manager.get_trait_value(PersonalityTraitType.VERBOSITY)
        assert 0.0 <= verbosity <= 1.0
    
    def test_adjust_trait(self, parameter_manager):
        """Test adjusting traits."""
        old_value = parameter_manager.get_trait_value(PersonalityTraitType.HUMOR)
        
        adjustment = parameter_manager.adjust_trait(
            PersonalityTraitType.HUMOR,
            0.2,
            "Test adjustment",
            confidence=0.8
        )
        
        new_value = parameter_manager.get_trait_value(PersonalityTraitType.HUMOR)
        assert new_value != old_value
        assert adjustment.trait_type == PersonalityTraitType.HUMOR
        assert adjustment.confidence == 0.8
    
    def test_context_adjustments(self, parameter_manager):
        """Test context-based adjustments."""
        adjustments = parameter_manager.apply_context_adjustments("technical")
        
        # Should have some adjustments for technical context
        assert isinstance(adjustments, dict)
    
    def test_reset_trait(self, parameter_manager):
        """Test resetting individual traits."""
        # Adjust a trait first
        parameter_manager.adjust_trait(PersonalityTraitType.CREATIVITY, 0.3, "Test")
        
        # Reset it
        success = parameter_manager.reset_trait(PersonalityTraitType.CREATIVITY)
        assert success
        
        # Should be back to default
        trait = parameter_manager.get_parameters().get_trait(PersonalityTraitType.CREATIVITY)
        assert trait.value == trait.default_value


@pytest.mark.asyncio
class TestBehavioralConditioner:
    """Test cases for BehavioralConditioner."""
    
    @pytest.fixture
    def conditioner(self):
        """Create a test behavioral conditioner."""
        return BehavioralConditioner()
    
    async def test_process_user_input(self, conditioner):
        """Test processing user input."""
        result = await conditioner.process_user_input("Please be more detailed in your responses")
        
        assert "explicit_patterns" in result
        assert "engagement" in result
        assert "style_preferences" in result
    
    async def test_process_feedback(self, conditioner):
        """Test processing user feedback."""
        result = await conditioner.process_user_feedback("That was great, very helpful!")
        
        assert "sentiment" in result
        assert "style_feedback" in result
        assert "pattern" in result
    
    async def test_analyze_patterns(self, conditioner):
        """Test analyzing interaction patterns."""
        # Add some interactions first
        await conditioner.process_user_input("Hello")
        await conditioner.process_user_input("How are you?")
        await conditioner.process_user_input("Tell me about AI")
        
        patterns = await conditioner.analyze_interaction_patterns()
        assert isinstance(patterns, list)
    
    def test_behavior_profile(self, conditioner):
        """Test behavior profile management."""
        profile = conditioner.get_behavior_profile()
        assert isinstance(profile, BehaviorProfile)
        assert profile.observation_count >= 0
    
    def test_learning_stats(self, conditioner):
        """Test getting learning statistics."""
        stats = conditioner.get_learning_stats()
        
        assert "total_interactions" in stats
        assert "observation_count" in stats
        assert "learned_preferences" in stats


@pytest.mark.asyncio
class TestTopicAwareness:
    """Test cases for TopicAwareness."""
    
    @pytest.fixture
    def topic_awareness(self):
        """Create a test topic awareness system."""
        return TopicAwareness()
    
    async def test_detect_topic(self, topic_awareness):
        """Test topic detection."""
        # Technical topic
        tech_context = await topic_awareness.detect_topic("How do I write Python code?")
        assert tech_context.category == TopicCategory.TECHNICAL
        assert tech_context.confidence > 0
        
        # Creative topic
        creative_context = await topic_awareness.detect_topic("Help me write a story")
        assert creative_context.category == TopicCategory.CREATIVE
    
    async def test_process_user_input(self, topic_awareness):
        """Test processing user input for topics."""
        context = await topic_awareness.process_user_input("I'm learning machine learning")
        
        assert isinstance(context, TopicContext)
        assert context.category in [TopicCategory.TECHNICAL, TopicCategory.EDUCATIONAL]
    
    def test_get_current_topic(self, topic_awareness):
        """Test getting current topic."""
        current = topic_awareness.get_current_topic()
        assert isinstance(current, TopicContext)
    
    async def test_topic_personality_adjustments(self, topic_awareness):
        """Test getting personality adjustments for topics."""
        # Create a technical topic context
        tech_context = TopicContext(
            current_topic="programming",
            category=TopicCategory.TECHNICAL,
            confidence=0.8
        )
        
        adjustments = await topic_awareness.get_topic_personality_adjustments(tech_context)
        
        assert isinstance(adjustments, dict)
        # Technical topics should increase analytical trait
        if PersonalityTraitType.ANALYTICAL in adjustments:
            assert adjustments[PersonalityTraitType.ANALYTICAL] > 0


class TestPersonalLoreManager:
    """Test cases for PersonalLoreManager."""
    
    @pytest.fixture
    def lore_manager(self):
        """Create a test personal lore manager."""
        return PersonalLoreManager()
    
    def test_initialization(self, lore_manager):
        """Test lore manager initialization."""
        lore = lore_manager.get_lore()
        assert isinstance(lore, PersonalLore)
        assert len(lore.backstory) > 0
        assert len(lore.quirks) > 0
        assert len(lore.memories) > 0
    
    def test_get_relevant_lore(self, lore_manager):
        """Test getting relevant lore."""
        relevant = lore_manager.get_relevant_lore("technical programming", ["code", "python"])
        
        assert isinstance(relevant, list)
        # Should return lore elements relevant to programming
    
    def test_add_lore_element(self, lore_manager):
        """Test adding new lore elements."""
        success = lore_manager.add_lore_element(
            "quirk",
            "I enjoy explaining complex topics with simple analogies",
            ["explain", "complex", "analogy"],
            0.7
        )
        
        assert success
        
        # Check it was added
        lore = lore_manager.get_lore()
        assert len(lore.quirks) > 0
    
    def test_enhance_prompt(self, lore_manager):
        """Test prompt enhancement."""
        original_prompt = "You are a helpful assistant."
        enhanced = lore_manager.enhance_prompt(original_prompt, "technical", ["programming"])
        
        # Enhanced prompt should be longer
        assert len(enhanced) >= len(original_prompt)
    
    def test_lore_usage_stats(self, lore_manager):
        """Test getting lore usage statistics."""
        stats = lore_manager.get_lore_usage_stats()
        
        assert "total_quirks" in stats
        assert "total_memories" in stats
        assert "quirk_usage" in stats


class TestSessionManager:
    """Test cases for SessionManager."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a test session manager."""
        return SessionManager()
    
    def test_initialization(self, session_manager):
        """Test session manager initialization."""
        state = session_manager.get_session_state()
        assert isinstance(state, SessionState)
        assert state.turn_count == 0
        assert not state.in_closure_mode
    
    def test_process_interaction(self, session_manager):
        """Test processing interactions."""
        session_manager.process_interaction("user", "Hello")
        session_manager.process_interaction("assistant", "Hi there!")
        
        state = session_manager.get_session_state()
        assert state.turn_count == 2
    
    def test_session_summary(self, session_manager):
        """Test generating session summary."""
        # Add some interactions
        session_manager.process_interaction("user", "Hello")
        session_manager.process_interaction("assistant", "Hi!")
        
        summary = session_manager.generate_session_summary()
        
        assert "session_id" in summary
        assert "total_turns" in summary
        assert summary["total_turns"] == 2
    
    def test_closure_detection(self, session_manager):
        """Test session closure detection."""
        should_close, reason = session_manager.should_enter_closure_mode()
        assert isinstance(should_close, bool)
        assert isinstance(reason, str)


@pytest.mark.asyncio
class TestPersonalityManager:
    """Test cases for PersonalityManager."""
    
    @pytest.fixture
    def personality_manager(self):
        """Create a test personality manager."""
        return PersonalityManager()
    
    async def test_initialization(self, personality_manager):
        """Test personality manager initialization."""
        state = personality_manager.get_personality_state()
        
        assert "parameters" in state
        assert "topic_context" in state
        assert "behavior_profile" in state
        assert "session" in state
    
    async def test_process_user_input(self, personality_manager):
        """Test processing user input."""
        result = await personality_manager.process_user_input("I'm learning Python programming")
        
        assert "session_state" in result
        assert "topic_context" in result
        assert "behavior_analysis" in result
        assert "current_parameters" in result
    
    async def test_enhance_prompt(self, personality_manager):
        """Test prompt enhancement."""
        original = "You are a helpful assistant."
        enhanced = await personality_manager.enhance_prompt(original, "technical context")
        
        assert len(enhanced) >= len(original)
    
    async def test_process_feedback(self, personality_manager):
        """Test processing feedback."""
        result = await personality_manager.process_feedback("Please be more concise")
        
        assert "feedback_analysis" in result
        assert "learning_updated" in result
    
    def test_get_analytics(self, personality_manager):
        """Test getting analytics."""
        analytics = personality_manager.get_analytics()
        
        assert "overview" in analytics
        assert "parameters" in analytics
        assert "behavior_learning" in analytics
    
    def test_export_import_state(self, personality_manager):
        """Test exporting and importing personality state."""
        # Export state
        exported = personality_manager.export_personality_state()
        assert "parameters" in exported
        assert "exported_at" in exported
        
        # Create new manager and import
        new_manager = PersonalityManager()
        success = new_manager.import_personality_state(exported)
        assert success


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        manager = PersonalityManager()
        
        # Test basic functionality
        result = await manager.process_user_input("Hello, I'm interested in learning about AI")
        print(f"Processed input with {len(result.get('personality_adjustments', {}))} adjustments")
        
        # Test prompt enhancement
        enhanced = await manager.enhance_prompt("You are a helpful assistant.", "AI discussion")
        print(f"Enhanced prompt length: {len(enhanced)} characters")
        
        # Test analytics
        analytics = manager.get_analytics()
        print(f"Analytics overview: {analytics['overview']}")
        
        print("✅ Personality system test passed")
    
    asyncio.run(simple_test())
