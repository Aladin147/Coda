"""
Personality parameter management for Coda.

This module provides the PersonalityParameterManager class for managing
dynamic personality traits that can be adjusted based on context and learning.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .interfaces import PersonalityParameterInterface
from .models import (
    PersonalityTrait,
    PersonalityParameters,
    PersonalityTraitType,
    PersonalityAdjustment,
    PersonalityParameterConfig,
)

logger = logging.getLogger("coda.personality.parameters")


class PersonalityParameterManager(PersonalityParameterInterface):
    """
    Manages personality parameters with dynamic adjustment capabilities.
    
    Features:
    - Dynamic trait value adjustment with bounds checking
    - Context-aware personality modifications
    - Adjustment history tracking for explainability
    - Default value restoration
    - Configuration-driven trait definitions
    """
    
    def __init__(self, config: Optional[PersonalityParameterConfig] = None):
        """
        Initialize the personality parameter manager.
        
        Args:
            config: Configuration for personality parameters
        """
        self.config = config or PersonalityParameterConfig()
        self.parameters = PersonalityParameters()
        self.adjustment_history: List[PersonalityAdjustment] = []
        
        # Initialize default traits
        self._initialize_default_traits()
        
        logger.info(f"PersonalityParameterManager initialized with {len(self.parameters.traits)} traits")
    
    def _initialize_default_traits(self) -> None:
        """Initialize default personality traits."""
        default_traits = {
            PersonalityTraitType.VERBOSITY: {
                "value": 0.5,
                "description": "Controls length and detail of responses",
                "context_adjustments": {
                    "technical": 0.2,
                    "casual": -0.1,
                    "formal": 0.1,
                    "emergency": -0.2
                }
            },
            PersonalityTraitType.ASSERTIVENESS: {
                "value": 0.5,
                "description": "Controls how assertive or tentative responses are",
                "context_adjustments": {
                    "information_request": 0.2,
                    "creative": -0.1,
                    "emergency": 0.3,
                    "personal": -0.1
                }
            },
            PersonalityTraitType.HUMOR: {
                "value": 0.3,
                "description": "Controls the level of humor in responses",
                "context_adjustments": {
                    "entertainment": 0.3,
                    "technical": -0.2,
                    "formal": -0.2,
                    "emergency": -0.4
                }
            },
            PersonalityTraitType.FORMALITY: {
                "value": 0.5,
                "description": "Controls formality level of language",
                "context_adjustments": {
                    "professional": 0.3,
                    "casual": -0.3,
                    "technical": 0.1,
                    "personal": -0.2
                }
            },
            PersonalityTraitType.PROACTIVITY: {
                "value": 0.4,
                "description": "Controls tendency to offer additional help or suggestions",
                "context_adjustments": {
                    "educational": 0.2,
                    "creative": 0.3,
                    "emergency": 0.4,
                    "casual": -0.1
                }
            },
            PersonalityTraitType.CONFIDENCE: {
                "value": 0.6,
                "description": "Controls confidence level in responses",
                "context_adjustments": {
                    "technical": 0.2,
                    "educational": 0.1,
                    "creative": -0.1,
                    "personal": -0.1
                }
            },
            PersonalityTraitType.EMPATHY: {
                "value": 0.7,
                "description": "Controls empathetic responses and emotional awareness",
                "context_adjustments": {
                    "personal": 0.2,
                    "emotional": 0.3,
                    "technical": -0.2,
                    "formal": -0.1
                }
            },
            PersonalityTraitType.CREATIVITY: {
                "value": 0.5,
                "description": "Controls creative and imaginative responses",
                "context_adjustments": {
                    "creative": 0.3,
                    "entertainment": 0.2,
                    "technical": -0.1,
                    "formal": -0.2
                }
            },
            PersonalityTraitType.ANALYTICAL: {
                "value": 0.6,
                "description": "Controls analytical and logical approach",
                "context_adjustments": {
                    "technical": 0.3,
                    "educational": 0.2,
                    "creative": -0.2,
                    "entertainment": -0.1
                }
            },
            PersonalityTraitType.ENTHUSIASM: {
                "value": 0.5,
                "description": "Controls enthusiasm and energy in responses",
                "context_adjustments": {
                    "entertainment": 0.3,
                    "creative": 0.2,
                    "educational": 0.1,
                    "formal": -0.2
                }
            }
        }
        
        # Merge with config if provided
        if self.config.default_traits:
            default_traits.update(self.config.default_traits)
        
        # Create trait objects
        for trait_type, trait_config in default_traits.items():
            trait = PersonalityTrait(
                name=trait_type,
                value=trait_config["value"],
                default_value=trait_config["value"],
                description=trait_config["description"],
                context_adjustments=trait_config.get("context_adjustments", {})
            )
            self.parameters.traits[trait_type] = trait
    
    def get_parameters(self) -> PersonalityParameters:
        """Get current personality parameters."""
        return self.parameters
    
    def get_trait_value(self, trait_type: PersonalityTraitType) -> float:
        """Get the current value of a specific trait."""
        return self.parameters.get_trait_value(trait_type)
    
    def adjust_trait(self, trait_type: PersonalityTraitType, delta: float, 
                    reason: str = "", confidence: float = 1.0) -> PersonalityAdjustment:
        """
        Adjust a personality trait value.
        
        Args:
            trait_type: Type of trait to adjust
            delta: Amount to adjust (can be negative)
            reason: Reason for the adjustment
            confidence: Confidence in the adjustment (0.0 to 1.0)
            
        Returns:
            PersonalityAdjustment record
        """
        trait = self.parameters.get_trait(trait_type)
        if not trait:
            logger.warning(f"Trait {trait_type} not found")
            return PersonalityAdjustment(
                trait_type=trait_type,
                old_value=0.5,
                new_value=0.5,
                delta=0.0,
                reason=f"Trait not found: {reason}",
                confidence=0.0
            )
        
        old_value = trait.value
        
        # Apply confidence weighting to delta
        weighted_delta = delta * confidence
        
        # Apply adjustment limits if configured
        if self.config.adjustment_limits:
            max_adjustment = self.config.adjustment_limits.get("max_single_adjustment", 0.3)
            weighted_delta = max(-max_adjustment, min(max_adjustment, weighted_delta))
        
        # Adjust the trait
        actual_delta = trait.adjust(weighted_delta, reason)
        new_value = trait.value
        
        # Create adjustment record
        adjustment = PersonalityAdjustment(
            trait_type=trait_type,
            old_value=old_value,
            new_value=new_value,
            delta=actual_delta,
            reason=reason,
            confidence=confidence
        )
        
        # Add to history
        self.adjustment_history.append(adjustment)
        
        # Trim history if needed
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
        
        logger.info(f"Adjusted {trait_type.value}: {old_value:.3f} → {new_value:.3f} (Δ{actual_delta:+.3f}) - {reason}")
        
        return adjustment
    
    def apply_context_adjustments(self, context: str) -> Dict[PersonalityTraitType, float]:
        """
        Apply context-based adjustments to personality traits.
        
        Args:
            context: Context identifier (e.g., "technical", "casual", "formal")
            
        Returns:
            Dictionary of actual adjustments made
        """
        adjustments = {}
        
        for trait_type, trait in self.parameters.traits.items():
            if context in trait.context_adjustments:
                adjustment_value = trait.context_adjustments[context]
                
                # Apply the adjustment
                adjustment = self.adjust_trait(
                    trait_type,
                    adjustment_value,
                    f"Context adjustment for '{context}'",
                    confidence=0.8  # High confidence for context adjustments
                )
                
                if adjustment.delta != 0:
                    adjustments[trait_type] = adjustment.delta
        
        if adjustments:
            logger.info(f"Applied context adjustments for '{context}': {adjustments}")
        
        return adjustments
    
    def reset_trait(self, trait_type: PersonalityTraitType) -> bool:
        """
        Reset a trait to its default value.
        
        Args:
            trait_type: Type of trait to reset
            
        Returns:
            True if successful, False otherwise
        """
        trait = self.parameters.get_trait(trait_type)
        if not trait:
            return False
        
        old_value = trait.value
        trait.reset_to_default()
        
        # Record the reset
        adjustment = PersonalityAdjustment(
            trait_type=trait_type,
            old_value=old_value,
            new_value=trait.value,
            delta=trait.value - old_value,
            reason="Manual reset to default",
            confidence=1.0
        )
        
        self.adjustment_history.append(adjustment)
        
        logger.info(f"Reset {trait_type.value} to default value: {trait.value}")
        return True
    
    def reset_all_traits(self) -> None:
        """Reset all traits to their default values."""
        for trait_type in self.parameters.traits.keys():
            self.reset_trait(trait_type)
        
        logger.info("Reset all personality traits to default values")
    
    def get_adjustment_history(self, limit: int = 10) -> List[PersonalityAdjustment]:
        """
        Get recent personality adjustments.
        
        Args:
            limit: Maximum number of adjustments to return
            
        Returns:
            List of recent adjustments
        """
        return self.adjustment_history[-limit:] if self.adjustment_history else []
    
    def get_trait_summary(self) -> Dict[str, Any]:
        """Get a summary of all personality traits."""
        summary = {}
        
        for trait_type, trait in self.parameters.traits.items():
            summary[trait_type.value] = {
                "current_value": trait.value,
                "default_value": trait.default_value,
                "deviation": trait.value - trait.default_value,
                "adjustment_count": trait.adjustment_count,
                "last_adjusted": trait.last_adjusted.isoformat() if trait.last_adjusted else None,
                "description": trait.description
            }
        
        return summary
    
    def get_context_impact(self, context: str) -> Dict[str, float]:
        """Get the potential impact of a context on all traits."""
        impact = {}
        
        for trait_type, trait in self.parameters.traits.items():
            if context in trait.context_adjustments:
                impact[trait_type.value] = trait.context_adjustments[context]
        
        return impact
    
    def validate_trait_values(self) -> List[str]:
        """Validate all trait values are within acceptable ranges."""
        issues = []
        
        for trait_type, trait in self.parameters.traits.items():
            if not (trait.min_value <= trait.value <= trait.max_value):
                issues.append(f"{trait_type.value} value {trait.value} is outside range [{trait.min_value}, {trait.max_value}]")
        
        return issues
    
    def export_parameters(self) -> Dict[str, Any]:
        """Export current parameters for persistence."""
        return {
            "parameters": self.parameters.model_dump(),
            "adjustment_history": [adj.model_dump() for adj in self.adjustment_history[-50:]],  # Last 50 adjustments
            "exported_at": datetime.now().isoformat()
        }
    
    def import_parameters(self, data: Dict[str, Any]) -> bool:
        """Import parameters from exported data."""
        try:
            if "parameters" in data:
                self.parameters = PersonalityParameters(**data["parameters"])
            
            if "adjustment_history" in data:
                self.adjustment_history = [
                    PersonalityAdjustment(**adj) for adj in data["adjustment_history"]
                ]
            
            logger.info("Successfully imported personality parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import parameters: {e}")
            return False
