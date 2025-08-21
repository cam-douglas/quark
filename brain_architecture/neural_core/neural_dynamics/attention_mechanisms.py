#!/usr/bin/env python3
"""
👁️ Advanced Attention Mechanisms
Implements sophisticated attention mechanisms including spatial attention, 
feature-based attention, and attentional modulation for enhanced cognitive processing.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class AttentionType(Enum):
    """Types of attention mechanisms"""
    SPATIAL = "SPATIAL"  # Spatial attention
    FEATURE = "FEATURE"  # Feature-based attention
    TEMPORAL = "TEMPORAL"  # Temporal attention
    OBJECT = "OBJECT"  # Object-based attention

@dataclass
class AttentionField:
    """Represents an attention field in space"""
    x: float
    y: float
    radius: float
    strength: float
    attention_type: AttentionType
    decay_rate: float = 0.95

@dataclass
class FeatureFilter:
    """Feature-based attention filter"""
    feature_type: str  # "color", "orientation", "motion", etc.
    feature_value: Any
    strength: float
    selectivity: float  # How selective the filter is

class SpatialAttention:
    """Spatial attention mechanism"""
    
    def __init__(self, spatial_resolution: Tuple[int, int] = (100, 100)):
        self.spatial_resolution = spatial_resolution
        self.attention_map = np.zeros(spatial_resolution)
        self.attention_fields: List[AttentionField] = []
        self.focus_point = (50, 50)  # Center of attention
        self.attention_strength = 1.0
        
        print(f"👁️ Spatial attention initialized ({spatial_resolution[0]}x{spatial_resolution[1]})")
    
    def add_attention_field(self, x: float, y: float, radius: float, 
                          strength: float, attention_type: AttentionType = AttentionType.SPATIAL):
        """Add a new attention field"""
        field = AttentionField(x, y, radius, strength, attention_type)
        self.attention_fields.append(field)
        print(f"👁️ Added attention field at ({x:.1f}, {y:.1f}) with radius {radius:.1f}")
    
    def update_attention_map(self):
        """Update the spatial attention map"""
        self.attention_map.fill(0.0)
        
        for field in self.attention_fields:
            # Create Gaussian attention field
            y_coords, x_coords = np.ogrid[:self.spatial_resolution[0], :self.spatial_resolution[1]]
            
            # Calculate distance from field center
            distance = np.sqrt((x_coords - field.x)**2 + (y_coords - field.y)**2)
            
            # Apply Gaussian attention field
            attention = field.strength * np.exp(-(distance**2) / (2 * field.radius**2))
            
            # Add to attention map
            self.attention_map += attention
            
            # Decay field strength
            field.strength *= field.decay_rate
    
    def get_attention_at_location(self, x: int, y: int) -> float:
        """Get attention strength at specific location"""
        if 0 <= x < self.spatial_resolution[1] and 0 <= y < self.spatial_resolution[0]:
            return self.attention_map[y, x]
        return 0.0
    
    def move_focus(self, new_x: float, new_y: float, speed: float = 0.1):
        """Move attention focus to new location"""
        current_x, current_y = self.focus_point
        
        # Smooth movement
        new_x_pos = current_x + speed * (new_x - current_x)
        new_y_pos = current_y + speed * (new_y - current_y)
        
        self.focus_point = (new_x_pos, new_y_pos)
        
        # Add attention field at new focus
        self.add_attention_field(new_x_pos, new_y_pos, 10.0, self.attention_strength)
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get attention statistics"""
        return {
            "focus_point": self.focus_point,
            "attention_strength": self.attention_strength,
            "num_fields": len(self.attention_fields),
            "max_attention": np.max(self.attention_map),
            "mean_attention": np.mean(self.attention_map)
        }

class FeatureAttention:
    """Feature-based attention mechanism"""
    
    def __init__(self):
        self.feature_filters: List[FeatureFilter] = []
        self.feature_weights: Dict[str, float] = {}
        self.attention_modulation = 1.0
        
        print("👁️ Feature-based attention initialized")
    
    def add_feature_filter(self, feature_type: str, feature_value: Any, 
                          strength: float, selectivity: float = 0.8):
        """Add a feature-based attention filter"""
        filter_obj = FeatureFilter(feature_type, feature_value, strength, selectivity)
        self.feature_filters.append(filter_obj)
        self.feature_weights[feature_type] = strength
        
        print(f"👁️ Added {feature_type} filter for value {feature_value} (strength: {strength:.2f})")
    
    def apply_feature_attention(self, features: Dict[str, Any]) -> float:
        """Apply feature-based attention to input features"""
        total_attention = 0.0
        
        for filter_obj in self.feature_filters:
            if filter_obj.feature_type in features:
                feature_value = features[filter_obj.feature_type]
                
                # Calculate feature similarity
                if isinstance(feature_value, (int, float)) and isinstance(filter_obj.feature_value, (int, float)):
                    # Numeric features
                    similarity = 1.0 - abs(feature_value - filter_obj.feature_value) / max(abs(feature_value), abs(filter_obj.feature_value), 1.0)
                elif isinstance(feature_value, str) and isinstance(filter_obj.feature_value, str):
                    # String features
                    similarity = 1.0 if feature_value == filter_obj.feature_value else 0.0
                else:
                    # Default similarity
                    similarity = 0.5
                
                # Apply selectivity
                attention = filter_obj.strength * similarity * filter_obj.selectivity
                total_attention += attention
        
        return total_attention * self.attention_modulation
    
    def update_feature_weights(self, learning_rate: float = 0.01):
        """Update feature weights based on recent performance"""
        for filter_obj in self.feature_filters:
            # Simple weight update based on attention strength
            filter_obj.strength = np.clip(filter_obj.strength + learning_rate, 0.0, 2.0)
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature attention statistics"""
        return {
            "num_filters": len(self.feature_filters),
            "feature_types": list(self.feature_weights.keys()),
            "attention_modulation": self.attention_modulation,
            "total_strength": sum(self.feature_weights.values())
        }

class TemporalAttention:
    """Temporal attention mechanism for time-based attention"""
    
    def __init__(self, time_window: float = 1000.0):  # ms
        self.time_window = time_window
        self.temporal_weights = np.ones(10)  # 10 time bins
        self.recent_events: List[Tuple[float, float]] = []  # (time, importance)
        self.temporal_focus = 0.5  # Focus on middle of time window
        
        print(f"👁️ Temporal attention initialized (window: {time_window}ms)")
    
    def add_temporal_event(self, event_time: float, importance: float):
        """Add a temporal event"""
        self.recent_events.append((event_time, importance))
        
        # Remove old events
        current_time = time.time() * 1000  # Convert to ms
        self.recent_events = [(t, i) for t, i in self.recent_events 
                             if current_time - t <= self.time_window]
    
    def update_temporal_weights(self):
        """Update temporal attention weights"""
        if not self.recent_events:
            return
        
        # Calculate temporal attention based on recent events
        current_time = time.time() * 1000
        temporal_attention = np.zeros(10)
        
        for event_time, importance in self.recent_events:
            # Calculate relative time position
            relative_time = (current_time - event_time) / self.time_window
            time_bin = int(relative_time * 9)  # 0-9 bins
            
            if 0 <= time_bin < 10:
                temporal_attention[time_bin] += importance
        
        # Update weights
        self.temporal_weights = 0.9 * self.temporal_weights + 0.1 * temporal_attention
    
    def get_temporal_attention(self, time_offset: float) -> float:
        """Get temporal attention at specific time offset"""
        time_bin = int((time_offset / self.time_window) * 9)
        if 0 <= time_bin < 10:
            return self.temporal_weights[time_bin]
        return 0.0

class AttentionManager:
    """Manages multiple attention mechanisms"""
    
    def __init__(self, spatial_resolution: Tuple[int, int] = (100, 100)):
        self.spatial_attention = SpatialAttention(spatial_resolution)
        self.feature_attention = FeatureAttention()
        self.temporal_attention = TemporalAttention()
        
        # Attention integration
        self.integration_weights = {
            "spatial": 0.4,
            "feature": 0.4,
            "temporal": 0.2
        }
        
        print("👁️ Attention Manager initialized")
    
    def apply_attention(self, spatial_location: Tuple[int, int], 
                       features: Dict[str, Any], time_offset: float = 0.0) -> float:
        """Apply all attention mechanisms to input"""
        # Spatial attention
        spatial_att = self.spatial_attention.get_attention_at_location(*spatial_location)
        
        # Feature attention
        feature_att = self.feature_attention.apply_feature_attention(features)
        
        # Temporal attention
        temporal_att = self.temporal_attention.get_temporal_attention(time_offset)
        
        # Integrate attention signals
        total_attention = (
            self.integration_weights["spatial"] * spatial_att +
            self.integration_weights["feature"] * feature_att +
            self.integration_weights["temporal"] * temporal_att
        )
        
        return total_attention
    
    def update_attention_systems(self):
        """Update all attention systems"""
        self.spatial_attention.update_attention_map()
        self.feature_attention.update_feature_weights()
        self.temporal_attention.update_temporal_weights()
    
    def add_spatial_focus(self, x: float, y: float, radius: float = 10.0, strength: float = 1.0):
        """Add spatial attention focus"""
        self.spatial_attention.add_attention_field(x, y, radius, strength)
    
    def add_feature_focus(self, feature_type: str, feature_value: Any, strength: float = 1.0):
        """Add feature-based attention focus"""
        self.feature_attention.add_feature_filter(feature_type, feature_value, strength)
    
    def add_temporal_event(self, importance: float):
        """Add temporal attention event"""
        self.temporal_attention.add_temporal_event(time.time() * 1000, importance)
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get comprehensive attention statistics"""
        return {
            "spatial": self.spatial_attention.get_attention_stats(),
            "feature": self.feature_attention.get_feature_stats(),
            "integration_weights": self.integration_weights.copy()
        }

def main():
    """Demonstrate advanced attention mechanisms"""
    print("👁️ QUARK Advanced Attention Mechanisms - Phase 2")
    print("=" * 55)
    
    # Create attention manager
    attention = AttentionManager(spatial_resolution=(50, 50))
    
    # Add spatial attention focus
    attention.add_spatial_focus(25, 25, radius=8.0, strength=1.5)
    attention.add_spatial_focus(35, 15, radius=5.0, strength=0.8)
    
    # Add feature-based attention
    attention.add_feature_focus("color", "red", strength=1.2)
    attention.add_feature_focus("orientation", 45, strength=0.9)
    attention.add_feature_focus("motion", "fast", strength=1.0)
    
    # Add temporal events
    attention.add_temporal_event(importance=0.8)
    attention.add_temporal_event(importance=0.6)
    
    # Update attention systems
    attention.update_attention_systems()
    
    # Test attention at different locations and features
    print("\n🔄 Testing attention mechanisms...")
    
    test_cases = [
        ((25, 25), {"color": "red", "orientation": 45, "motion": "fast"}),
        ((35, 15), {"color": "blue", "orientation": 90, "motion": "slow"}),
        ((10, 10), {"color": "red", "orientation": 0, "motion": "medium"})
    ]
    
    for i, (location, features) in enumerate(test_cases):
        attention_strength = attention.apply_attention(location, features)
        print(f"   Location {location}, Features {features}: Attention = {attention_strength:.3f}")
    
    # Show statistics
    stats = attention.get_attention_stats()
    print(f"\n📊 Attention Statistics:")
    print(f"   Spatial focus: {stats['spatial']['focus_point']}")
    print(f"   Feature filters: {stats['feature']['num_filters']}")
    print(f"   Integration weights: {stats['integration_weights']}")
    
    print("\n✅ Advanced attention mechanisms demonstration completed!")
    return attention

if __name__ == "__main__":
    try:
        attention = main()
    except Exception as e:
        print(f"❌ Attention mechanisms failed: {e}")
        import traceback
        traceback.print_exc()
