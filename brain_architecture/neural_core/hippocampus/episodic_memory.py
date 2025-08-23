#!/usr/bin/env python3
"""
🧠 Hippocampus - Episodic Memory Module
Handles episodic memory formation, consolidation, and pattern completion
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class MemoryEpisode:
    """Represents an episodic memory"""
    episode_id: str
    content: Dict[str, Any]
    context: Dict[str, Any]
    emotional_valence: float
    importance: float
    created_at: float
    last_accessed: float
    access_count: int = 0
    consolidation_strength: float = 0.5

@dataclass
class MemoryPattern:
    """Represents a memory pattern for pattern completion"""
    pattern_id: str
    features: np.ndarray
    associated_episodes: List[str]
    strength: float
    last_updated: float

class EpisodicMemory:
    """Hippocampus episodic memory system"""
    
    def __init__(self, max_episodes: int = 1000, pattern_dim: int = 64):
        self.max_episodes = max_episodes
        self.pattern_dim = pattern_dim
        
        # Memory storage
        self.episodes: Dict[str, MemoryEpisode] = {}
        self.patterns: Dict[str, MemoryPattern] = {}
        
        # Neural representations
        self.episode_neurons = np.random.rand(max_episodes, pattern_dim)
        self.pattern_neurons = np.random.rand(200, pattern_dim)  # 200 pattern neurons
        
        # Memory consolidation parameters
        self.consolidation_rate = 0.1
        self.forgetting_rate = 0.05
        self.pattern_threshold = 0.7
        
        # Context associations
        self.context_index: Dict[str, List[str]] = defaultdict(list)
        
    def store_episode(self, content: Dict[str, Any], context: Dict[str, Any], 
                     emotional_valence: float = 0.0, importance: float = 0.5) -> str:
        """Store new episodic memory"""
        episode_id = f"episode_{len(self.episodes) + 1}_{int(time.time())}"
        
        episode = MemoryEpisode(
            episode_id=episode_id,
            content=content,
            context=context,
            emotional_valence=emotional_valence,
            importance=importance,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        # Store episode
        self.episodes[episode_id] = episode
        
        # Index by context
        for context_key, context_value in context.items():
            context_str = f"{context_key}:{context_value}"
            self.context_index[context_str].append(episode_id)
        
        # Create or update memory patterns
        self._update_memory_patterns(episode)
        
        # Manage memory capacity
        if len(self.episodes) > self.max_episodes:
            self._consolidate_memories()
        
        return episode_id
    
    def _update_memory_patterns(self, episode: MemoryEpisode):
        """Update memory patterns based on new episode"""
        # Extract features from episode content and context
        features = self._extract_features(episode)
        
        # Find similar patterns
        best_pattern = None
        best_similarity = 0.0
        
        for pattern in self.patterns.values():
            similarity = np.dot(features, pattern.features) / (np.linalg.norm(features) * np.linalg.norm(pattern.features))
            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern = pattern
        
        if best_similarity > self.pattern_threshold and best_pattern:
            # Update existing pattern
            best_pattern.features = (best_pattern.features + features) / 2.0
            best_pattern.associated_episodes.append(episode.episode_id)
            best_pattern.strength += 0.1
            best_pattern.last_updated = time.time()
        else:
            # Create new pattern
            pattern_id = f"pattern_{len(self.patterns) + 1}"
            new_pattern = MemoryPattern(
                pattern_id=pattern_id,
                features=features,
                associated_episodes=[episode.episode_id],
                strength=0.5,
                last_updated=time.time()
            )
            self.patterns[pattern_id] = new_pattern
    
    def _extract_features(self, episode: MemoryEpisode) -> np.ndarray:
        """Extract feature vector from episode"""
        features = np.zeros(self.pattern_dim)
        
        # Content features (simple hash-based)
        content_str = str(episode.content)
        for i, char in enumerate(content_str[:self.pattern_dim]):
            features[i] = ord(char) / 255.0
        
        # Context features
        context_str = str(episode.context)
        for i, char in enumerate(context_str[:self.pattern_dim//2]):
            features[i + self.pattern_dim//2] = ord(char) / 255.0
        
        # Emotional and importance features
        features[0] = episode.emotional_valence
        features[1] = episode.importance
        
        return features
    
    def retrieve_episode(self, query: Dict[str, Any], max_results: int = 5) -> List[MemoryEpisode]:
        """Retrieve episodes based on query"""
        query_features = self._extract_query_features(query)
        
        # Score episodes based on similarity
        episode_scores = []
        for episode in self.episodes.values():
            episode_features = self._extract_features(episode)
            similarity = np.dot(query_features, episode_features) / (np.linalg.norm(query_features) * np.linalg.norm(episode_features))
            
            # Boost score for recent and important episodes
            recency_boost = 1.0 / (time.time() - episode.last_accessed + 1)
            importance_boost = episode.importance
            
            final_score = similarity * (1.0 + recency_boost + importance_boost)
            episode_scores.append((episode, final_score))
        
        # Sort by score and return top results
        episode_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update access times for retrieved episodes
        for episode, _ in episode_scores[:max_results]:
            episode.last_accessed = time.time()
            episode.access_count += 1
        
        return [episode for episode, _ in episode_scores[:max_results]]
    
    def _extract_query_features(self, query: Dict[str, Any]) -> np.ndarray:
        """Extract features from query"""
        features = np.zeros(self.pattern_dim)
        
        query_str = str(query)
        for i, char in enumerate(query_str[:self.pattern_dim]):
            features[i] = ord(char) / 255.0
        
        return features
    
    def pattern_completion(self, partial_features: np.ndarray, threshold: float = 0.6) -> List[MemoryEpisode]:
        """Complete memory patterns from partial information"""
        completed_episodes = []
        
        for pattern in self.patterns.values():
            # Calculate similarity with partial features
            if len(partial_features) == len(pattern.features):
                similarity = np.dot(partial_features, pattern.features) / (np.linalg.norm(partial_features) * np.linalg.norm(pattern.features))
                
                if similarity > threshold:
                    # Retrieve associated episodes
                    for episode_id in pattern.associated_episodes:
                        if episode_id in self.episodes:
                            episode = self.episodes[episode_id]
                            episode.last_accessed = time.time()
                            episode.access_count += 1
                            completed_episodes.append(episode)
        
        return completed_episodes
    
    def _consolidate_memories(self):
        """Consolidate memories to manage capacity"""
        # Remove least important and least accessed memories
        episode_scores = []
        for episode in self.episodes.values():
            # Score based on importance, access count, and recency
            importance_score = episode.importance
            access_score = min(episode.access_count / 10.0, 1.0)
            recency_score = 1.0 / (time.time() - episode.last_accessed + 1)
            
            final_score = importance_score * 0.5 + access_score * 0.3 + recency_score * 0.2
            episode_scores.append((episode, final_score))
        
        # Keep top memories
        episode_scores.sort(key=lambda x: x[1], reverse=True)
        episodes_to_keep = episode_scores[:self.max_episodes]
        
        # Rebuild episodes dict
        self.episodes = {episode.episode_id: episode for episode, _ in episodes_to_keep}
        
        # Rebuild context index
        self.context_index.clear()
        for episode in self.episodes.values():
            for context_key, context_value in episode.context.items():
                context_str = f"{context_key}:{context_value}"
                self.context_index[context_str].append(episode.episode_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_episodes = len(self.episodes)
        total_patterns = len(self.patterns)
        
        if total_episodes > 0:
            avg_importance = np.mean([ep.importance for ep in self.episodes.values()])
            avg_emotional_valence = np.mean([ep.emotional_valence for ep in self.episodes.values()])
            avg_access_count = np.mean([ep.access_count for ep in self.episodes.values()])
        else:
            avg_importance = avg_emotional_valence = avg_access_count = 0.0
        
        return {
            "total_episodes": total_episodes,
            "total_patterns": total_patterns,
            "avg_importance": avg_importance,
            "avg_emotional_valence": avg_emotional_valence,
            "avg_access_count": avg_access_count,
            "memory_utilization": total_episodes / self.max_episodes,
            "context_index_size": len(self.context_index)
        }
    
    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Step episodic memory system forward"""
        # Process new episodes
        if "new_episodes" in inputs:
            for episode_data in inputs["new_episodes"]:
                self.store_episode(
                    content=episode_data["content"],
                    context=episode_data.get("context", {}),
                    emotional_valence=episode_data.get("emotional_valence", 0.0),
                    importance=episode_data.get("importance", 0.5)
                )
        
        # Process retrieval requests
        retrieved_episodes = []
        if "retrieval_requests" in inputs:
            for request in inputs["retrieval_requests"]:
                episodes = self.retrieve_episode(
                    query=request["query"],
                    max_results=request.get("max_results", 5)
                )
                retrieved_episodes.extend(episodes)
        
        # Process pattern completion requests
        completed_episodes = []
        if "pattern_completion_requests" in inputs:
            for request in inputs["pattern_completion_requests"]:
                partial_features = np.array(request["partial_features"])
                episodes = self.pattern_completion(partial_features, request.get("threshold", 0.6))
                completed_episodes.extend(episodes)
        
        return {
            "memory_stats": self.get_memory_stats(),
            "retrieved_episodes": len(retrieved_episodes),
            "completed_episodes": len(completed_episodes),
            "total_episodes": len(self.episodes)
        }
