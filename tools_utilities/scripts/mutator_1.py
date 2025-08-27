#!/usr/bin/env python3
"""
🧬 Mutator - Tests system robustness with prompt mutations
Generates variants and scores them for biological compliance
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

META_PATH = Path("memory/metadata.json")
MUTATION_LOG_PATH = Path("memory/mutation_log.json")

class PromptMutator:
    """Generates and tests prompt mutations for robustness"""
    
    def __init__(self):
        self.metadata = self._load_metadata()
        self.mutation_history = []
        
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from DNA controller"""
        if not META_PATH.exists():
            raise FileNotFoundError("metadata.json not found. Run dna_controller.py first.")
        
        with open(META_PATH) as f:
            return json.load(f)
    
    def generate_mutations(self, original_prompt: str, num_variants: int = 5) -> List[str]:
        """Generate prompt variants for testing"""
        mutations = [original_prompt]  # Original prompt
        
        words = original_prompt.split()
        
        for i in range(num_variants - 1):
            if len(words) > 3:
                # Random word replacement
                mutation = words.copy()
                replace_idx = random.randint(0, len(mutation) - 1)
                mutation[replace_idx] = f"[MUTATED_{i}]"
                mutations.append(" ".join(mutation))
                
                # Word reordering
                if len(words) > 5:
                    reorder_mutation = words.copy()
                    idx1, idx2 = random.sample(range(len(reorder_mutation)), 2)
                    reorder_mutation[idx1], reorder_mutation[idx2] = reorder_mutation[idx2], reorder_mutation[idx1]
                    mutations.append(" ".join(reorder_mutation))
                    
                    # Add extra words
                    extra_mutation = words.copy()
                    extra_mutation.insert(random.randint(0, len(extra_mutation)), f"[EXTRA_{i}]"
                    mutations.append(" ".join(extra_mutation))
            else:
                # Simple mutations for short prompts
                mutations.append(f"{original_prompt} [EXTRA_{i}]")
                mutations.append(f"[PREFIX_{i}] {original_prompt}")
        
        return mutations[:num_variants]
    
    def score_mutation(self, chunks: List[Dict], prompt: str) -> Dict[str, float]:
        """Score a mutation based on multiple criteria"""
        if not chunks:
            return {
                "overall_score": 0.0,
                "relevance_score": 0.0,
                "weight_score": 0.0,
                "biological_score": 0.0,
                "diversity_score": 0.0
            }
        
        # Relevance score (average relevance of returned chunks)
        avg_relevance = sum(chunk["relevance"] for chunk in chunks) / len(chunks)
        
        # Weight score (average activation weight)
        avg_weight = sum(chunk["weight"] for chunk in chunks) / len(chunks)
        
        # Biological marker diversity
        all_markers = set()
        for chunk in chunks:
            all_markers.update(chunk["markers"])
        marker_diversity = len(all_markers) / 10.0  # Normalize
        
        # Critical marker coverage
        critical_markers = {"GFAP", "NeuN"}
        critical_coverage = len(all_markers & critical_markers) / len(critical_markers)
        
        # Combined scores
        relevance_score = avg_relevance * 0.3
        weight_score = avg_weight * 0.3
        biological_score = marker_diversity * 0.2
        diversity_score = critical_coverage * 0.2
        
        overall_score = relevance_score + weight_score + biological_score + diversity_score
        
        return {
            "overall_score": min(overall_score, 1.0),
            "relevance_score": relevance_score,
            "weight_score": weight_score,
            "biological_score": biological_score,
            "diversity_score": diversity_score
        }
    
    def test_robustness(self, original_prompt: str, num_variants: int = 5) -> List[Dict]:
        """Test system robustness with prompt mutations"""
        print(f"🧬 Testing robustness with {num_variants} prompt mutations...")
        
        mutations = self.generate_mutations(original_prompt, num_variants)
        mutation_results = []
        
        for i, mutation in enumerate(mutations):
            print(f"   Testing mutation {i+1}: {mutation[:50]}...")
            
            # Simulate chunk retrieval (in real system, this would call cortical_recall)
            chunks = self._simulate_chunk_retrieval(mutation)
            
            # Score the mutation
            scores = self.score_mutation(chunks, mutation)
            
            # Calculate biological coverage
            biological_coverage = self._calculate_biological_coverage(chunks)
            
            result = {
                "mutation_id": i,
                "prompt": mutation,
                "chunks": chunks,
                "scores": scores,
                "biological_coverage": biological_coverage,
                "timestamp": datetime.now().isoformat(),
                "hash": hashlib.md5(mutation.encode()).hexdigest()[:8]
            }
            
            mutation_results.append(result)
            
            print(f"     Score: {scores['overall_score']:.3f}")
            print(f"     Biological markers: {len(biological_coverage)}")
        
        # Sort by overall score
        mutation_results.sort(key=lambda x: x["scores"]["overall_score"], reverse=True)
        
        # Log mutations
        self._log_mutations(mutation_results)
        
        print(f"\n✓ Robustness testing completed.")
        print(f"   Best score: {mutation_results[0]['scores']['overall_score']:.3f}")
        print(f"   Worst score: {mutation_results[-1]['scores']['overall_score']:.3f}")
        
        return mutation_results
    
    def _simulate_chunk_retrieval(self, prompt: str) -> List[Dict]:
        """Simulate chunk retrieval for testing purposes"""
        # This simulates what cortical_recall would return
        # In the real system, this would call the actual retrieval function
        
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())
        
        relevant_chunks = []
        
        for entry in self.metadata:
            chunk_text = entry["text"].lower()
            chunk_words = set(chunk_text.split())
            
            # Calculate word overlap
            overlap = len(prompt_words & chunk_words)
            if overlap > 0:
                relevance = overlap / len(prompt_words)
                
                if relevance > 0.1:  # Threshold for relevance
                    relevant_chunks.append({
                        "chunk_id": entry["chunk_id"],
                        "text": entry["text"],
                        "relevance": relevance,
                        "weight": entry["weight"],
                        "markers": entry["markers"],
                        "file": entry["file"]
                    })
        
        # Sort by relevance and return top chunks
        relevant_chunks.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_chunks[:5]  # Return top 5 chunks
    
    def _calculate_biological_coverage(self, chunks: List[Dict]) -> Dict[str, float]:
        """Calculate coverage of different biological markers"""
        marker_counts = {}
        total_chunks = len(chunks)
        
        for chunk in chunks:
            for marker in chunk["markers"]:
                marker_counts[marker] = marker_counts.get(marker, 0) + 1
        
        # Convert to percentages
        coverage = {marker: count / total_chunks for marker, count in marker_counts.items()}
        return coverage
    
    def _log_mutations(self, mutation_results: List[Dict]):
        """Log mutation results for analysis"""
        os.makedirs(MUTATION_LOG_PATH.parent, exist_ok=True)
        
        # Load existing log
        existing_log = []
        if MUTATION_LOG_PATH.exists():
            with open(MUTATION_LOG_PATH) as f:
                existing_log = json.load(f)
        
        # Add new results
        existing_log.extend(mutation_results)
        
        # Save updated log
        with open(MUTATION_LOG_PATH, 'w') as f:
            json.dump(existing_log, f, indent=2)
        
        print(f"✓ Mutation log saved to {MUTATION_LOG_PATH}")
    
    def analyze_mutation_history(self) -> Dict[str, any]:
        """Analyze mutation testing history"""
        if not MUTATION_LOG_PATH.exists():
            return {"error": "No mutation log found"}
        
        with open(MUTATION_LOG_PATH) as f:
            history = json.load(f)
        
        if not history:
            return {"error": "Empty mutation log"}
        
        # Calculate statistics
        scores = [result["scores"]["overall_score"] for result in history]
        
        analysis = {
            "total_mutations": len(history),
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "score_variance": np.var(scores) if len(scores) > 1 else 0,
            "robustness_rating": self._calculate_robustness_rating(scores)
        }
        
        return analysis
    
    def _calculate_robustness_rating(self, scores: List[float]) -> str:
        """Calculate overall robustness rating"""
        if not scores:
            return "Unknown"
        
        avg_score = sum(scores) / len(scores)
        variance = np.var(scores) if len(scores) > 1 else 0
        
        if avg_score > 0.8 and variance < 0.1:
            return "Excellent"
        elif avg_score > 0.6 and variance < 0.2:
            return "Good"
        elif avg_score > 0.4:
            return "Fair"
        else:
            return "Poor"


def main():
    """Main execution function"""
    print("🧬 Prompt Mutator - Testing system robustness...")
    
    try:
        mutator = PromptMutator()
        
        # Test with a sample prompt
        test_prompt = "security rules and compliance requirements"
        results = mutator.test_robustness(test_prompt, num_variants=5)
        
        # Analyze history
        analysis = mutator.analyze_mutation_history()
        
        print(f"\n[📊] Mutation Analysis:")
        for key, value in analysis.items():
            if key != "error":
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure to run dna_controller.py first to generate metadata")


if __name__ == "__main__":
    import os
    import numpy as np
    main()
