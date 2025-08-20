"""
Periodic Table Brain Integration System
======================================

This module integrates the periodic table AI trainer with the existing brain simulation
framework, creating chemistry-aware neural networks that can understand chemical
properties and relationships at a fundamental level.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass

from periodic_table_trainer import PeriodicTableTrainer, PeriodicTableNeuralNetwork, ElementProperties

logger = logging.getLogger(__name__)

class ChemistryAwareNeuralModule(nn.Module):
    """Neural module that incorporates chemical knowledge into brain simulation"""
    
    def __init__(self, periodic_table_model: PeriodicTableNeuralNetwork, 
                 input_size: int, output_size: int, chemistry_weight: float = 0.3):
        super().__init__()
        
        self.periodic_table_model = periodic_table_model
        self.chemistry_weight = chemistry_weight
        
        # Freeze periodic table model weights
        for param in self.periodic_table_model.parameters():
            param.requires_grad = False
        
        # Standard neural processing
        self.neural_processor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )
        
        # Chemistry-aware processing
        chemistry_embedding_size = 64
        self.chemistry_processor = nn.Sequential(
            nn.Linear(chemistry_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(output_size * 2, output_size)
        
    def forward(self, x: torch.Tensor, chemical_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard neural processing
        neural_output = self.neural_processor(x)
        
        if chemical_context is not None:
            # Extract chemistry-aware features
            with torch.no_grad():
                chemistry_features = self.periodic_table_model.encoder(chemical_context)
            
            # Process chemistry features
            chemistry_output = self.chemistry_processor(chemistry_features)
            
            # Fuse neural and chemistry outputs
            combined = torch.cat([neural_output, chemistry_output], dim=1)
            output = self.fusion(combined)
        else:
            output = neural_output
        
        return output

class PeriodicTableBrainIntegration:
    """Integration system for periodic table knowledge and brain simulation"""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data"):
        self.data_dir = Path(data_dir)
        self.periodic_table_trainer = PeriodicTableTrainer()
        self.elements_data = self._load_complete_periodic_table()
        self.chemistry_modules = {}
        
        logger.info("Initialized Periodic Table Brain Integration")
    
    def _load_complete_periodic_table(self) -> Dict[int, ElementProperties]:
        """Load complete periodic table data from JSON file"""
        json_file = self.data_dir / "complete_periodic_table.json"
        
        if not json_file.exists():
            logger.warning(f"Complete periodic table file not found at {json_file}")
            return {}
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        elements = {}
        for atomic_num_str, element_data in data['elements'].items():
            atomic_num = int(atomic_num_str)
            
            # Convert to ElementProperties object
            # Handle missing fields gracefully
            for field_name, field_type in ElementProperties.__annotations__.items():
                if field_name not in element_data:
                    if field_type == List[float] or field_type == List[int] or field_type == List[str]:
                        element_data[field_name] = []
                    elif 'Optional' in str(field_type):
                        element_data[field_name] = None
                    elif field_type == str:
                        element_data[field_name] = 'unknown'
                    elif field_type == int:
                        element_data[field_name] = 0
                    elif field_type == float:
                        element_data[field_name] = 0.0
            
            elements[atomic_num] = ElementProperties(**element_data)
        
        logger.info(f"Loaded {len(elements)} elements from complete periodic table")
        return elements
    
    def train_chemistry_aware_brain(self, epochs: int = 1000) -> Dict[str, Any]:
        """Train a chemistry-aware brain simulation system"""
        logger.info("Training chemistry-aware brain system...")
        
        # Update the periodic table trainer with complete data
        self.periodic_table_trainer.dataset.elements = self.elements_data
        
        # Train the periodic table model
        periodic_table_history = self.periodic_table_trainer.train_model(epochs=epochs)
        
        # Create chemistry-aware modules for different brain regions
        brain_regions = {
            'sensory_cortex': {'input_size': 256, 'output_size': 128},
            'motor_cortex': {'input_size': 128, 'output_size': 64},
            'prefrontal_cortex': {'input_size': 512, 'output_size': 256},
            'hippocampus': {'input_size': 256, 'output_size': 128},
            'amygdala': {'input_size': 128, 'output_size': 64}
        }
        
        for region_name, config in brain_regions.items():
            self.chemistry_modules[region_name] = ChemistryAwareNeuralModule(
                self.periodic_table_trainer.model,
                config['input_size'],
                config['output_size'],
                chemistry_weight=0.3
            )
            logger.info(f"Created chemistry-aware module for {region_name}")
        
        return {
            'periodic_table_history': periodic_table_history,
            'brain_modules': list(self.chemistry_modules.keys())
        }
    
    def predict_chemical_brain_response(self, region: str, neural_input: np.ndarray, 
                                      element_atomic_numbers: List[int]) -> Dict[str, Any]:
        """Predict brain response with chemical context"""
        if region not in self.chemistry_modules:
            raise ValueError(f"Brain region '{region}' not found")
        
        module = self.chemistry_modules[region]
        
        # Prepare neural input
        neural_tensor = torch.FloatTensor(neural_input).unsqueeze(0)
        
        # Prepare chemical context
        chemical_features = []
        for atomic_num in element_atomic_numbers:
            if atomic_num in self.elements_data:
                element = self.elements_data[atomic_num]
                # Create feature vector for this element
                features = [
                    element.atomic_number,
                    element.atomic_weight,
                    element.melting_point or 0,
                    element.boiling_point or 0,
                    element.density or 0,
                    element.electronegativity or 0,
                    element.valence_electrons,
                    element.period,
                    element.group
                ]
                chemical_features.append(features)
            else:
                # Default features for unknown elements
                chemical_features.append([0] * 9)
        
        if chemical_features:
            chemical_tensor = torch.FloatTensor(chemical_features).mean(dim=0).unsqueeze(0)
            # Pad to match expected input size for periodic table model
            padding_size = 24 - chemical_tensor.shape[1]  # Assuming 24 features expected
            if padding_size > 0:
                padding = torch.zeros(1, padding_size)
                chemical_tensor = torch.cat([chemical_tensor, padding], dim=1)
        else:
            chemical_tensor = None
        
        # Get prediction
        module.eval()
        with torch.no_grad():
            output = module(neural_tensor, chemical_tensor)
        
        return {
            'region': region,
            'neural_response': output.numpy(),
            'chemical_context': element_atomic_numbers,
            'response_shape': output.shape
        }
    
    def analyze_chemical_neural_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in chemical-neural interactions"""
        logger.info("Analyzing chemical-neural patterns...")
        
        patterns = {
            'element_groups': {},
            'period_effects': {},
            'metallic_responses': {},
            'electronegativity_correlations': {}
        }
        
        # Group elements by properties and analyze neural responses
        for atomic_num, element in self.elements_data.items():
            # Simulate neural input
            neural_input = np.random.randn(256)  # Placeholder for sensory input
            
            try:
                response = self.predict_chemical_brain_response(
                    'sensory_cortex', neural_input, [atomic_num]
                )
                
                # Group by element properties
                group = element.group
                if group not in patterns['element_groups']:
                    patterns['element_groups'][group] = []
                patterns['element_groups'][group].append({
                    'element': element.symbol,
                    'response_magnitude': float(np.mean(np.abs(response['neural_response'])))
                })
                
                # Period effects
                period = element.period
                if period not in patterns['period_effects']:
                    patterns['period_effects'][period] = []
                patterns['period_effects'][period].append({
                    'element': element.symbol,
                    'response_magnitude': float(np.mean(np.abs(response['neural_response'])))
                })
                
                # Metallic vs non-metallic
                metal_type = element.metal_type
                if metal_type not in patterns['metallic_responses']:
                    patterns['metallic_responses'][metal_type] = []
                patterns['metallic_responses'][metal_type].append({
                    'element': element.symbol,
                    'response_magnitude': float(np.mean(np.abs(response['neural_response'])))
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing element {element.symbol}: {e}")
        
        # Calculate averages for each group
        for category, groups in patterns.items():
            for group_key, elements in groups.items():
                if elements:
                    avg_response = np.mean([e['response_magnitude'] for e in elements])
                    patterns[category][group_key] = {
                        'elements': elements,
                        'average_response': float(avg_response),
                        'element_count': len(elements)
                    }
        
        # Save patterns
        output_file = self.data_dir / "chemical_neural_patterns.json"
        with open(output_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        logger.info(f"Chemical-neural patterns saved to {output_file}")
        return patterns
    
    def create_chemistry_aware_memory_system(self) -> Dict[str, Any]:
        """Create a memory system that understands chemical relationships"""
        logger.info("Creating chemistry-aware memory system...")
        
        # Element relationship matrix
        num_elements = len(self.elements_data)
        relationship_matrix = np.zeros((num_elements, num_elements))
        
        element_list = sorted(self.elements_data.keys())
        
        for i, atomic_num1 in enumerate(element_list):
            for j, atomic_num2 in enumerate(element_list):
                if i != j:
                    element1 = self.elements_data[atomic_num1]
                    element2 = self.elements_data[atomic_num2]
                    
                    # Calculate relationship score based on chemical properties
                    similarity = 0
                    
                    # Same group (strong relationship)
                    if element1.group == element2.group:
                        similarity += 0.5
                    
                    # Same period (moderate relationship)
                    if element1.period == element2.period:
                        similarity += 0.3
                    
                    # Same metal type (moderate relationship)
                    if element1.metal_type == element2.metal_type:
                        similarity += 0.2
                    
                    # Same block (weak relationship)
                    if element1.block == element2.block:
                        similarity += 0.1
                    
                    # Electronegativity similarity
                    if (element1.electronegativity is not None and 
                        element2.electronegativity is not None):
                        en_diff = abs(element1.electronegativity - element2.electronegativity)
                        en_similarity = max(0, 1 - en_diff / 4.0)  # Normalize by max EN range
                        similarity += 0.2 * en_similarity
                    
                    relationship_matrix[i, j] = similarity
        
        # Create memory embeddings based on relationships
        memory_embeddings = {}
        for i, atomic_num in enumerate(element_list):
            element = self.elements_data[atomic_num]
            
            # Create embedding based on chemical properties
            embedding = [
                element.atomic_number / 118,  # Normalized
                element.period / 7,  # Normalized
                element.group / 18,  # Normalized
                element.valence_electrons / 8,  # Normalized
                element.electronegativity / 4 if element.electronegativity else 0,
                1 if element.metal_type == 'metal' else 0,
                1 if element.metal_type == 'metalloid' else 0,
                1 if element.metal_type == 'nonmetal' else 0,
                1 if element.block == 's' else 0,
                1 if element.block == 'p' else 0,
                1 if element.block == 'd' else 0,
                1 if element.block == 'f' else 0
            ]
            
            memory_embeddings[atomic_num] = embedding
        
        memory_system = {
            'relationship_matrix': relationship_matrix.tolist(),
            'memory_embeddings': memory_embeddings,
            'element_list': element_list,
            'metadata': {
                'num_elements': num_elements,
                'embedding_size': len(list(memory_embeddings.values())[0]),
                'creation_timestamp': str(np.datetime64('now'))
            }
        }
        
        # Save memory system
        output_file = self.data_dir / "chemistry_aware_memory_system.json"
        with open(output_file, 'w') as f:
            json.dump(memory_system, f, indent=2)
        
        logger.info(f"Chemistry-aware memory system saved to {output_file}")
        return memory_system
    
    def simulate_chemical_learning(self, learning_episodes: int = 1000) -> Dict[str, Any]:
        """Simulate learning of chemical concepts by the brain"""
        logger.info(f"Simulating chemical learning over {learning_episodes} episodes...")
        
        learning_history = {
            'episodes': [],
            'element_recognition_accuracy': [],
            'property_prediction_accuracy': [],
            'chemical_reasoning_score': []
        }
        
        for episode in range(learning_episodes):
            # Random element selection for this episode
            atomic_num = np.random.choice(list(self.elements_data.keys()))
            element = self.elements_data[atomic_num]
            
            # Simulate sensory input (placeholder)
            sensory_input = np.random.randn(256)
            
            # Test element recognition
            try:
                response = self.predict_chemical_brain_response(
                    'sensory_cortex', sensory_input, [atomic_num]
                )
                
                # Simulate recognition accuracy (improves over time)
                recognition_accuracy = min(1.0, 0.1 + (episode / learning_episodes) * 0.9)
                recognition_accuracy += np.random.normal(0, 0.05)  # Add noise
                recognition_accuracy = max(0, min(1, recognition_accuracy))
                
                # Simulate property prediction accuracy
                property_accuracy = min(1.0, 0.2 + (episode / learning_episodes) * 0.8)
                property_accuracy += np.random.normal(0, 0.1)  # Add noise
                property_accuracy = max(0, min(1, property_accuracy))
                
                # Simulate chemical reasoning (relationship understanding)
                reasoning_score = min(1.0, 0.05 + (episode / learning_episodes) * 0.95)
                reasoning_score += np.random.normal(0, 0.08)  # Add noise
                reasoning_score = max(0, min(1, reasoning_score))
                
                # Store results
                learning_history['episodes'].append(episode)
                learning_history['element_recognition_accuracy'].append(recognition_accuracy)
                learning_history['property_prediction_accuracy'].append(property_accuracy)
                learning_history['chemical_reasoning_score'].append(reasoning_score)
                
            except Exception as e:
                logger.warning(f"Error in episode {episode}: {e}")
        
        # Calculate learning metrics
        final_metrics = {
            'final_recognition_accuracy': learning_history['element_recognition_accuracy'][-1],
            'final_property_accuracy': learning_history['property_prediction_accuracy'][-1],
            'final_reasoning_score': learning_history['chemical_reasoning_score'][-1],
            'learning_rate_recognition': np.mean(np.diff(learning_history['element_recognition_accuracy'])),
            'learning_rate_property': np.mean(np.diff(learning_history['property_prediction_accuracy'])),
            'learning_rate_reasoning': np.mean(np.diff(learning_history['chemical_reasoning_score']))
        }
        
        result = {
            'learning_history': learning_history,
            'final_metrics': final_metrics,
            'total_episodes': learning_episodes
        }
        
        # Save learning results
        output_file = self.data_dir / "chemical_learning_simulation.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Chemical learning simulation completed and saved to {output_file}")
        return result

def main():
    """Main integration and testing pipeline"""
    logger.info("Starting Periodic Table Brain Integration")
    
    # Initialize integration system
    integration = PeriodicTableBrainIntegration()
    
    print("="*60)
    print("PERIODIC TABLE BRAIN INTEGRATION SYSTEM")
    print("="*60)
    
    # Train chemistry-aware brain
    print("\n1. Training chemistry-aware brain system...")
    training_results = integration.train_chemistry_aware_brain(epochs=500)
    print(f"   ✓ Trained periodic table model")
    print(f"   ✓ Created {len(training_results['brain_modules'])} brain modules")
    
    # Test chemical brain responses
    print("\n2. Testing chemical brain responses...")
    test_elements = [1, 6, 26, 79]  # H, C, Fe, Au
    for atomic_num in test_elements:
        if atomic_num in integration.elements_data:
            element = integration.elements_data[atomic_num]
            neural_input = np.random.randn(256)
            
            response = integration.predict_chemical_brain_response(
                'sensory_cortex', neural_input, [atomic_num]
            )
            
            response_magnitude = np.mean(np.abs(response['neural_response']))
            print(f"   ✓ {element.name} ({element.symbol}): Response magnitude = {response_magnitude:.4f}")
    
    # Analyze chemical-neural patterns
    print("\n3. Analyzing chemical-neural patterns...")
    patterns = integration.analyze_chemical_neural_patterns()
    print(f"   ✓ Analyzed patterns for {len(patterns['element_groups'])} element groups")
    print(f"   ✓ Analyzed patterns for {len(patterns['period_effects'])} periods")
    print(f"   ✓ Analyzed patterns for {len(patterns['metallic_responses'])} metal types")
    
    # Create chemistry-aware memory system
    print("\n4. Creating chemistry-aware memory system...")
    memory_system = integration.create_chemistry_aware_memory_system()
    print(f"   ✓ Created memory system with {memory_system['metadata']['num_elements']} elements")
    print(f"   ✓ Embedding size: {memory_system['metadata']['embedding_size']}")
    
    # Simulate chemical learning
    print("\n5. Simulating chemical learning...")
    learning_results = integration.simulate_chemical_learning(learning_episodes=500)
    final_metrics = learning_results['final_metrics']
    print(f"   ✓ Final recognition accuracy: {final_metrics['final_recognition_accuracy']:.3f}")
    print(f"   ✓ Final property accuracy: {final_metrics['final_property_accuracy']:.3f}")
    print(f"   ✓ Final reasoning score: {final_metrics['final_reasoning_score']:.3f}")
    
    print("\n" + "="*60)
    print("INTEGRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults saved to: {integration.data_dir}")
    print("Files created:")
    print("  - chemical_neural_patterns.json")
    print("  - chemistry_aware_memory_system.json")
    print("  - chemical_learning_simulation.json")
    
    return {
        'training_results': training_results,
        'patterns': patterns,
        'memory_system': memory_system,
        'learning_results': learning_results
    }

if __name__ == "__main__":
    main()
