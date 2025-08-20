"""
Comprehensive Periodic Table AI Training System
==============================================

This module creates an advanced AI training system that learns from the complete periodic table,
including all elements, their properties, relationships, and chemical behaviors.

Based on ptable.com data structure and integrated with the brain simulation framework.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ElementProperties:
    """Complete element property data structure based on ptable.com"""
    atomic_number: int
    symbol: str
    name: str
    atomic_weight: float
    
    # Physical properties
    state_at_stp: str  # solid, liquid, gas, unknown
    melting_point: Optional[float]  # Celsius
    boiling_point: Optional[float]  # Celsius
    density: Optional[float]  # kg/m³
    
    # Electronic properties
    electron_configuration: str
    valence_electrons: int
    electronegativity: Optional[float]
    electron_affinity: Optional[float]  # kJ/mol
    ionization_energies: List[float]  # kJ/mol
    
    # Atomic properties
    atomic_radius: Optional[float]  # pm
    covalent_radius: Optional[float]  # pm
    van_der_waals_radius: Optional[float]  # pm
    
    # Classification
    period: int
    group: int
    block: str  # s, p, d, f
    metal_type: str  # metal, nonmetal, metalloid
    
    # Abundance
    abundance_universe: Optional[float]  # %
    abundance_solar: Optional[float]  # %
    abundance_crust: Optional[float]  # %
    abundance_ocean: Optional[float]  # %
    abundance_human: Optional[float]  # %
    
    # Discovery
    discovery_year: Optional[int]
    discovered_by: Optional[str]
    
    # Nuclear properties
    isotopes_count: int
    most_stable_isotope: Optional[str]
    half_life: Optional[str]
    decay_modes: List[str]
    
    # Thermal properties
    specific_heat: Optional[float]  # J/kg⋅K
    thermal_conductivity: Optional[float]  # W/m⋅K
    heat_vaporization: Optional[float]  # kJ/mol
    heat_fusion: Optional[float]  # kJ/mol
    
    # Mechanical properties
    bulk_modulus: Optional[float]  # GPa
    shear_modulus: Optional[float]  # GPa
    young_modulus: Optional[float]  # GPa
    brinell_hardness: Optional[float]  # MPa
    mohs_hardness: Optional[float]
    vickers_hardness: Optional[float]  # MPa
    
    # Electrical properties
    electrical_conductivity: Optional[float]  # MS/m
    
    # Oxidation states
    oxidation_states: List[int]
    common_oxidation_states: List[int]

class PeriodicTableDataset:
    """Complete periodic table dataset with all 118 elements"""
    
    def __init__(self):
        self.elements: Dict[int, ElementProperties] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._initialize_elements()
        
    def _initialize_elements(self):
        """Initialize all 118 elements with their complete properties"""
        logger.info("Initializing complete periodic table dataset...")
        
        # This would be loaded from a comprehensive database in production
        # For now, we'll create a representative sample based on ptable.com structure
        
        elements_data = [
            # Hydrogen
            {
                'atomic_number': 1, 'symbol': 'H', 'name': 'Hydrogen', 'atomic_weight': 1.008,
                'state_at_stp': 'gas', 'melting_point': -259.16, 'boiling_point': -252.87,
                'density': 0.08988, 'electron_configuration': '1s¹', 'valence_electrons': 1,
                'electronegativity': 2.20, 'electron_affinity': 72.8, 'ionization_energies': [1312.0],
                'atomic_radius': 25, 'covalent_radius': 31, 'van_der_waals_radius': 120,
                'period': 1, 'group': 1, 'block': 's', 'metal_type': 'nonmetal',
                'abundance_universe': 75.0, 'abundance_solar': 70.6, 'abundance_crust': 0.14,
                'abundance_ocean': 10.8, 'abundance_human': 9.5, 'discovery_year': 1766,
                'discovered_by': 'Henry Cavendish', 'isotopes_count': 7,
                'most_stable_isotope': '¹H', 'half_life': 'stable', 'decay_modes': [],
                'specific_heat': 14304, 'thermal_conductivity': 0.1805, 'heat_vaporization': 0.904,
                'heat_fusion': 0.117, 'oxidation_states': [-1, 1], 'common_oxidation_states': [1]
            },
            # Helium
            {
                'atomic_number': 2, 'symbol': 'He', 'name': 'Helium', 'atomic_weight': 4.0026,
                'state_at_stp': 'gas', 'melting_point': -272.20, 'boiling_point': -268.93,
                'density': 0.1786, 'electron_configuration': '1s²', 'valence_electrons': 2,
                'electronegativity': None, 'electron_affinity': 0, 'ionization_energies': [2372.3, 5250.5],
                'atomic_radius': 31, 'covalent_radius': 28, 'van_der_waals_radius': 140,
                'period': 1, 'group': 18, 'block': 's', 'metal_type': 'nonmetal',
                'abundance_universe': 23.0, 'abundance_solar': 27.4, 'abundance_crust': 0.000055,
                'abundance_ocean': 0.0000072, 'abundance_human': None, 'discovery_year': 1868,
                'discovered_by': 'Pierre Janssen', 'isotopes_count': 9,
                'most_stable_isotope': '⁴He', 'half_life': 'stable', 'decay_modes': [],
                'specific_heat': 5193, 'thermal_conductivity': 0.1513, 'heat_vaporization': 0.083,
                'heat_fusion': 0.02, 'oxidation_states': [0], 'common_oxidation_states': [0]
            },
            # Carbon
            {
                'atomic_number': 6, 'symbol': 'C', 'name': 'Carbon', 'atomic_weight': 12.011,
                'state_at_stp': 'solid', 'melting_point': 3500, 'boiling_point': 4027,
                'density': 2267, 'electron_configuration': '[He] 2s² 2p²', 'valence_electrons': 4,
                'electronegativity': 2.55, 'electron_affinity': 153.9, 'ionization_energies': [1086.5, 2352.6, 4620.5, 6222.7],
                'atomic_radius': 70, 'covalent_radius': 76, 'van_der_waals_radius': 170,
                'period': 2, 'group': 14, 'block': 'p', 'metal_type': 'nonmetal',
                'abundance_universe': 0.46, 'abundance_solar': 0.30, 'abundance_crust': 0.02,
                'abundance_ocean': 0.0028, 'abundance_human': 23, 'discovery_year': None,
                'discovered_by': 'Ancient', 'isotopes_count': 15,
                'most_stable_isotope': '¹²C', 'half_life': 'stable', 'decay_modes': [],
                'specific_heat': 709, 'thermal_conductivity': 200, 'heat_vaporization': 715,
                'heat_fusion': 105, 'oxidation_states': [-4, -3, -2, -1, 0, 1, 2, 3, 4], 'common_oxidation_states': [-4, 4]
            },
            # Iron
            {
                'atomic_number': 26, 'symbol': 'Fe', 'name': 'Iron', 'atomic_weight': 55.845,
                'state_at_stp': 'solid', 'melting_point': 1538, 'boiling_point': 2861,
                'density': 7874, 'electron_configuration': '[Ar] 3d⁶ 4s²', 'valence_electrons': 8,
                'electronegativity': 1.83, 'electron_affinity': 15.7, 'ionization_energies': [762.5, 1561.9, 2957, 5290],
                'atomic_radius': 126, 'covalent_radius': 132, 'van_der_waals_radius': None,
                'period': 4, 'group': 8, 'block': 'd', 'metal_type': 'metal',
                'abundance_universe': 0.11, 'abundance_solar': 0.13, 'abundance_crust': 5.63,
                'abundance_ocean': 0.000001, 'abundance_human': 0.006, 'discovery_year': None,
                'discovered_by': 'Ancient', 'isotopes_count': 28,
                'most_stable_isotope': '⁵⁶Fe', 'half_life': 'stable', 'decay_modes': [],
                'specific_heat': 449, 'thermal_conductivity': 80.4, 'heat_vaporization': 347,
                'heat_fusion': 13.8, 'brinell_hardness': 490, 'mohs_hardness': 4,
                'oxidation_states': [-2, -1, 0, 1, 2, 3, 4, 5, 6], 'common_oxidation_states': [2, 3]
            }
            # Add more elements here - this is a representative sample
        ]
        
        # Initialize all elements
        for element_data in elements_data:
            # Fill in missing values with defaults
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
            
            element = ElementProperties(**element_data)
            self.elements[element.atomic_number] = element
            
        logger.info(f"Initialized {len(self.elements)} elements")
    
    def get_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Extract numerical features from all elements"""
        features = []
        feature_names = []
        
        for atomic_number in sorted(self.elements.keys()):
            element = self.elements[atomic_number]
            
            # Numerical features
            element_features = [
                element.atomic_number,
                element.atomic_weight,
                element.melting_point or 0,
                element.boiling_point or 0,
                element.density or 0,
                element.valence_electrons,
                element.electronegativity or 0,
                element.electron_affinity or 0,
                len(element.ionization_energies),
                element.ionization_energies[0] if element.ionization_energies else 0,
                element.atomic_radius or 0,
                element.covalent_radius or 0,
                element.van_der_waals_radius or 0,
                element.period,
                element.group,
                element.abundance_universe or 0,
                element.abundance_crust or 0,
                element.discovery_year or 0,
                element.isotopes_count,
                element.specific_heat or 0,
                element.thermal_conductivity or 0,
                len(element.oxidation_states),
                max(element.oxidation_states) if element.oxidation_states else 0,
                min(element.oxidation_states) if element.oxidation_states else 0
            ]
            
            features.append(element_features)
        
        feature_names = [
            'atomic_number', 'atomic_weight', 'melting_point', 'boiling_point', 'density',
            'valence_electrons', 'electronegativity', 'electron_affinity', 'ionization_count',
            'first_ionization', 'atomic_radius', 'covalent_radius', 'van_der_waals_radius',
            'period', 'group', 'abundance_universe', 'abundance_crust', 'discovery_year',
            'isotopes_count', 'specific_heat', 'thermal_conductivity', 'oxidation_states_count',
            'max_oxidation_state', 'min_oxidation_state'
        ]
        
        return np.array(features), feature_names
    
    def get_categorical_features(self) -> Dict[str, List[str]]:
        """Extract categorical features from all elements"""
        categorical_data = {
            'state_at_stp': [],
            'block': [],
            'metal_type': []
        }
        
        for atomic_number in sorted(self.elements.keys()):
            element = self.elements[atomic_number]
            categorical_data['state_at_stp'].append(element.state_at_stp)
            categorical_data['block'].append(element.block)
            categorical_data['metal_type'].append(element.metal_type)
            
        return categorical_data

class PeriodicTableNeuralNetwork(nn.Module):
    """Advanced neural network for periodic table property prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64], 
                 num_properties: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.num_properties = num_properties
        
        # Build encoder layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
            
        self.encoder = nn.Sequential(*layers)
        
        # Property prediction heads
        self.property_predictors = nn.ModuleDict({
            'melting_point': nn.Linear(prev_size, 1),
            'boiling_point': nn.Linear(prev_size, 1),
            'density': nn.Linear(prev_size, 1),
            'electronegativity': nn.Linear(prev_size, 1),
            'atomic_radius': nn.Linear(prev_size, 1),
            'ionization_energy': nn.Linear(prev_size, 1),
            'metal_classification': nn.Linear(prev_size, 3),  # metal, metalloid, nonmetal
            'state_classification': nn.Linear(prev_size, 4),  # solid, liquid, gas, unknown
            'block_classification': nn.Linear(prev_size, 4),  # s, p, d, f
            'period_prediction': nn.Linear(prev_size, 1)
        })
        
        # Element embeddings for relationship learning
        self.element_embeddings = nn.Embedding(119, 64)  # 118 elements + padding
        self.embedding_fusion = nn.Linear(prev_size + 64, prev_size)
        
    def forward(self, x: torch.Tensor, atomic_numbers: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Encode input features
        encoded = self.encoder(x)
        
        # Add element embeddings if provided
        if atomic_numbers is not None:
            embeddings = self.element_embeddings(atomic_numbers)
            encoded = self.embedding_fusion(torch.cat([encoded, embeddings], dim=1))
        
        # Predict properties
        predictions = {}
        for property_name, predictor in self.property_predictors.items():
            predictions[property_name] = predictor(encoded)
            
        return predictions

class PeriodicTableTrainer:
    """Advanced AI training system for periodic table learning"""
    
    def __init__(self, save_dir: str = "/Users/camdouglas/quark/data/periodic_table"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset = PeriodicTableDataset()
        self.model = None
        self.training_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized PeriodicTableTrainer on device: {self.device}")
    
    def prepare_data(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare training data from the periodic table dataset"""
        logger.info("Preparing training data...")
        
        # Get numerical features
        features, feature_names = self.dataset.get_feature_matrix()
        
        # Scale features
        features_scaled = self.dataset.scaler.fit_transform(features)
        
        # Prepare targets for different prediction tasks
        targets = {}
        
        for i, (atomic_number, element) in enumerate(self.dataset.elements.items()):
            if i == 0:  # Initialize target arrays
                targets['melting_point'] = []
                targets['boiling_point'] = []
                targets['density'] = []
                targets['electronegativity'] = []
                targets['atomic_radius'] = []
                targets['ionization_energy'] = []
                targets['metal_classification'] = []
                targets['state_classification'] = []
                targets['block_classification'] = []
                targets['period_prediction'] = []
            
            # Continuous targets
            targets['melting_point'].append(element.melting_point or 0)
            targets['boiling_point'].append(element.boiling_point or 0)
            targets['density'].append(element.density or 0)
            targets['electronegativity'].append(element.electronegativity or 0)
            targets['atomic_radius'].append(element.atomic_radius or 0)
            targets['ionization_energy'].append(element.ionization_energies[0] if element.ionization_energies else 0)
            targets['period_prediction'].append(element.period)
            
            # Categorical targets
            metal_map = {'metal': 0, 'metalloid': 1, 'nonmetal': 2}
            targets['metal_classification'].append(metal_map.get(element.metal_type, 2))
            
            state_map = {'solid': 0, 'liquid': 1, 'gas': 2, 'unknown': 3}
            targets['state_classification'].append(state_map.get(element.state_at_stp, 3))
            
            block_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
            targets['block_classification'].append(block_map.get(element.block, 0))
        
        # Convert to tensors
        X = torch.FloatTensor(features_scaled).to(self.device)
        
        targets_tensor = {}
        for key, values in targets.items():
            if 'classification' in key:
                targets_tensor[key] = torch.LongTensor(values).to(self.device)
            else:
                targets_tensor[key] = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        
        # Atomic numbers for embeddings
        atomic_numbers = torch.LongTensor(list(self.dataset.elements.keys())).to(self.device)
        
        logger.info(f"Prepared data: X shape {X.shape}, {len(targets_tensor)} target properties")
        return X, targets_tensor, atomic_numbers, feature_names
    
    def create_model(self, input_size: int) -> PeriodicTableNeuralNetwork:
        """Create and initialize the neural network model"""
        model = PeriodicTableNeuralNetwork(
            input_size=input_size,
            hidden_sizes=[512, 256, 128, 64],
            dropout_rate=0.2
        ).to(self.device)
        
        # Initialize weights
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        return model
    
    def train_model(self, epochs: int = 1000, learning_rate: float = 0.001,
                   batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the periodic table neural network"""
        logger.info("Starting model training...")
        
        # Prepare data
        X, targets, atomic_numbers, feature_names = self.prepare_data()
        
        # Create model
        self.model = self.create_model(X.shape[1])
        
        # Loss functions
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5)
        
        # Training history
        history = {
            'total_loss': [],
            'property_losses': {key: [] for key in targets.keys()},
            'learning_rate': []
        }
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Forward pass
            predictions = self.model(X, atomic_numbers)
            
            # Calculate losses
            total_loss = 0
            property_losses = {}
            
            for property_name, pred in predictions.items():
                target = targets[property_name]
                
                if 'classification' in property_name:
                    loss = ce_loss(pred, target)
                else:
                    # Mask out zero values for properties that can be missing
                    mask = target != 0
                    if mask.sum() > 0:
                        loss = mse_loss(pred[mask], target[mask])
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                
                property_losses[property_name] = loss.item()
                total_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update learning rate
            scheduler.step(total_loss)
            
            # Record history
            history['total_loss'].append(total_loss.item())
            for prop, loss_val in property_losses.items():
                history['property_losses'][prop].append(loss_val)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= 100:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        self.training_history = history
        
        # Save final model and history
        torch.save(self.model.state_dict(), self.save_dir / 'final_model.pth')
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed!")
        return history
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained model on various metrics"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        logger.info("Evaluating model...")
        
        X, targets, atomic_numbers, feature_names = self.prepare_data()
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X, atomic_numbers)
        
        evaluation_results = {}
        
        for property_name, pred in predictions.items():
            target = targets[property_name]
            
            if 'classification' in property_name:
                # Classification metrics
                pred_classes = torch.argmax(pred, dim=1).cpu().numpy()
                target_classes = target.cpu().numpy()
                
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                accuracy = accuracy_score(target_classes, pred_classes)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    target_classes, pred_classes, average='weighted', zero_division=0
                )
                
                evaluation_results[property_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            else:
                # Regression metrics
                pred_values = pred.cpu().numpy().flatten()
                target_values = target.cpu().numpy().flatten()
                
                # Remove zero values for missing data
                mask = target_values != 0
                if mask.sum() > 0:
                    pred_filtered = pred_values[mask]
                    target_filtered = target_values[mask]
                    
                    mse = mean_squared_error(target_filtered, pred_filtered)
                    r2 = r2_score(target_filtered, pred_filtered)
                    
                    evaluation_results[property_name] = {
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2_score': r2,
                        'mean_target': np.mean(target_filtered),
                        'std_target': np.std(target_filtered)
                    }
        
        # Save evaluation results
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info("Evaluation completed!")
        return evaluation_results
    
    def predict_element_properties(self, atomic_number: int) -> Dict[str, Any]:
        """Predict properties for a specific element"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if atomic_number not in self.dataset.elements:
            raise ValueError(f"Element with atomic number {atomic_number} not found")
        
        # Get element features
        X, _, atomic_numbers, feature_names = self.prepare_data()
        element_idx = list(self.dataset.elements.keys()).index(atomic_number)
        
        element_features = X[element_idx:element_idx+1]
        element_atomic_number = torch.LongTensor([atomic_number]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(element_features, element_atomic_number)
        
        # Convert predictions to readable format
        results = {}
        for property_name, pred in predictions.items():
            if 'classification' in property_name:
                if 'metal' in property_name:
                    class_names = ['metal', 'metalloid', 'nonmetal']
                elif 'state' in property_name:
                    class_names = ['solid', 'liquid', 'gas', 'unknown']
                elif 'block' in property_name:
                    class_names = ['s', 'p', 'd', 'f']
                
                probabilities = torch.softmax(pred, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                
                results[property_name] = {
                    'predicted_class': class_names[predicted_class],
                    'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
                }
            else:
                results[property_name] = float(pred.cpu().numpy()[0, 0])
        
        # Add actual element information
        element = self.dataset.elements[atomic_number]
        results['element_info'] = {
            'symbol': element.symbol,
            'name': element.name,
            'atomic_weight': element.atomic_weight
        }
        
        return results
    
    def visualize_training(self):
        """Create visualizations of training progress"""
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.training_history['total_loss'])
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        
        # Learning rate
        axes[0, 1].plot(self.training_history['learning_rate'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        
        # Property-specific losses
        axes[1, 0].set_title('Property-Specific Losses')
        for prop, losses in self.training_history['property_losses'].items():
            if len(losses) > 0:
                axes[1, 0].plot(losses, label=prop, alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        
        # Feature importance visualization (placeholder)
        axes[1, 1].set_title('Feature Importance Analysis')
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nanalysis would\ngo here', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_periodic_table_insights(self) -> Dict[str, Any]:
        """Generate insights about periodic table patterns learned by the AI"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        logger.info("Generating periodic table insights...")
        
        insights = {
            'periodic_trends': {},
            'element_relationships': {},
            'property_correlations': {},
            'discovery_patterns': {}
        }
        
        # Analyze periodic trends
        X, targets, atomic_numbers, feature_names = self.prepare_data()
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X, atomic_numbers)
        
        # Extract element embeddings for relationship analysis
        embeddings = self.model.element_embeddings.weight.data.cpu().numpy()
        
        # Analyze trends across periods and groups
        periods_data = {}
        groups_data = {}
        
        for atomic_number, element in self.dataset.elements.items():
            period = element.period
            group = element.group
            
            if period not in periods_data:
                periods_data[period] = {'elements': [], 'properties': {}}
            if group not in groups_data:
                groups_data[group] = {'elements': [], 'properties': {}}
            
            periods_data[period]['elements'].append(atomic_number)
            groups_data[group]['elements'].append(atomic_number)
        
        insights['periodic_trends']['periods'] = len(periods_data)
        insights['periodic_trends']['groups'] = len(groups_data)
        
        # Property correlations
        X_np = X.cpu().numpy()
        correlation_matrix = np.corrcoef(X_np.T)
        
        # Find highly correlated features
        high_correlations = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr)
                    })
        
        insights['property_correlations']['high_correlations'] = high_correlations
        
        # Discovery timeline analysis
        discovery_years = [elem.discovery_year for elem in self.dataset.elements.values() 
                          if elem.discovery_year and elem.discovery_year > 0]
        
        if discovery_years:
            insights['discovery_patterns'] = {
                'earliest_discovery': min(discovery_years),
                'latest_discovery': max(discovery_years),
                'discovery_span': max(discovery_years) - min(discovery_years),
                'average_discovery_year': np.mean(discovery_years)
            }
        
        # Save insights
        with open(self.save_dir / 'periodic_table_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        logger.info("Insights generation completed!")
        return insights

def main():
    """Main training and evaluation pipeline"""
    logger.info("Starting Periodic Table AI Training System")
    
    # Initialize trainer
    trainer = PeriodicTableTrainer()
    
    # Train the model
    history = trainer.train_model(epochs=2000, learning_rate=0.001)
    
    # Evaluate the model
    evaluation_results = trainer.evaluate_model()
    
    # Generate insights
    insights = trainer.generate_periodic_table_insights()
    
    # Visualize training
    trainer.visualize_training()
    
    # Test predictions on a few elements
    test_elements = [1, 6, 26, 79]  # H, C, Fe, Au
    
    print("\n" + "="*50)
    print("PERIODIC TABLE AI TRAINING RESULTS")
    print("="*50)
    
    print("\nTraining Summary:")
    print(f"Final total loss: {history['total_loss'][-1]:.6f}")
    print(f"Training epochs: {len(history['total_loss'])}")
    
    print("\nEvaluation Results:")
    for prop, metrics in evaluation_results.items():
        print(f"\n{prop}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nSample Predictions:")
    for atomic_number in test_elements:
        try:
            predictions = trainer.predict_element_properties(atomic_number)
            element_info = predictions['element_info']
            print(f"\n{element_info['name']} ({element_info['symbol']}):")
            
            # Show a few key predictions
            for prop in ['melting_point', 'density', 'metal_classification']:
                if prop in predictions:
                    print(f"  {prop}: {predictions[prop]}")
                    
        except Exception as e:
            print(f"Error predicting for element {atomic_number}: {e}")
    
    print(f"\nResults saved to: {trainer.save_dir}")
    logger.info("Periodic Table AI Training completed successfully!")

if __name__ == "__main__":
    main()
