#!/usr/bin/env python3
"""
Wolfram Alpha Enhanced Brain Training
====================================

Complete integration of Wolfram Alpha with your Quark brain simulation.
This script demonstrates how to enhance your existing brain simulation
with Wolfram Alpha's computational power for:

1. Real-time parameter optimization
2. Mathematical validation of neural dynamics
3. Connectivity matrix analysis
4. Neural oscillation pattern analysis
5. Training enhancement and guidance

Run this to see your brain simulation supercharged with Wolfram Alpha!
"""

import sys
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our Wolfram integration
from development.src.core.wolfram_brain_integration import WolframBrainConnector, BrainState, create_sample_brain_state

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WolframEnhancedBrainSimulation:
    """
    Brain simulation enhanced with Wolfram Alpha computational intelligence
    """
    
    def __init__(self):
        self.wolfram_connector = WolframBrainConnector()
        self.training_history = []
        self.wolfram_enhancements = []
        
        # Create enhanced results directory
        self.results_dir = Path("./data/wolfram_enhanced_training")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠 Initialized Wolfram Enhanced Brain Simulation")

    def create_enhanced_neural_network(self, input_size: int = 64, hidden_size: int = 128, output_size: int = 32):
        """Create a neural network that will be enhanced by Wolfram Alpha"""
        
        class EnhancedBrainNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.cortex = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, output_size)
                )
                
                self.hippocampus = nn.Sequential(
                    nn.Linear(input_size, hidden_size // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_size // 2, output_size // 2)
                )
                
                self.thalamus = nn.Sequential(
                    nn.Linear(input_size, hidden_size // 4),
                    nn.Sigmoid(),
                    nn.Linear(hidden_size // 4, output_size // 4)
                )
            
            def forward(self, x):
                cortex_out = self.cortex(x)
                hippocampus_out = self.hippocampus(x)
                thalamus_out = self.thalamus(x)
                
                # Combine outputs (simple concatenation)
                combined = torch.cat([
                    cortex_out,
                    hippocampus_out,
                    thalamus_out
                ], dim=1)
                
                return {
                    'cortex': cortex_out,
                    'hippocampus': hippocampus_out,
                    'thalamus': thalamus_out,
                    'combined': combined
                }
        
        return EnhancedBrainNetwork(input_size, hidden_size, output_size)

    def extract_brain_state_from_model(self, model, input_data):
        """Extract brain state from current model for Wolfram analysis"""
        model.eval()
        with torch.no_grad():
            outputs = model(input_data)
            
            # Extract neural activities
            cortex_activity = outputs['cortex'].flatten().cpu().numpy()
            hippocampus_activity = outputs['hippocampus'].flatten().cpu().numpy()
            thalamus_activity = outputs['thalamus'].flatten().cpu().numpy()
            
            # Create simplified connectivity matrix from model weights
            cortex_weights = model.cortex[0].weight.data.cpu().numpy()
            connectivity_matrix = np.corrcoef(cortex_weights[:8, :8])  # Take subset for analysis
            
            # Extract current parameters
            parameters = {
                'learning_rate': 0.001,  # Would come from optimizer
                'batch_size': input_data.shape[0],
                'hidden_size': cortex_weights.shape[0],
                'cortex_mean_activity': float(np.mean(cortex_activity)),
                'hippocampus_mean_activity': float(np.mean(hippocampus_activity)),
                'thalamus_mean_activity': float(np.mean(thalamus_activity))
            }
            
            return BrainState(
                cortex_activity=cortex_activity,
                hippocampus_activity=hippocampus_activity,
                thalamus_activity=thalamus_activity,
                connectivity_matrix=connectivity_matrix,
                timestamp=datetime.now().timestamp(),
                parameters=parameters
            )

    def wolfram_enhanced_training_step(self, model, optimizer, criterion, train_data, train_targets, epoch):
        """Enhanced training step with Wolfram Alpha analysis"""
        
        # Regular training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_data)
        loss = criterion(outputs['combined'], train_targets)
        
        loss.backward()
        optimizer.step()
        
        # Wolfram Alpha enhancement every few epochs
        if epoch % 5 == 0:  # Analyze every 5 epochs
            logger.info(f"🔬 Epoch {epoch}: Running Wolfram Alpha analysis...")
            
            # Extract current brain state
            brain_state = self.extract_brain_state_from_model(model, train_data)
            
            # Get Wolfram Alpha enhancement
            enhancement = self.wolfram_connector.enhance_brain_state(brain_state)
            
            if enhancement.success:
                # Apply parameter optimizations
                self.apply_wolfram_optimizations(optimizer, enhancement.parameter_optimizations)
                
                # Store enhancement for analysis
                self.wolfram_enhancements.append({
                    'epoch': epoch,
                    'enhancement': enhancement,
                    'loss': float(loss.item())
                })
                
                logger.info(f"✨ Applied Wolfram Alpha optimizations at epoch {epoch}")
            else:
                logger.warning(f"⚠️ Wolfram Alpha analysis failed at epoch {epoch}")
        
        return float(loss.item())

    def apply_wolfram_optimizations(self, optimizer, optimizations: Dict[str, float]):
        """Apply Wolfram Alpha parameter optimizations to the optimizer"""
        
        # Apply learning rate optimization if available
        if 'learning_rate' in optimizations:
            new_lr = optimizations['learning_rate']
            # Clamp to reasonable bounds
            new_lr = max(1e-6, min(0.1, new_lr))
            
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = new_lr
                logger.info(f"📈 Updated learning rate: {old_lr:.6f} → {new_lr:.6f}")

    def run_enhanced_training(self, epochs: int = 50, batch_size: int = 32):
        """Run complete Wolfram Alpha enhanced training"""
        
        logger.info("🚀 Starting Wolfram Alpha Enhanced Brain Training...")
        
        # Create model
        model = self.create_enhanced_neural_network()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Generate training data
        np.random.seed(42)
        train_data = torch.randn(batch_size, 64)
        train_targets = torch.randn(batch_size, 56)  # 32 + 16 + 8 = 56 total output
        
        # Training loop
        losses = []
        
        for epoch in range(epochs):
            # Enhanced training step
            loss = self.wolfram_enhanced_training_step(
                model, optimizer, criterion, train_data, train_targets, epoch
            )
            
            losses.append(loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        # Final analysis
        final_brain_state = self.extract_brain_state_from_model(model, train_data)
        final_enhancement = self.wolfram_connector.enhance_brain_state(final_brain_state)
        
        # Save results
        results = {
            'training_losses': losses,
            'wolfram_enhancements': len(self.wolfram_enhancements),
            'final_enhancement': final_enhancement.__dict__ if final_enhancement.success else None,
            'final_loss': losses[-1],
            'improvement': (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] != 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        results_file = self.results_dir / f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Training results saved to {results_file}")
        
        # Visualize results
        self.visualize_enhanced_training(losses, results)
        
        return results

    def visualize_enhanced_training(self, losses: List[float], results: Dict):
        """Create visualizations of the enhanced training"""
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss curve
        axes[0, 0].plot(losses, 'b-', alpha=0.8, linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Wolfram Alpha Enhanced Training Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark Wolfram enhancements
        enhancement_epochs = [enh['epoch'] for enh in self.wolfram_enhancements]
        if enhancement_epochs:
            enhancement_losses = [losses[epoch] for epoch in enhancement_epochs]
            axes[0, 0].scatter(enhancement_epochs, enhancement_losses, 
                             color='red', s=100, alpha=0.8, marker='*', 
                             label='Wolfram Enhancement', zorder=5)
            axes[0, 0].legend()
        
        # Loss improvement over time
        if len(losses) > 1:
            improvements = [(losses[0] - loss) / losses[0] * 100 for loss in losses]
            axes[0, 1].plot(improvements, 'g-', alpha=0.8, linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Improvement (%)')
            axes[0, 1].set_title('Training Improvement Over Time')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Enhancement frequency
        if self.wolfram_enhancements:
            enhancement_losses = [enh['loss'] for enh in self.wolfram_enhancements]
            axes[1, 0].bar(range(len(enhancement_losses)), enhancement_losses, 
                          alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_xlabel('Enhancement Number')
            axes[1, 0].set_ylabel('Loss at Enhancement')
            axes[1, 0].set_title('Loss at Each Wolfram Enhancement')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        summary_text = f"""
Wolfram Alpha Enhanced Training Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Training Epochs: {len(losses)}
🎯 Final Loss: {losses[-1]:.6f}
📈 Total Improvement: {results['improvement']:.1f}%
⚡ Wolfram Enhancements: {len(self.wolfram_enhancements)}
🧠 Enhanced Brain Components: Cortex, Hippocampus, Thalamus

🏆 Performance Metrics:
   • Initial Loss: {losses[0]:.6f}
   • Final Loss: {losses[-1]:.6f}
   • Best Loss: {min(losses):.6f}
   • Avg Enhancement Impact: {np.mean([enh['loss'] for enh in self.wolfram_enhancements]) if self.wolfram_enhancements else 0:.6f}

🔬 Wolfram Alpha Contributions:
   • Parameter optimization
   • Connectivity analysis  
   • Neural dynamics validation
   • Oscillation pattern analysis
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"enhanced_training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"📊 Training visualization saved to {plot_file}")

def main():
    """Main demonstration function"""
    
    print("🌟 WOLFRAM ALPHA ENHANCED BRAIN SIMULATION")
    print("=" * 60)
    print("Integrating Wolfram Alpha's computational power with your brain simulation!")
    print("=" * 60)
    
    # Create enhanced simulation
    enhanced_sim = WolframEnhancedBrainSimulation()
    
    # Run enhanced training
    results = enhanced_sim.run_enhanced_training(epochs=30, batch_size=32)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🧠 Final Loss: {results['final_loss']:.6f}")
    print(f"📈 Total Improvement: {results['improvement']:.1f}%")
    print(f"⚡ Wolfram Enhancements Applied: {results['wolfram_enhancements']}")
    
    if results['final_enhancement']:
        print(f"🔬 Final Wolfram Analysis: Success")
        print(f"📊 Parameter Optimizations: {len(results['final_enhancement']['parameter_optimizations'])}")
    
    print(f"\n💾 All results saved to: {enhanced_sim.results_dir}")
    
    print("\n🚀 Your brain simulation is now enhanced with Wolfram Alpha!")
    print("Next steps:")
    print("1. Integrate with your existing brain_launcher_v3.py")
    print("2. Use Wolfram validations in your neural training")
    print("3. Apply mathematical insights to improve accuracy")
    print("4. Scale up with your full brain simulation architecture")

if __name__ == "__main__":
    main()
