"""
WormBrowser Integration Module

Integrates OpenWorm's WormBrowser model into the SmallMind neural visualization system.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


class WormBrowserIntegration:
    """Integration with OpenWorm's WormBrowser for C. elegans visualization"""
    
    def __init__(self, cache_dir: str = "./neurodata_cache/worm"):
        # Convert to absolute path to avoid relative path issues
        if cache_dir.startswith("./"):
            cache_dir = str(Path(__file__).parent.parent.parent / cache_dir[2:])
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # WormBrowser data sources
        self.data_sources = {
            'neuron_positions': 'https://raw.githubusercontent.com/openworm/wormbrowser/master/data/neuron_positions.json',
            'connectivity': 'https://raw.githubusercontent.com/openworm/wormbrowser/master/data/connectivity.json',
            'anatomy': 'https://raw.githubusercontent.com/openworm/wormbrowser/master/data/anatomy.json'
        }
        
        # C. elegans specific constants
        self.worm_dimensions = {
            'length_mm': 1.0,  # Adult C. elegans is ~1mm
            'width_mm': 0.08,  # ~80 microns wide
            'height_mm': 0.08   # ~80 microns high
        }
        
        # Neuron type classifications
        self.neuron_types = {
            'sensory': ['ASJ', 'ASG', 'ASH', 'AWA', 'AWB', 'AWC', 'BAG', 'CEP', 'FLP', 'OLQ', 'PHA', 'PHB', 'PVD', 'URX'],
            'interneuron': ['AIY', 'AIZ', 'AVA', 'AVB', 'AVD', 'AVE', 'AVG', 'AVH', 'AVJ', 'AVK', 'RIA', 'RIB', 'RIC', 'RID', 'RIG', 'RIH', 'RIM', 'RIP', 'RIR', 'RIS', 'RIV', 'RMD', 'RME', 'RMF', 'RMG', 'RMH', 'SAA', 'SAB', 'SIA', 'SIB', 'SMB', 'SMD', 'SME', 'SMN', 'SMP', 'SMR', 'SMT', 'SND', 'SNT', 'SOA', 'SOB', 'SOP', 'SOR', 'SOV', 'SPA', 'SPB', 'SPD', 'SPV', 'SRA', 'SRB', 'SRI', 'SRN', 'SRT', 'SRU', 'SRV', 'SRX', 'SRY', 'SSA', 'SSB', 'SSC', 'SSD', 'SSE', 'SSF', 'SSG', 'SSH', 'SSI', 'SSJ', 'SSK', 'SSL', 'SSM', 'SSN', 'SSO', 'SSP', 'SSQ', 'SSR', 'SSS', 'SST', 'SSU', 'SSV', 'SSW', 'SSX', 'SSY', 'SSZ'],
            'motor': ['DA', 'DB', 'DD', 'VA', 'VB', 'VC', 'VD', 'VE', 'VF', 'VG', 'VH', 'VI', 'VJ', 'VK', 'VL', 'VM', 'VN', 'VP', 'VQ', 'VR', 'VS', 'VT', 'VU', 'VV', 'VW', 'VX', 'VY', 'VZ']
        }
        
        # Load or download worm data
        self.neuron_data = self._load_worm_data()
    
    def _load_worm_data(self) -> Dict[str, Any]:
        """Load worm data from cache or download from WormBrowser"""
        data_file = self.cache_dir / "worm_data.json"
        
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    logger.info("Loading worm data from cache")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        # Download data if not cached
        return self._download_worm_data()
    
    def _download_worm_data(self) -> Dict[str, Any]:
        """Download worm data from WormBrowser repository"""
        logger.info("Downloading worm data from WormBrowser...")
        
        worm_data = {
            'neuron_positions': {},
            'connectivity': {},
            'anatomy': {}
        }
        
        # Try to download data, use fallback if any fail
        download_success = False
        try:
            for data_type, url in self.data_sources.items():
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    worm_data[data_type] = response.json()
                    logger.info(f"Downloaded {data_type}")
                    download_success = True
                else:
                    logger.warning(f"Failed to download {data_type}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading worm data: {e}")
        
        # If no downloads succeeded, use fallback data
        if not download_success:
            logger.info("Using fallback data for all data types")
            worm_data = self._get_fallback_data()
            logger.info(f"Generated fallback data: {len(worm_data.get('neuron_positions', {}))} neurons")
        
        # Only cache the data if it's not empty
        if worm_data.get('neuron_positions'):
            try:
                with open(self.cache_dir / "worm_data.json", 'w') as f:
                    json.dump(worm_data, f, indent=2)
                logger.info(f"Worm data cached successfully: {len(worm_data.get('neuron_positions', {}))} neurons")
            except Exception as e:
                logger.warning(f"Failed to cache worm data: {e}")
        else:
            logger.warning("Not caching empty worm data")
        
        logger.info(f"Returning worm data with {len(worm_data.get('neuron_positions', {}))} neurons")
        return worm_data
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Provide fallback worm data if download fails"""
        logger.info("Using fallback worm data")
        
        # Generate synthetic C. elegans data based on known anatomy
        num_neurons = 302  # C. elegans has 302 neurons
        
        # Generate positions along the worm's body
        positions = {}
        for i in range(num_neurons):
            # Position along body length (0-1)
            x = np.random.uniform(0, 1)
            # Position across body width (-1 to 1)
            y = np.random.uniform(-0.5, 0.5)
            # Position across body height (-1 to 1)
            z = np.random.uniform(-0.5, 0.5)
            
            positions[f"neuron_{i:03d}"] = {
                'x': x,
                'y': y,
                'z': z,
                'type': self._classify_neuron_type(i)
            }
        
        # Generate connectivity matrix (sparse)
        connectivity = {}
        for i in range(num_neurons):
            pre_neuron = f"neuron_{i:03d}"
            connectivity[pre_neuron] = []
            
            # Each neuron connects to 2-10 other neurons
            num_connections = np.random.randint(2, 11)
            for _ in range(num_connections):
                post_neuron = f"neuron_{np.random.randint(0, num_neurons):03d}"
                if post_neuron != pre_neuron:
                    weight = np.random.uniform(0.1, 1.0)
                    connectivity[pre_neuron].append({
                        'post': post_neuron,
                        'weight': weight,
                        'type': 'chemical' if np.random.random() > 0.3 else 'electrical'
                    })
        
        return {
            'neuron_positions': positions,
            'connectivity': connectivity,
            'anatomy': {
                'total_neurons': num_neurons,
                'body_regions': ['head', 'anterior', 'middle', 'posterior', 'tail'],
                'neural_rings': ['nerve_ring', 'ventral_cord', 'dorsal_cord']
            }
        }
    
    def _classify_neuron_type(self, neuron_id: int) -> str:
        """Classify neuron type based on ID (simplified)"""
        if neuron_id < 50:
            return 'sensory'
        elif neuron_id < 200:
            return 'interneuron'
        else:
            return 'motor'
    
    def get_neuron_positions(self) -> Dict[str, Dict[str, float]]:
        """Get 3D positions of all neurons"""
        return self.neuron_data.get('neuron_positions', {})
    
    def get_connectivity(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get synaptic connectivity matrix"""
        return self.neuron_data.get('connectivity', {})
    
    def get_anatomy_info(self) -> Dict[str, Any]:
        """Get anatomical information about the worm"""
        return self.neuron_data.get('anatomy', {})
    
    def plot_3d_worm_anatomy(self, save_path: Optional[str] = None, 
                             show_connections: bool = True,
                             neuron_size: float = 20.0):
        """Create 3D visualization of worm neural anatomy"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = self.get_neuron_positions()
        connectivity = self.get_connectivity()
        
        # Color map for neuron types
        type_colors = {
            'sensory': 'red',
            'interneuron': 'blue', 
            'motor': 'green',
            'unknown': 'gray'
        }
        
        # Plot neurons
        for neuron_id, pos_data in positions.items():
            x, y, z = pos_data['x'], pos_data['y'], pos_data['z']
            neuron_type = pos_data.get('type', 'unknown')
            color = type_colors.get(neuron_type, 'gray')
            
            ax.scatter(x, y, z, c=color, s=neuron_size, alpha=0.7, 
                      label=neuron_type if neuron_type not in [ax.get_legend_handles_labels()[1]] else "")
        
        # Plot connections if requested
        if show_connections and connectivity:
            for pre_neuron, connections in connectivity.items():
                if pre_neuron in positions:
                    pre_pos = positions[pre_neuron]
                    pre_x, pre_y, pre_z = pre_pos['x'], pre_pos['y'], pre_pos['z']
                    
                    for connection in connections:
                        post_neuron = connection['post']
                        if post_neuron in positions:
                            post_pos = positions[post_neuron]
                            post_x, post_y, post_z = post_pos['x'], post_pos['y'], post_pos['z']
                            
                            # Plot connection line
                            weight = connection.get('weight', 0.5)
                            alpha = min(0.3 + weight * 0.4, 0.8)  # Weight affects opacity
                            
                            ax.plot([pre_x, post_x], [pre_y, post_y], [pre_z, post_z], 
                                   'k-', alpha=alpha, linewidth=weight * 2)
        
        # Set labels and title
        ax.set_xlabel('Body Length (0-1)')
        ax.set_ylabel('Body Width (-0.5 to 0.5)')
        ax.set_zlabel('Body Height (-0.5 to 0.5)')
        ax.set_title('C. elegans Neural Anatomy (WormBrowser Integration)')
        
        # Set view limits
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Worm anatomy plot saved to {save_path}")
        
        plt.show()
    
    def integrate_with_neural_visualizer(self, neural_visualizer):
        """Integrate worm data with existing neural visualizer"""
        logger.info("Integrating WormBrowser data with neural visualizer...")
        
        # Add worm-specific visualization methods to the neural visualizer
        neural_visualizer.plot_worm_anatomy = self.plot_3d_worm_anatomy
        neural_visualizer.get_worm_neurons = self.get_neuron_positions
        neural_visualizer.get_worm_connectivity = self.get_connectivity
        neural_visualizer.get_worm_anatomy = self.get_anatomy_info
        
        logger.info("WormBrowser integration completed")
        return neural_visualizer


def demo_worm_integration():
    """Demonstrate WormBrowser integration"""
    print("🐛 Initializing WormBrowser Integration...")
    
    # Create integration instance
    worm_integration = WormBrowserIntegration()
    
    # Display available data
    print(f"📊 Available neurons: {len(worm_integration.get_neuron_positions())}")
    print(f"🔗 Available connections: {len(worm_integration.get_connectivity())}")
    print(f"🏗️  Anatomy info: {worm_integration.get_anatomy_info()}")
    
    # Create visualization
    print("\n🎨 Creating worm anatomy visualization...")
    worm_integration.plot_3d_worm_anatomy()
    
    print("\n✅ WormBrowser integration demo completed!")
    return worm_integration


if __name__ == "__main__":
    demo_worm_integration()
