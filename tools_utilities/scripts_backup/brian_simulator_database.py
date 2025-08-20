#!/usr/bin/env python3
"""
Brian Simulator Database
Integration with Biological AGI Development

This module provides a comprehensive database interface for Brian simulator,
supporting biological neural network simulations with STDP, neuromodulation,
and cortical architecture validation.

Reference: https://briansimulator.org/
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrianSimulatorDatabase:
    """
    Database interface for Brian simulator integration with biological AGI.
    
    This class provides comprehensive database functionality for:
    - Neural network configurations
    - STDP parameters and validation
    - Neuromodulatory system data
    - Cortical architecture specifications
    - Simulation results and metrics
    - Biological validation data
    """
    
    def __init__(self, db_path: str = "data/brian_simulator.db"):
        """
        Initialize Brian simulator database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize database with required tables."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.create_tables()
            logger.info(f"Brian simulator database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create all required database tables."""
        cursor = self.connection.cursor()
        
        # Neural Network Configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS neural_networks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                network_type TEXT NOT NULL,
                neuron_count INTEGER NOT NULL,
                synapse_count INTEGER NOT NULL,
                stdp_enabled BOOLEAN DEFAULT TRUE,
                neuromodulation_enabled BOOLEAN DEFAULT TRUE,
                cortical_layers INTEGER DEFAULT 6,
                configuration_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # STDP Parameters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stdp_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id INTEGER,
                tau_plus REAL DEFAULT 20.0,
                tau_minus REAL DEFAULT 20.0,
                a_plus REAL DEFAULT 0.01,
                a_minus REAL DEFAULT 0.01,
                w_max REAL DEFAULT 1.0,
                w_min REAL DEFAULT 0.0,
                dopamine_modulation BOOLEAN DEFAULT TRUE,
                validation_accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (network_id) REFERENCES neural_networks (id)
            )
        """)
        
        # Neuromodulatory Systems
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS neuromodulatory_systems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id INTEGER,
                dopamine_level REAL DEFAULT 0.0,
                norepinephrine_level REAL DEFAULT 0.0,
                acetylcholine_level REAL DEFAULT 0.0,
                serotonin_level REAL DEFAULT 0.0,
                modulation_factors_json TEXT,
                validation_accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (network_id) REFERENCES neural_networks (id)
            )
        """)
        
        # Cortical Architecture
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cortical_architecture (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id INTEGER,
                layer_count INTEGER DEFAULT 6,
                layer_configurations_json TEXT,
                minicolumns_enabled BOOLEAN DEFAULT TRUE,
                connectivity_patterns_json TEXT,
                validation_accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (network_id) REFERENCES neural_networks (id)
            )
        """)
        
        # Simulation Results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id INTEGER,
                simulation_name TEXT NOT NULL,
                duration REAL NOT NULL,
                neuron_count INTEGER NOT NULL,
                spike_count INTEGER NOT NULL,
                stdp_accuracy REAL,
                neuromodulation_accuracy REAL,
                cortical_accuracy REAL,
                overall_accuracy REAL,
                performance_metrics_json TEXT,
                biological_validation_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (network_id) REFERENCES neural_networks (id)
            )
        """)
        
        # Biological Validation Data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS biological_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id INTEGER,
                validation_type TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                expected_value REAL,
                actual_value REAL,
                accuracy REAL,
                validation_details_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (network_id) REFERENCES neural_networks (id)
            )
        """)
        
        # Cloud Computing Configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cloud_configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id INTEGER,
                deployment_type TEXT NOT NULL,
                platform TEXT NOT NULL,
                instance_type TEXT,
                gpu_count INTEGER DEFAULT 0,
                memory_gb INTEGER,
                cost_per_hour REAL,
                scaling_configuration_json TEXT,
                performance_metrics_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (network_id) REFERENCES neural_networks (id)
            )
        """)
        
        self.connection.commit()
        logger.info("Database tables created successfully")
    
    def add_neural_network(self, name: str, description: str, network_type: str,
                          neuron_count: int, synapse_count: int,
                          stdp_enabled: bool = True, neuromodulation_enabled: bool = True,
                          cortical_layers: int = 6, configuration: Dict = None) -> int:
        """
        Add a new neural network configuration to the database.
        
        Args:
            name: Network name
            description: Network description
            network_type: Type of network (e.g., 'cortical_column', 'hippocampus')
            neuron_count: Number of neurons
            synapse_count: Number of synapses
            stdp_enabled: Whether STDP is enabled
            neuromodulation_enabled: Whether neuromodulation is enabled
            cortical_layers: Number of cortical layers
            configuration: Additional configuration as JSON
            
        Returns:
            Network ID
        """
        cursor = self.connection.cursor()
        configuration_json = json.dumps(configuration) if configuration else None
        
        cursor.execute("""
            INSERT INTO neural_networks 
            (name, description, network_type, neuron_count, synapse_count,
             stdp_enabled, neuromodulation_enabled, cortical_layers, configuration_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, description, network_type, neuron_count, synapse_count,
              stdp_enabled, neuromodulation_enabled, cortical_layers, configuration_json))
        
        network_id = cursor.lastrowid
        self.connection.commit()
        logger.info(f"Added neural network '{name}' with ID {network_id}")
        return network_id
    
    def add_stdp_parameters(self, network_id: int, tau_plus: float = 20.0,
                           tau_minus: float = 20.0, a_plus: float = 0.01,
                           a_minus: float = 0.01, w_max: float = 1.0,
                           w_min: float = 0.0, dopamine_modulation: bool = True,
                           validation_accuracy: float = None) -> int:
        """
        Add STDP parameters for a neural network.
        
        Args:
            network_id: ID of the neural network
            tau_plus: LTP time constant
            tau_minus: LTD time constant
            a_plus: LTP amplitude
            a_minus: LTD amplitude
            w_max: Maximum synaptic weight
            w_min: Minimum synaptic weight
            dopamine_modulation: Whether dopamine modulation is enabled
            validation_accuracy: Biological validation accuracy
            
        Returns:
            STDP parameters ID
        """
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO stdp_parameters 
            (network_id, tau_plus, tau_minus, a_plus, a_minus, w_max, w_min,
             dopamine_modulation, validation_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (network_id, tau_plus, tau_minus, a_plus, a_minus, w_max, w_min,
              dopamine_modulation, validation_accuracy))
        
        stdp_id = cursor.lastrowid
        self.connection.commit()
        logger.info(f"Added STDP parameters for network {network_id}")
        return stdp_id
    
    def add_neuromodulatory_system(self, network_id: int, dopamine_level: float = 0.0,
                                  norepinephrine_level: float = 0.0,
                                  acetylcholine_level: float = 0.0,
                                  serotonin_level: float = 0.0,
                                  modulation_factors: Dict = None,
                                  validation_accuracy: float = None) -> int:
        """
        Add neuromodulatory system configuration.
        
        Args:
            network_id: ID of the neural network
            dopamine_level: Dopamine level
            norepinephrine_level: Norepinephrine level
            acetylcholine_level: Acetylcholine level
            serotonin_level: Serotonin level
            modulation_factors: Additional modulation factors
            validation_accuracy: Biological validation accuracy
            
        Returns:
            Neuromodulatory system ID
        """
        cursor = self.connection.cursor()
        modulation_factors_json = json.dumps(modulation_factors) if modulation_factors else None
        
        cursor.execute("""
            INSERT INTO neuromodulatory_systems 
            (network_id, dopamine_level, norepinephrine_level, acetylcholine_level,
             serotonin_level, modulation_factors_json, validation_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (network_id, dopamine_level, norepinephrine_level, acetylcholine_level,
              serotonin_level, modulation_factors_json, validation_accuracy))
        
        neuromod_id = cursor.lastrowid
        self.connection.commit()
        logger.info(f"Added neuromodulatory system for network {network_id}")
        return neuromod_id
    
    def add_cortical_architecture(self, network_id: int, layer_count: int = 6,
                                 layer_configurations: Dict = None,
                                 minicolumns_enabled: bool = True,
                                 connectivity_patterns: Dict = None,
                                 validation_accuracy: float = None) -> int:
        """
        Add cortical architecture configuration.
        
        Args:
            network_id: ID of the neural network
            layer_count: Number of cortical layers
            layer_configurations: Layer-specific configurations
            minicolumns_enabled: Whether minicolumns are enabled
            connectivity_patterns: Connectivity patterns
            validation_accuracy: Biological validation accuracy
            
        Returns:
            Cortical architecture ID
        """
        cursor = self.connection.cursor()
        layer_config_json = json.dumps(layer_configurations) if layer_configurations else None
        connectivity_json = json.dumps(connectivity_patterns) if connectivity_patterns else None
        
        cursor.execute("""
            INSERT INTO cortical_architecture 
            (network_id, layer_count, layer_configurations_json, minicolumns_enabled,
             connectivity_patterns_json, validation_accuracy)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (network_id, layer_count, layer_config_json, minicolumns_enabled,
              connectivity_json, validation_accuracy))
        
        cortical_id = cursor.lastrowid
        self.connection.commit()
        logger.info(f"Added cortical architecture for network {network_id}")
        return cortical_id
    
    def add_simulation_result(self, network_id: int, simulation_name: str,
                             duration: float, neuron_count: int, spike_count: int,
                             stdp_accuracy: float = None, neuromodulation_accuracy: float = None,
                             cortical_accuracy: float = None, overall_accuracy: float = None,
                             performance_metrics: Dict = None,
                             biological_validation: Dict = None) -> int:
        """
        Add simulation results to the database.
        
        Args:
            network_id: ID of the neural network
            simulation_name: Name of the simulation
            duration: Simulation duration
            neuron_count: Number of neurons in simulation
            spike_count: Total number of spikes
            stdp_accuracy: STDP accuracy
            neuromodulation_accuracy: Neuromodulation accuracy
            cortical_accuracy: Cortical architecture accuracy
            overall_accuracy: Overall biological accuracy
            performance_metrics: Performance metrics
            biological_validation: Biological validation results
            
        Returns:
            Simulation result ID
        """
        cursor = self.connection.cursor()
        performance_json = json.dumps(performance_metrics) if performance_metrics else None
        biological_json = json.dumps(biological_validation) if biological_validation else None
        
        cursor.execute("""
            INSERT INTO simulation_results 
            (network_id, simulation_name, duration, neuron_count, spike_count,
             stdp_accuracy, neuromodulation_accuracy, cortical_accuracy, overall_accuracy,
             performance_metrics_json, biological_validation_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (network_id, simulation_name, duration, neuron_count, spike_count,
              stdp_accuracy, neuromodulation_accuracy, cortical_accuracy, overall_accuracy,
              performance_json, biological_json))
        
        result_id = cursor.lastrowid
        self.connection.commit()
        logger.info(f"Added simulation result '{simulation_name}' for network {network_id}")
        return result_id
    
    def add_biological_validation(self, network_id: int, validation_type: str,
                                 benchmark_name: str, expected_value: float,
                                 actual_value: float, accuracy: float,
                                 validation_details: Dict = None) -> int:
        """
        Add biological validation data.
        
        Args:
            network_id: ID of the neural network
            validation_type: Type of validation (e.g., 'stdp', 'neuromodulation')
            benchmark_name: Name of the benchmark
            expected_value: Expected biological value
            actual_value: Actual simulated value
            accuracy: Validation accuracy
            validation_details: Additional validation details
            
        Returns:
            Validation ID
        """
        cursor = self.connection.cursor()
        details_json = json.dumps(validation_details) if validation_details else None
        
        cursor.execute("""
            INSERT INTO biological_validation 
            (network_id, validation_type, benchmark_name, expected_value,
             actual_value, accuracy, validation_details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (network_id, validation_type, benchmark_name, expected_value,
              actual_value, accuracy, details_json))
        
        validation_id = cursor.lastrowid
        self.connection.commit()
        logger.info(f"Added biological validation for network {network_id}")
        return validation_id
    
    def add_cloud_configuration(self, network_id: int, deployment_type: str,
                               platform: str, instance_type: str = None,
                               gpu_count: int = 0, memory_gb: int = None,
                               cost_per_hour: float = None,
                               scaling_configuration: Dict = None,
                               performance_metrics: Dict = None) -> int:
        """
        Add cloud computing configuration.
        
        Args:
            network_id: ID of the neural network
            deployment_type: Type of deployment (laptop, cloud_burst, production)
            platform: Cloud platform (AWS, GCP, Azure)
            instance_type: Instance type
            gpu_count: Number of GPUs
            memory_gb: Memory in GB
            cost_per_hour: Cost per hour
            scaling_configuration: Scaling configuration
            performance_metrics: Performance metrics
            
        Returns:
            Cloud configuration ID
        """
        cursor = self.connection.cursor()
        scaling_json = json.dumps(scaling_configuration) if scaling_configuration else None
        performance_json = json.dumps(performance_metrics) if performance_metrics else None
        
        cursor.execute("""
            INSERT INTO cloud_configurations 
            (network_id, deployment_type, platform, instance_type, gpu_count,
             memory_gb, cost_per_hour, scaling_configuration_json, performance_metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (network_id, deployment_type, platform, instance_type, gpu_count,
              memory_gb, cost_per_hour, scaling_json, performance_json))
        
        config_id = cursor.lastrowid
        self.connection.commit()
        logger.info(f"Added cloud configuration for network {network_id}")
        return config_id
    
    def get_neural_network(self, network_id: int) -> Dict:
        """Get neural network configuration by ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM neural_networks WHERE id = ?", (network_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [description[0] for description in cursor.description]
            network_data = dict(zip(columns, row))
            if network_data['configuration_json']:
                network_data['configuration'] = json.loads(network_data['configuration_json'])
            return network_data
        return None
    
    def get_networks_by_type(self, network_type: str) -> List[Dict]:
        """Get all networks of a specific type."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM neural_networks WHERE network_type = ?", (network_type,))
        rows = cursor.fetchall()
        
        networks = []
        columns = [description[0] for description in cursor.description]
        for row in rows:
            network_data = dict(zip(columns, row))
            if network_data['configuration_json']:
                network_data['configuration'] = json.loads(network_data['configuration_json'])
            networks.append(network_data)
        
        return networks
    
    def get_simulation_results(self, network_id: int) -> List[Dict]:
        """Get all simulation results for a network."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM simulation_results WHERE network_id = ?", (network_id,))
        rows = cursor.fetchall()
        
        results = []
        columns = [description[0] for description in cursor.description]
        for row in rows:
            result_data = dict(zip(columns, row))
            if result_data['performance_metrics_json']:
                result_data['performance_metrics'] = json.loads(result_data['performance_metrics_json'])
            if result_data['biological_validation_json']:
                result_data['biological_validation'] = json.loads(result_data['biological_validation_json'])
            results.append(result_data)
        
        return results
    
    def get_biological_validation_summary(self, network_id: int) -> Dict:
        """Get biological validation summary for a network."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT validation_type, AVG(accuracy) as avg_accuracy, 
                   COUNT(*) as validation_count
            FROM biological_validation 
            WHERE network_id = ?
            GROUP BY validation_type
        """, (network_id,))
        rows = cursor.fetchall()
        
        summary = {}
        for row in rows:
            validation_type, avg_accuracy, count = row
            summary[validation_type] = {
                'average_accuracy': avg_accuracy,
                'validation_count': count
            }
        
        return summary
    
    def get_cloud_configurations(self, network_id: int) -> List[Dict]:
        """Get all cloud configurations for a network."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM cloud_configurations WHERE network_id = ?", (network_id,))
        rows = cursor.fetchall()
        
        configurations = []
        columns = [description[0] for description in cursor.description]
        for row in rows:
            config_data = dict(zip(columns, row))
            if config_data['scaling_configuration_json']:
                config_data['scaling_configuration'] = json.loads(config_data['scaling_configuration_json'])
            if config_data['performance_metrics_json']:
                config_data['performance_metrics'] = json.loads(config_data['performance_metrics_json'])
            configurations.append(config_data)
        
        return configurations
    
    def export_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """Export a table to pandas DataFrame."""
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql_query(query, self.connection)
    
    def get_database_summary(self) -> Dict:
        """Get database summary statistics."""
        cursor = self.connection.cursor()
        
        summary = {}
        
        # Count networks by type
        cursor.execute("""
            SELECT network_type, COUNT(*) as count 
            FROM neural_networks 
            GROUP BY network_type
        """)
        summary['networks_by_type'] = dict(cursor.fetchall())
        
        # Average accuracies
        cursor.execute("SELECT AVG(overall_accuracy) FROM simulation_results WHERE overall_accuracy IS NOT NULL")
        summary['average_overall_accuracy'] = cursor.fetchone()[0]
        
        # Total simulations
        cursor.execute("SELECT COUNT(*) FROM simulation_results")
        summary['total_simulations'] = cursor.fetchone()[0]
        
        # Cloud deployments
        cursor.execute("SELECT deployment_type, COUNT(*) FROM cloud_configurations GROUP BY deployment_type")
        summary['cloud_deployments'] = dict(cursor.fetchall())
        
        return summary
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Example usage and testing
def create_sample_data():
    """Create sample data for testing."""
    db = BrianSimulatorDatabase()
    
    # Add a cortical column network
    network_id = db.add_neural_network(
        name="Cortical Column v1",
        description="6-layer cortical column with STDP and neuromodulation",
        network_type="cortical_column",
        neuron_count=100000,
        synapse_count=1000000,
        stdp_enabled=True,
        neuromodulation_enabled=True,
        cortical_layers=6,
        configuration={
            "learning_rate": 0.01,
            "plasticity_type": "stdp",
            "neuromodulation": ["dopamine", "norepinephrine", "acetylcholine", "serotonin"]
        }
    )
    
    # Add STDP parameters
    db.add_stdp_parameters(
        network_id=network_id,
        tau_plus=20.0,
        tau_minus=20.0,
        a_plus=0.01,
        a_minus=0.01,
        dopamine_modulation=True,
        validation_accuracy=0.95
    )
    
    # Add neuromodulatory system
    db.add_neuromodulatory_system(
        network_id=network_id,
        dopamine_level=0.5,
        norepinephrine_level=0.3,
        acetylcholine_level=0.4,
        serotonin_level=0.2,
        validation_accuracy=0.92
    )
    
    # Add cortical architecture
    db.add_cortical_architecture(
        network_id=network_id,
        layer_count=6,
        minicolumns_enabled=True,
        validation_accuracy=0.88
    )
    
    # Add simulation result
    db.add_simulation_result(
        network_id=network_id,
        simulation_name="Cortical Column Test 1",
        duration=100.0,
        neuron_count=100000,
        spike_count=5000000,
        stdp_accuracy=0.95,
        neuromodulation_accuracy=0.92,
        cortical_accuracy=0.88,
        overall_accuracy=0.92
    )
    
    # Add cloud configuration
    db.add_cloud_configuration(
        network_id=network_id,
        deployment_type="cloud_burst",
        platform="AWS",
        instance_type="g4dn.xlarge",
        gpu_count=1,
        memory_gb=16,
        cost_per_hour=0.50
    )
    
    # Print summary
    summary = db.get_database_summary()
    print("Database Summary:")
    print(json.dumps(summary, indent=2))
    
    db.close()


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data()
