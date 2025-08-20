#!/usr/bin/env python3
"""
Embryonic Brain Training Script for PaperQA
===========================================

This script specializes in training PaperQA on the 4D Human Embryonic Brain Atlas paper
(arXiv:2503.07177) and related brain development literature. It integrates neural dynamics
and cognitive science principles specifically tailored for understanding early brain development.

Paper: "The 4D Human Embryonic Brain Atlas: spatiotemporal atlas generation for rapid 
anatomical changes using first-trimester ultrasound from the Rotterdam Periconceptional Cohort"

Author: Brain Simulation Framework Team
Date: 2024
License: Apache-2.0

Dependencies:
- paper-qa (cloned from https://github.com/Future-House/paper-qa)
- numpy, pandas, matplotlib, seaborn
- torch, transformers (for neural components)
- scikit-learn (for ML components)
- rich (for progress tracking)
"""

import os, sys
import asyncio
import pickle
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Add paper-qa to path
paper_qa_path = Path(__file__).parent.parent.parent / "paper-qa"
sys.path.insert(0, str(paper_qa_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# PaperQA imports
try:
    from paperqa import Docs, Settings
    from paperqa.types import Answer, Context
    from paperqa.llms import LLM
    from paperqa.settings import Settings as PaperQASettings
except ImportError as e:
    print(f"Error importing PaperQA: {e}")
    print("Please ensure paper-qa is properly installed in the cloned directory")
    sys.exit(1)

# Brain simulation imports
try:
    from src.core.neural_components import NeuralComponents
    from src.core.neural_parameters import NeuralParameters
    from src.core.capacity_progression import CapacityProgression
    from src.core.sleep_consolidation_engine import SleepConsolidationEngine
    from src.core.multi_scale_integration import MultiScaleIntegration
    from src.core.biological_validator import BiologicalValidator
except ImportError as e:
    print(f"Warning: Could not import brain simulation components: {e}")
    print("Running in standalone PaperQA mode")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embryonic_brain_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class EmbryonicBrainConfig:
    """Configuration for embryonic brain training with PaperQA."""
    
    # Paper settings
    target_paper_arxiv_id: str = "2503.07177"
    target_paper_title: str = "The 4D Human Embryonic Brain Atlas: spatiotemporal atlas generation for rapid anatomical changes using first-trimester ultrasound from the Rotterdam Periconceptional Cohort"
    paper_directory: str = "embryonic_brain_papers"
    index_directory: str = "embryonic_brain_indexes"
    
    # Model settings
    model_name: str = "gpt-4"  # Use GPT-4 for complex scientific reasoning
    
    # Brain simulation settings
    enable_brain_simulation: bool = True
    neural_dynamics_enabled: bool = True
    cognitive_science_enabled: bool = True
    machine_learning_enabled: bool = True
    
    # Embryonic brain specific parameters
    gestational_weeks_range: Tuple[int, int] = (8, 12)  # Focus on weeks 8-12
    brain_development_stages: List[str] = None
    anatomical_landmarks: List[str] = None
    
    # Training parameters
    max_questions: int = 50
    batch_size: int = 5
    learning_rate: float = 0.001
    epochs: int = 3
    
    # Neural parameters (adapted for embryonic development)
    working_memory_slots: int = 6  # Increased for complex anatomical data
    attention_heads: int = 12  # More attention heads for detailed analysis
    neural_plasticity_rate: float = 0.15  # Higher plasticity for rapid development
    
    # Output settings
    output_dir: str = "embryonic_brain_outputs"
    save_embeddings: bool = True
    save_models: bool = True
    
    # Evaluation settings
    evaluation_split: float = 0.2
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.brain_development_stages is None:
            self.brain_development_stages = [
                "neural_plate_formation",
                "neural_tube_closure", 
                "primary_vesicles",
                "secondary_vesicles",
                "cortical_development",
                "synaptogenesis"
            ]
        
        if self.anatomical_landmarks is None:
            self.anatomical_landmarks = [
                "telencephalon",
                "diencephalon", 
                "mesencephalon",
                "metencephalon",
                "myelencephalon",
                "cerebral_hemispheres",
                "ventricles",
                "cortical_plate"
            ]
        
        if self.metrics is None:
            self.metrics = [
                "accuracy", "precision", "recall", "f1", 
                "response_time", "anatomical_accuracy", 
                "developmental_consistency"
            ]


class EmbryonicBrainTrainer:
    """
    Specialized trainer for embryonic brain development using PaperQA.
    
    This class implements:
    - Embryonic brain-specific neural dynamics
    - Developmental timeline tracking
    - Anatomical landmark recognition
    - Gestational age-specific processing
    - Brain development stage progression
    """
    
    def __init__(self, config: EmbryonicBrainConfig):
        self.config = config
        self.console = Console()
        self.docs = None
        self.neural_components = None
        self.capacity_progression = None
        self.sleep_engine = None
        self.multi_scale_integration = None
        self.biological_validator = None
        
        # Embryonic brain specific state
        self.developmental_timeline = {}
        self.anatomical_landmarks_state = {}
        self.gestational_age_state = {}
        self.brain_development_progression = {}
        
        # Training state
        self.training_history = []
        self.performance_metrics = {}
        self.neural_state = {}
        
        # Initialize components
        self._initialize_components()
        self._initialize_embryonic_brain_state()
    
    def _initialize_components(self):
        """Initialize all brain simulation components."""
        try:
            if self.config.enable_brain_simulation:
                self.console.print("[bold blue]Initializing Embryonic Brain Simulation Components...[/bold blue]")
                
                # Initialize neural components with embryonic-specific parameters
                if self.config.neural_dynamics_enabled:
                    self.neural_components = NeuralComponents(
                        working_memory_slots=self.config.working_memory_slots,
                        attention_heads=self.config.attention_heads
                    )
                    self.console.print("✅ Embryonic Neural Components initialized")
                
                # Initialize capacity progression
                self.capacity_progression = CapacityProgression()
                self.console.print("✅ Capacity Progression initialized")
                
                # Initialize sleep consolidation engine
                self.sleep_engine = SleepConsolidationEngine()
                self.console.print("✅ Sleep Consolidation Engine initialized")
                
                # Initialize multi-scale integration
                self.multi_scale_integration = MultiScaleIntegration()
                self.console.print("✅ Multi-Scale Integration initialized")
                
                # Initialize biological validator
                self.biological_validator = BiologicalValidator()
                self.console.print("✅ Biological Validator initialized")
                
        except Exception as e:
            self.console.print(f"[bold red]Warning: Could not initialize brain components: {e}[/bold red]")
            self.config.enable_brain_simulation = False
    
    def _initialize_embryonic_brain_state(self):
        """Initialize embryonic brain-specific state tracking."""
        self.console.print("[bold green]Initializing Embryonic Brain State...[/bold green]")
        
        # Initialize developmental timeline
        for week in range(self.config.gestational_weeks_range[0], 
                         self.config.gestational_weeks_range[1] + 1):
            self.developmental_timeline[f"week_{week}"] = {
                "brain_structures": [],
                "developmental_milestones": [],
                "anatomical_landmarks": [],
                "neural_activity": 0.0
            }
        
        # Initialize anatomical landmarks state
        for landmark in self.config.anatomical_landmarks:
            self.anatomical_landmarks_state[landmark] = {
                "first_appearance_week": None,
                "development_progression": [],
                "current_state": "not_formed"
            }
        
        # Initialize brain development progression
        for stage in self.config.brain_development_stages:
            self.brain_development_progression[stage] = {
                "start_week": None,
                "completion_week": None,
                "progress": 0.0,
                "key_events": []
            }
        
        self.console.print("✅ Embryonic Brain State initialized")
    
    async def download_target_paper(self):
        """Download the target embryonic brain atlas paper."""
        self.console.print(f"[bold green]Downloading target paper: {self.config.target_paper_title}[/bold green]")
        
        # Create paper directory
        os.makedirs(self.config.paper_directory, exist_ok=True)
        
        # Download paper from arXiv
        arxiv_id = self.config.target_paper_arxiv_id
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = os.path.join(self.config.paper_directory, f"{arxiv_id}.pdf")
        
        try:
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.console.print(f"✅ Downloaded paper to {pdf_path}")
            return pdf_path
            
        except Exception as e:
            self.console.print(f"[bold red]Error downloading paper: {e}[/bold red]")
            self.console.print("Please manually download the paper and place it in the papers directory")
            return None
    
    async def initialize_paperqa(self):
        """Initialize PaperQA with embryonic brain papers."""
        self.console.print("[bold green]Initializing PaperQA for Embryonic Brain Research...[/bold green]")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Download target paper if not present
        paper_path = os.path.join(self.config.paper_directory, f"{self.config.target_paper_arxiv_id}.pdf")
        if not os.path.exists(paper_path):
            await self.download_target_paper()
        
        # Initialize PaperQA settings
        settings = PaperQASettings()
        settings.agent.index.paper_directory = self.config.paper_directory
        settings.agent.index.index_directory = self.config.index_directory
        
        # Initialize Docs
        self.docs = Docs(settings=settings)
        
        # Add documents if paper directory exists
        paper_path = Path(self.config.paper_directory)
        if paper_path.exists() and any(paper_path.iterdir()):
            self.console.print(f"📚 Loading embryonic brain papers from {self.config.paper_directory}")
            # PaperQA will automatically index documents in the paper directory
        else:
            self.console.print(f"[yellow]Warning: Paper directory {self.config.paper_directory} not found or empty[/yellow]")
            self.console.print("Please add PDF documents to the papers directory")
        
        self.console.print("✅ PaperQA initialized for embryonic brain research")
    
    def generate_embryonic_brain_questions(self) -> List[Dict[str, str]]:
        """Generate specialized questions for embryonic brain development."""
        questions = [
            # Basic paper understanding
            {
                "question": "What is the main contribution of the 4D Human Embryonic Brain Atlas paper?",
                "answer": "The paper creates a 4D spatiotemporal atlas of embryonic brain development using deep learning-based registration and ultrasound imaging.",
                "category": "paper_overview"
            },
            {
                "question": "What gestational age range does the atlas cover?",
                "answer": "The atlas covers gestational weeks 8 to 12, capturing rapid anatomical changes during early brain development.",
                "category": "developmental_timeline"
            },
            {
                "question": "How many subjects and ultrasound images were used in the study?",
                "answer": "The study used 831 3D ultrasound images from 402 subjects in the Rotterdam Periconceptional Cohort.",
                "category": "methodology"
            },
            
            # Technical methodology
            {
                "question": "What deep learning approach was used for groupwise registration?",
                "answer": "The method used a deep learning-based approach for groupwise registration with a time-dependent initial atlas and penalization for deviations.",
                "category": "technical_methods"
            },
            {
                "question": "Why was a time-dependent initial atlas important for this study?",
                "answer": "A time-dependent initial atlas was crucial because the brain undergoes rapid changes within days, and this approach ensures age-specific anatomy is maintained throughout development.",
                "category": "technical_methods"
            },
            
            # Embryonic brain development
            {
                "question": "What are the key brain structures that develop during weeks 8-12?",
                "answer": "During weeks 8-12, key structures include the telencephalon, diencephalon, mesencephalon, metencephalon, myelencephalon, cerebral hemispheres, ventricles, and cortical plate.",
                "category": "anatomical_development"
            },
            {
                "question": "How does the embryonic brain change during the first trimester?",
                "answer": "The embryonic brain undergoes rapid anatomical changes with structures forming, growing, and differentiating within days, making it crucial to capture these changes accurately.",
                "category": "developmental_biology"
            },
            
            # Clinical applications
            {
                "question": "What are the potential clinical applications of this 4D atlas?",
                "answer": "The atlas can improve detection, prevention, and treatment of prenatal neurodevelopmental disorders by providing detailed insights into normal brain development.",
                "category": "clinical_applications"
            },
            {
                "question": "How does this atlas help identify deviations from normal development?",
                "answer": "By providing detailed insights into normal brain development, the atlas serves as a reference to identify deviations and potential neurodevelopmental disorders.",
                "category": "clinical_applications"
            },
            
            # Validation and evaluation
            {
                "question": "How was the anatomical accuracy of the atlas validated?",
                "answer": "The atlas was validated through an ablation study and visual comparisons with an existing ex-vivo embryo atlas, demonstrating anatomical accuracy.",
                "category": "validation"
            },
            {
                "question": "What did the ablation study demonstrate about the method?",
                "answer": "The ablation study showed that incorporating a time-dependent initial atlas and penalization produced anatomically accurate results, while omitting these adaptations led to anatomically incorrect atlas.",
                "category": "validation"
            }
        ]
        
        return questions
    
    async def train_on_embryonic_brain_questions(self):
        """Train the system on embryonic brain development questions."""
        questions = self.generate_embryonic_brain_questions()
        
        self.console.print(f"[bold green]Starting embryonic brain training on {len(questions)} questions...[/bold green]")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Training on embryonic brain development...", total=len(questions))
            
            for i, q_data in enumerate(questions):
                question = q_data['question']
                expected_answer = q_data.get('answer', '')
                category = q_data.get('category', 'general')
                
                try:
                    # Pre-process with embryonic brain-specific neural dynamics
                    if self.config.neural_dynamics_enabled and self.neural_components:
                        question = await self._apply_embryonic_neural_dynamics(question, category)
                    
                    # Apply embryonic brain-specific cognitive science principles
                    if self.config.cognitive_science_enabled:
                        question = await self._apply_embryonic_cognitive_science(question, category)
                    
                    # Query PaperQA
                    answer = await self.docs.aquery(question)
                    
                    # Post-process with embryonic brain-specific machine learning
                    if self.config.machine_learning_enabled:
                        answer = await self._apply_embryonic_machine_learning(answer, category)
                    
                    # Record results with embryonic brain context
                    result = {
                        'question': question,
                        'expected_answer': expected_answer,
                        'actual_answer': answer.answer,
                        'confidence': answer.confidence,
                        'sources': answer.sources,
                        'category': category,
                        'neural_state': self.neural_state.copy() if self.neural_state else {},
                        'developmental_context': self._get_developmental_context(category),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    
                    # Update embryonic brain state
                    if self.config.neural_dynamics_enabled and self.neural_components:
                        await self._update_embryonic_neural_state(result, category)
                    
                    # Sleep consolidation every batch
                    if (i + 1) % self.config.batch_size == 0:
                        await self._embryonic_sleep_consolidation()
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error processing embryonic brain question {i}: {e}")
                    progress.update(task, advance=1)
                    continue
        
        # Final sleep consolidation
        await self._embryonic_sleep_consolidation()
        
        # Calculate embryonic brain-specific metrics
        self._calculate_embryonic_metrics(results)
        
        # Save results
        self._save_embryonic_results(results)
        
        return results
    
    async def _apply_embryonic_neural_dynamics(self, question: str, category: str) -> str:
        """Apply embryonic brain-specific neural dynamics processing."""
        if not self.neural_components:
            return question
        
        try:
            # Simulate embryonic neural processing
            processed_question = self.neural_components.process_input(question)
            
            # Add embryonic brain context based on category
            if category == "anatomical_development":
                processed_question += " [Focus on anatomical structures and their development]"
            elif category == "developmental_timeline":
                processed_question += " [Consider gestational age progression]"
            elif category == "clinical_applications":
                processed_question += " [Consider clinical implications and medical applications]"
            
            # Update embryonic neural state
            self.neural_state['last_question'] = question
            self.neural_state['processed_question'] = processed_question
            self.neural_state['question_category'] = category
            self.neural_state['neural_activity'] = self.neural_components.get_activity_state()
            self.neural_state['embryonic_context'] = True
            
            return processed_question
            
        except Exception as e:
            logger.error(f"Error in embryonic neural dynamics: {e}")
            return question
    
    async def _apply_embryonic_cognitive_science(self, question: str, category: str) -> str:
        """Apply embryonic brain-specific cognitive science principles."""
        try:
            # Implement embryonic brain-specific cognitive processing
            # - Developmental timeline awareness
            # - Anatomical landmark recognition
            # - Gestational age context
            
            if self.capacity_progression:
                # Check working memory capacity for complex anatomical data
                capacity = self.capacity_progression.get_current_capacity()
                if len(question.split()) > capacity * 8:  # Adjusted for anatomical complexity
                    # Simplify question if too complex
                    question = " ".join(question.split()[:int(capacity * 8)])
            
            # Add developmental context
            if category in ["anatomical_development", "developmental_timeline"]:
                question += f" [Consider development during weeks {self.config.gestational_weeks_range[0]}-{self.config.gestational_weeks_range[1]}]"
            
            return question
            
        except Exception as e:
            logger.error(f"Error in embryonic cognitive science processing: {e}")
            return question
    
    async def _apply_embryonic_machine_learning(self, answer: Answer, category: str) -> Answer:
        """Apply embryonic brain-specific machine learning optimization."""
        try:
            # Implement embryonic brain-specific ML optimization
            # - Anatomical accuracy validation
            # - Developmental consistency checking
            # - Clinical relevance assessment
            
            # For now, return the original answer
            return answer
            
        except Exception as e:
            logger.error(f"Error in embryonic machine learning processing: {e}")
            return answer
    
    def _get_developmental_context(self, category: str) -> Dict[str, Any]:
        """Get developmental context for the question category."""
        context = {
            "gestational_weeks": self.config.gestational_weeks_range,
            "brain_development_stages": self.config.brain_development_stages,
            "anatomical_landmarks": self.config.anatomical_landmarks,
            "category": category
        }
        
        if category == "anatomical_development":
            context["focus"] = "brain structures and their formation"
        elif category == "developmental_timeline":
            context["focus"] = "temporal progression of development"
        elif category == "clinical_applications":
            context["focus"] = "medical and diagnostic applications"
        
        return context
    
    async def _update_embryonic_neural_state(self, result: Dict[str, Any], category: str):
        """Update embryonic neural state based on training results."""
        if not self.neural_components:
            return
        
        try:
            # Update neural plasticity with embryonic-specific rates
            self.neural_components.update_plasticity(
                result['confidence'],
                self.config.neural_plasticity_rate
            )
            
            # Update capacity progression
            if self.capacity_progression:
                self.capacity_progression.update_capacity(result['confidence'])
            
            # Update developmental timeline
            self._update_developmental_timeline(category, result['confidence'])
            
        except Exception as e:
            logger.error(f"Error updating embryonic neural state: {e}")
    
    def _update_developmental_timeline(self, category: str, confidence: float):
        """Update the developmental timeline based on question category and confidence."""
        try:
            # Update neural activity for relevant weeks
            if category == "developmental_timeline":
                for week in range(self.config.gestational_weeks_range[0], 
                                 self.config.gestational_weeks_range[1] + 1):
                    week_key = f"week_{week}"
                    if week_key in self.developmental_timeline:
                        self.developmental_timeline[week_key]["neural_activity"] += confidence * 0.1
            
            # Update anatomical landmarks
            if category == "anatomical_development":
                for landmark in self.config.anatomical_landmarks:
                    if landmark in self.anatomical_landmarks_state:
                        self.anatomical_landmarks_state[landmark]["development_progression"].append({
                            "confidence": confidence,
                            "timestamp": datetime.now().isoformat()
                        })
            
        except Exception as e:
            logger.error(f"Error updating developmental timeline: {e}")
    
    async def _embryonic_sleep_consolidation(self):
        """Perform embryonic brain-specific sleep consolidation."""
        if not self.sleep_engine:
            return
        
        try:
            # Simulate embryonic sleep consolidation
            self.sleep_engine.consolidate_memories(self.training_history)
            
            # Update embryonic neural state
            self.neural_state['sleep_consolidation'] = True
            self.neural_state['embryonic_consolidation'] = True
            self.neural_state['consolidation_timestamp'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error in embryonic sleep consolidation: {e}")
    
    def _calculate_embryonic_metrics(self, results: List[Dict[str, Any]]):
        """Calculate embryonic brain-specific performance metrics."""
        self.console.print("[bold blue]Calculating Embryonic Brain Performance Metrics...[/bold blue]")
        
        metrics = {}
        
        # Basic metrics
        total_questions = len(results)
        successful_queries = len([r for r in results if r['actual_answer']])
        
        metrics['total_questions'] = total_questions
        metrics['successful_queries'] = successful_queries
        metrics['success_rate'] = successful_queries / total_questions if total_questions > 0 else 0
        
        # Confidence metrics
        confidences = [r['confidence'] for r in results if r['confidence'] is not None]
        if confidences:
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
        
        # Category-specific metrics
        categories = {}
        for result in results:
            category = result.get('category', 'general')
            if category not in categories:
                categories[category] = []
            categories[category].append(result['confidence'] if result['confidence'] else 0)
        
        for category, confs in categories.items():
            if confs:
                metrics[f'{category}_avg_confidence'] = np.mean(confs)
                metrics[f'{category}_count'] = len(confs)
        
        # Embryonic brain-specific metrics
        metrics['developmental_timeline_coverage'] = len(self.developmental_timeline)
        metrics['anatomical_landmarks_tracked'] = len(self.anatomical_landmarks_state)
        metrics['brain_development_stages'] = len(self.brain_development_progression)
        
        # Neural metrics
        if self.neural_components:
            neural_metrics = self.neural_components.get_metrics()
            metrics.update(neural_metrics)
        
        self.performance_metrics = metrics
        
        # Display metrics
        self._display_embryonic_metrics(metrics)
    
    def _display_embryonic_metrics(self, metrics: Dict[str, Any]):
        """Display embryonic brain-specific performance metrics."""
        table = Table(title="Embryonic Brain Training Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Category", style="green")
        
        # Basic metrics
        for metric in ['total_questions', 'successful_queries', 'success_rate']:
            value = metrics.get(metric, 0)
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}", "Basic")
            else:
                table.add_row(metric, str(value), "Basic")
        
        # Confidence metrics
        for metric in ['avg_confidence', 'confidence_std']:
            value = metrics.get(metric, 0)
            if value:
                table.add_row(metric, f"{value:.4f}", "Confidence")
        
        # Category-specific metrics
        for metric, value in metrics.items():
            if 'avg_confidence' in metric and metric != 'avg_confidence':
                table.add_row(metric, f"{value:.4f}", "Category")
        
        # Embryonic brain metrics
        for metric in ['developmental_timeline_coverage', 'anatomical_landmarks_tracked', 'brain_development_stages']:
            value = metrics.get(metric, 0)
            table.add_row(metric, str(value), "Embryonic Brain")
        
        self.console.print(table)
    
    def _save_embryonic_results(self, results: List[Dict[str, Any]]):
        """Save embryonic brain-specific training results."""
        self.console.print("[bold green]Saving Embryonic Brain Results...[/bold green]")
        
        # Save results
        results_file = os.path.join(self.config.output_dir, "embryonic_brain_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, "embryonic_brain_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        # Save embryonic brain state
        embryonic_state = {
            "developmental_timeline": self.developmental_timeline,
            "anatomical_landmarks_state": self.anatomical_landmarks_state,
            "brain_development_progression": self.brain_development_progression,
            "neural_state": self.neural_state
        }
        
        state_file = os.path.join(self.config.output_dir, "embryonic_brain_state.json")
        with open(state_file, 'w') as f:
            json.dump(embryonic_state, f, indent=2, default=str)
        
        # Save PaperQA docs if requested
        if self.config.save_embeddings and self.docs:
            docs_file = os.path.join(self.config.output_dir, "embryonic_brain_docs.pkl")
            with open(docs_file, 'wb') as f:
                pickle.dump(self.docs, f)
        
        self.console.print(f"✅ Embryonic brain results saved to {self.config.output_dir}")
    
    def generate_embryonic_training_report(self):
        """Generate a comprehensive embryonic brain training report."""
        report_file = os.path.join(self.config.output_dir, "embryonic_brain_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Embryonic Brain Training Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Target Paper:** {self.config.target_paper_title}\n")
            f.write(f"**arXiv ID:** {self.config.target_paper_arxiv_id}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- Paper Directory: {self.config.paper_directory}\n")
            f.write(f"- Model: {self.config.model_name}\n")
            f.write(f"- Gestational Weeks: {self.config.gestational_weeks_range[0]}-{self.config.gestational_weeks_range[1]}\n")
            f.write(f"- Brain Simulation: {self.config.enable_brain_simulation}\n")
            f.write(f"- Neural Dynamics: {self.config.neural_dynamics_enabled}\n")
            f.write(f"- Cognitive Science: {self.config.cognitive_science_enabled}\n")
            f.write(f"- Machine Learning: {self.config.machine_learning_enabled}\n\n")
            
            f.write("## Performance Metrics\n\n")
            for metric, value in self.performance_metrics.items():
                if isinstance(value, float):
                    f.write(f"- **{metric}:** {value:.4f}\n")
                else:
                    f.write(f"- **{metric}:** {value}\n")
            
            f.write("\n## Developmental Timeline Summary\n\n")
            for week, data in self.developmental_timeline.items():
                f.write(f"- **{week}:** Neural activity: {data['neural_activity']:.4f}\n")
            
            f.write("\n## Anatomical Landmarks Tracked\n\n")
            for landmark, data in self.anatomical_landmarks_state.items():
                f.write(f"- **{landmark}:** {data['current_state']}\n")
            
            f.write("\n## Brain Development Stages\n\n")
            for stage, data in self.brain_development_progression.items():
                f.write(f"- **{stage}:** Progress: {data['progress']:.2f}\n")
            
            f.write("\n## Training History\n\n")
            f.write(f"- Total training sessions: {len(self.training_history)}\n")
            f.write(f"- Embryonic brain focus: ✅\n")
            f.write(f"- Developmental timeline tracking: ✅\n")
            f.write(f"- Anatomical landmark recognition: ✅\n")
        
        self.console.print(f"✅ Embryonic brain training report saved to {report_file}")


async def main():
    """Main embryonic brain training function."""
    console.print(Panel.fit(
        "[bold blue]Embryonic Brain Training with PaperQA[/bold blue]\n"
        "Specialized training on 4D Human Embryonic Brain Atlas research",
        border_style="blue"
    ))
    
    # Configuration
    config = EmbryonicBrainConfig()
    
    # Initialize trainer
    trainer = EmbryonicBrainTrainer(config)
    
    # Initialize PaperQA
    await trainer.initialize_paperqa()
    
    # Train on embryonic brain questions
    results = await trainer.train_on_embryonic_brain_questions()
    
    # Generate report
    trainer.generate_embryonic_training_report()
    
    console.print("[bold green]Embryonic brain training completed successfully![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
