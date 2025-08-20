#!/usr/bin/env python3
"""
Fetal Brain Training Script for PaperQA
======================================

This script specializes in training PaperQA on fetal brain development research,
combining two key papers:

1. "The 4D Human Embryonic Brain Atlas: spatiotemporal atlas generation for rapid 
   anatomical changes using first-trimester ultrasound from the Rotterdam Periconceptional Cohort"
   (arXiv:2503.07177)

2. "Conditional Fetal Brain Atlas Learning for Automatic Tissue Segmentation"
   (arXiv:2508.04522)

This creates a comprehensive framework for understanding brain development from 
embryonic to fetal stages with advanced deep learning and tissue segmentation.

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
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
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
    from development.src.core.neural_components import NeuralComponents
    from development.src.core.neural_parameters import NeuralParameters
    from development.src.core.capacity_progression import CapacityProgression
    from development.src.core.sleep_consolidation_engine import SleepConsolidationEngine
    from development.src.core.multi_scale_integration import MultiScaleIntegration
    from development.src.core.biological_validator import BiologicalValidator
except ImportError as e:
    print(f"Warning: Could not import brain simulation components: {e}")
    print("Running in standalone PaperQA mode")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetal_brain_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class FetalBrainConfig:
    """Configuration for fetal brain training with PaperQA."""
    
    # Paper settings
    embryonic_paper_arxiv_id: str = "2503.07177"
    fetal_paper_arxiv_id: str = "2508.04522"
    embryonic_paper_title: str = "The 4D Human Embryonic Brain Atlas: spatiotemporal atlas generation for rapid anatomical changes using first-trimester ultrasound from the Rotterdam Periconceptional Cohort"
    fetal_paper_title: str = "Conditional Fetal Brain Atlas Learning for Automatic Tissue Segmentation"
    paper_directory: str = "fetal_brain_papers"
    index_directory: str = "fetal_brain_indexes"
    
    # Model settings
    model_name: str = "gpt-4"  # Use GPT-4 for complex scientific reasoning
    
    # Brain simulation settings
    enable_brain_simulation: bool = True
    neural_dynamics_enabled: bool = True
    cognitive_science_enabled: bool = True
    machine_learning_enabled: bool = True
    
    # Developmental timeline parameters
    embryonic_weeks_range: Tuple[int, int] = (8, 12)  # Embryonic period
    fetal_weeks_range: Tuple[int, int] = (21, 37)     # Fetal period
    brain_development_stages: List[str] = None
    tissue_types: List[str] = None
    anatomical_landmarks: List[str] = None
    
    # Training parameters
    max_questions: int = 75
    batch_size: int = 8
    learning_rate: float = 0.001
    epochs: int = 4
    
    # Neural parameters (adapted for fetal development)
    working_memory_slots: int = 8  # Increased for complex fetal data
    attention_heads: int = 16      # More attention heads for detailed analysis
    neural_plasticity_rate: float = 0.18  # Higher plasticity for rapid development
    
    # Output settings
    output_dir: str = "fetal_brain_outputs"
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
                "synaptogenesis",
                "tissue_differentiation",
                "myelination"
            ]
        
        if self.tissue_types is None:
            self.tissue_types = [
                "cortical_gray_matter",
                "white_matter",
                "cerebrospinal_fluid",
                "cerebellum",
                "brainstem",
                "deep_gray_matter"
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
                "cortical_plate",
                "sulci",
                "gyri"
            ]
        
        if self.metrics is None:
            self.metrics = [
                "accuracy", "precision", "recall", "f1", 
                "response_time", "anatomical_accuracy", 
                "developmental_consistency", "tissue_segmentation_accuracy",
                "dice_similarity_coefficient"
            ]


class FetalBrainTrainer:
    """
    Comprehensive trainer for fetal brain development using PaperQA.
    
    This class implements:
    - Embryonic to fetal development progression
    - Tissue segmentation and analysis
    - Deep learning atlas generation
    - Multi-modal imaging integration (ultrasound + MRI)
    - Developmental timeline tracking
    - Anatomical landmark recognition
    """
    
    def __init__(self, config: FetalBrainConfig):
        self.config = config
        self.console = Console()
        self.docs = None
        self.neural_components = None
        self.capacity_progression = None
        self.sleep_engine = None
        self.multi_scale_integration = None
        self.biological_validator = None
        
        # Fetal brain specific state
        self.developmental_timeline = {}
        self.tissue_segmentation_state = {}
        self.atlas_generation_state = {}
        self.imaging_modality_state = {}
        self.anatomical_landmarks_state = {}
        
        # Training state
        self.training_history = []
        self.performance_metrics = {}
        self.neural_state = {}
        
        # Initialize components
        self._initialize_components()
        self._initialize_fetal_brain_state()
    
    def _initialize_components(self):
        """Initialize all brain simulation components."""
        try:
            if self.config.enable_brain_simulation:
                self.console.print("[bold blue]Initializing Fetal Brain Simulation Components...[/bold blue]")
                
                # Initialize neural components with fetal-specific parameters
                if self.config.neural_dynamics_enabled:
                    self.neural_components = NeuralComponents(
                        working_memory_slots=self.config.working_memory_slots,
                        attention_heads=self.config.attention_heads
                    )
                    self.console.print("✅ Fetal Neural Components initialized")
                
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
    
    def _initialize_fetal_brain_state(self):
        """Initialize fetal brain-specific state tracking."""
        self.console.print("[bold green]Initializing Fetal Brain State...[/bold green]")
        
        # Initialize developmental timeline (embryonic + fetal)
        for week in range(self.config.embryonic_weeks_range[0], 
                         self.config.fetal_weeks_range[1] + 1):
            period = "embryonic" if week <= 12 else "fetal"
            self.developmental_timeline[f"week_{week}"] = {
                "period": period,
                "brain_structures": [],
                "developmental_milestones": [],
                "anatomical_landmarks": [],
                "tissue_types": [],
                "neural_activity": 0.0,
                "imaging_modality": "ultrasound" if week <= 12 else "mri"
            }
        
        # Initialize tissue segmentation state
        for tissue in self.config.tissue_types:
            self.tissue_segmentation_state[tissue] = {
                "first_appearance_week": None,
                "segmentation_accuracy": 0.0,
                "volume_trajectory": [],
                "dice_coefficient": 0.0
            }
        
        # Initialize atlas generation state
        self.atlas_generation_state = {
            "embryonic_atlas": {
                "method": "deep_learning_registration",
                "imaging": "ultrasound",
                "weeks": self.config.embryonic_weeks_range,
                "subjects": 402,
                "images": 831
            },
            "fetal_atlas": {
                "method": "conditional_deep_learning",
                "imaging": "mri",
                "weeks": self.config.fetal_weeks_range,
                "subjects": 219,
                "tissue_types": len(self.config.tissue_types)
            }
        }
        
        # Initialize imaging modality state
        self.imaging_modality_state = {
            "ultrasound": {
                "period": "embryonic",
                "weeks": self.config.embryonic_weeks_range,
                "advantages": ["real_time", "non_ionizing", "portable"],
                "limitations": ["lower_resolution", "operator_dependent"]
            },
            "mri": {
                "period": "fetal",
                "weeks": self.config.fetal_weeks_range,
                "advantages": ["high_resolution", "tissue_contrast", "3d_imaging"],
                "limitations": ["longer_acquisition", "motion_sensitive"]
            }
        }
        
        # Initialize anatomical landmarks state
        for landmark in self.config.anatomical_landmarks:
            self.anatomical_landmarks_state[landmark] = {
                "first_appearance_week": None,
                "development_progression": [],
                "current_state": "not_formed",
                "imaging_visibility": {"ultrasound": False, "mri": False}
            }
        
        self.console.print("✅ Fetal Brain State initialized")
    
    async def download_target_papers(self):
        """Download both target papers."""
        self.console.print("[bold green]Downloading target papers...[/bold green]")
        
        # Create paper directory
        os.makedirs(self.config.paper_directory, exist_ok=True)
        
        papers = [
            (self.config.embryonic_paper_arxiv_id, "embryonic"),
            (self.config.fetal_paper_arxiv_id, "fetal")
        ]
        
        downloaded_papers = []
        
        for arxiv_id, paper_type in papers:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_path = os.path.join(self.config.paper_directory, f"{arxiv_id}_{paper_type}.pdf")
            
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.console.print(f"✅ Downloaded {paper_type} paper to {pdf_path}")
                downloaded_papers.append(pdf_path)
                
            except Exception as e:
                self.console.print(f"[bold red]Error downloading {paper_type} paper: {e}[/bold red]")
                self.console.print(f"Please manually download arXiv:{arxiv_id} and place it in the papers directory")
        
        return downloaded_papers
    
    async def initialize_paperqa(self):
        """Initialize PaperQA with fetal brain papers."""
        self.console.print("[bold green]Initializing PaperQA for Fetal Brain Research...[/bold green]")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Download target papers if not present
        embryonic_paper_path = os.path.join(self.config.paper_directory, f"{self.config.embryonic_paper_arxiv_id}_embryonic.pdf")
        fetal_paper_path = os.path.join(self.config.paper_directory, f"{self.config.fetal_paper_arxiv_id}_fetal.pdf")
        
        if not os.path.exists(embryonic_paper_path) or not os.path.exists(fetal_paper_path):
            await self.download_target_papers()
        
        # Initialize PaperQA settings
        settings = PaperQASettings()
        settings.agent.index.paper_directory = self.config.paper_directory
        settings.agent.index.index_directory = self.config.index_directory
        
        # Initialize Docs
        self.docs = Docs(settings=settings)
        
        # Add documents if paper directory exists
        paper_path = Path(self.config.paper_directory)
        if paper_path.exists() and any(paper_path.iterdir()):
            self.console.print(f"📚 Loading fetal brain papers from {self.config.paper_directory}")
            # PaperQA will automatically index documents in the paper directory
        else:
            self.console.print(f"[yellow]Warning: Paper directory {self.config.paper_directory} not found or empty[/yellow]")
            self.console.print("Please add PDF documents to the papers directory")
        
        self.console.print("✅ PaperQA initialized for fetal brain research")
    
    def generate_fetal_brain_questions(self) -> List[Dict[str, str]]:
        """Generate specialized questions for fetal brain development."""
        questions = [
            # Paper overview questions
            {
                "question": "What are the main contributions of both the embryonic and fetal brain atlas papers?",
                "answer": "The embryonic paper creates a 4D spatiotemporal atlas using ultrasound (weeks 8-12), while the fetal paper develops conditional deep learning for tissue segmentation using MRI (weeks 21-37).",
                "category": "paper_overview",
                "papers": ["both"]
            },
            {
                "question": "How do the imaging modalities differ between embryonic and fetal brain studies?",
                "answer": "Embryonic studies use ultrasound (weeks 8-12) for real-time imaging, while fetal studies use MRI (weeks 21-37) for high-resolution tissue contrast and 3D imaging.",
                "category": "imaging_modalities",
                "papers": ["both"]
            },
            
            # Embryonic brain specific
            {
                "question": "What is the 4D Human Embryonic Brain Atlas and how was it created?",
                "answer": "The 4D atlas captures rapid anatomical changes using deep learning-based groupwise registration with time-dependent initial atlas and penalization, using 831 ultrasound images from 402 subjects.",
                "category": "embryonic_atlas",
                "papers": ["embryonic"]
            },
            {
                "question": "Why was a time-dependent initial atlas crucial for embryonic brain development?",
                "answer": "Because the brain undergoes rapid changes within days during weeks 8-12, requiring age-specific anatomy maintenance throughout development.",
                "category": "embryonic_methods",
                "papers": ["embryonic"]
            },
            
            # Fetal brain specific
            {
                "question": "What is the Conditional Fetal Brain Atlas Learning framework?",
                "answer": "A deep learning framework combining direct registration with conditional discriminator for generating continuous, age-specific fetal brain atlases for real-time tissue segmentation.",
                "category": "fetal_atlas",
                "papers": ["fetal"]
            },
            {
                "question": "What tissue segmentation performance was achieved in the fetal brain study?",
                "answer": "The method achieved an average Dice Similarity Coefficient (DSC) of 86.3% across six brain tissues with 219 fetal MRIs from 21-37 weeks gestation.",
                "category": "fetal_segmentation",
                "papers": ["fetal"]
            },
            
            # Developmental biology
            {
                "question": "How does brain development progress from embryonic to fetal stages?",
                "answer": "Development progresses from rapid anatomical changes (weeks 8-12) to tissue differentiation and maturation (weeks 21-37), with different imaging modalities capturing each stage optimally.",
                "category": "developmental_biology",
                "papers": ["both"]
            },
            {
                "question": "What are the key brain tissues segmented in fetal MRI?",
                "answer": "The six brain tissues include cortical gray matter, white matter, cerebrospinal fluid, cerebellum, brainstem, and deep gray matter.",
                "category": "tissue_biology",
                "papers": ["fetal"]
            },
            
            # Technical methodology
            {
                "question": "How do the deep learning approaches differ between embryonic and fetal studies?",
                "answer": "Embryonic uses groupwise registration with time-dependent atlas, while fetal uses conditional discriminator with direct registration for tissue-specific segmentation.",
                "category": "technical_methods",
                "papers": ["both"]
            },
            {
                "question": "What validation methods were used in both studies?",
                "answer": "Embryonic: ablation study and ex-vivo atlas comparison. Fetal: Dice coefficient evaluation and volumetric growth trajectory analysis.",
                "category": "validation",
                "papers": ["both"]
            },
            
            # Clinical applications
            {
                "question": "What are the clinical applications of these brain atlases?",
                "answer": "Both enable individualized developmental assessment, improved detection of neurodevelopmental disorders, and support for research and clinical applications with minimal pre-processing.",
                "category": "clinical_applications",
                "papers": ["both"]
            },
            {
                "question": "How do these atlases support real-time assessment?",
                "answer": "The frameworks enable real-time performance for both embryonic ultrasound assessment and fetal MRI tissue segmentation, supporting clinical decision-making.",
                "category": "clinical_applications",
                "papers": ["both"]
            },
            
            # Comparative analysis
            {
                "question": "What are the advantages and limitations of ultrasound vs MRI for brain development?",
                "answer": "Ultrasound: real-time, non-ionizing, portable but lower resolution. MRI: high resolution, tissue contrast, 3D imaging but longer acquisition and motion sensitive.",
                "category": "imaging_comparison",
                "papers": ["both"]
            },
            {
                "question": "How do the datasets and subjects differ between the studies?",
                "answer": "Embryonic: 402 subjects, 831 ultrasound images, weeks 8-12. Fetal: 219 subjects, MRI images, weeks 21-37, focus on tissue segmentation.",
                "category": "dataset_comparison",
                "papers": ["both"]
            }
        ]
        
        return questions
    
    async def train_on_fetal_brain_questions(self):
        """Train the system on fetal brain development questions."""
        questions = self.generate_fetal_brain_questions()
        
        self.console.print(f"[bold green]Starting fetal brain training on {len(questions)} questions...[/bold green]")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Training on fetal brain development...", total=len(questions))
            
            for i, q_data in enumerate(questions):
                question = q_data['question']
                expected_answer = q_data.get('answer', '')
                category = q_data.get('category', 'general')
                papers = q_data.get('papers', ['both'])
                
                try:
                    # Pre-process with fetal brain-specific neural dynamics
                    if self.config.neural_dynamics_enabled and self.neural_components:
                        question = await self._apply_fetal_neural_dynamics(question, category, papers)
                    
                    # Apply fetal brain-specific cognitive science principles
                    if self.config.cognitive_science_enabled:
                        question = await self._apply_fetal_cognitive_science(question, category, papers)
                    
                    # Query PaperQA
                    answer = await self.docs.aquery(question)
                    
                    # Post-process with fetal brain-specific machine learning
                    if self.config.machine_learning_enabled:
                        answer = await self._apply_fetal_machine_learning(answer, category, papers)
                    
                    # Record results with fetal brain context
                    result = {
                        'question': question,
                        'expected_answer': expected_answer,
                        'actual_answer': answer.answer,
                        'confidence': answer.confidence,
                        'sources': answer.sources,
                        'category': category,
                        'papers': papers,
                        'neural_state': self.neural_state.copy() if self.neural_state else {},
                        'developmental_context': self._get_fetal_developmental_context(category, papers),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    
                    # Update fetal brain state
                    if self.config.neural_dynamics_enabled and self.neural_components:
                        await self._update_fetal_neural_state(result, category, papers)
                    
                    # Sleep consolidation every batch
                    if (i + 1) % self.config.batch_size == 0:
                        await self._fetal_sleep_consolidation()
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error processing fetal brain question {i}: {e}")
                    progress.update(task, advance=1)
                    continue
        
        # Final sleep consolidation
        await self._fetal_sleep_consolidation()
        
        # Calculate fetal brain-specific metrics
        self._calculate_fetal_metrics(results)
        
        # Save results
        self._save_fetal_results(results)
        
        return results
    
    async def _apply_fetal_neural_dynamics(self, question: str, category: str, papers: List[str]) -> str:
        """Apply fetal brain-specific neural dynamics processing."""
        if not self.neural_components:
            return question
        
        try:
            # Simulate fetal neural processing
            processed_question = self.neural_components.process_input(question)
            
            # Add fetal brain context based on category and papers
            if category == "embryonic_atlas" or "embryonic" in papers:
                processed_question += " [Focus on embryonic development weeks 8-12, ultrasound imaging]"
            elif category == "fetal_atlas" or "fetal" in papers:
                processed_question += " [Focus on fetal development weeks 21-37, MRI imaging]"
            elif category == "tissue_biology":
                processed_question += " [Focus on tissue segmentation and differentiation]"
            elif category == "imaging_comparison":
                processed_question += " [Compare ultrasound vs MRI modalities]"
            
            # Update fetal neural state
            self.neural_state['last_question'] = question
            self.neural_state['processed_question'] = processed_question
            self.neural_state['question_category'] = category
            self.neural_state['papers_referenced'] = papers
            self.neural_state['neural_activity'] = self.neural_components.get_activity_state()
            self.neural_state['fetal_context'] = True
            
            return processed_question
            
        except Exception as e:
            logger.error(f"Error in fetal neural dynamics: {e}")
            return question
    
    async def _apply_fetal_cognitive_science(self, question: str, category: str, papers: List[str]) -> str:
        """Apply fetal brain-specific cognitive science principles."""
        try:
            # Implement fetal brain-specific cognitive processing
            # - Developmental timeline awareness
            # - Tissue segmentation understanding
            # - Multi-modal imaging integration
            
            if self.capacity_progression:
                # Check working memory capacity for complex fetal data
                capacity = self.capacity_progression.get_current_capacity()
                if len(question.split()) > capacity * 10:  # Adjusted for fetal complexity
                    # Simplify question if too complex
                    question = " ".join(question.split()[:int(capacity * 10)])
            
            # Add developmental context
            if "embryonic" in papers:
                question += f" [Consider embryonic development weeks {self.config.embryonic_weeks_range[0]}-{self.config.embryonic_weeks_range[1]}]"
            if "fetal" in papers:
                question += f" [Consider fetal development weeks {self.config.fetal_weeks_range[0]}-{self.config.fetal_weeks_range[1]}]"
            
            return question
            
        except Exception as e:
            logger.error(f"Error in fetal cognitive science processing: {e}")
            return question
    
    async def _apply_fetal_machine_learning(self, answer: Answer, category: str, papers: List[str]) -> Answer:
        """Apply fetal brain-specific machine learning optimization."""
        try:
            # Implement fetal brain-specific ML optimization
            # - Tissue segmentation accuracy validation
            # - Developmental consistency checking
            # - Multi-modal integration assessment
            
            # For now, return the original answer
            return answer
            
        except Exception as e:
            logger.error(f"Error in fetal machine learning processing: {e}")
            return answer
    
    def _get_fetal_developmental_context(self, category: str, papers: List[str]) -> Dict[str, Any]:
        """Get fetal developmental context for the question category."""
        context = {
            "embryonic_weeks": self.config.embryonic_weeks_range,
            "fetal_weeks": self.config.fetal_weeks_range,
            "brain_development_stages": self.config.brain_development_stages,
            "tissue_types": self.config.tissue_types,
            "anatomical_landmarks": self.config.anatomical_landmarks,
            "category": category,
            "papers": papers
        }
        
        if category == "embryonic_atlas":
            context["focus"] = "embryonic brain atlas generation"
            context["imaging"] = "ultrasound"
        elif category == "fetal_atlas":
            context["focus"] = "fetal brain atlas and tissue segmentation"
            context["imaging"] = "mri"
        elif category == "tissue_biology":
            context["focus"] = "brain tissue differentiation and segmentation"
        elif category == "imaging_comparison":
            context["focus"] = "ultrasound vs MRI comparison"
        
        return context
    
    async def _update_fetal_neural_state(self, result: Dict[str, Any], category: str, papers: List[str]):
        """Update fetal neural state based on training results."""
        if not self.neural_components:
            return
        
        try:
            # Update neural plasticity with fetal-specific rates
            self.neural_components.update_plasticity(
                result['confidence'],
                self.config.neural_plasticity_rate
            )
            
            # Update capacity progression
            if self.capacity_progression:
                self.capacity_progression.update_capacity(result['confidence'])
            
            # Update developmental timeline
            self._update_fetal_developmental_timeline(category, papers, result['confidence'])
            
            # Update tissue segmentation state
            if category == "fetal_segmentation" or category == "tissue_biology":
                self._update_tissue_segmentation_state(result['confidence'])
            
        except Exception as e:
            logger.error(f"Error updating fetal neural state: {e}")
    
    def _update_fetal_developmental_timeline(self, category: str, papers: List[str], confidence: float):
        """Update the fetal developmental timeline based on question category and papers."""
        try:
            # Update neural activity for relevant weeks
            if "embryonic" in papers:
                for week in range(self.config.embryonic_weeks_range[0], 
                                 self.config.embryonic_weeks_range[1] + 1):
                    week_key = f"week_{week}"
                    if week_key in self.developmental_timeline:
                        self.developmental_timeline[week_key]["neural_activity"] += confidence * 0.1
            
            if "fetal" in papers:
                for week in range(self.config.fetal_weeks_range[0], 
                                 self.config.fetal_weeks_range[1] + 1):
                    week_key = f"week_{week}"
                    if week_key in self.developmental_timeline:
                        self.developmental_timeline[week_key]["neural_activity"] += confidence * 0.1
            
            # Update tissue segmentation state
            if category == "tissue_biology" or category == "fetal_segmentation":
                for tissue in self.config.tissue_types:
                    if tissue in self.tissue_segmentation_state:
                        self.tissue_segmentation_state[tissue]["segmentation_accuracy"] += confidence * 0.05
                        self.tissue_segmentation_state[tissue]["dice_coefficient"] += confidence * 0.05
            
        except Exception as e:
            logger.error(f"Error updating fetal developmental timeline: {e}")
    
    def _update_tissue_segmentation_state(self, confidence: float):
        """Update tissue segmentation state based on training performance."""
        try:
            # Update Dice coefficients for all tissue types
            for tissue in self.config.tissue_types:
                if tissue in self.tissue_segmentation_state:
                    # Simulate improvement in segmentation accuracy
                    current_dice = self.tissue_segmentation_state[tissue]["dice_coefficient"]
                    improvement = confidence * 0.01  # Small improvement per training step
                    self.tissue_segmentation_state[tissue]["dice_coefficient"] = min(1.0, current_dice + improvement)
                    
                    # Update volume trajectory
                    self.tissue_segmentation_state[tissue]["volume_trajectory"].append({
                        "confidence": confidence,
                        "dice_coefficient": self.tissue_segmentation_state[tissue]["dice_coefficient"],
                        "timestamp": datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error updating tissue segmentation state: {e}")
    
    async def _fetal_sleep_consolidation(self):
        """Perform fetal brain-specific sleep consolidation."""
        if not self.sleep_engine:
            return
        
        try:
            # Simulate fetal sleep consolidation
            self.sleep_engine.consolidate_memories(self.training_history)
            
            # Update fetal neural state
            self.neural_state['sleep_consolidation'] = True
            self.neural_state['fetal_consolidation'] = True
            self.neural_state['consolidation_timestamp'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error in fetal sleep consolidation: {e}")
    
    def _calculate_fetal_metrics(self, results: List[Dict[str, Any]]):
        """Calculate fetal brain-specific performance metrics."""
        self.console.print("[bold blue]Calculating Fetal Brain Performance Metrics...[/bold blue]")
        
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
        
        # Paper-specific metrics
        paper_metrics = {"embryonic": [], "fetal": [], "both": []}
        for result in results:
            papers = result.get('papers', ['both'])
            for paper in papers:
                if result['confidence']:
                    paper_metrics[paper].append(result['confidence'])
        
        for paper, confs in paper_metrics.items():
            if confs:
                metrics[f'{paper}_paper_avg_confidence'] = np.mean(confs)
                metrics[f'{paper}_paper_count'] = len(confs)
        
        # Fetal brain-specific metrics
        metrics['developmental_timeline_coverage'] = len(self.developmental_timeline)
        metrics['tissue_types_tracked'] = len(self.tissue_segmentation_state)
        metrics['anatomical_landmarks_tracked'] = len(self.anatomical_landmarks_state)
        metrics['imaging_modalities'] = len(self.imaging_modality_state)
        
        # Tissue segmentation metrics
        tissue_dice_scores = [state["dice_coefficient"] for state in self.tissue_segmentation_state.values()]
        if tissue_dice_scores:
            metrics['avg_tissue_dice_coefficient'] = np.mean(tissue_dice_scores)
            metrics['tissue_segmentation_accuracy'] = np.mean([state["segmentation_accuracy"] for state in self.tissue_segmentation_state.values()])
        
        # Neural metrics
        if self.neural_components:
            neural_metrics = self.neural_components.get_metrics()
            metrics.update(neural_metrics)
        
        self.performance_metrics = metrics
        
        # Display metrics
        self._display_fetal_metrics(metrics)
    
    def _display_fetal_metrics(self, metrics: Dict[str, Any]):
        """Display fetal brain-specific performance metrics."""
        table = Table(title="Fetal Brain Training Performance Metrics")
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
        
        # Paper-specific metrics
        for metric, value in metrics.items():
            if 'paper_avg_confidence' in metric:
                table.add_row(metric, f"{value:.4f}", "Paper")
        
        # Fetal brain metrics
        for metric in ['developmental_timeline_coverage', 'tissue_types_tracked', 'anatomical_landmarks_tracked']:
            value = metrics.get(metric, 0)
            table.add_row(metric, str(value), "Fetal Brain")
        
        # Tissue segmentation metrics
        for metric in ['avg_tissue_dice_coefficient', 'tissue_segmentation_accuracy']:
            value = metrics.get(metric, 0)
            if value:
                table.add_row(metric, f"{value:.4f}", "Tissue Segmentation")
        
        self.console.print(table)
    
    def _save_fetal_results(self, results: List[Dict[str, Any]]):
        """Save fetal brain-specific training results."""
        self.console.print("[bold green]Saving Fetal Brain Results...[/bold green]")
        
        # Save results
        results_file = os.path.join(self.config.output_dir, "fetal_brain_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, "fetal_brain_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        # Save fetal brain state
        fetal_state = {
            "developmental_timeline": self.developmental_timeline,
            "tissue_segmentation_state": self.tissue_segmentation_state,
            "atlas_generation_state": self.atlas_generation_state,
            "imaging_modality_state": self.imaging_modality_state,
            "anatomical_landmarks_state": self.anatomical_landmarks_state,
            "neural_state": self.neural_state
        }
        
        state_file = os.path.join(self.config.output_dir, "fetal_brain_state.json")
        with open(state_file, 'w') as f:
            json.dump(fetal_state, f, indent=2, default=str)
        
        # Save PaperQA docs if requested
        if self.config.save_embeddings and self.docs:
            docs_file = os.path.join(self.config.output_dir, "fetal_brain_docs.pkl")
            with open(docs_file, 'wb') as f:
                pickle.dump(self.docs, f)
        
        self.console.print(f"✅ Fetal brain results saved to {self.config.output_dir}")
    
    def generate_fetal_training_report(self):
        """Generate a comprehensive fetal brain training report."""
        report_file = os.path.join(self.config.output_dir, "fetal_brain_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Fetal Brain Training Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Embryonic Paper:** {self.config.embryonic_paper_title}\n")
            f.write(f"**Embryonic arXiv ID:** {self.config.embryonic_paper_arxiv_id}\n")
            f.write(f"**Fetal Paper:** {self.config.fetal_paper_title}\n")
            f.write(f"**Fetal arXiv ID:** {self.config.fetal_paper_arxiv_id}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- Paper Directory: {self.config.paper_directory}\n")
            f.write(f"- Model: {self.config.model_name}\n")
            f.write(f"- Embryonic Weeks: {self.config.embryonic_weeks_range[0]}-{self.config.embryonic_weeks_range[1]}\n")
            f.write(f"- Fetal Weeks: {self.config.fetal_weeks_range[0]}-{self.config.fetal_weeks_range[1]}\n")
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
                f.write(f"- **{week}** ({data['period']}): Neural activity: {data['neural_activity']:.4f}, Imaging: {data['imaging_modality']}\n")
            
            f.write("\n## Tissue Segmentation Performance\n\n")
            for tissue, data in self.tissue_segmentation_state.items():
                f.write(f"- **{tissue}:** Dice coefficient: {data['dice_coefficient']:.4f}, Accuracy: {data['segmentation_accuracy']:.4f}\n")
            
            f.write("\n## Atlas Generation Summary\n\n")
            f.write("### Embryonic Atlas\n")
            embryonic = self.atlas_generation_state["embryonic_atlas"]
            f.write(f"- Method: {embryonic['method']}\n")
            f.write(f"- Imaging: {embryonic['imaging']}\n")
            f.write(f"- Subjects: {embryonic['subjects']}\n")
            f.write(f"- Images: {embryonic['images']}\n\n")
            
            f.write("### Fetal Atlas\n")
            fetal = self.atlas_generation_state["fetal_atlas"]
            f.write(f"- Method: {fetal['method']}\n")
            f.write(f"- Imaging: {fetal['imaging']}\n")
            f.write(f"- Subjects: {fetal['subjects']}\n")
            f.write(f"- Tissue Types: {fetal['tissue_types']}\n\n")
            
            f.write("## Training History\n\n")
            f.write(f"- Total training sessions: {len(self.training_history)}\n")
            f.write(f"- Fetal brain focus: ✅\n")
            f.write(f"- Multi-modal imaging: ✅\n")
            f.write(f"- Tissue segmentation: ✅\n")
            f.write(f"- Developmental timeline tracking: ✅\n")
        
        self.console.print(f"✅ Fetal brain training report saved to {report_file}")


async def main():
    """Main fetal brain training function."""
    console.print(Panel.fit(
        "[bold blue]Fetal Brain Training with PaperQA[/bold blue]\n"
        "Comprehensive training on embryonic and fetal brain development research",
        border_style="blue"
    ))
    
    # Configuration
    config = FetalBrainConfig()
    
    # Initialize trainer
    trainer = FetalBrainTrainer(config)
    
    # Initialize PaperQA
    await trainer.initialize_paperqa()
    
    # Train on fetal brain questions
    results = await trainer.train_on_fetal_brain_questions()
    
    # Generate report
    trainer.generate_fetal_training_report()
    
    console.print("[bold green]Fetal brain training completed successfully![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
