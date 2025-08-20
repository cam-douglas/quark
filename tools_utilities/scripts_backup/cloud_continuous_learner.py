#!/usr/bin/env python3
"""
Cloud-Powered Continuous Learning System
Runs continuously from startup and constantly learns from trusted sources
"""

import os, sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import requests
from datetime import datetime, timedelta
import threading
import queue

# Add small-mind to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudContinuousLearner:
    """Cloud-powered continuous learning system"""
    
    def __init__(self):
        self.smallmind_path = Path("ROOT")
        self.data_dir = self.smallmind_path / "continuous_learning_data"
        self.models_dir = self.smallmind_path / "models"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Learning queues
        self.short_prompt_queue = queue.Queue()
        self.long_prompt_queue = queue.Queue()
        self.synthetic_data_queue = queue.Queue()
        
        # Trusted sources
        self.trusted_sources = {
            "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/",
            "arxiv": "http://export.arxiv.org/api/query?search_query=",
            "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        }
        
        # Learning categories
        self.learning_categories = [
            "computational_neuroscience",
            "physics_simulation", 
            "ml_optimization",
            "data_visualization",
            "natural_language_processing"
        ]
        
        # Background training status
        self.is_training = False
        self.training_thread = None
        
    def start_continuous_learning(self):
        """Start continuous learning in background"""
        if not self.is_training:
            self.is_training = True
            self.training_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
            self.training_thread.start()
            logger.info("🚀 Continuous learning started in background")
    
    def _continuous_learning_loop(self):
        """Main continuous learning loop"""
        while self.is_training:
            try:
                # Generate synthetic data
                self._generate_synthetic_data_batch()
                
                # Process training queues
                self._process_training_queues()
                
                # Cloud-based training
                self._cloud_training_iteration()
                
                # Sleep before next iteration
                time.sleep(300)  # 5 minutes between iterations
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(60)
    
    def _generate_synthetic_data_batch(self):
        """Generate synthetic training data from trusted sources"""
        try:
            for category in self.learning_categories:
                # Crawl Wikipedia
                wiki_data = self._crawl_wikipedia(category, max_results=20)
                
                # Add to synthetic data queue
                for data in wiki_data:
                    self.synthetic_data_queue.put({
                        "data": data,
                        "category": category,
                        "priority": "high" if "neuroscience" in category else "medium"
                    })
                
                logger.info(f"Generated {len(wiki_data)} synthetic data points for {category}")
                    
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
    
    def _crawl_wikipedia(self, category: str, max_results: int) -> List[Dict[str, Any]]:
        """Crawl Wikipedia for category-specific data"""
        data = []
        
        # Category-specific search terms
        search_terms = self._get_category_search_terms(category)
        
        for term in search_terms[:max_results]:
            try:
                url = f"{self.trusted_sources['wikipedia']}{term.replace(' ', '_')}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    content = response.json()
                    data.append({
                        "source": "wikipedia",
                        "title": content.get("title", ""),
                        "extract": content.get("extract", ""),
                        "category": category,
                        "url": content.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.debug(f"Error crawling Wikipedia for {term}: {e}")
                continue
        
        return data
    
    def _get_category_search_terms(self, category: str) -> List[str]:
        """Get search terms for a specific category"""
        search_terms = {
            "computational_neuroscience": [
                "neural networks", "synaptic plasticity", "brain development",
                "cognitive neuroscience", "neural dynamics", "brain modeling"
            ],
            "physics_simulation": [
                "fluid dynamics", "particle systems", "collision detection",
                "rigid body dynamics", "soft body physics"
            ],
            "ml_optimization": [
                "hyperparameter optimization", "neural architecture search",
                "machine learning optimization", "deep learning training"
            ],
            "data_visualization": [
                "3D visualization", "scientific visualization", "data plotting",
                "interactive visualization", "3D rendering"
            ],
            "natural_language_processing": [
                "natural language understanding", "language models", "text generation",
                "semantic analysis", "language processing"
            ]
        }
        
        return search_terms.get(category, [category])
    
    def _process_training_queues(self):
        """Process training queues for short and long prompts"""
        try:
            # Process short prompt queue
            while not self.short_prompt_queue.empty():
                prompt_data = self.short_prompt_queue.get_nowait()
                self._train_on_prompt(prompt_data, prompt_type="short")
            
            # Process long prompt queue
            while not self.long_prompt_queue.empty():
                prompt_data = self.long_prompt_queue.get_nowait()
                self._train_on_prompt(prompt_data, prompt_type="long")
                
        except Exception as e:
            logger.error(f"Error processing training queues: {e}")
    
    def _train_on_prompt(self, prompt_data: Dict[str, Any], prompt_type: str):
        """Train on user prompt data"""
        try:
            # Extract training information
            user_input = prompt_data.get("input", "")
            user_feedback = prompt_data.get("feedback", "")
            category = prompt_data.get("category", "general")
            
            # Create training example
            training_example = {
                "input": user_input,
                "expected_output": user_feedback,
                "category": category,
                "prompt_type": prompt_type,
                "timestamp": datetime.now().isoformat(),
                "source": "user_interaction"
            }
            
            # Save training example
            self._save_training_example(training_example, category)
            
        except Exception as e:
            logger.error(f"Error training on prompt: {e}")
    
    def _save_training_example(self, example: Dict[str, Any], category: str):
        """Save training example to appropriate category file"""
        try:
            category_file = self.data_dir / f"{category}_training_data.json"
            
            # Load existing data
            if category_file.exists():
                with open(category_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Add new example
            data.append(example)
            
            # Save updated data
            with open(category_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving training example: {e}")
    
    def _cloud_training_iteration(self):
        """Perform cloud-based training iteration"""
        try:
            # Check cloud resources (simplified)
            available_instances = 2  # Simulate cloud availability
            
            if available_instances > 0:
                # Submit training jobs
                for category in self.learning_categories:
                    if self._should_train_category(category):
                        training_job = self._create_training_job(category)
                        self._submit_cloud_training_job(training_job)
                        
        except Exception as e:
            logger.error(f"Error in cloud training iteration: {e}")
    
    def _should_train_category(self, category: str) -> bool:
        """Determine if a category should be trained"""
        try:
            # Check if enough new data exists
            category_file = self.data_dir / f"{category}_training_data.json"
            
            if not category_file.exists():
                return True
            
            # Check data freshness
            with open(category_file, 'r') as f:
                data = json.load(f)
            
            if len(data) < 10:  # Need minimum data
                return True
            
            # Check if data is recent
            latest_timestamp = max(item.get("timestamp", "") for item in data)
            if latest_timestamp:
                latest_time = datetime.fromisoformat(latest_timestamp)
                if datetime.now() - latest_time > timedelta(hours=1):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if should train {category}: {e}")
            return False
    
    def _create_training_job(self, category: str) -> Dict[str, Any]:
        """Create training job for a category"""
        return {
            "category": category,
            "priority": "normal",
            "timestamp": datetime.now().isoformat(),
            "type": "scheduled",
            "config": {
                "batch_size": 64,
                "learning_rate": 1e-4,
                "epochs": 5,
                "auto_optimization": True
            }
        }
    
    def _submit_cloud_training_job(self, training_job: Dict[str, Any]):
        """Submit training job to cloud"""
        try:
            # Save job to queue
            job_file = self.data_dir / "cloud_training_queue.json"
            
            if job_file.exists():
                with open(job_file, 'r') as f:
                    queue = json.load(f)
            else:
                queue = []
            
            queue.append(training_job)
            
            with open(job_file, 'w') as f:
                json.dump(queue, f, indent=2)
            
            logger.info(f"Submitted cloud training job for {training_job['category']}")
            
        except Exception as e:
            logger.error(f"Error submitting cloud training job: {e}")
    
    def add_user_interaction(self, user_input: str, response: str, feedback: str = "", category: str = "general"):
        """Add user interaction for continuous learning"""
        try:
            interaction = {
                "input": user_input,
                "response": response,
                "feedback": feedback,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(user_input),
                "response_length": len(response)
            }
            
            # Save interaction
            interaction_file = self.data_dir / "user_interactions.json"
            
            if interaction_file.exists():
                with open(interaction_file, 'r') as f:
                    interactions = json.load(f)
            else:
                interactions = []
            
            interactions.append(interaction)
            
            with open(interaction_file, 'w') as f:
                json.dump(interactions, f, indent=2)
            
            # Add to appropriate training queue
            if len(user_input) < 50:
                self.short_prompt_queue.put({
                    "input": user_input,
                    "feedback": feedback,
                    "category": category,
                    "priority": "high" if feedback else "medium"
                })
            else:
                self.long_prompt_queue.put({
                    "input": user_input,
                    "feedback": feedback,
                    "category": category,
                    "priority": "high" if feedback else "medium"
                })
            
            logger.info(f"Added user interaction for {category}")
            
        except Exception as e:
            logger.error(f"Error adding user interaction: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status"""
        try:
            status = {
                "is_training": self.is_training,
                "categories": {},
                "queue_sizes": {
                    "short_prompts": self.short_prompt_queue.qsize(),
                    "long_prompts": self.long_prompt_queue.qsize(),
                    "synthetic_data": self.synthetic_data_queue.qsize()
                },
                "last_update": datetime.now().isoformat()
            }
            
            # Add category-specific status
            for category in self.learning_categories:
                category_file = self.data_dir / f"{category}_training_data.json"
                if category_file.exists():
                    with open(category_file, 'r') as f:
                        data = json.load(f)
                    status["categories"][category] = {
                        "training_examples": len(data),
                        "last_training": data[-1]["timestamp"] if data else "Never"
                    }
                else:
                    status["categories"][category] = {
                        "training_examples": 0,
                        "last_training": "Never"
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting learning status: {e}")
            return {"error": str(e)}
    
    def stop_continuous_learning(self):
        """Stop continuous learning"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        logger.info("🛑 Continuous learning stopped")

def main():
    """Main function for command-line usage"""
    learner = CloudContinuousLearner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            learner.start_continuous_learning()
            print("🚀 Continuous learning started")
        elif command == "stop":
            learner.stop_continuous_learning()
            print("🛑 Continuous learning stopped")
        elif command == "status":
            status = learner.get_learning_status()
            print(json.dumps(status, indent=2))
        elif command == "add_interaction":
            if len(sys.argv) >= 4:
                user_input = sys.argv[2]
                response = sys.argv[3]
                feedback = sys.argv[4] if len(sys.argv) > 4 else ""
                category = sys.argv[5] if len(sys.argv) > 5 else "general"
                learner.add_user_interaction(user_input, response, feedback, category)
                print("✅ User interaction added")
            else:
                print("Usage: python cloud_continuous_learner.py add_interaction <input> <response> [feedback] [category]")
        else:
            print("Usage: python cloud_continuous_learner.py [start|stop|status|add_interaction]")
    else:
        # Default: show status
        status = learner.get_learning_status()
        print("Cloud-Powered Continuous Learning System")
        print("=" * 50)
        print(f"Status: {'🟢 Running' if status.get('is_training') else '🔴 Stopped'}")
        print(f"Queue Sizes: {status.get('queue_sizes', {})}")

if __name__ == "__main__":
    main()
