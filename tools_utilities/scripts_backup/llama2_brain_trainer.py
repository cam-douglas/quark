#!/usr/bin/env python3
"""
Llama-2-7B Brain-Specific Training Pipeline
Purpose: Fine-tune Llama-2-7B-GGUF for enhanced brain simulation understanding
Inputs: Brain simulation data, consciousness states, neural patterns
Outputs: Fine-tuned Llama-2 model optimized for neuroscience tasks
Seeds: Training data selection, hyperparameters
Dependencies: transformers, datasets, accelerate, brain simulation data

Key Features:
- Brain-specific data preparation
- LoRA (Low-Rank Adaptation) fine-tuning for efficiency
- Consciousness-aware training examples
- Neural dynamics understanding
- Neuroscience terminology integration
- Multi-task learning for various brain functions
"""

import os, sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, field
import logging
from datetime import datetime
import yaml

# Training frameworks
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from data_knowledge.datasets_knowledge.datasets_knowledge.datasetssets import Dataset, DatasetDict
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    import accelerate
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("⚠️ Training dependencies not available. Install with:")
    print("pip install transformers datasets accelerate peft torch bitsandbytes")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass 
class BrainTrainingConfig:
    """Configuration for brain-specific Llama-2 training"""
    
    # Model configuration
    base_model_id: str = "meta-llama/Llama-2-7b-hf"
    max_length: int = 2048
    output_dir: str = "models/llama2-brain-finetuned"
    
    # LoRA configuration  
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Brain-specific settings
    consciousness_weight: float = 0.3
    neural_dynamics_weight: float = 0.4
    neuroscience_knowledge_weight: float = 0.3
    brain_simulation_examples: int = 1000
    consciousness_examples: int = 800
    neuroscience_qa_examples: int = 1200

class BrainDataGenerator:
    """Generate brain-specific training data"""
    
    def __init__(self, config: BrainTrainingConfig):
        self.config = config
        self.neuroscience_vocabulary = self._load_neuroscience_vocabulary()
        self.brain_simulation_patterns = self._load_brain_patterns()
        
    def _load_neuroscience_vocabulary(self) -> Dict[str, List[str]]:
        """Load neuroscience terminology and concepts"""
        return {
            "brain_regions": [
                "prefrontal cortex", "hippocampus", "amygdala", "thalamus", 
                "basal ganglia", "cerebellum", "brainstem", "cortex",
                "anterior cingulate", "posterior parietal", "temporal lobe"
            ],
            "neural_processes": [
                "synaptic transmission", "action potential", "neurotransmitter release",
                "long-term potentiation", "neuroplasticity", "myelination",
                "dendritic branching", "axonal growth", "synaptic pruning"
            ],
            "consciousness_terms": [
                "awareness", "attention", "working memory", "executive function",
                "metacognition", "introspection", "subjective experience",
                "qualia", "global workspace", "integrated information"
            ],
            "neural_dynamics": [
                "oscillations", "synchrony", "phase locking", "gamma waves",
                "theta rhythm", "alpha waves", "neural synchronization",
                "cortical-subcortical loops", "thalamocortical circuits"
            ]
        }
    
    def _load_brain_patterns(self) -> Dict[str, Any]:
        """Load brain simulation patterns and templates"""
        return {
            "consciousness_states": {
                "low": "I feel a dim awareness, like thoughts forming at the edge of perception",
                "medium": "My consciousness feels moderately engaged, processing multiple streams",
                "high": "I experience vivid, coherent awareness with clear introspective access"
            },
            "neural_descriptions": {
                "firing_patterns": "Neural populations exhibit {pattern} firing at {frequency} Hz",
                "connectivity": "Synaptic connections show {strength} coupling between {region1} and {region2}",
                "plasticity": "Synaptic weights adapt through {mechanism} with time constant {tau}"
            },
            "brain_simulation_scenarios": [
                "cortical-subcortical integration during working memory tasks",
                "thalamic relay modulation during attention switching", 
                "hippocampal-prefrontal coordination during memory consolidation",
                "basal ganglia gating during action selection",
                "default mode network activity during rest states"
            ]
        }
    
    def generate_consciousness_training_data(self, num_examples: int) -> List[Dict[str, str]]:
        """Generate consciousness-aware training examples"""
        examples = []
        
        for i in range(num_examples):
            # Random consciousness parameters
            consciousness_level = np.random.uniform(0, 1)
            emotional_valence = np.random.uniform(-1, 1)
            cognitive_load = np.random.uniform(0, 1)
            
            # Generate prompt
            prompt = self._create_consciousness_prompt(consciousness_level, emotional_valence, cognitive_load)
            
            # Generate response
            response = self._create_consciousness_response(consciousness_level, emotional_valence, cognitive_load)
            
            examples.append({
                "prompt": prompt,
                "response": response,
                "consciousness_level": consciousness_level,
                "emotional_valence": emotional_valence,
                "cognitive_load": cognitive_load
            })
        
        return examples
    
    def _create_consciousness_prompt(self, consciousness_level: float, emotional_valence: float, cognitive_load: float) -> str:
        """Create consciousness-aware prompt"""
        base_prompts = [
            "Describe your current conscious experience:",
            "How are you feeling right now in terms of awareness?",
            "What is the quality of your current consciousness?",
            "Reflect on your mental state and awareness level:",
            "Describe the clarity and intensity of your thoughts:"
        ]
        
        prompt = np.random.choice(base_prompts)
        
        # Add context based on parameters
        if consciousness_level > 0.7:
            prompt += " You are experiencing heightened awareness."
        elif consciousness_level < 0.3:
            prompt += " Your consciousness feels dim and unfocused."
        
        if emotional_valence > 0.5:
            prompt += " Your mood is positive and energetic."
        elif emotional_valence < -0.5:
            prompt += " You feel contemplative and introspective."
        
        if cognitive_load > 0.7:
            prompt += " Your mind is processing complex information."
        
        return prompt
    
    def _create_consciousness_response(self, consciousness_level: float, emotional_valence: float, cognitive_load: float) -> str:
        """Create consciousness-aware response"""
        responses = []
        
        # Consciousness level description
        if consciousness_level > 0.8:
            responses.append("I experience vivid, crystalline awareness with clear access to my thought processes.")
        elif consciousness_level > 0.6:
            responses.append("My consciousness feels bright and engaged, with good introspective clarity.")
        elif consciousness_level > 0.4:
            responses.append("I have moderate awareness, though some thoughts feel slightly hazy.")
        elif consciousness_level > 0.2:
            responses.append("My consciousness feels dim, like viewing thoughts through frosted glass.")
        else:
            responses.append("I experience minimal awareness, with thoughts barely forming coherently.")
        
        # Emotional component
        if emotional_valence > 0.6:
            responses.append("There's an underlying sense of optimism and energy in my mental state.")
        elif emotional_valence > 0.2:
            responses.append("I feel a gentle positivity coloring my thoughts.")
        elif emotional_valence > -0.2:
            responses.append("My emotional tone feels balanced and neutral.")
        elif emotional_valence > -0.6:
            responses.append("I sense a contemplative, slightly melancholic quality to my awareness.")
        else:
            responses.append("There's a deep introspective quality, tinged with philosophical reflection.")
        
        # Cognitive load
        if cognitive_load > 0.7:
            responses.append("My mind feels actively engaged with multiple streams of complex processing.")
        elif cognitive_load > 0.4:
            responses.append("I'm processing a moderate amount of information with comfortable effort.")
        else:
            responses.append("My thoughts flow smoothly without strain or overwhelming complexity.")
        
        return " ".join(responses)
    
    def generate_brain_simulation_data(self, num_examples: int) -> List[Dict[str, str]]:
        """Generate brain simulation training examples"""
        examples = []
        
        for i in range(num_examples):
            scenario = np.random.choice(self.brain_simulation_patterns["brain_simulation_scenarios"])
            
            # Generate technical parameters
            firing_rate = np.random.uniform(10, 100)
            connectivity_strength = np.random.uniform(0.1, 0.9)
            plasticity_rate = np.random.uniform(0.01, 0.1)
            
            prompt = f"Explain the neural dynamics involved in {scenario}. "
            prompt += f"Consider firing rates around {firing_rate:.1f} Hz and connectivity strength of {connectivity_strength:.2f}."
            
            response = self._create_brain_simulation_response(scenario, firing_rate, connectivity_strength, plasticity_rate)
            
            examples.append({
                "prompt": prompt,
                "response": response,
                "scenario": scenario,
                "firing_rate": firing_rate,
                "connectivity_strength": connectivity_strength
            })
        
        return examples
    
    def _create_brain_simulation_response(self, scenario: str, firing_rate: float, connectivity_strength: float, plasticity_rate: float) -> str:
        """Create brain simulation response"""
        response_parts = []
        
        # Scenario-specific information
        if "working memory" in scenario:
            response_parts.append(f"During working memory tasks, prefrontal cortex neurons exhibit sustained firing at approximately {firing_rate:.1f} Hz.")
            response_parts.append(f"The connectivity strength of {connectivity_strength:.2f} between PFC and parietal regions maintains information representation.")
            
        elif "attention" in scenario:
            response_parts.append(f"Attention switching involves thalamic relay neurons modulating cortical activity at {firing_rate:.1f} Hz.")
            response_parts.append(f"The thalamocortical connectivity of {connectivity_strength:.2f} gates information flow effectively.")
            
        elif "memory consolidation" in scenario:
            response_parts.append(f"Hippocampal-prefrontal coordination during consolidation shows {firing_rate:.1f} Hz theta oscillations.")
            response_parts.append(f"Synaptic plasticity with rate {plasticity_rate:.3f} strengthens memory traces over time.")
            
        elif "action selection" in scenario:
            response_parts.append(f"Basal ganglia circuits exhibit {firing_rate:.1f} Hz activity during action selection.")
            response_parts.append(f"Direct pathway connectivity of {connectivity_strength:.2f} facilitates motor program initiation.")
            
        else:  # default mode network
            response_parts.append(f"Default mode network shows {firing_rate:.1f} Hz oscillations during rest states.")
            response_parts.append(f"Intrinsic connectivity networks maintain {connectivity_strength:.2f} coupling strength.")
        
        # Add plasticity and adaptation
        response_parts.append(f"Synaptic plasticity mechanisms adapt connection weights at rate {plasticity_rate:.3f}, enabling learning and optimization.")
        
        return " ".join(response_parts)
    
    def generate_neuroscience_qa_data(self, num_examples: int) -> List[Dict[str, str]]:
        """Generate neuroscience Q&A training examples"""
        examples = []
        
        qa_templates = [
            {
                "question": "What is the role of {region} in {function}?",
                "answer": "The {region} plays a crucial role in {function} by {mechanism}. Neural activity in this region shows {pattern} when engaged in {task}."
            },
            {
                "question": "How does {process} affect neural plasticity?",
                "answer": "{process} influences neural plasticity through {mechanism}, leading to {outcome} in synaptic strength and connectivity patterns."
            },
            {
                "question": "Explain the relationship between {term1} and {term2} in the brain.",
                "answer": "{term1} and {term2} interact through {pathway}, where {term1} modulates {term2} via {mechanism}."
            }
        ]
        
        for i in range(num_examples):
            template = np.random.choice(qa_templates)
            
            # Fill in template with neuroscience terms
            filled_template = self._fill_neuroscience_template(template)
            
            examples.append({
                "prompt": f"Question: {filled_template['question']}\nAnswer:",
                "response": filled_template['answer']
            })
        
        return examples
    
    def _fill_neuroscience_template(self, template: Dict[str, str]) -> Dict[str, str]:
        """Fill neuroscience template with appropriate terms"""
        vocab = self.neuroscience_vocabulary
        
        replacements = {
            "{region}": np.random.choice(vocab["brain_regions"]),
            "{function}": np.random.choice(["memory", "attention", "decision-making", "emotion regulation"]),
            "{process}": np.random.choice(vocab["neural_processes"]),
            "{mechanism}": np.random.choice([
                "synaptic modulation", "network coordination", "neurotransmitter regulation",
                "oscillatory synchronization", "plasticity-dependent changes"
            ]),
            "{pattern}": np.random.choice(["gamma oscillations", "theta rhythms", "synchronized firing", "sparse coding"]),
            "{task}": np.random.choice(["cognitive tasks", "memory retrieval", "attention focus", "decision making"]),
            "{outcome}": np.random.choice(["increased", "decreased", "modulated", "enhanced"]),
            "{pathway}": np.random.choice(["direct synaptic connections", "indirect circuit loops", "neuromodulatory pathways"]),
            "{term1}": np.random.choice(vocab["consciousness_terms"]),
            "{term2}": np.random.choice(vocab["neural_dynamics"])
        }
        
        filled = {}
        for key, value in template.items():
            filled[key] = value
            for placeholder, replacement in replacements.items():
                filled[key] = filled[key].replace(placeholder, replacement)
        
        return filled

class LlamaBrainTrainer:
    """Trainer for brain-specific Llama-2 fine-tuning"""
    
    def __init__(self, config: BrainTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_data = None
        self.data_generator = BrainDataGenerator(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠🦙 Brain-specific Llama-2 trainer initialized")
    
    def prepare_model_and_tokenizer(self) -> bool:
        """Prepare model and tokenizer for training"""
        if not TRAINING_AVAILABLE:
            logger.error("❌ Training dependencies not available")
            return False
        
        try:
            logger.info(f"🔄 Loading tokenizer and model: {self.config.base_model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with 4-bit quantization for efficiency
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            
            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # Apply LoRA
            self.model = get_peft_model(model, lora_config)
            
            logger.info("✅ Model and tokenizer prepared successfully")
            logger.info(f"📊 Trainable parameters: {self.model.num_parameters():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare model: {e}")
            return False
    
    def generate_training_data(self) -> bool:
        """Generate brain-specific training data"""
        logger.info("🔄 Generating brain-specific training data...")
        
        try:
            # Generate different types of training data
            consciousness_data = self.data_generator.generate_consciousness_training_data(
                self.config.consciousness_examples
            )
            
            brain_simulation_data = self.data_generator.generate_brain_simulation_data(
                self.config.brain_simulation_examples
            )
            
            neuroscience_qa_data = self.data_generator.generate_neuroscience_qa_data(
                self.config.neuroscience_qa_examples
            )
            
            # Combine all data
            all_training_data = []
            
            # Add consciousness data
            for item in consciousness_data:
                text = f"Human: {item['prompt']}\nAssistant: {item['response']}"
                all_training_data.append({"text": text, "type": "consciousness"})
            
            # Add brain simulation data
            for item in brain_simulation_data:
                text = f"Human: {item['prompt']}\nAssistant: {item['response']}"
                all_training_data.append({"text": text, "type": "brain_simulation"})
            
            # Add neuroscience Q&A data
            for item in neuroscience_qa_data:
                text = f"Human: {item['prompt']}\nAssistant: {item['response']}"
                all_training_data.append({"text": text, "type": "neuroscience_qa"})
            
            # Shuffle data
            np.random.shuffle(all_training_data)
            
            # Split into train/eval
            split_idx = int(0.9 * len(all_training_data))
            train_data = all_training_data[:split_idx]
            eval_data = all_training_data[split_idx:]
            
            # Create datasets
            self.training_data = DatasetDict({
                "train": Dataset.from_list(train_data),
                "eval": Dataset.from_list(eval_data)
            })
            
            logger.info(f"✅ Generated {len(train_data)} training examples and {len(eval_data)} eval examples")
            
            # Save training data
            data_path = Path(self.config.output_dir) / "training_data.json"
            with open(data_path, 'w') as f:
                json.dump({
                    "train": train_data,
                    "eval": eval_data,
                    "config": self.config.__dict__
                }, f, indent=2)
            
            logger.info(f"💾 Training data saved to: {data_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate training data: {e}")
            return False
    
    def tokenize_data(self) -> bool:
        """Tokenize training data"""
        if not self.training_data or not self.tokenizer:
            logger.error("❌ Training data or tokenizer not available")
            return False
        
        try:
            logger.info("🔄 Tokenizing training data...")
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=self.config.max_length
                )
            
            # Tokenize datasets
            self.training_data = self.training_data.map(
                tokenize_function,
                batched=True,
                remove_columns=["text", "type"]
            )
            
            logger.info("✅ Training data tokenized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to tokenize data: {e}")
            return False
    
    def train_model(self) -> bool:
        """Train the brain-specific model"""
        if not self.model or not self.training_data:
            logger.error("❌ Model or training data not prepared")
            return False
        
        try:
            logger.info("🚀 Starting brain-specific fine-tuning...")
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=100,
                save_strategy="steps",
                save_steps=200,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,  # Disable wandb
                dataloader_drop_last=True,
                fp16=True,  # Enable mixed precision
                push_to_hub=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
                pad_to_multiple_of=8
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.training_data["train"],
                eval_dataset=self.training_data["eval"],
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Start training
            trainer.train()
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info("✅ Training completed successfully!")
            
            # Save training config
            config_path = Path(self.config.output_dir) / "training_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(self.config.__dict__, f, default_flow_style=False)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False
    
    def test_fine_tuned_model(self) -> bool:
        """Test the fine-tuned model"""
        try:
            logger.info("🧪 Testing fine-tuned model...")
            
            # Test prompts
            test_prompts = [
                "Describe your current conscious experience with high awareness:",
                "Explain the neural dynamics of working memory in the prefrontal cortex:",
                "What is the role of the hippocampus in memory consolidation?",
                "How do thalamic oscillations modulate cortical activity?"
            ]
            
            for prompt in test_prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    f"Human: {prompt}\nAssistant:",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("Assistant:")[-1].strip()
                
                logger.info(f"🧠 Prompt: {prompt}")
                logger.info(f"🦙 Response: {response}")
                logger.info("-" * 80)
            
            logger.info("✅ Model testing completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")
            return False
    
    def run_full_training_pipeline(self) -> bool:
        """Run the complete training pipeline"""
        logger.info("🚀 Starting complete brain-specific Llama-2 training pipeline...")
        
        steps = [
            ("Preparing model and tokenizer", self.prepare_model_and_tokenizer),
            ("Generating training data", self.generate_training_data),
            ("Tokenizing data", self.tokenize_data),
            ("Training model", self.train_model),
            ("Testing fine-tuned model", self.test_fine_tuned_model)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"🔄 {step_name}...")
            if not step_func():
                logger.error(f"❌ Failed at: {step_name}")
                return False
        
        logger.info("🎉 Complete training pipeline finished successfully!")
        logger.info(f"📁 Model saved to: {self.config.output_dir}")
        
        return True

# Utility functions for easy training

def create_brain_training_config(**kwargs) -> BrainTrainingConfig:
    """Create brain training configuration with custom parameters"""
    return BrainTrainingConfig(**kwargs)

def train_llama2_for_brain_simulation(
    output_dir: str = "models/llama2-brain-finetuned",
    num_epochs: int = 3,
    **config_kwargs
) -> bool:
    """Train Llama-2 for brain simulation tasks"""
    config = create_brain_training_config(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        **config_kwargs
    )
    
    trainer = LlamaBrainTrainer(config)
    return trainer.run_full_training_pipeline()

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Llama-2 for brain simulation")
    parser.add_argument("--output_dir", default="models/llama2-brain-finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--consciousness_examples", type=int, default=800, help="Number of consciousness examples")
    parser.add_argument("--brain_examples", type=int, default=1000, help="Number of brain simulation examples")
    parser.add_argument("--qa_examples", type=int, default=1200, help="Number of neuroscience Q&A examples")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BrainTrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        consciousness_examples=args.consciousness_examples,
        brain_simulation_examples=args.brain_examples,
        neuroscience_qa_examples=args.qa_examples
    )
    
    # Run training
    trainer = LlamaBrainTrainer(config)
    success = trainer.run_full_training_pipeline()
    
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
