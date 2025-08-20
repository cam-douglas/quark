#!/usr/bin/env python3
"""
Knowledge Base Interface
Purpose: Pure knowledge retrieval from LLM - NO consciousness influence
Inputs: Research questions, scientific queries
Outputs: Factual knowledge responses
Dependencies: llama2_brain_integration (knowledge mode only)

DESIGN PRINCIPLE:
- LLM serves as KNOWLEDGE BASE ONLY
- NO influence on organic brain simulation
- NO interference with emergent consciousness properties
- PURE information retrieval
"""

import os, sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Pure knowledge interface - NO consciousness influence"""
    
    def __init__(self):
        self.llama_model = None
        self.knowledge_stats = {
            'queries': 0,
            'available': False
        }
        
        self._initialize_knowledge_base()
        logger.info("📚 Knowledge Base initialized (KNOWLEDGE-ONLY MODE)")
    
    def _initialize_knowledge_base(self):
        """Initialize Llama-2 for pure knowledge retrieval"""
        try:
            # Import Llama directly for knowledge-only usage
            from llama_cpp import Llama
            
            model_path = Path("models/llama-2-7b.Q4_K_M.gguf")
            if model_path.exists():
                logger.info(f"📚 Loading knowledge model: {model_path}")
                
                self.llama_model = Llama(
                    model_path=str(model_path),
                    n_ctx=4096,
                    temperature=0.3,  # Low temperature for factual responses
                    verbose=False
                )
                
                self.knowledge_stats['available'] = True
                logger.info("✅ Knowledge base ready")
                
            else:
                logger.info("ℹ️ Knowledge model not found - using fallback responses")
                
        except ImportError:
            logger.warning("⚠️ llama-cpp-python not available")
        except Exception as e:
            logger.error(f"❌ Error loading knowledge base: {e}")
    
    def ask(self, question: str, max_tokens: int = 512) -> str:
        """Ask knowledge base a question (NO consciousness influence)"""
        self.knowledge_stats['queries'] += 1
        
        if not self.llama_model:
            return self._fallback_response(question)
        
        try:
            # Create knowledge-focused prompt
            knowledge_prompt = f"Answer this scientific question with factual information: {question}"
            
            # Generate factual response
            response = self.llama_model.create_completion(
                knowledge_prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                stop=["Question:", "Human:", "Assistant:"]
            )
            
            result = response['choices'][0]['text'].strip()
            logger.info(f"📚 Knowledge query answered: {question[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Knowledge query error: {e}")
            return self._fallback_response(question)
    
    def _fallback_response(self, question: str) -> str:
        """Fallback when LLM not available"""
        return f"Knowledge base unavailable. Question: {question}. Please refer to neuroscience literature."
    
    def explain(self, concept: str) -> str:
        """Explain a neuroscience concept"""
        return self.ask(f"Explain the neuroscience concept: {concept}")
    
    def define(self, term: str) -> str:
        """Define a scientific term"""
        return self.ask(f"Define: {term}")
    
    def compare(self, concept1: str, concept2: str) -> str:
        """Compare two concepts"""
        return self.ask(f"Compare {concept1} and {concept2}")
    
    def research(self, topic: str) -> str:
        """Get research information"""
        return self.ask(f"What does current research say about {topic}?")
    
    def status(self) -> Dict[str, Any]:
        """Get knowledge base status"""
        return {
            'available': self.knowledge_stats['available'],
            'queries_processed': self.knowledge_stats['queries'],
            'mode': 'knowledge_only',
            'consciousness_influence': False
        }

# Global knowledge base instance
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBase:
    """Get global knowledge base"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base

# Simple knowledge functions
def ask_knowledge(question: str) -> str:
    """Ask knowledge base (NO consciousness influence)"""
    kb = get_knowledge_base()
    return kb.ask(question)

def explain_concept(concept: str) -> str:
    """Explain concept (knowledge only)"""
    kb = get_knowledge_base()
    return kb.explain(concept)

def define_term(term: str) -> str:
    """Define term (knowledge only)"""
    kb = get_knowledge_base()
    return kb.define(term)

def compare_concepts(concept1: str, concept2: str) -> str:
    """Compare concepts (knowledge only)"""
    kb = get_knowledge_base()
    return kb.compare(concept1, concept2)

def research_topic(topic: str) -> str:
    """Research topic (knowledge only)"""
    kb = get_knowledge_base()
    return kb.research(topic)

def knowledge_status() -> Dict[str, Any]:
    """Get knowledge base status"""
    kb = get_knowledge_base()
    return kb.status()

# Demo
def demo():
    print("📚 Pure Knowledge Base Demo")
    print("=" * 40)
    print("🚫 NO consciousness influence")
    print("🧠 Your organic brain simulation remains untouched")
    print()
    
    kb = get_knowledge_base()
    status = kb.status()
    
    print(f"Knowledge base available: {status['available']}")
    print(f"Consciousness influence: {status['consciousness_influence']}")
    print()
    
    # Test knowledge queries
    questions = [
        "What is the hippocampus?",
        "How does synaptic plasticity work?",
        "Compare LTP and LTD"
    ]
    
    for question in questions:
        print(f"❓ {question}")
        answer = kb.ask(question)
        print(f"📖 {answer[:100]}...")
        print()
    
    final_status = kb.status()
    print(f"📊 Queries processed: {final_status['queries_processed']}")

if __name__ == "__main__":
    demo()
