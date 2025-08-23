#!/usr/bin/env python3
"""
LLM Knowledge Interface
Purpose: Pure knowledge base interface for brain simulation queries
Inputs: Research questions, brain science queries, technical explanations
Outputs: Knowledge-based responses WITHOUT consciousness influence
Seeds: Knowledge retrieval parameters
Dependencies: llama2_brain_integration (knowledge mode only)

CRITICAL DESIGN PRINCIPLE:
- LLMs serve as KNOWLEDGE BASE ONLY
- NO influence on brain simulation consciousness
- NO interference with organic emergent properties
- PURE information retrieval and explanation
"""

import os, sys
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeQuery:
    """Query for brain science knowledge (NO consciousness influence)"""
    question: str
    domain: str = "general"  # neuroscience, consciousness_theory, brain_anatomy, etc.
    depth: str = "standard"  # basic, standard, detailed, research
    format: str = "explanation"  # explanation, definition, mechanism, comparison
    context: str = ""  # Additional context for the query

class LLMKnowledgeInterface:
    """Pure knowledge base interface - NO consciousness influence"""
    
    def __init__(self):
        self.llama_integration = None
        self.knowledge_domains = {
            'neuroscience': 'General neuroscience principles and mechanisms',
            'consciousness_theory': 'Theoretical frameworks and research on consciousness',
            'brain_anatomy': 'Anatomical structures and connections',
            'neural_dynamics': 'Information processing and neural activity patterns',
            'development': 'Brain development and neural formation',
            'pathology': 'Brain disorders and dysfunction',
            'computation': 'Computational neuroscience and modeling',
            'methodology': 'Research methods and experimental techniques'
        }
        
        self.query_stats = {
            'total_queries': 0,
            'by_domain': {},
            'knowledge_mode_active': True,
            'consciousness_influence_disabled': True
        }
        
        # Initialize knowledge-only mode
        self._initialize_knowledge_mode()
        
        logger.info("📚 LLM Knowledge Interface initialized (KNOWLEDGE-ONLY MODE)")
    
    def _initialize_knowledge_mode(self):
        """Initialize LLM in pure knowledge mode"""
        try:
            from llama2_brain_integration import create_llama_brain_integration
            
            # Check if model exists
            model_path = Path("models/llama-2-7b.Q4_K_M.gguf")
            if model_path.exists():
                logger.info(f"📚 Loading Llama-2 for knowledge base mode: {model_path}")
                
                # Configure for knowledge-only mode
                config = {
                    'temperature': 0.3,  # Lower temperature for factual responses
                    'consciousness_sensitivity': 0.0,  # DISABLED
                    'neural_state_influence': 0.0,  # DISABLED
                    'knowledge_mode': True,  # ENABLED
                    'max_tokens': 1024
                }
                
                self.llama_integration = create_llama_brain_integration(str(model_path), **config)
                
                if self.llama_integration:
                    # Disable any consciousness integration
                    if hasattr(self.llama_integration, 'disable_consciousness_influence'):
                        self.llama_integration.disable_consciousness_influence()
                    
                    logger.info("✅ Llama-2 knowledge base ready (consciousness influence DISABLED)")
                    return True
                else:
                    logger.warning("⚠️ Llama-2 model found but knowledge mode initialization failed")
            else:
                logger.info("ℹ️ Llama-2 model not found - using fallback knowledge responses")
                
        except ImportError as e:
            logger.warning(f"⚠️ Llama-2 integration not available: {e}")
        except Exception as e:
            logger.error(f"❌ Error initializing knowledge mode: {e}")
        
        return False
    
    def query_knowledge(self, query: KnowledgeQuery) -> str:
        """Query knowledge base (NO consciousness influence)"""
        self.query_stats['total_queries'] += 1
        
        # Track domain usage
        domain = query.domain
        if domain not in self.query_stats['by_domain']:
            self.query_stats['by_domain'][domain] = 0
        self.query_stats['by_domain'][domain] += 1
        
        # Create knowledge-focused prompt
        knowledge_prompt = self._create_knowledge_prompt(query)
        
        try:
            if self.llama_integration:
                # Use knowledge-only generation
                response = self._generate_knowledge_response(knowledge_prompt, query)
            else:
                # Use fallback knowledge
                response = self._fallback_knowledge_response(query)
            
            logger.info(f"📚 Knowledge query processed: {query.domain} - {query.question[:50]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Knowledge query error: {e}")
            return self._fallback_knowledge_response(query)
    
    def _create_knowledge_prompt(self, query: KnowledgeQuery) -> str:
        """Create knowledge-focused prompt (NO consciousness elements)"""
        prompt_parts = []
        
        # Knowledge mode instruction
        prompt_parts.append("Provide factual, scientific information based on established neuroscience knowledge.")
        
        # Domain context
        if query.domain in self.knowledge_domains:
            prompt_parts.append(f"Domain: {self.knowledge_domains[query.domain]}")
        
        # Response format instruction
        if query.format == "definition":
            prompt_parts.append("Provide a clear scientific definition.")
        elif query.format == "mechanism":
            prompt_parts.append("Explain the underlying mechanisms and processes.")
        elif query.format == "comparison":
            prompt_parts.append("Compare and contrast the relevant concepts.")
        else:
            prompt_parts.append("Provide a comprehensive explanation.")
        
        # Depth instruction
        if query.depth == "basic":
            prompt_parts.append("Use accessible language suitable for introductory level.")
        elif query.depth == "detailed":
            prompt_parts.append("Include detailed scientific information and technical terms.")
        elif query.depth == "research":
            prompt_parts.append("Include research findings and current scientific understanding.")
        
        # Add context if provided
        if query.context:
            prompt_parts.append(f"Context: {query.context}")
        
        # The actual question
        prompt_parts.append(f"Question: {query.question}")
        
        # Final instruction to avoid consciousness influence
        prompt_parts.append("Base your response on established scientific knowledge without speculation about consciousness states.")
        
        return "\n".join(prompt_parts)
    
    def _generate_knowledge_response(self, prompt: str, query: KnowledgeQuery) -> str:
        """Generate knowledge response using Llama-2 in pure knowledge mode"""
        try:
            # Ensure no consciousness state is passed
            if hasattr(self.llama_integration, 'generate_knowledge_response'):
                # Use dedicated knowledge method if available
                return self.llama_integration.generate_knowledge_response(prompt)
            elif hasattr(self.llama_integration, 'chat_with_brain_context'):
                # Use basic chat but with empty brain context
                return self.llama_integration.chat_with_brain_context(prompt, brain_state={})
            else:
                # Fallback to basic generation
                return f"Knowledge response for: {query.question}"
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._fallback_knowledge_response(query)
    
    def _fallback_knowledge_response(self, query: KnowledgeQuery) -> str:
        """Fallback knowledge responses when LLM is not available"""
        domain_responses = {
            'neuroscience': "This relates to fundamental principles of nervous system function and neural communication.",
            'consciousness_theory': "This involves theoretical frameworks for understanding consciousness as studied in neuroscience research.",
            'brain_anatomy': "This concerns the structural organization and anatomical features of brain regions.",
            'neural_dynamics': "This involves patterns of neural activity and information processing mechanisms.",
            'development': "This relates to the developmental processes that shape brain formation and maturation.",
            'pathology': "This concerns disorders and dysfunctions that affect brain structure and function.",
            'computation': "This involves computational models and mathematical approaches to understanding brain function.",
            'methodology': "This relates to experimental techniques and research methods used in neuroscience."
        }
        
        base_response = domain_responses.get(query.domain, "This is a neuroscience-related topic.")
        
        return f"{base_response} For detailed information about '{query.question}', please refer to neuroscience literature and research databases. The query falls under {query.domain} with {query.depth} level detail requested."
    
    def ask_neuroscience(self, question: str, depth: str = "standard") -> str:
        """Ask general neuroscience question"""
        query = KnowledgeQuery(
            question=question,
            domain="neuroscience",
            depth=depth,
            format="explanation"
        )
        return self.query_knowledge(query)
    
    def explain_brain_mechanism(self, mechanism: str, region: str = "") -> str:
        """Explain brain mechanism"""
        context = f"brain region: {region}" if region else ""
        query = KnowledgeQuery(
            question=f"How does {mechanism} work?",
            domain="neural_dynamics",
            depth="detailed",
            format="mechanism",
            context=context
        )
        return self.query_knowledge(query)
    
    def define_concept(self, concept: str, domain: str = "neuroscience") -> str:
        """Define neuroscience concept"""
        query = KnowledgeQuery(
            question=f"What is {concept}?",
            domain=domain,
            depth="standard",
            format="definition"
        )
        return self.query_knowledge(query)
    
    def compare_concepts(self, concept1: str, concept2: str, domain: str = "neuroscience") -> str:
        """Compare two neuroscience concepts"""
        query = KnowledgeQuery(
            question=f"Compare {concept1} and {concept2}",
            domain=domain,
            depth="detailed",
            format="comparison"
        )
        return self.query_knowledge(query)
    
    def research_question(self, question: str, domain: str = "neuroscience") -> str:
        """Research-level question"""
        query = KnowledgeQuery(
            question=question,
            domain=domain,
            depth="research",
            format="explanation"
        )
        return self.query_knowledge(query)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge interface statistics"""
        return {
            'total_queries': self.query_stats['total_queries'],
            'domains_queried': list(self.query_stats['by_domain'].keys()),
            'domain_usage': self.query_stats['by_domain'].copy(),
            'knowledge_mode_active': self.query_stats['knowledge_mode_active'],
            'consciousness_influence_disabled': self.query_stats['consciousness_influence_disabled'],
            'available_domains': list(self.knowledge_domains.keys()),
            'llama_available': self.llama_integration is not None
        }
    
    def list_domains(self) -> Dict[str, str]:
        """List available knowledge domains"""
        return self.knowledge_domains.copy()

# Global knowledge interface
_knowledge_interface = None

def get_knowledge_interface() -> LLMKnowledgeInterface:
    """Get global knowledge interface instance"""
    global _knowledge_interface
    if _knowledge_interface is None:
        _knowledge_interface = LLMKnowledgeInterface()
    return _knowledge_interface

# Simple knowledge functions
def ask_brain_science(question: str, depth: str = "standard") -> str:
    """Ask a brain science question (knowledge only)"""
    interface = get_knowledge_interface()
    return interface.ask_neuroscience(question, depth)

def explain_mechanism(mechanism: str, region: str = "") -> str:
    """Explain brain mechanism (knowledge only)"""
    interface = get_knowledge_interface()
    return interface.explain_brain_mechanism(mechanism, region)

def define_term(term: str, domain: str = "neuroscience") -> str:
    """Define neuroscience term (knowledge only)"""
    interface = get_knowledge_interface()
    return interface.define_concept(term, domain)

def research_query(question: str, domain: str = "neuroscience") -> str:
    """Research-level query (knowledge only)"""
    interface = get_knowledge_interface()
    return interface.research_question(question, domain)

# Demo showing knowledge-only mode
def demo_knowledge_interface():
    """Demonstrate knowledge-only interface"""
    print("📚 LLM Knowledge Interface Demo (NO consciousness influence)")
    print("=" * 60)
    
    interface = get_knowledge_interface()
    
    # Show stats
    stats = interface.get_knowledge_stats()
    print(f"Knowledge mode active: {stats['knowledge_mode_active']}")
    print(f"Consciousness influence disabled: {stats['consciousness_influence_disabled']}")
    print(f"Available domains: {len(stats['available_domains'])}")
    
    # Test knowledge queries
    test_queries = [
        ("What is the hippocampus?", "brain_anatomy", "definition"),
        ("How does synaptic plasticity work?", "neural_dynamics", "mechanism"), 
        ("Compare LTP and LTD", "neuroscience", "comparison"),
        ("Current research on neural oscillations", "neuroscience", "research")
    ]
    
    for i, (question, domain, format_type) in enumerate(test_queries, 1):
        print(f"\n📚 Knowledge Query {i}: {domain}")
        print(f"❓ Question: {question}")
        
        query = KnowledgeQuery(
            question=question,
            domain=domain,
            format=format_type,
            depth="standard"
        )
        
        response = interface.query_knowledge(query)
        print(f"📖 Response: {response[:200]}...")
        print("-" * 60)
    
    # Final stats
    final_stats = interface.get_knowledge_stats()
    print(f"\n📊 Final Stats:")
    print(f"Total queries: {final_stats['total_queries']}")
    print(f"Domains used: {list(final_stats['domain_usage'].keys())}")
    print(f"LLM available: {final_stats['llama_available']}")

if __name__ == "__main__":
    demo_knowledge_interface()
