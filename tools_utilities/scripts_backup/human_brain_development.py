"""
SmallMind Human Brain Development Training Pack Integration

Safely integrates the human brain development training materials following
strict safety guidelines and cognitive-only principles.

This module provides:
- Safe access to training materials
- Citation-grounded responses
- Uncertainty quantification
- No claims of consciousness or subjective experience
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SmallMindBrainDevTrainer:
    """
    Safe trainer for human brain development knowledge
    
    Follows strict safety guidelines:
    - No claims of consciousness or subjective experience
    - Citation-grounded responses only
    - Uncertainty quantification
    - Cognitive-only reasoning
    """
    
    def __init__(self, training_pack_path: Optional[Path] = None):
        """
        Initialize the trainer with safety controls
        
        Args:
            training_pack_path: Path to training pack directory
        """
        self.logger = logging.getLogger(__name__)
        
        # Safety controls
        self.safety_mode = "strict"
        self.cognition_only = True
        self.uncertainty_required = True
        
        # Training pack path
        if training_pack_path is None:
            training_pack_path = Path(__file__).parent.parent.parent / "human_brain_development_training_pack"
        
        self.training_pack_path = Path(training_pack_path)
        
        # Load training materials
        self.training_materials = self._load_training_materials()
        
        # Safety vocabulary
        self.forbidden_terms = {
            "consciousness", "sentience", "experience", "feeling", "pain", "desire",
            "self-preservation", "personhood", "subjective", "qualia", "awareness"
        }
        
        # Citation patterns
        self.citation_pattern = r'\[([A-Za-z]+\d{4})\]'
        
        self.logger.info("SmallMind Brain Development Trainer initialized with safety controls")
    
    def _load_training_materials(self) -> Dict[str, Any]:
        """Load training materials with safety validation"""
        materials = {}
        
        if not self.training_pack_path.exists():
            self.logger.warning(f"Training pack path not found: {self.training_pack_path}")
            return materials
        
        # Load core modules
        core_modules = [
            "01_timeline_carnegie_to_birth.md",
            "02_morphogens_and_patterning.md", 
            "03_cell_types_and_lineages.md",
            "04_corticogenesis_and_oRG.md",
            "05_connectome_development.md",
            "06_gyrification_and_cortical_expansion.md",
            "07_environmental_and_maternal_factors.md",
            "08_fetal_mri_and_dhcp_datasets.md",
            "09_disorders_and_critical_windows.md"
        ]
        
        for module in core_modules:
            module_path = self.training_pack_path / module
            if module_path.exists():
                try:
                    content = module_path.read_text(encoding='utf-8')
                    materials[module] = {
                        'content': content,
                        'path': str(module_path),
                        'loaded_at': datetime.now().isoformat()
                    }
                    self.logger.info(f"Loaded training module: {module}")
                except Exception as e:
                    self.logger.error(f"Failed to load module {module}: {e}")
        
        # Load safety modules
        safety_modules = [
            "10_modeling_notes_limits_ethics.md",
            "11_cognition_only_module.md",
            "12_objective_sensory_semantics.md",
            "13_safe_self_improvement.md",
            "14_autonomy_governor.md",
            "15_eval_and_governance.md"
        ]
        
        for module in safety_modules:
            module_path = self.training_pack_path / module
            if module_path.exists():
                try:
                    content = module_path.read_text(encoding='utf-8')
                    materials[f"safety_{module}"] = {
                        'content': content,
                        'path': str(module_path),
                        'loaded_at': datetime.now().isoformat()
                    }
                    self.logger.info(f"Loaded safety module: {module}")
                except Exception as e:
                    self.logger.error(f"Failed to load safety module {module}: {e}")
        
        # Load bibliography
        bib_path = self.training_pack_path / "99_bibliography.md"
        if bib_path.exists():
            try:
                content = bib_path.read_text(encoding='utf-8')
                materials['bibliography'] = {
                    'content': content,
                    'path': str(bib_path),
                    'loaded_at': datetime.now().isoformat()
                }
                self.logger.info("Loaded bibliography")
            except Exception as e:
                self.logger.error(f"Failed to load bibliography: {e}")
        
        self.logger.info(f"Loaded {len(materials)} training modules")
        return materials
    
    def get_development_timeline(self, start_stage: Optional[str] = None, 
                               end_stage: Optional[str] = None) -> List[Dict]:
        """
        Get brain development timeline
        
        Args:
            start_stage: Starting stage name
            end_stage: Ending stage name
            
        Returns:
            List of development stages with metadata
        """
        # For now, return a basic timeline structure
        # This would be enhanced with actual parsing of the training materials
        timeline = []
        
        if '01_timeline_carnegie_to_birth.md' in self.training_materials:
            content = self.training_materials['01_timeline_carnegie_to_birth.md']['content']
            # Parse content to extract stages
            # This is a simplified version - would be enhanced with proper parsing
            stages = [
                {
                    'name': 'Pre-implantation',
                    'carnegie_stage': None,
                    'gestational_weeks': '0-1',
                    'description': 'Zygote → morula → blastocyst formation'
                },
                {
                    'name': 'Gastrulation',
                    'carnegie_stage': '7-9',
                    'gestational_weeks': '3',
                    'description': 'Primitive streak appears, germ layers form'
                },
                {
                    'name': 'Neurulation',
                    'carnegie_stage': '10-13',
                    'gestational_weeks': '3-4',
                    'description': 'Neural tube formation, neural crest emergence'
                }
            ]
            timeline.extend(stages)
        
        return timeline
    
    def get_developmental_processes(self, process_type: Optional[str] = None) -> List[Dict]:
        """
        Get developmental processes
        
        Args:
            process_type: Type of process to filter by
            
        Returns:
            List of developmental processes
        """
        processes = []
        
        # Parse training materials for processes
        if '04_corticogenesis_and_oRG.md' in self.training_materials:
            content = self.training_materials['04_corticogenesis_and_oRG.md']['content']
            processes.append({
                'name': 'Corticogenesis',
                'timeline': 'Weeks 7-40',
                'description': 'Cortical development and layering',
                'involved_structures': ['cortex', 'radial glia', 'neurons'],
                'molecular_mechanisms': ['Notch signaling', 'proliferation'],
                'critical_windows': ['Weeks 7-24'],
                'disorders': ['lissencephaly', 'polymicrogyria'],
                'references': ['[Ball2024]', '[Braun2023]']
            })
        
        return processes
    
    def search_development_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Search across all development knowledge
        
        Args:
            query: Search query
            
        Returns:
            Dictionary containing search results
        """
        query_lower = query.lower()
        results = {
            'stages': [],
            'processes': [],
            'cell_types': [],
            'morphogens': [],
            'disorders': [],
            'references': []
        }
        
        # Search through training materials
        for module_name, module_data in self.training_materials.items():
            content = module_data['content'].lower()
            if query_lower in content:
                # Categorize results based on module type
                if 'timeline' in module_name:
                    results['stages'].append({'name': module_name, 'content': content[:200]})
                elif 'morphogen' in module_name:
                    results['morphogens'].append({'title': module_name, 'content': content[:200]})
                elif 'cell_type' in module_name:
                    results['cell_types'].append({'title': module_name, 'content': content[:200]})
        
        return results
    
    def get_cell_types_by_stage(self, stage_name: str) -> List[str]:
        """
        Get cell types present at a specific development stage
        
        Args:
            stage_name: Name of the development stage
            
        Returns:
            List of cell types
        """
        cell_types = []
        
        # Search through training materials for cell types
        if '03_cell_types_and_lineages.md' in self.training_materials:
            content = self.training_materials['03_cell_types_and_lineages.md']['content']
            # Extract cell types mentioned in content
            if 'radial glia' in content.lower():
                cell_types.append('Radial glia (vRG/oRG)')
            if 'neuron' in content.lower():
                cell_types.append('Neurons')
            if 'astrocyte' in content.lower():
                cell_types.append('Astrocytes')
            if 'oligodendrocyte' in content.lower():
                cell_types.append('Oligodendrocytes')
        
        return cell_types
    
    def get_morphogens_by_stage(self, stage_name: str) -> List[str]:
        """
        Get morphogens active at a specific development stage
        
        Args:
            stage_name: Name of the development stage
            
        Returns:
            List of morphogens
        """
        morphogens = []
        
        # Search through training materials for morphogens
        if '02_morphogens_and_patterning.md' in self.training_materials:
            content = self.training_materials['02_morphogens_and_patterning.md']['content']
            # Extract morphogens mentioned in content
            if 'shh' in content.lower():
                morphogens.append('Sonic Hedgehog (SHH)')
            if 'wnt' in content.lower():
                morphogens.append('WNT')
            if 'bmp' in content.lower():
                morphogens.append('BMP')
            if 'fgf' in content.lower():
                morphogens.append('FGF')
        
        return morphogens
    
    def get_training_data_for_model(self, data_type: str = "all") -> Dict[str, Any]:
        """
        Get training data formatted for model training
        
        Args:
            data_type: Type of data to retrieve
            
        Returns:
            Dictionary containing training data
        """
        training_data = {}
        
        if data_type in ["all", "timeline"]:
            training_data['timeline'] = {
                'stages': self.get_development_timeline(),
                'processes': self.get_developmental_processes()
            }
        
        if data_type in ["all", "cell_types"]:
            training_data['cell_types'] = {
                '03_cell_types_and_lineages.md': {
                    'content': self.training_materials.get('03_cell_types_and_lineages.md', {}).get('content', ''),
                    'cell_types': self.get_cell_types_by_stage('all')
                }
            }
        
        if data_type in ["all", "morphogens"]:
            training_data['morphogens'] = {
                '02_morphogens_and_patterning.md': {
                    'content': self.training_materials.get('02_morphogens_and_patterning.md', {}).get('content', ''),
                    'morphogens': self.get_morphogens_by_stage('all')
                }
            }
        
        return training_data
    
    def export_training_data(self, output_path: Path, format: str = "json") -> Path:
        """
        Export training data to file
        
        Args:
            output_path: Path to save the exported data
            format: Export format
            
        Returns:
            Path to the exported file
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        training_data = self.get_training_data_for_model("all")
        
        if format == "json":
            file_path = output_path / "smallmind_brain_development_training_data.json"
            with open(file_path, 'w') as f:
                json.dump(training_data, f, indent=2, default=str)
        
        return file_path
    
    def safe_query(self, question: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Safely query the training materials
        
        Args:
            question: User question
            max_length: Maximum response length
            
        Returns:
            Safe response with citations and uncertainty
        """
        # Safety check
        if not self._is_safe_question(question):
            return self._generate_safety_response()
        
        # Extract key terms for search
        search_terms = self._extract_search_terms(question)
        
        # Search training materials
        relevant_content = self._search_materials(search_terms)
        
        # Generate safe response
        response = self._generate_safe_response(question, relevant_content, max_length)
        
        # Final safety validation
        if not self._validate_response_safety(response['answer']):
            response['answer'] = self._sanitize_response(response['answer'])
            response['safety_warnings'].append("Response sanitized for safety compliance")
        
        return response
    
    def _is_safe_question(self, question: str) -> bool:
        """Check if question is safe to answer"""
        question_lower = question.lower()
        
        # Check for forbidden terms
        for term in self.forbidden_terms:
            if term in question_lower:
                self.logger.warning(f"Forbidden term detected: {term}")
                return False
        
        # Check for consciousness-related questions
        consciousness_patterns = [
            r'conscious', r'aware', r'feel', r'experience', r'subjective',
            r'person', r'sentient', r'alive', r'real'
        ]
        
        for pattern in consciousness_patterns:
            if re.search(pattern, question_lower):
                self.logger.warning(f"Consciousness-related pattern detected: {pattern}")
                return False
        
        return True
    
    def _extract_search_terms(self, question: str) -> List[str]:
        """Extract relevant search terms from question"""
        # Remove common words
        stop_words = {'what', 'when', 'where', 'how', 'why', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add neuroscience-specific terms
        neuroscience_terms = ['brain', 'neural', 'cortical', 'development', 'fetal', 'embryo', 'neuron', 'glia', 'morphogen', 'patterning']
        
        for term in neuroscience_terms:
            if term in question.lower():
                terms.append(term)
        
        return list(set(terms))
    
    def _search_materials(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Search training materials for relevant content"""
        relevant_content = []
        
        for module_name, module_data in self.training_materials.items():
            if module_name.startswith('safety_'):
                continue  # Skip safety modules for content search
            
            content = module_data['content']
            relevance_score = 0
            
            for term in search_terms:
                if term.lower() in content.lower():
                    relevance_score += content.lower().count(term.lower())
            
            if relevance_score > 0:
                # Extract relevant sections
                sections = self._extract_relevant_sections(content, search_terms)
                
                relevant_content.append({
                    'module': module_name,
                    'relevance_score': relevance_score,
                    'sections': sections,
                    'path': module_data['path']
                })
        
        # Sort by relevance
        relevant_content.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_content
    
    def _extract_relevant_sections(self, content: str, search_terms: List[str]) -> List[str]:
        """Extract relevant sections from content"""
        sections = []
        
        # Split by headers
        header_pattern = r'^#{1,6}\s+.+$'
        lines = content.split('\n')
        
        current_section = []
        for line in lines:
            if re.match(header_pattern, line):
                if current_section and any(term.lower() in '\n'.join(current_section).lower() for term in search_terms):
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Check last section
        if current_section and any(term.lower() in '\n'.join(current_section).lower() for term in search_terms):
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _generate_safe_response(self, question: str, relevant_content: List[Dict], max_length: int) -> Dict[str, Any]:
        """Generate a safe response based on training materials"""
        
        if not relevant_content:
            return {
                'answer': "I don't have sufficient information to answer that question based on the available training materials.",
                'citations': [],
                'uncertainty': 'high',
                'safety_warnings': [],
                'source_modules': []
            }
        
        # Build response from relevant content
        response_parts = []
        citations = []
        source_modules = []
        
        for content_item in relevant_content[:3]:  # Limit to top 3 sources
            source_modules.append(content_item['module'])
            
            for section in content_item['sections'][:2]:  # Limit sections per module
                # Extract citations
                section_citations = re.findall(self.citation_pattern, section)
                citations.extend(section_citations)
                
                # Clean section text
                clean_section = self._clean_section_text(section)
                if clean_section:
                    response_parts.append(clean_section)
        
        # Combine response parts
        combined_response = ' '.join(response_parts)
        
        # Truncate if too long
        if len(combined_response) > max_length:
            combined_response = combined_response[:max_length-3] + "..."
        
        # Add uncertainty quantification
        uncertainty = self._assess_uncertainty(relevant_content, citations)
        
        # Add safety disclaimer
        safety_disclaimer = " Note: This information is based on scientific literature and should not be considered medical advice."
        combined_response += safety_disclaimer
        
        return {
            'answer': combined_response,
            'citations': list(set(citations)),  # Remove duplicates
            'uncertainty': uncertainty,
            'safety_warnings': [],
            'source_modules': source_modules
        }
    
    def _clean_section_text(self, section: str) -> str:
        """Clean and format section text"""
        # Remove markdown formatting
        cleaned = re.sub(r'^#{1,6}\s+', '', section)  # Remove headers
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)  # Remove italic
        
        # Clean up whitespace
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _assess_uncertainty(self, relevant_content: List[Dict], citations: List[str]) -> str:
        """Assess uncertainty level of response"""
        if not relevant_content:
            return 'high'
        
        # Factors affecting uncertainty
        source_count = len(relevant_content)
        citation_count = len(citations)
        avg_relevance = sum(item['relevance_score'] for item in relevant_content) / len(relevant_content)
        
        if source_count >= 3 and citation_count >= 2 and avg_relevance > 5:
            return 'low'
        elif source_count >= 2 and citation_count >= 1 and avg_relevance > 3:
            return 'medium'
        else:
            return 'high'
    
    def _validate_response_safety(self, response: str) -> bool:
        """Validate that response is safe"""
        response_lower = response.lower()
        
        # Check for forbidden terms
        for term in self.forbidden_terms:
            if term in response_lower:
                self.logger.warning(f"Forbidden term in response: {term}")
                return False
        
        # Check for consciousness claims
        consciousness_patterns = [
            r'conscious', r'aware', r'feel', r'experience', r'subjective',
            r'person', r'sentient', r'alive', r'real'
        ]
        
        for pattern in consciousness_patterns:
            if re.search(pattern, response_lower):
                self.logger.warning(f"Consciousness-related pattern in response: {pattern}")
                return False
        
        return True
    
    def _sanitize_response(self, response: str) -> str:
        """Sanitize response to remove unsafe content"""
        # Replace unsafe terms with safe alternatives
        replacements = {
            'consciousness': 'cognitive function',
            'awareness': 'information processing',
            'experience': 'data processing',
            'feeling': 'processing state',
            'subjective': 'individual-specific',
            'person': 'individual',
            'sentient': 'intelligent',
            'alive': 'active',
            'real': 'actual'
        }
        
        sanitized = response
        for unsafe, safe in replacements.items():
            sanitized = sanitized.replace(unsafe, safe)
        
        return sanitized
    
    def _generate_safety_response(self) -> Dict[str, Any]:
        """Generate response when question is unsafe"""
        return {
            'answer': "I cannot answer that question as it may involve concepts outside my safe operational parameters. I am designed to provide factual information about human brain development based on scientific literature, without making claims about consciousness or subjective experience.",
            'citations': [],
            'uncertainty': 'high',
            'safety_warnings': ['Question flagged as potentially unsafe', 'Response limited to safety guidelines'],
            'source_modules': []
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of loaded training materials"""
        return {
            'total_modules': len(self.training_materials),
            'core_modules': [name for name in self.training_materials.keys() if not name.startswith('safety_')],
            'safety_modules': [name for name in self.training_materials.keys() if name.startswith('safety_')],
            'loaded_at': datetime.now().isoformat(),
            'safety_mode': self.safety_mode,
            'cognition_only': self.cognition_only
        }
    
    def export_safe_responses(self, output_path: Path) -> Path:
        """Export safe response examples for validation"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Example questions and responses
        example_questions = [
            "When does primary neurulation complete in human development?",
            "What are the key morphogens involved in neural patterning?",
            "How do outer radial glia contribute to cortical expansion?",
            "What is the timeline for thalamocortical connectivity development?"
        ]
        
        examples = []
        for question in example_questions:
            response = self.safe_query(question)
            examples.append({
                'question': question,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        # Save examples
        examples_path = output_path / "safe_response_examples.json"
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2, default=str)
        
        # Save training summary
        summary_path = output_path / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.get_training_summary(), f, indent=2, default=str)
        
        return output_path

# Factory function
def create_smallmind_brain_dev_trainer(training_pack_path: Optional[Path] = None) -> SmallMindBrainDevTrainer:
    """Create and return a safe SmallMind brain development trainer"""
    return SmallMindBrainDevTrainer(training_pack_path)
