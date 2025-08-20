"""
Question Generator - Generates questions to drive curiosity-based exploration
"""

from typing import List, Dict, Any, Optional
import random
import re

class QuestionGenerator:
    """
    Generates questions about observations to drive exploration and learning.
    
    Uses templates, context analysis, and pattern recognition to create
    meaningful questions that can guide learning.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Question templates organized by type
        self.question_templates = {
            'causal': [
                "What causes {} to happen?",
                "Why does {} occur in this context?",
                "What are the underlying reasons for {}?",
                "How does {} lead to other effects?",
                "What would prevent {} from happening?"
            ],
            'functional': [
                "How does {} work?",
                "What is the purpose of {}?",
                "How can {} be used effectively?",
                "What are the mechanisms behind {}?",
                "How can {} be improved?"
            ],
            'relational': [
                "How is {} related to {}?",
                "What connections exist between {} and other concepts?",
                "How does {} affect {}?",
                "What similarities exist between {} and {}?",
                "How do {} and {} interact?"
            ],
            'comparative': [
                "How does {} compare to alternatives?",
                "What makes {} different from {}?",
                "What are the advantages and disadvantages of {}?",
                "When is {} better than {}?",
                "What are the trade-offs with {}?"
            ],
            'exploratory': [
                "What would happen if {} were changed?",
                "What are the boundaries of {}?",
                "What unexplored aspects of {} exist?",
                "How might {} evolve over time?",
                "What patterns exist in {}?"
            ],
            'meta': [
                "How do we know that {} is true?",
                "What evidence supports {}?",
                "How reliable is our understanding of {}?",
                "What assumptions are we making about {}?",
                "How can our knowledge of {} be validated?"
            ]
        }
        
        # Context-specific question generators
        self.context_generators = {
            'ml_model': self._generate_ml_questions,
            'code': self._generate_code_questions,
            'data': self._generate_data_questions,
            'system': self._generate_system_questions,
            'experiment': self._generate_experiment_questions
        }
        
    def generate_questions(self, 
                          observation: Any,
                          context: Optional[Dict] = None,
                          question_types: Optional[List[str]] = None,
                          num_questions: int = 5) -> List[str]:
        """
        Generate questions about an observation.
        
        Args:
            observation: The thing to ask questions about
            context: Additional context information
            question_types: Types of questions to generate
            num_questions: Maximum number of questions to generate
            
        Returns:
            List of generated questions
        """
        context = context or {}
        question_types = question_types or ['causal', 'functional', 'exploratory']
        
        questions = []
        observation_str = self._extract_observation_string(observation)
        
        # Generate template-based questions
        template_questions = self._generate_template_questions(
            observation_str, question_types, num_questions // 2
        )
        questions.extend(template_questions)
        
        # Generate context-specific questions
        context_questions = self._generate_context_questions(
            observation, context, num_questions - len(questions)
        )
        questions.extend(context_questions)
        
        # Clean and filter questions
        clean_questions = self._clean_questions(questions)
        
        return clean_questions[:num_questions]
        
    def generate_follow_up_questions(self,
                                   original_question: str,
                                   answer: str,
                                   num_follow_ups: int = 3) -> List[str]:
        """Generate follow-up questions based on an answer."""
        follow_ups = []
        
        # Analyze the answer for new concepts
        concepts = self._extract_concepts(answer)
        
        # Generate questions about new concepts mentioned
        for concept in concepts[:2]:
            follow_ups.extend([
                f"Can you elaborate on {concept}?",
                f"How does {concept} relate to the original question?",
                f"What are the implications of {concept}?"
            ])
            
        # Generate deeper questions
        follow_ups.extend([
            f"What evidence supports this answer?",
            f"Are there alternative explanations?",
            f"How does this answer connect to broader patterns?",
            f"What new questions does this answer raise?"
        ])
        
        return follow_ups[:num_follow_ups]
        
    def _extract_observation_string(self, observation: Any) -> str:
        """Extract a string representation of the observation."""
        if isinstance(observation, str):
            return observation
        elif isinstance(observation, dict):
            # Extract key information from dict
            if 'description' in observation:
                return observation['description']
            elif 'content' in observation:
                return str(observation['content'])
            else:
                return str(observation)
        else:
            return str(observation)
            
    def _generate_template_questions(self,
                                   observation_str: str,
                                   question_types: List[str],
                                   num_questions: int) -> List[str]:
        """Generate questions using templates."""
        questions = []
        
        # Prepare observation for templates
        clean_observation = self._clean_observation_for_template(observation_str)
        
        for question_type in question_types:
            if question_type in self.question_templates:
                templates = self.question_templates[question_type]
                
                # Select a few templates randomly
                selected_templates = random.sample(
                    templates, 
                    min(len(templates), max(1, num_questions // len(question_types)))
                )
                
                for template in selected_templates:
                    try:
                        if template.count('{}') == 1:
                            question = template.format(clean_observation)
                        elif template.count('{}') == 2:
                            # For templates with two placeholders
                            related_concept = self._find_related_concept(clean_observation)
                            question = template.format(clean_observation, related_concept)
                        else:
                            # Skip templates with more complex formatting
                            continue
                            
                        questions.append(question)
                        
                    except (IndexError, KeyError):
                        # Skip malformed templates
                        continue
                        
        return questions
        
    def _generate_context_questions(self,
                                  observation: Any,
                                  context: Dict,
                                  num_questions: int) -> List[str]:
        """Generate context-specific questions."""
        questions = []
        
        # Determine context type
        context_type = self._determine_context_type(observation, context)
        
        if context_type in self.context_generators:
            generator = self.context_generators[context_type]
            context_questions = generator(observation, context, num_questions)
            questions.extend(context_questions)
        else:
            # Generate generic questions
            questions.extend(self._generate_generic_questions(observation, num_questions))
            
        return questions
        
    def _determine_context_type(self, observation: Any, context: Dict) -> str:
        """Determine the type of context for specialized question generation."""
        
        # Check explicit context type
        if 'type' in context:
            return context['type']
            
        # Infer from observation content
        obs_str = str(observation).lower()
        
        if any(keyword in obs_str for keyword in ['model', 'training', 'neural', 'ml', 'ai']):
            return 'ml_model'
        elif any(keyword in obs_str for keyword in ['function', 'class', 'code', 'def', 'import']):
            return 'code'
        elif any(keyword in obs_str for keyword in ['data', 'dataset', 'csv', 'json', 'database']):
            return 'data'
        elif any(keyword in obs_str for keyword in ['system', 'architecture', 'infrastructure']):
            return 'system'
        elif any(keyword in obs_str for keyword in ['experiment', 'test', 'hypothesis', 'trial']):
            return 'experiment'
        else:
            return 'generic'
            
    def _generate_ml_questions(self, observation: Any, context: Dict, num_questions: int) -> List[str]:
        """Generate ML-specific questions."""
        return [
            "What is the model architecture and why was it chosen?",
            "How does the training process work for this model?",
            "What are the key hyperparameters and their effects?",
            "How is the model's performance evaluated?",
            "What are potential failure modes or biases?",
            "How does this model compare to state-of-the-art alternatives?",
            "What data preprocessing is required?",
            "How can the model be improved or optimized?"
        ][:num_questions]
        
    def _generate_code_questions(self, observation: Any, context: Dict, num_questions: int) -> List[str]:
        """Generate code-specific questions."""
        return [
            "What is the purpose and functionality of this code?",
            "How does the algorithm work step by step?",
            "What are the input and output specifications?",
            "What are potential edge cases or error conditions?",
            "How can this code be optimized for performance?",
            "What dependencies does this code have?",
            "How is this code tested and validated?",
            "What design patterns are being used?"
        ][:num_questions]
        
    def _generate_data_questions(self, observation: Any, context: Dict, num_questions: int) -> List[str]:
        """Generate data-specific questions."""
        return [
            "What is the source and quality of this data?",
            "What patterns or trends exist in the data?",
            "How representative is this data of the population?",
            "What are potential biases or limitations?",
            "How should this data be preprocessed or cleaned?",
            "What relationships exist between variables?",
            "How can this data be validated or verified?",
            "What insights can be derived from this data?"
        ][:num_questions]
        
    def _generate_system_questions(self, observation: Any, context: Dict, num_questions: int) -> List[str]:
        """Generate system-specific questions."""
        return [
            "How are the system components interconnected?",
            "What are the system's scalability characteristics?",
            "How does the system handle failures or errors?",
            "What are the security considerations?",
            "How is system performance monitored and optimized?",
            "What are the deployment and maintenance requirements?",
            "How does the system integrate with other systems?",
            "What are the cost and resource implications?"
        ][:num_questions]
        
    def _generate_experiment_questions(self, observation: Any, context: Dict, num_questions: int) -> List[str]:
        """Generate experiment-specific questions."""
        return [
            "What hypothesis is being tested?",
            "What variables are controlled vs. manipulated?",
            "How is success or failure measured?",
            "What are potential confounding factors?",
            "How can the experimental design be improved?",
            "What are the statistical analysis methods?",
            "How reproducible are the results?",
            "What are the broader implications of the findings?"
        ][:num_questions]
        
    def _generate_generic_questions(self, observation: Any, num_questions: int) -> List[str]:
        """Generate generic questions for unknown contexts."""
        obs_str = str(observation)[:100]  # Truncate for safety
        
        return [
            f"What is the significance of {obs_str}?",
            f"How does {obs_str} work?",
            f"What are the implications of {obs_str}?",
            f"How can {obs_str} be improved or optimized?",
            f"What patterns exist in {obs_str}?",
            f"What questions does {obs_str} raise?",
            f"How does {obs_str} relate to other concepts?",
            f"What evidence supports claims about {obs_str}?"
        ][:num_questions]
        
    def _clean_observation_for_template(self, observation_str: str) -> str:
        """Clean observation string for use in templates."""
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', observation_str.strip())
        
        # Truncate if too long
        if len(clean) > 50:
            clean = clean[:47] + "..."
            
        return clean
        
    def _find_related_concept(self, observation_str: str) -> str:
        """Find a related concept for two-placeholder templates."""
        # Simple approach: extract a keyword from the observation
        words = observation_str.split()
        
        # Look for interesting words (not common articles/prepositions)
        interesting_words = [
            word for word in words 
            if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'they', 'their']
        ]
        
        if interesting_words:
            return random.choice(interesting_words)
        else:
            return "related concepts"
            
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for follow-up questions."""
        # Simple concept extraction using keywords
        words = text.split()
        
        # Look for capitalized words (potential proper nouns) and technical terms
        concepts = []
        
        for word in words:
            word = word.strip('.,!?;:')
            if (len(word) > 4 and 
                (word[0].isupper() or 
                 any(tech_indicator in word.lower() for tech_indicator in 
                     ['_', 'tion', 'ment', 'ness', 'ism', 'ogy']))):
                concepts.append(word)
                
        return list(set(concepts))[:5]  # Return unique concepts, max 5
        
    def _clean_questions(self, questions: List[str]) -> List[str]:
        """Clean and filter generated questions."""
        cleaned = []
        
        for question in questions:
            # Remove duplicates and very similar questions
            if not any(self._questions_too_similar(question, existing) for existing in cleaned):
                # Clean up formatting
                clean_question = question.strip()
                if clean_question and not clean_question.endswith('?'):
                    clean_question += '?'
                    
                # Basic quality checks
                if len(clean_question) > 10 and len(clean_question) < 200:
                    cleaned.append(clean_question)
                    
        return cleaned
        
    def _questions_too_similar(self, q1: str, q2: str, threshold: float = 0.8) -> bool:
        """Check if two questions are too similar."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        
        if not words1 or not words2:
            return False
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
