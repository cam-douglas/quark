#!/usr/bin/env python3
"""
Response Generator for Exponential Learning System
Learns from research and generates intelligent responses to prompts
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass
from datetime import datetime
import json
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ResponseContext:
    """Context for generating responses"""
    prompt: str
    research_results: Dict[str, Any]
    synthesized_knowledge: Any
    confidence: float
    sources: List[str]
    timestamp: datetime

@dataclass
class GeneratedResponse:
    """A generated response to a prompt"""
    response: str
    confidence: float
    sources: List[str]
    knowledge_used: List[str]
    reasoning: str
    timestamp: datetime
    learning_improvements: List[str]

class ResponseGenerator:
    """
    Generates intelligent responses based on learned knowledge
    Continuously improves through research and synthesis
    """
    
    def __init__(self):
        self.response_patterns = {}
        self.knowledge_base = {}
        self.learning_history = []
        self.response_quality_metrics = defaultdict(list)
        self.improvement_suggestions = []
        
    async def generate_response(self, prompt: str, research_results: Dict[str, Any], 
                              synthesized_knowledge: Any) -> GeneratedResponse:
        """Generate a response to a prompt using learned knowledge"""
        logger.info(f"🧠 Generating response for: {prompt[:100]}...")
        
        # Create response context
        context = ResponseContext(
            prompt=prompt,
            research_results=research_results,
            synthesized_knowledge=synthesized_knowledge,
            confidence=0.0,
            sources=[],
            timestamp=datetime.now()
        )
        
        # Analyze the prompt
        prompt_analysis = self.analyze_prompt(prompt)
        
        # Extract relevant knowledge
        relevant_knowledge = self.extract_relevant_knowledge(prompt_analysis, synthesized_knowledge)
        
        # Generate response
        response_text = self.construct_response(prompt_analysis, relevant_knowledge, research_results)
        
        # Calculate confidence
        confidence = self.calculate_response_confidence(relevant_knowledge, research_results)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(prompt_analysis, relevant_knowledge)
        
        # Identify learning improvements
        improvements = self.identify_learning_improvements(prompt, response_text, confidence)
        
        # Create response object
        generated_response = GeneratedResponse(
            response=response_text,
            confidence=confidence,
            sources=list(set(relevant_knowledge.get("sources", []))),
            knowledge_used=relevant_knowledge.get("concepts", []),
            reasoning=reasoning,
            timestamp=datetime.now(),
            learning_improvements=improvements
        )
        
        # Learn from this response generation
        await self.learn_from_response(prompt, generated_response, research_results)
        
        logger.info(f"✅ Generated response with confidence {confidence:.2f}")
        return generated_response
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the prompt to understand what's being asked"""
        analysis = {
            "intent": self.detect_intent(prompt),
            "domain": self.detect_domain(prompt),
            "complexity": self.assess_complexity(prompt),
            "key_concepts": self.extract_key_concepts(prompt),
            "question_type": self.classify_question_type(prompt)
        }
        
        return analysis
    
    def detect_intent(self, prompt: str) -> str:
        """Detect the intent behind the prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["what is", "define", "explain", "describe"]):
            return "definition"
        elif any(word in prompt_lower for word in ["how", "process", "method", "steps"]):
            return "process"
        elif any(word in prompt_lower for word in ["why", "reason", "cause", "because"]):
            return "explanation"
        elif any(word in prompt_lower for word in ["compare", "difference", "similar", "versus"]):
            return "comparison"
        elif any(word in prompt_lower for word in ["example", "instance", "case", "scenario"]):
            return "example"
        else:
            return "general"
    
    def detect_domain(self, prompt: str) -> str:
        """Detect the domain/topic of the prompt"""
        prompt_lower = prompt.lower()
        
        domains = {
            "technology": ["computer", "software", "hardware", "programming", "ai", "machine learning"],
            "science": ["physics", "chemistry", "biology", "research", "experiment", "theory"],
            "medicine": ["health", "medical", "disease", "treatment", "symptoms", "diagnosis"],
            "business": ["company", "market", "finance", "strategy", "management", "economics"],
            "education": ["learning", "teaching", "student", "school", "university", "course"]
        }
        
        for domain, keywords in domains.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return domain
        
        return "general"
    
    def assess_complexity(self, prompt: str) -> str:
        """Assess the complexity level of the prompt"""
        word_count = len(prompt.split())
        technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', prompt))
        
        if word_count > 50 or technical_terms > 5:
            return "high"
        elif word_count > 25 or technical_terms > 2:
            return "medium"
        else:
            return "low"
    
    def extract_key_concepts(self, prompt: str) -> List[str]:
        """Extract key concepts from the prompt"""
        # Extract capitalized phrases (potential proper nouns)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', prompt)
        
        # Extract technical terms
        technical_terms = re.findall(r'\b(?:algorithm|theory|method|system|framework|model|approach)\b', prompt, re.IGNORECASE)
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', prompt)
        
        all_concepts = concepts + technical_terms + quoted_terms
        return list(set(all_concepts))
    
    def classify_question_type(self, prompt: str) -> str:
        """Classify the type of question being asked"""
        prompt_lower = prompt.lower()
        
        if "?" in prompt:
            if prompt_lower.startswith(("what", "which", "where", "when", "who")):
                return "wh_question"
            elif prompt_lower.startswith("how"):
                return "how_question"
            elif prompt_lower.startswith("why"):
                return "why_question"
            else:
                return "yes_no_question"
        else:
            return "statement"
    
    def extract_relevant_knowledge(self, prompt_analysis: Dict[str, Any], 
                                 synthesized_knowledge: Any) -> Dict[str, Any]:
        """Extract knowledge relevant to the prompt"""
        relevant_knowledge = {
            "concepts": [],
            "definitions": {},
            "relationships": [],
            "insights": [],
            "sources": []
        }
        
        if not synthesized_knowledge:
            return relevant_knowledge
        
        # Extract concepts that match prompt analysis
        prompt_concepts = set(prompt_analysis["key_concepts"])
        
        if hasattr(synthesized_knowledge, 'core_concepts'):
            for concept in synthesized_knowledge.core_concepts:
                if any(pc.lower() in concept.lower() or concept.lower() in pc.lower() 
                       for pc in prompt_concepts):
                    relevant_knowledge["concepts"].append(concept)
        
        # Extract relevant definitions
        if hasattr(synthesized_knowledge, 'definitions'):
            for word, definition in synthesized_knowledge.definitions.items():
                if any(pc.lower() in word.lower() or word.lower() in pc.lower() 
                       for pc in prompt_concepts):
                    relevant_knowledge["definitions"][word] = definition
        
        # Extract relevant relationships
        if hasattr(synthesized_knowledge, 'relationships'):
            for rel in synthesized_knowledge.relationships:
                if any(pc.lower() in str(rel).lower() for pc in prompt_concepts):
                    relevant_knowledge["relationships"].append(rel)
        
        # Extract relevant insights
        if hasattr(synthesized_knowledge, 'insights'):
            for insight in synthesized_knowledge.insights:
                if any(pc.lower() in insight.lower() for pc in prompt_concepts):
                    relevant_knowledge["insights"].append(insight)
        
        # Extract sources
        if hasattr(synthesized_knowledge, 'sources'):
            relevant_knowledge["sources"] = synthesized_knowledge.sources
        
        return relevant_knowledge
    
    def construct_response(self, prompt_analysis: Dict[str, Any], 
                         relevant_knowledge: Dict[str, Any], 
                         research_results: Dict[str, Any]) -> str:
        """Construct a coherent response based on analysis and knowledge"""
        
        # Start building response
        response_parts = []
        
        # Add introduction based on intent
        intro = self.generate_introduction(prompt_analysis)
        if intro:
            response_parts.append(intro)
        
        # Add main content based on knowledge
        main_content = self.generate_main_content(prompt_analysis, relevant_knowledge)
        if main_content:
            response_parts.append(main_content)
        
        # Add examples if requested
        if prompt_analysis["intent"] == "example":
            examples = self.generate_examples(relevant_knowledge, research_results)
            if examples:
                response_parts.append(examples)
        
        # Add relationships if available
        if relevant_knowledge["relationships"]:
            relationships = self.explain_relationships(relevant_knowledge["relationships"])
            if relationships:
                response_parts.append(relationships)
        
        # Add insights if available
        if relevant_knowledge["insights"]:
            insights = self.present_insights(relevant_knowledge["insights"])
            if insights:
                response_parts.append(insights)
        
        # Add conclusion
        conclusion = self.generate_conclusion(prompt_analysis, relevant_knowledge)
        if conclusion:
            response_parts.append(conclusion)
        
        # Combine all parts
        response = "\n\n".join(response_parts)
        
        # Clean up response
        response = self.clean_response(response)
        
        return response
    
    def generate_introduction(self, prompt_analysis: Dict[str, Any]) -> str:
        """Generate an introduction based on prompt analysis"""
        domain = prompt_analysis["domain"]
        intent = prompt_analysis["intent"]
        
        if intent == "definition":
            return f"Based on current research and knowledge, here's what I understand about this topic:"
        elif intent == "process":
            return f"Here's how this process works, according to available research:"
        elif intent == "explanation":
            return f"Let me explain the reasoning behind this, based on current knowledge:"
        elif intent == "comparison":
            return f"Here's a comparison based on available research:"
        elif intent == "example":
            return f"Here are some examples and instances I've learned about:"
        else:
            return f"Based on my research in {domain}, here's what I can tell you:"
    
    def generate_main_content(self, prompt_analysis: Dict[str, Any], 
                            relevant_knowledge: Dict[str, Any]) -> str:
        """Generate the main content of the response"""
        content_parts = []
        
        # Add definitions
        if relevant_knowledge["definitions"]:
            for term, definition in relevant_knowledge["definitions"].items():
                content_parts.append(f"**{term}**: {definition}")
        
        # Add concepts
        if relevant_knowledge["concepts"]:
            concepts_text = ", ".join(relevant_knowledge["concepts"][:5])  # Limit to 5
            content_parts.append(f"Key concepts include: {concepts_text}")
        
        # Add insights
        if relevant_knowledge["insights"]:
            insights_text = relevant_knowledge["insights"][0]  # Use first insight
            content_parts.append(f"An important insight is: {insights_text}")
        
        return "\n\n".join(content_parts)
    
    def generate_examples(self, relevant_knowledge: Dict[str, Any], 
                         research_results: Dict[str, Any]) -> str:
        """Generate examples based on knowledge and research"""
        examples = []
        
        # Extract examples from research results
        for source, results in research_results.items():
            if isinstance(results, list):
                for result in results[:2]:  # Limit to 2 examples per source
                    if hasattr(result, 'content'):
                        # Look for example patterns in content
                        example_matches = re.findall(r'(?:example|instance|case|scenario):\s*([^.]*?)(?:\.|$)', 
                                                   result.content, re.IGNORECASE)
                        examples.extend(example_matches)
        
        if examples:
            examples_text = "\n".join([f"• {example}" for example in examples[:3]])  # Limit to 3
            return f"**Examples:**\n{examples_text}"
        
        return ""
    
    def explain_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        """Explain relationships between concepts"""
        if not relationships:
            return ""
        
        relationship_texts = []
        for rel in relationships[:3]:  # Limit to 3 relationships
            source = rel.get("source", "unknown")
            target = rel.get("target", "unknown")
            rel_type = rel.get("type", "related to")
            strength = rel.get("strength", 0.0)
            
            if strength > 0.3:  # Only show strong relationships
                relationship_texts.append(f"• **{source}** {rel_type} **{target}**")
        
        if relationship_texts:
            return f"**Key Relationships:**\n" + "\n".join(relationship_texts)
        
        return ""
    
    def present_insights(self, insights: List[str]) -> str:
        """Present insights in a structured way"""
        if not insights:
            return ""
        
        # Select most relevant insights
        selected_insights = insights[:2]  # Limit to 2 insights
        
        insights_text = "\n".join([f"• {insight}" for insight in selected_insights])
        return f"**Key Insights:**\n{insights_text}"
    
    def generate_conclusion(self, prompt_analysis: Dict[str, Any], 
                          relevant_knowledge: Dict[str, Any]) -> str:
        """Generate a conclusion for the response"""
        if prompt_analysis["intent"] == "definition":
            return "This represents the current understanding based on available research. As new information becomes available, this understanding may evolve."
        elif prompt_analysis["intent"] == "process":
            return "This process represents the current best practices based on research. There may be variations or improvements as the field advances."
        elif prompt_analysis["intent"] == "explanation":
            return "This explanation is based on current research and understanding. Further research may reveal additional factors or perspectives."
        else:
            return "This response is based on current knowledge and research. The field continues to evolve with new discoveries and insights."
    
    def clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove extra whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        
        # Ensure proper sentence endings
        response = re.sub(r'([a-z])\n([A-Z])', r'\1. \2', response)
        
        # Limit response length
        if len(response) > 2000:
            response = response[:2000] + "..."
        
        return response.strip()
    
    def calculate_response_confidence(self, relevant_knowledge: Dict[str, Any], 
                                   research_results: Dict[str, Any]) -> float:
        """Calculate confidence in the generated response"""
        confidence = 0.0
        
        # Base confidence on knowledge availability
        if relevant_knowledge["concepts"]:
            confidence += 0.3
        
        if relevant_knowledge["definitions"]:
            confidence += 0.2
        
        if relevant_knowledge["relationships"]:
            confidence += 0.2
        
        if relevant_knowledge["insights"]:
            confidence += 0.1
        
        # Boost confidence based on source diversity
        source_count = len(relevant_knowledge.get("sources", []))
        if source_count > 1:
            confidence += min(0.2, source_count * 0.05)
        
        # Reduce confidence if knowledge is limited
        if not any(relevant_knowledge.values()):
            confidence = 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def generate_reasoning(self, prompt_analysis: Dict[str, Any], 
                          relevant_knowledge: Dict[str, Any]) -> str:
        """Generate reasoning for the response"""
        reasoning_parts = []
        
        reasoning_parts.append(f"I analyzed this as a {prompt_analysis['complexity']} complexity {prompt_analysis['intent']} question in the {prompt_analysis['domain']} domain.")
        
        if relevant_knowledge["concepts"]:
            reasoning_parts.append(f"I found {len(relevant_knowledge['concepts'])} relevant concepts to work with.")
        
        if relevant_knowledge["sources"]:
            reasoning_parts.append(f"I drew from {len(relevant_knowledge['sources'])} different knowledge sources.")
        
        if relevant_knowledge["relationships"]:
            reasoning_parts.append(f"I identified {len(relevant_knowledge['relationships'])} key relationships between concepts.")
        
        return " ".join(reasoning_parts)
    
    def identify_learning_improvements(self, prompt: str, response: str, 
                                     confidence: float) -> List[str]:
        """Identify areas where the system can learn and improve"""
        improvements = []
        
        if confidence < 0.5:
            improvements.append("Need more research on this topic to improve confidence")
        
        if len(response) < 100:
            improvements.append("Response could be more detailed with additional knowledge")
        
        if "unknown" in response.lower() or "unclear" in response.lower():
            improvements.append("Need to clarify unclear aspects through additional research")
        
        # Identify specific areas for improvement
        prompt_concepts = self.extract_key_concepts(prompt)
        if prompt_concepts:
            improvements.append(f"Could research more about: {', '.join(prompt_concepts[:3])}")
        
        return improvements
    
    async def learn_from_response(self, prompt: str, response: GeneratedResponse, 
                                research_results: Dict[str, Any]):
        """Learn from the response generation process to improve future responses"""
        
        # Record learning
        learning_entry = {
            "prompt": prompt,
            "response_confidence": response.confidence,
            "knowledge_used": len(response.knowledge_used),
            "sources_used": len(response.sources),
            "timestamp": datetime.now(),
            "improvements_needed": response.learning_improvements
        }
        
        self.learning_history.append(learning_entry)
        
        # Update response patterns
        prompt_type = self.classify_question_type(prompt)
        if prompt_type not in self.response_patterns:
            self.response_patterns[prompt_type] = []
        
        self.response_patterns[prompt_type].append({
            "prompt": prompt,
            "response": response.response,
            "confidence": response.confidence
        })
        
        # Track quality metrics
        self.response_quality_metrics["confidence"].append(response.confidence)
        self.response_quality_metrics["knowledge_utilization"].append(len(response.knowledge_used))
        
        # Generate improvement suggestions
        if response.confidence < 0.7:
            self.improvement_suggestions.append({
                "area": "response_confidence",
                "suggestion": f"Research more about: {', '.join(response.learning_improvements)}",
                "priority": "high" if response.confidence < 0.5 else "medium"
            })
        
        logger.info(f"📚 Learned from response generation. Confidence: {response.confidence:.2f}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning progress"""
        if not self.learning_history:
            return {"status": "No learning history yet"}
        
        avg_confidence = sum(entry["response_confidence"] for entry in self.learning_history) / len(self.learning_history)
        avg_knowledge = sum(entry["knowledge_used"] for entry in self.learning_history) / len(self.learning_history)
        
        return {
            "total_responses": len(self.learning_history),
            "average_confidence": avg_confidence,
            "average_knowledge_used": avg_knowledge,
            "response_patterns": len(self.response_patterns),
            "improvement_suggestions": len(self.improvement_suggestions),
            "recent_improvements": self.improvement_suggestions[-5:] if self.improvement_suggestions else []
        }

async def main():
    """Test the response generator"""
    generator = ResponseGenerator()
    
    # Mock research results and knowledge
    mock_research = {
        "wikipedia": [type('MockResult', (), {'content': 'Quantum computing uses quantum mechanics for computation.'})()],
        "arxiv": [type('MockResult', (), {'content': 'Recent advances show quantum supremacy in specific tasks.'})()]
    }
    
    mock_knowledge = type('MockKnowledge', (), {
        'core_concepts': ['quantum computing', 'quantum mechanics', 'quantum supremacy'],
        'definitions': {'quantum computing': 'Computing using quantum mechanical phenomena'},
        'relationships': [{'source': 'quantum computing', 'target': 'quantum mechanics', 'type': 'based_on'}],
        'insights': ['Quantum computing represents a paradigm shift in computational power'],
        'sources': ['wikipedia', 'arxiv']
    })()
    
    # Test response generation
    prompt = "What is quantum computing and how does it work?"
    
    response = await generator.generate_response(prompt, mock_research, mock_knowledge)
    
    print(f"🧠 Generated Response:")
    print(f"Prompt: {prompt}")
    print(f"Response: {response.response}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Sources: {response.sources}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Improvements: {response.learning_improvements}")
    
    # Get learning stats
    stats = generator.get_learning_stats()
    print(f"\n📊 Learning Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
