#!/usr/bin/env python3
"""
Research Agents for Exponential Learning System
Gathers knowledge from multiple sources: Wikipedia, ArXiv, PubMed, etc.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from datetime import datetime
import json
import re
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

@dataclass
class ResearchResult:
    """Represents a research result from a knowledge source"""
    source: str
    topic: str
    content: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    connections: List[str]

class BaseResearchAgent:
    """Base class for all research agents"""
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.session = None
        self.rate_limit_delay = 1.0  # Delay between requests
        
    async def initialize(self):
        """Initialize the agent's HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    async def search(self, query: str) -> List[ResearchResult]:
        """Search for information (to be implemented by subclasses)"""
        raise NotImplementedError
    
    async def extract_knowledge(self, content: str) -> Dict[str, Any]:
        """Extract structured knowledge from content"""
        # Basic knowledge extraction
        knowledge = {
            "concepts": self.extract_concepts(content),
            "definitions": self.extract_definitions(content),
            "connections": self.extract_connections(content),
            "insights": self.extract_insights(content)
        }
        return knowledge
    
    def extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Simple concept extraction (can be enhanced with NLP)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        return list(set(words[:20]))  # Top 20 unique concepts
    
    def extract_definitions(self, content: str) -> List[str]:
        """Extract definitions from content"""
        # Look for definition patterns
        definitions = re.findall(r'is\s+(?:a|an)\s+([^.]*?)(?:\.|,|;|$)', content, re.IGNORECASE)
        return definitions[:10]
    
    def extract_connections(self, content: str) -> List[str]:
        """Extract connections between concepts"""
        # Look for relationship words
        connections = re.findall(r'(?:related to|connected to|similar to|differs from)\s+([^.]*?)(?:\.|,|;|$)', content, re.IGNORECASE)
        return connections[:10]
    
    def extract_insights(self, content: str) -> List[str]:
        """Extract insights and implications"""
        # Look for insight patterns
        insights = re.findall(r'(?:this means|this suggests|this implies|therefore|consequently)\s+([^.]*?)(?:\.|,|;|$)', content, re.IGNORECASE)
        return insights[:10]

class WikipediaResearchAgent(BaseResearchAgent):
    """Agent for gathering knowledge from Wikipedia"""
    
    def __init__(self):
        super().__init__("Wikipedia", "https://en.wikipedia.org/api/rest_v1")
        self.search_url = "https://en.wikipedia.org/w/api.php"
    
    async def search(self, query: str) -> List[ResearchResult]:
        """Search Wikipedia for information"""
        await self.initialize()
        
        try:
            # Search for pages
            search_params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 5
            }
            
            async with self.session.get(self.search_url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    search_results = data.get("query", {}).get("search", [])
                    
                    results = []
                    for result in search_results:
                        # Get page content
                        page_content = await self.get_page_content(result["pageid"])
                        if page_content:
                            knowledge = await self.extract_knowledge(page_content)
                            
                            research_result = ResearchResult(
                                source="Wikipedia",
                                topic=result["title"],
                                content=page_content[:1000],  # First 1000 chars
                                confidence=0.8,
                                timestamp=datetime.now(),
                                metadata={
                                    "pageid": result["pageid"],
                                    "snippet": result["snippet"],
                                    "wordcount": result["wordcount"]
                                },
                                connections=knowledge.get("connections", [])
                            )
                            results.append(research_result)
                    
                    logger.info(f"🔍 Wikipedia: Found {len(results)} results for '{query}'")
                    return results
                
        except Exception as e:
            logger.error(f"❌ Wikipedia search error: {e}")
        
        return []
    
    async def get_page_content(self, pageid: int) -> Optional[str]:
        """Get the content of a Wikipedia page"""
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "pageids": pageid,
                "exintro": True,  # Only introduction
                "explaintext": True  # Plain text
            }
            
            async with self.session.get(self.search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})
                    if str(pageid) in pages:
                        return pages[str(pageid)].get("extract", "")
            
        except Exception as e:
            logger.error(f"❌ Error getting page content: {e}")
        
        return None

class ArxivResearchAgent(BaseResearchAgent):
    """Agent for gathering knowledge from ArXiv scientific papers"""
    
    def __init__(self):
        super().__init__("ArXiv", "http://export.arxiv.org/api/query")
    
    async def search(self, query: str) -> List[ResearchResult]:
        """Search ArXiv for scientific papers"""
        await self.initialize()
        
        try:
            # Search ArXiv
            search_params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": 5,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            async with self.session.get(self.base_url, params=search_params) as response:
                if response.status == 200:
                    content = await response.text()
                    results = self.parse_arxiv_response(content, query)
                    
                    logger.info(f"🔬 ArXiv: Found {len(results)} papers for '{query}'")
                    return results
                
        except Exception as e:
            logger.error(f"❌ ArXiv search error: {e}")
        
        return []
    
    def parse_arxiv_response(self, content: str, query: str) -> List[ResearchResult]:
        """Parse ArXiv XML response"""
        results = []
        
        # Simple XML parsing (can be enhanced with proper XML parser)
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
        
        for entry in entries:
            try:
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                title = title_match.group(1).strip() if title_match else "Unknown Title"
                
                # Extract summary
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else "No summary available"
                
                # Extract authors
                authors = re.findall(r'<name>(.*?)</name>', entry)
                
                # Extract categories
                categories = re.findall(r'<category term="(.*?)"', entry)
                
                knowledge = self.extract_knowledge(summary)
                
                research_result = ResearchResult(
                    source="ArXiv",
                    topic=title,
                    content=summary,
                    confidence=0.9,
                    timestamp=datetime.now(),
                    metadata={
                        "authors": authors,
                        "categories": categories,
                        "query": query
                    },
                    connections=knowledge.get("connections", [])
                )
                results.append(research_result)
                
            except Exception as e:
                logger.error(f"❌ Error parsing ArXiv entry: {e}")
                continue
        
        return results

class DictionaryResearchAgent(BaseResearchAgent):
    """Agent for gathering definitions and word knowledge"""
    
    def __init__(self):
        super().__init__("Dictionary", "https://api.dictionaryapi.dev/api/v2/entries")
    
    async def search(self, query: str) -> List[ResearchResult]:
        """Search dictionary for word definitions"""
        await self.initialize()
        
        try:
            # Clean query for dictionary search
            clean_query = re.sub(r'[^a-zA-Z\s]', '', query).strip().split()[0]
            
            if not clean_query:
                return []
            
            url = f"{self.base_url}/en/{clean_query}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = self.parse_dictionary_response(data, clean_query)
                    
                    logger.info(f"📚 Dictionary: Found {len(results)} definitions for '{clean_query}'")
                    return results
                
        except Exception as e:
            logger.error(f"❌ Dictionary search error: {e}")
        
        return []
    
    def parse_dictionary_response(self, data: List[Dict], word: str) -> List[ResearchResult]:
        """Parse dictionary API response"""
        results = []
        
        for entry in data:
            try:
                word_info = entry.get("word", word)
                meanings = entry.get("meanings", [])
                
                for meaning in meanings:
                    part_of_speech = meaning.get("partOfSpeech", "unknown")
                    definitions = meaning.get("definitions", [])
                    
                    for definition in definitions:
                        definition_text = definition.get("definition", "")
                        example = definition.get("example", "")
                        
                        # Combine definition and example
                        content = definition_text
                        if example:
                            content += f" Example: {example}"
                        
                        knowledge = self.extract_knowledge(content)
                        
                        research_result = ResearchResult(
                            source="Dictionary",
                            topic=f"{word_info} ({part_of_speech})",
                            content=content,
                            confidence=0.95,
                            timestamp=datetime.now(),
                            metadata={
                                "word": word_info,
                                "part_of_speech": part_of_speech,
                                "definition": definition_text,
                                "example": example
                            },
                            connections=knowledge.get("connections", [])
                        )
                        results.append(research_result)
                
            except Exception as e:
                logger.error(f"❌ Error parsing dictionary entry: {e}")
                continue
        
        return results

class PubMedResearchAgent(BaseResearchAgent):
    """Agent for gathering medical and scientific knowledge from PubMed"""
    
    def __init__(self):
        super().__init__("PubMed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
    
    async def search(self, query: str) -> List[ResearchResult]:
        """Search PubMed for medical/scientific articles"""
        await self.initialize()
        
        try:
            # Search PubMed
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": 5,
                "retmode": "json",
                "tool": "SmallMind",
                "email": "research@smallmind.ai"
            }
            
            search_url = f"{self.base_url}/esearch.fcgi"
            async with self.session.get(search_url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    id_list = data.get("esearchresult", {}).get("idlist", [])
                    
                    results = []
                    for pmid in id_list[:3]:  # Limit to 3 articles
                        article_data = await self.get_article_data(pmid)
                        if article_data:
                            results.append(article_data)
                    
                    logger.info(f"🏥 PubMed: Found {len(results)} articles for '{query}'")
                    return results
                
        except Exception as e:
            logger.error(f"❌ PubMed search error: {e}")
        
        return []
    
    async def get_article_data(self, pmid: str) -> Optional[ResearchResult]:
        """Get article data from PubMed"""
        try:
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
                "tool": "SmallMind",
                "email": "research@smallmind.ai"
            }
            
            fetch_url = f"{self.base_url}/efetch.fcgi"
            async with self.session.get(fetch_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Extract title and abstract
                    title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', content, re.DOTALL)
                    title = title_match.group(1).strip() if title_match else "Unknown Title"
                    
                    abstract_match = re.search(r'<AbstractText>(.*?)</AbstractText>', content, re.DOTALL)
                    abstract = abstract_match.group(1).strip() if abstract_match else "No abstract available"
                    
                    knowledge = self.extract_knowledge(abstract)
                    
                    research_result = ResearchResult(
                        source="PubMed",
                        topic=title,
                        content=abstract,
                        confidence=0.9,
                        timestamp=datetime.now(),
                        metadata={
                            "pmid": pmid,
                            "title": title,
                            "source": "PubMed"
                        },
                        connections=knowledge.get("connections", [])
                    )
                    return research_result
            
        except Exception as e:
            logger.error(f"❌ Error getting article data: {e}")
        
        return None

class ResearchAgentHub:
    """Hub that coordinates all research agents"""
    
    def __init__(self):
        self.agents = {
            "wikipedia": WikipediaResearchAgent(),
            "arxiv": ArxivResearchAgent(),
            "dictionary": DictionaryResearchAgent(),
            "pubmed": PubMedResearchAgent()
        }
        self.active_searches = {}
    
    async def initialize_all(self):
        """Initialize all research agents"""
        for name, agent in self.agents.items():
            await agent.initialize()
            logger.info(f"✅ Initialized {name} research agent")
    
    async def cleanup_all(self):
        """Clean up all research agents"""
        for name, agent in self.agents.items():
            await agent.cleanup()
            logger.info(f"🧹 Cleaned up {name} research agent")
    
    async def search_all_sources(self, query: str) -> Dict[str, List[ResearchResult]]:
        """Search all knowledge sources simultaneously"""
        logger.info(f"🔍 Searching all sources for: {query}")
        
        # Create search tasks for all agents
        search_tasks = {}
        for name, agent in self.agents.items():
            task = asyncio.create_task(agent.search(query))
            search_tasks[name] = task
        
        # Wait for all searches to complete
        results = {}
        for name, task in search_tasks.items():
            try:
                agent_results = await task
                results[name] = agent_results
                logger.info(f"✅ {name}: {len(agent_results)} results")
            except Exception as e:
                logger.error(f"❌ {name} search failed: {e}")
                results[name] = []
        
        return results
    
    async def synthesize_knowledge(self, query: str) -> Dict[str, Any]:
        """Synthesize knowledge from all sources"""
        # Search all sources
        all_results = await self.search_all_sources(query)
        
        # Synthesize knowledge
        synthesized = {
            "query": query,
            "timestamp": datetime.now(),
            "sources": {},
            "concepts": set(),
            "definitions": set(),
            "connections": set(),
            "insights": set(),
            "total_results": 0
        }
        
        for source, results in all_results.items():
            synthesized["sources"][source] = len(results)
            synthesized["total_results"] += len(results)
            
            for result in results:
                # Extract knowledge from each result
                knowledge = await self.agents[source].extract_knowledge(result.content)
                
                # Merge knowledge
                synthesized["concepts"].update(knowledge.get("concepts", []))
                synthesized["definitions"].update(knowledge.get("definitions", []))
                synthesized["connections"].update(knowledge.get("connections", []))
                synthesized["insights"].update(knowledge.get("insights", []))
        
        # Convert sets to lists for JSON serialization
        synthesized["concepts"] = list(synthesized["concepts"])
        synthesized["definitions"] = list(synthesized["definitions"])
        synthesized["connections"] = list(synthesized["connections"])
        synthesized["insights"] = list(synthesized["insights"])
        
        logger.info(f"🔬 Synthesized knowledge: {len(synthesized['concepts'])} concepts, {len(synthesized['connections'])} connections")
        return synthesized

async def main():
    """Test the research agent hub"""
    hub = ResearchAgentHub()
    
    try:
        await hub.initialize_all()
        
        # Test search
        query = "quantum computing"
        results = await hub.search_all_sources(query)
        
        print(f"\n🔍 Search results for '{query}':")
        for source, source_results in results.items():
            print(f"\n{source.upper()}: {len(source_results)} results")
            for result in source_results[:2]:  # Show first 2 results
                print(f"  - {result.topic}")
                print(f"    {result.content[:100]}...")
        
        # Test knowledge synthesis
        synthesized = await hub.synthesize_knowledge(query)
        print(f"\n🔬 Synthesized knowledge:")
        print(f"  Concepts: {len(synthesized['concepts'])}")
        print(f"  Connections: {len(synthesized['connections'])}")
        print(f"  Insights: {len(synthesized['insights'])}")
        
    finally:
        await hub.cleanup_all()

if __name__ == "__main__":
    asyncio.run(main())
