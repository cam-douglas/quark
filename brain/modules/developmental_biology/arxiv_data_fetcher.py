#!/usr/bin/env python3
"""arXiv Data Fetcher for Experimental Lineage Studies.

Fetches developmental biology research papers from arXiv using the arXiv API
to find experimental lineage tracing data and supplementary materials for
validation of lineage tracking systems.

Integration: arXiv data fetcher for lineage validation framework
Rationale: Alternative data source for experimental validation
"""

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)

class ArxivDataFetcher:
    """Fetcher for developmental biology papers from arXiv.
    
    Uses arXiv API to search for and retrieve developmental biology
    research papers that may contain experimental lineage tracing
    data and supplementary materials.
    """
    
    def __init__(self):
        """Initialize arXiv data fetcher."""
        self.base_url = "http://export.arxiv.org/api/query"
        
        # Rate limiting (arXiv recommends 3 second delays)
        self.last_request_time = 0
        self.request_interval = 3.0
        
        logger.info("Initialized ArxivDataFetcher")
        logger.info(f"Base URL: {self.base_url}")
    
    def search_developmental_biology_papers(self, max_results: int = 50) -> List[Dict[str, any]]:
        """Search arXiv for developmental biology papers with lineage data.
        
        Args:
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of relevant papers with metadata
        """
        # Search queries for developmental biology lineage papers
        search_queries = [
            'all:"neural tube" AND all:"lineage tracing"',
            'all:"developmental biology" AND all:"cell fate"',
            'all:"neuroepithelial" AND all:"progenitor"',
            'all:"embryonic development" AND all:"clonal analysis"',
            'ti:"lineage" AND abs:"neural development"',
            'all:"neural progenitor" AND all:"fate mapping"'
        ]
        
        all_papers = []
        
        for query in search_queries:
            papers = self._search_arxiv(query, max_results=10)
            all_papers.extend(papers)
            
            # Rate limiting between queries
            time.sleep(1.0)
        
        # Remove duplicates and sort by relevance
        unique_papers = {}
        for paper in all_papers:
            arxiv_id = paper['arxiv_id']
            if arxiv_id not in unique_papers or paper['relevance_score'] > unique_papers[arxiv_id]['relevance_score']:
                unique_papers[arxiv_id] = paper
        
        sorted_papers = sorted(unique_papers.values(), 
                              key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Found {len(sorted_papers)} unique developmental biology papers")
        
        return sorted_papers[:max_results]
    
    def _search_arxiv(self, search_query: str, max_results: int = 10) -> List[Dict[str, any]]:
        """Search arXiv with specific query.
        
        Args:
            search_query: arXiv search query
            max_results: Maximum results to retrieve
            
        Returns:
            List of paper metadata
        """
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_arxiv_response(response.text)
            
            logger.info(f"Retrieved {len(papers)} papers for query: {search_query[:50]}...")
            
            return papers
            
        except Exception as e:
            logger.error(f"arXiv search failed for query '{search_query}': {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, any]]:
        """Parse arXiv API XML response.
        
        Args:
            xml_content: XML response from arXiv API
            
        Returns:
            List of parsed paper metadata
        """
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom',
                'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
            }
            
            papers = []
            
            # Extract papers from entries
            for entry in root.findall('.//atom:entry', namespaces):
                paper_data = self._extract_paper_metadata(entry, namespaces)
                if paper_data:
                    papers.append(paper_data)
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to parse arXiv response: {e}")
            return []
    
    def _extract_paper_metadata(self, entry, namespaces: Dict[str, str]) -> Optional[Dict[str, any]]:
        """Extract metadata from single arXiv entry.
        
        Args:
            entry: XML entry element
            namespaces: XML namespaces
            
        Returns:
            Paper metadata dictionary
        """
        try:
            # Extract basic metadata
            title_elem = entry.find('atom:title', namespaces)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            id_elem = entry.find('atom:id', namespaces)
            arxiv_url = id_elem.text if id_elem is not None else ""
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
            
            summary_elem = entry.find('atom:summary', namespaces)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            published_elem = entry.find('atom:published', namespaces)
            published_date = published_elem.text if published_elem is not None else ""
            
            # Extract authors
            authors = []
            for author_elem in entry.findall('atom:author', namespaces):
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Extract categories
            categories = []
            for cat_elem in entry.findall('atom:category', namespaces):
                term = cat_elem.get('term')
                if term:
                    categories.append(term)
            
            # Extract arXiv-specific metadata
            primary_cat_elem = entry.find('arxiv:primary_category', namespaces)
            primary_category = primary_cat_elem.get('term') if primary_cat_elem is not None else ""
            
            comment_elem = entry.find('arxiv:comment', namespaces)
            comment = comment_elem.text if comment_elem is not None else ""
            
            journal_ref_elem = entry.find('arxiv:journal_ref', namespaces)
            journal_ref = journal_ref_elem.text if journal_ref_elem is not None else ""
            
            # Calculate relevance score
            relevance_score = self._calculate_developmental_biology_relevance(title, abstract, categories)
            
            paper_data = {
                'arxiv_id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'published_date': published_date,
                'primary_category': primary_category,
                'categories': categories,
                'comment': comment,
                'journal_reference': journal_ref,
                'arxiv_url': arxiv_url,
                'pdf_url': arxiv_url.replace('/abs/', '/pdf/') + '.pdf',
                'relevance_score': relevance_score
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to extract paper metadata: {e}")
            return None
    
    def _calculate_developmental_biology_relevance(self, title: str, abstract: str, 
                                                 categories: List[str]) -> float:
        """Calculate relevance score for developmental biology lineage studies.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            categories: arXiv categories
            
        Returns:
            Relevance score (0-1)
        """
        content = (title + " " + abstract).lower()
        
        # High relevance terms for lineage tracing
        high_relevance = [
            "lineage tracing", "clonal analysis", "fate mapping", "neural tube",
            "neuroepithelial", "cell fate", "progenitor", "neural development"
        ]
        
        # Medium relevance terms
        medium_relevance = [
            "developmental biology", "embryonic", "differentiation", "neural",
            "cell division", "stem cell", "morphogen", "patterning"
        ]
        
        # Calculate relevance score
        score = 0.0
        
        # High relevance terms (0.2 points each)
        for term in high_relevance:
            if term in content:
                score += 0.2
        
        # Medium relevance terms (0.05 points each)
        for term in medium_relevance:
            if term in content:
                score += 0.05
        
        # Category bonus
        bio_categories = ['q-bio', 'physics.bio-ph']
        for cat in categories:
            if any(bio_cat in cat for bio_cat in bio_categories):
                score += 0.1
        
        return min(1.0, score)
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting for arXiv API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            wait_time = self.request_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def get_paper_full_text_url(self, arxiv_id: str) -> str:
        """Get URL for full text PDF of arXiv paper.
        
        Args:
            arxiv_id: arXiv identifier
            
        Returns:
            URL for PDF download
        """
        return f"http://export.arxiv.org/pdf/{arxiv_id}.pdf"
    
    def export_search_summary(self, papers: List[Dict[str, any]]) -> Dict[str, any]:
        """Export summary of arXiv search results.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            Search summary
        """
        if not papers:
            return {'no_results': True}
        
        # Calculate summary statistics
        categories_found = set()
        authors_found = set()
        
        for paper in papers:
            categories_found.update(paper['categories'])
            authors_found.update(paper['authors'])
        
        relevance_scores = [paper['relevance_score'] for paper in papers]
        
        summary = {
            'total_papers': len(papers),
            'average_relevance': sum(relevance_scores) / len(relevance_scores),
            'high_relevance_papers': len([p for p in papers if p['relevance_score'] > 0.5]),
            'categories_found': len(categories_found),
            'unique_authors': len(authors_found),
            'date_range': {
                'earliest': min(paper['published_date'] for paper in papers if paper['published_date']),
                'latest': max(paper['published_date'] for paper in papers if paper['published_date'])
            },
            'top_papers': sorted(papers, key=lambda x: x['relevance_score'], reverse=True)[:5]
        }
        
        return summary
