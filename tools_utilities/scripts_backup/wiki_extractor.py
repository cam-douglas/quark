#!/usr/bin/env python3
"""
Wikipedia Extractor
==================

Efficient extraction and preprocessing of Wikipedia XML dumps.
Handles large-scale extraction with multiprocessing and memory optimization.

Purpose: Extract clean text from Wikipedia XML dumps for training
Inputs: Compressed Wikipedia XML dumps
Outputs: Clean article text with metadata
Seeds: N/A (deterministic extraction)
Dependencies: lxml, multiprocessing, bz2, regex
"""

import os
import re
import bz2
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import regex as regex_lib  # More robust than standard re


@dataclass
class WikiArticle:
    """Represents a single Wikipedia article."""
    id: str
    title: str
    text: str
    categories: List[str]
    links: List[str]
    redirect: Optional[str] = None
    disambiguation: bool = False
    stub: bool = False
    

class WikiTextCleaner:
    """Cleans and normalizes Wikipedia markup text."""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.patterns = {
            # Remove Wiki markup
            'templates': regex_lib.compile(r'\{\{[^{}]*?\}\}'),
            'links': regex_lib.compile(r'\[\[([^|\]]*?)(?:\|([^\]]*?))?\]\]'),
            'external_links': regex_lib.compile(r'\[http[s]?://[^\]]*?\]'),
            'refs': regex_lib.compile(r'<ref[^>]*?>.*?</ref>', regex_lib.DOTALL | regex_lib.IGNORECASE),
            'comments': regex_lib.compile(r'<!--.*?-->', regex_lib.DOTALL),
            'categories': regex_lib.compile(r'\[\[Category:[^\]]*?\]\]'),
            'files': regex_lib.compile(r'\[\[File:[^\]]*?\]\]'),
            'images': regex_lib.compile(r'\[\[Image:[^\]]*?\]\]'),
            
            # Clean formatting
            'bold': regex_lib.compile(r"'''([^']*?)'''"),
            'italic': regex_lib.compile(r"''([^']*?)''"),
            'headings': regex_lib.compile(r'^=+\s*([^=]*?)\s*=+$', regex_lib.MULTILINE),
            
            # Clean special characters
            'multiple_spaces': regex_lib.compile(r'\s+'),
            'multiple_newlines': regex_lib.compile(r'\n{3,}'),
            
            # Extract useful info
            'extract_categories': regex_lib.compile(r'\[\[Category:([^\]]*?)\]\]'),
            'extract_links': regex_lib.compile(r'\[\[([^|\]]*?)(?:\|[^\]]*?)?\]\]'),
        }
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia markup from text."""
        # Remove templates and infoboxes
        text = self.patterns['templates'].sub('', text)
        
        # Remove references
        text = self.patterns['refs'].sub('', text)
        text = self.patterns['comments'].sub('', text)
        
        # Remove file and category links
        text = self.patterns['files'].sub('', text)
        text = self.patterns['images'].sub('', text)
        text = self.patterns['categories'].sub('', text)
        
        # Convert wiki links to plain text
        text = self.patterns['links'].sub(lambda m: m.group(2) if m.group(2) else m.group(1), text)
        text = self.patterns['external_links'].sub('', text)
        
        # Clean formatting
        text = self.patterns['bold'].sub(r'\1', text)
        text = self.patterns['italic'].sub(r'\1', text)
        text = self.patterns['headings'].sub(r'\1', text)
        
        # Normalize whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text)
        text = self.patterns['multiple_newlines'].sub('\n\n', text)
        
        return text.strip()
    
    def extract_categories(self, text: str) -> List[str]:
        """Extract categories from article text."""
        categories = self.patterns['extract_categories'].findall(text)
        return [cat.strip() for cat in categories]
    
    def extract_links(self, text: str) -> List[str]:
        """Extract internal links from article text."""
        links = self.patterns['extract_links'].findall(text)
        return [link.strip() for link in links if ':' not in link]
    
    def is_disambiguation(self, text: str) -> bool:
        """Check if article is a disambiguation page."""
        return 'disambiguation' in text.lower() or 'may refer to' in text.lower()
    
    def is_stub(self, text: str) -> bool:
        """Check if article is a stub."""
        return 'stub' in text.lower() and len(text) < 1000


class WikipediaXMLParser:
    """Parses Wikipedia XML dumps efficiently."""
    
    def __init__(self, min_text_length: int = 100, max_text_length: int = 50000):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.cleaner = WikiTextCleaner()
        self.logger = logging.getLogger(__name__)
        
    def parse_article(self, page_element: ET.Element) -> Optional[WikiArticle]:
        """Parse a single Wikipedia page element."""
        try:
            # Extract basic info
            title_elem = page_element.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
            id_elem = page_element.find('.//{http://www.mediawiki.org/xml/export-0.10/}id')
            text_elem = page_element.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')
            redirect_elem = page_element.find('.//{http://www.mediawiki.org/xml/export-0.10/}redirect')
            
            if title_elem is None or text_elem is None:
                return None
                
            title = title_elem.text or ""
            article_id = id_elem.text if id_elem is not None else ""
            raw_text = text_elem.text or ""
            
            # Skip certain namespaces
            if any(title.startswith(prefix) for prefix in [
                'File:', 'Category:', 'Template:', 'Wikipedia:', 'Help:', 
                'Portal:', 'User:', 'Talk:', 'Media:'
            ]):
                return None
            
            # Handle redirects
            redirect_title = None
            if redirect_elem is not None:
                redirect_title = redirect_elem.get('title')
                
            # Extract metadata before cleaning
            categories = self.cleaner.extract_categories(raw_text)
            links = self.cleaner.extract_links(raw_text)
            
            # Clean text
            clean_text = self.cleaner.clean_text(raw_text)
            
            # Apply length filters
            if len(clean_text) < self.min_text_length or len(clean_text) > self.max_text_length:
                return None
            
            # Create article object
            article = WikiArticle(
                id=article_id,
                title=title,
                text=clean_text,
                categories=categories,
                links=links,
                redirect=redirect_title,
                disambiguation=self.cleaner.is_disambiguation(clean_text),
                stub=self.cleaner.is_stub(clean_text)
            )
            
            return article
            
        except Exception as e:
            self.logger.warning(f"Failed to parse article: {e}")
            return None
    
    def parse_xml_dump(self, dump_path: str) -> Iterator[WikiArticle]:
        """Parse Wikipedia XML dump file."""
        self.logger.info(f"Parsing XML dump: {dump_path}")
        
        # Handle compressed files
        if dump_path.endswith('.bz2'):
            file_obj = bz2.open(dump_path, 'rt', encoding='utf-8')
        else:
            file_obj = open(dump_path, 'r', encoding='utf-8')
        
        try:
            # Parse XML iteratively to handle large files
            context = ET.iterparse(file_obj, events=('start', 'end'))
            context = iter(context)
            
            # Get root element
            event, root = next(context)
            
            article_count = 0
            for event, elem in context:
                if event == 'end' and elem.tag.endswith('}page'):
                    article = self.parse_article(elem)
                    if article:
                        yield article
                        article_count += 1
                        
                        if article_count % 10000 == 0:
                            self.logger.info(f"Processed {article_count} articles")
                    
                    # Clear element to save memory
                    elem.clear()
                    root.clear()
                    
        finally:
            file_obj.close()


def process_chunk(args: Tuple[str, int, int, int, int]) -> List[Dict]:
    """Process a chunk of articles in a separate process."""
    dump_path, start_idx, end_idx, min_length, max_length = args
    
    parser = WikipediaXMLParser(min_length, max_length)
    articles = []
    
    try:
        for i, article in enumerate(parser.parse_xml_dump(dump_path)):
            if start_idx <= i < end_idx:
                articles.append({
                    'id': article.id,
                    'title': article.title,
                    'text': article.text,
                    'categories': article.categories,
                    'links': article.links,
                    'redirect': article.redirect,
                    'disambiguation': article.disambiguation,
                    'stub': article.stub
                })
            elif i >= end_idx:
                break
                
    except Exception as e:
        logging.error(f"Error processing chunk {start_idx}-{end_idx}: {e}")
    
    return articles


class WikiExtractor:
    """Main Wikipedia extraction class with multiprocessing support."""
    
    def __init__(
        self, 
        input_file: str,
        output_dir: str,
        processes: int = 8,
        min_text_length: int = 100,
        max_text_length: int = 50000,
        chunk_size: int = 10000
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.processes = processes
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def count_articles(self) -> int:
        """Count total number of articles in dump."""
        self.logger.info("Counting articles in dump...")
        
        if self.input_file.endswith('.bz2'):
            file_obj = bz2.open(self.input_file, 'rt', encoding='utf-8')
        else:
            file_obj = open(self.input_file, 'r', encoding='utf-8')
        
        count = 0
        try:
            for line in file_obj:
                if '<page>' in line:
                    count += 1
                    if count % 100000 == 0:
                        self.logger.info(f"Counted {count} articles...")
        finally:
            file_obj.close()
            
        self.logger.info(f"Total articles in dump: {count}")
        return count
    
    def extract(self) -> Iterator[Dict]:
        """Extract articles from Wikipedia dump."""
        self.logger.info(f"Starting extraction with {self.processes} processes")
        
        # For simplicity, use single-process extraction for now
        # Multi-processing XML parsing is complex due to file seeking issues
        parser = WikipediaXMLParser(self.min_text_length, self.max_text_length)
        
        for article in parser.parse_xml_dump(self.input_file):
            yield {
                'id': article.id,
                'title': article.title,
                'text': article.text,
                'categories': article.categories,
                'links': article.links,
                'redirect': article.redirect,
                'disambiguation': article.disambiguation,
                'stub': article.stub
            }
    
    def extract_to_files(self, articles_per_file: int = 50000) -> List[str]:
        """Extract articles and save to multiple JSON files."""
        output_files = []
        current_articles = []
        file_count = 0
        
        for article in self.extract():
            current_articles.append(article)
            
            if len(current_articles) >= articles_per_file:
                # Save current batch
                output_file = os.path.join(self.output_dir, f"wikipedia_articles_{file_count:04d}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(current_articles, f, ensure_ascii=False, indent=2)
                
                output_files.append(output_file)
                self.logger.info(f"Saved {len(current_articles)} articles to {output_file}")
                
                current_articles = []
                file_count += 1
        
        # Save remaining articles
        if current_articles:
            output_file = os.path.join(self.output_dir, f"wikipedia_articles_{file_count:04d}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_articles, f, ensure_ascii=False, indent=2)
            
            output_files.append(output_file)
            self.logger.info(f"Saved {len(current_articles)} articles to {output_file}")
        
        self.logger.info(f"Extraction complete. Created {len(output_files)} files.")
        return output_files


def main():
    """Command-line interface for Wikipedia extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from Wikipedia XML dumps")
    parser.add_argument("input_file", help="Path to Wikipedia XML dump file")
    parser.add_argument("output_dir", help="Output directory for extracted articles")
    parser.add_argument("--processes", type=int, default=8, help="Number of processes")
    parser.add_argument("--min-length", type=int, default=100, help="Minimum article length")
    parser.add_argument("--max-length", type=int, default=50000, help="Maximum article length")
    parser.add_argument("--articles-per-file", type=int, default=50000, help="Articles per output file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create extractor
    extractor = WikiExtractor(
        input_file=args.input_file,
        output_dir=args.output_dir,
        processes=args.processes,
        min_text_length=args.min_length,
        max_text_length=args.max_length
    )
    
    # Extract articles
    output_files = extractor.extract_to_files(args.articles_per_file)
    print(f"Extraction complete. Output files: {output_files}")


if __name__ == "__main__":
    main()
