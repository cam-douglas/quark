#!/usr/bin/env python3
"""
Urban Dictionary Scraper for Small-Mind Language Database
A comprehensive system to collect slang definitions while respecting rate limits
"""

import requests
import time
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from urllib.parse import urljoin, urlparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('urban_dict_scraper.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class UrbanDefinition:
    """Data structure for Urban Dictionary definitions"""
    word: str
    definition: str
    example: str
    author: str
    date: str
    thumbs_up: int
    thumbs_down: int
    definition_id: str
    url: str
    tags: List[str]
    scraped_at: datetime

class UrbanDictionaryScraper:
    """Main scraper class for Urban Dictionary"""
    
    def __init__(self, base_url: str = "https://www.urbandictionary.com", 
                 delay_range: tuple = (2, 5), max_workers: int = 3):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.delay_range = delay_range
        self.max_workers = max_workers
        self.db_path = Path("urban_dict_database.db")
        self.scraped_words: Set[str] = set()
        self.failed_words: Set[str] = set()
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing definitions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS definitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                definition TEXT NOT NULL,
                example TEXT,
                author TEXT,
                date TEXT,
                thumbs_up INTEGER DEFAULT 0,
                thumbs_down INTEGER DEFAULT 0,
                definition_id TEXT UNIQUE,
                url TEXT,
                tags TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_hash TEXT UNIQUE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS word_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                first_letter TEXT,
                word_length INTEGER,
                definition_count INTEGER DEFAULT 0,
                last_scraped TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_word ON definitions(word);
            CREATE INDEX IF NOT EXISTS idx_definition_id ON definitions(definition_id);
            CREATE INDEX IF NOT EXISTS idx_word_hash ON definitions(word_hash);
        ''')
        
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully")
    
    def _get_random_delay(self) -> float:
        """Get random delay between requests"""
        return random.uniform(*self.delay_range)
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and rate limiting"""
        for attempt in range(retries):
            try:
                delay = self._get_random_delay()
                time.sleep(delay)
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                logging.info(f"Successfully scraped: {url}")
                return response
                
            except requests.RequestException as e:
                logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"Failed to scrape {url} after {retries} attempts")
                    return None
    
    def _extract_definitions_from_page(self, html_content: str, word: str) -> List[UrbanDefinition]:
        """Extract definition data from Urban Dictionary HTML page"""
        definitions = []
        
        # This is a simplified parser - in practice you'd want more robust HTML parsing
        # Using regex as a starting point, but BeautifulSoup would be more reliable
        
        # Extract definition blocks
        definition_pattern = r'<div class="definition">(.*?)</div>'
        example_pattern = r'<div class="example">(.*?)</div>'
        author_pattern = r'<a class="author".*?>(.*?)</a>'
        date_pattern = r'<div class="date">(.*?)</div>'
        thumbs_pattern = r'<span class="count">(\d+)</span>'
        
        # Find all definition sections
        definition_sections = re.findall(definition_pattern, html_content, re.DOTALL)
        
        for i, section in enumerate(definition_sections):
            try:
                # Extract components
                definition_text = re.sub(r'<[^>]+>', '', section).strip()
                
                # Find corresponding example
                examples = re.findall(example_pattern, html_content, re.DOTALL)
                example_text = examples[i] if i < len(examples) else ""
                example_text = re.sub(r'<[^>]+>', '', example_text).strip()
                
                # Find corresponding author
                authors = re.findall(author_pattern, html_content, re.DOTALL)
                author_text = authors[i] if i < len(authors) else ""
                
                # Find corresponding date
                dates = re.findall(date_pattern, html_content, re.DOTALL)
                date_text = dates[i] if i < len(dates) else ""
                
                # Find thumbs up/down
                thumbs = re.findall(thumbs_pattern, html_content)
                thumbs_up = int(thumbs[i*2]) if i*2 < len(thumbs) else 0
                thumbs_down = int(thumbs[i*2+1]) if i*2+1 < len(thumbs) else 0
                
                # Generate unique ID
                definition_id = f"{word}_{i}_{hash(definition_text) % 10000}"
                
                # Create definition object
                urban_def = UrbanDefinition(
                    word=word,
                    definition=definition_text,
                    example=example_text,
                    author=author_text,
                    date=date_text,
                    thumbs_up=thumbs_up,
                    thumbs_down=thumbs_down,
                    definition_id=definition_id,
                    url=f"{self.base_url}/define.php?term={word}",
                    tags=[],  # Would need additional parsing for tags
                    scraped_at=datetime.now()
                )
                
                definitions.append(urban_def)
                
            except Exception as e:
                logging.warning(f"Error parsing definition {i} for word '{word}': {e}")
                continue
        
        return definitions
    
    def _save_definitions_to_db(self, definitions: List[UrbanDefinition]) -> int:
        """Save definitions to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for definition in definitions:
            try:
                # Create word hash for deduplication
                word_hash = hashlib.md5(
                    f"{definition.word}_{definition.definition}".encode()
                ).hexdigest()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO definitions 
                    (word, definition, example, author, date, thumbs_up, thumbs_down, 
                     definition_id, url, tags, scraped_at, word_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    definition.word, definition.definition, definition.example,
                    definition.author, definition.date, definition.thumbs_up,
                    definition.thumbs_down, definition.definition_id, definition.url,
                    json.dumps(definition.tags), definition.scraped_at, word_hash
                ))
                
                if cursor.rowcount > 0:
                    saved_count += 1
                    
            except sqlite3.IntegrityError as e:
                logging.warning(f"Duplicate definition for {definition.word}: {e}")
                continue
            except Exception as e:
                logging.error(f"Error saving definition for {definition.word}: {e}")
                continue
        
        conn.commit()
        conn.close()
        return saved_count
    
    def scrape_word(self, word: str) -> bool:
        """Scrape definitions for a specific word"""
        if word in self.scraped_words:
            logging.info(f"Word '{word}' already scraped, skipping")
            return True
            
        url = f"{self.base_url}/define.php?term={word}"
        response = self._make_request(url)
        
        if not response:
            self.failed_words.add(word)
            return False
        
        try:
            definitions = self._extract_definitions_from_page(response.text, word)
            if definitions:
                saved_count = self._save_definitions_to_db(definitions)
                logging.info(f"Saved {saved_count} definitions for word '{word}'")
                self.scraped_words.add(word)
                return True
            else:
                logging.warning(f"No definitions found for word '{word}'")
                return False
                
        except Exception as e:
            logging.error(f"Error processing word '{word}': {e}")
            self.failed_words.add(word)
            return False
    
    def scrape_alphabetical_section(self, letter: str, max_words: int = 100) -> int:
        """Scrape words starting with a specific letter"""
        logging.info(f"Starting to scrape words beginning with '{letter}'")
        
        # Get words from Urban Dictionary's alphabetical browse
        url = f"{self.base_url}/browse.php?character={letter}"
        response = self._make_request(url)
        
        if not response:
            logging.error(f"Failed to get word list for letter '{letter}'")
            return 0
        
        # Extract word links from browse page
        word_pattern = r'<a href="/define\.php\?term=([^&"]+)">([^<]+)</a>'
        words = re.findall(word_pattern, response.text)
        
        # Limit the number of words to scrape
        words = words[:max_words]
        
        logging.info(f"Found {len(words)} words starting with '{letter}'")
        
        # Scrape each word
        successful_scrapes = 0
        for word, display_name in words:
            if self.scrape_word(word):
                successful_scrapes += 1
        
        logging.info(f"Successfully scraped {successful_scrapes}/{len(words)} words for letter '{letter}'")
        return successful_scrapes
    
    def scrape_all_alphabetical(self, max_words_per_letter: int = 50) -> Dict[str, int]:
        """Scrape all alphabetical sections"""
        results = {}
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['#']
        
        for letter in letters:
            try:
                count = self.scrape_alphabetical_section(letter, max_words_per_letter)
                results[letter] = count
                
                # Add delay between letters to be respectful
                time.sleep(random.uniform(5, 10))
                
            except Exception as e:
                logging.error(f"Error scraping letter '{letter}': {e}")
                results[letter] = 0
        
        return results
    
    def scrape_trending_words(self, max_words: int = 100) -> int:
        """Scrape trending/popular words"""
        logging.info("Starting to scrape trending words")
        
        url = f"{self.base_url}/"
        response = self._make_request(url)
        
        if not response:
            logging.error("Failed to get trending words page")
            return 0
        
        # Extract trending words from homepage
        trending_pattern = r'<a href="/define\.php\?term=([^&"]+)">([^<]+)</a>'
        trending_words = re.findall(trending_pattern, response.text)
        
        # Limit words to scrape
        trending_words = trending_words[:max_words]
        
        logging.info(f"Found {len(trending_words)} trending words")
        
        # Scrape each trending word
        successful_scrapes = 0
        for word, display_name in trending_words:
            if self.scrape_word(word):
                successful_scrapes += 1
        
        logging.info(f"Successfully scraped {successful_scrapes}/{len(trending_words)} trending words")
        return successful_scrapes
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the scraped database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total definitions
        cursor.execute("SELECT COUNT(*) FROM definitions")
        total_definitions = cursor.fetchone()[0]
        
        # Unique words
        cursor.execute("SELECT COUNT(DISTINCT word) FROM definitions")
        unique_words = cursor.fetchone()[0]
        
        # Most popular words
        cursor.execute("""
            SELECT word, COUNT(*) as def_count 
            FROM definitions 
            GROUP BY word 
            ORDER BY def_count DESC 
            LIMIT 10
        """)
        popular_words = cursor.fetchall()
        
        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM definitions 
            WHERE scraped_at > datetime('now', '-1 day')
        """)
        recent_definitions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_definitions': total_definitions,
            'unique_words': unique_words,
            'popular_words': popular_words,
            'recent_definitions': recent_definitions,
            'scraped_words': len(self.scraped_words),
            'failed_words': len(self.failed_words)
        }
    
    def export_to_json(self, output_path: str = "urban_dict_database.json"):
        """Export database to JSON format"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT word, definition, example, author, date, thumbs_up, thumbs_down, 
                   definition_id, url, tags, scraped_at
            FROM definitions
            ORDER BY word, thumbs_up DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        data = []
        for row in rows:
            data.append({
                'word': row[0],
                'definition': row[1],
                'example': row[2],
                'author': row[3],
                'date': row[4],
                'thumbs_up': row[5],
                'thumbs_down': row[6],
                'definition_id': row[7],
                'url': row[8],
                'tags': json.loads(row[9]) if row[9] else [],
                'scraped_at': row[10]
            })
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Exported {len(data)} definitions to {output_path}")
        return len(data)

def main():
    """Main execution function"""
    print("Urban Dictionary Scraper for Small-Mind Language Database")
    print("=" * 60)
    
    # Initialize scraper
    scraper = UrbanDictionaryScraper(delay_range=(3, 7), max_workers=2)
    
    try:
        # Start with trending words
        print("\n1. Scraping trending words...")
        trending_count = scraper.scrape_trending_words(max_words=50)
        print(f"   Scraped {trending_count} trending words")
        
        # Then scrape alphabetical sections
        print("\n2. Scraping alphabetical sections...")
        results = scraper.scrape_all_alphabetical(max_words_per_letter=30)
        
        print("\nAlphabetical scraping results:")
        for letter, count in results.items():
            print(f"   {letter}: {count} words")
        
        # Show final statistics
        print("\n3. Final database statistics:")
        stats = scraper.get_database_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Export to JSON
        print("\n4. Exporting to JSON...")
        export_count = scraper.export_to_json()
        print(f"   Exported {export_count} definitions")
        
        print("\nScraping completed successfully!")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"\nError during scraping: {e}")
        logging.error(f"Scraping error: {e}")

if __name__ == "__main__":
    main()
