"""A simple, robust web scraper for the SelfLearningOrchestrator.

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Operational tooling invoked by agents/simulators when required.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional

class InternetScraper:
    """
    A simple web scraper to fetch and parse the textual content of a webpage.
    """
    def __init__(self, timeout: int = 10):
        """
        Initializes the scraper.
        Args:
            timeout: The timeout in seconds for web requests.
        """
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Quark-SelfLearning-Agent/1.0'
        }

    def fetch_text_from_url(self, url: str) -> Optional[str]:
        """
        Fetches the main textual content from a given URL.
        Args:
            url: The URL to scrape.
        Returns:
            The extracted text content, or None if an error occurred.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception for bad status codes

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
