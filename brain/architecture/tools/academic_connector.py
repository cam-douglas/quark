"""A connector for academic journal APIs like NCBI and IEEE.

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Operational tooling invoked by agents/simulators when required.
"""

import requests
import configparser
from typing import List, Dict

class AcademicConnector:
    """
    Handles interaction with academic APIs to search for and retrieve articles.
    """
    def __init__(self, config_path: str = "brain_architecture/config/config.ini"):
        """Initializes the academic connector."""
        config = configparser.ConfigParser()
        config.read(config_path)

        # NCBI (PubMed) configuration
        self.ncbi_api_key = config.get("API_KEYS", "ncbi_api_key", fallback=None)
        self.ncbi_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # IEEE configuration
        self.ieee_api_key = config.get("API_KEYS", "ieee_api_key", fallback=None)

    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Searches the NCBI PubMed database.
        Reference: https://www.ncbi.nlm.nih.gov/books/NBK25500/
        Args:
            query: The search term.
            max_results: The maximum number of results to return.
        Returns:
            A list of dictionaries, each containing info about an article.
        """
        if not self.ncbi_api_key or self.ncbi_api_key == "YOUR_NCBI_API_KEY_HERE":
            print("NCBI API key not configured. Skipping PubMed search.")
            return []

        # ESearch to get UIDs
        esearch_url = f"{self.ncbi_base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "apikey": self.ncbi_api_key,
            "format": "json"
        }
        try:
            response = requests.get(esearch_url, params=params)
            response.raise_for_status()
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # ESummary to get details for the UIDs
            esummary_url = f"{self.ncbi_base_url}esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "apikey": self.ncbi_api_key,
                "format": "json"
            }
            summary_response = requests.get(esummary_url, params=summary_params)
            summary_response.raise_for_status()
            summary_data = summary_response.json()

            results = []
            for uid in id_list:
                item = summary_data.get("result", {}).get(uid, {})
                results.append({
                    "uid": item.get("uid"),
                    "title": item.get("title"),
                    "authors": [author["name"] for author in item.get("authors", [])],
                    "journal": item.get("fulljournalname"),
                    "pub_date": item.get("pubdate"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                })
            return results

        except requests.exceptions.RequestException as e:
            print(f"Error searching PubMed: {e}")
            return []

    def search_ieee(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Searches the IEEE Xplore database.
        Reference: https://open.ieee.org/publishing-options/topical-journals/
        Args:
            query: The search term.
            max_results: The maximum number of results to return.
        Returns:
            A list of dictionaries, each containing info about an article.
        """
        if not self.ieee_api_key:
            print("IEEE API key not configured. Skipping IEEE search.")
            return []

        ieee_base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"
        params = {
            "querytext": query,
            "max_records": max_results,
            "sort_order": "desc",
            "sort_field": "article_title",
            "apikey": self.ieee_api_key,
            "format": "json"
        }

        try:
            response = requests.get(ieee_base_url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("articles", []):
                results.append({
                    "doi": item.get("doi"),
                    "title": item.get("title"),
                    "authors": [author["full_name"] for author in item.get("authors", {}).get("authors", [])],
                    "journal": item.get("publication_title"),
                    "pub_date": item.get("publication_year"),
                    "url": item.get("html_url")
                })
            return results

        except requests.exceptions.RequestException as e:
            print(f"Error searching IEEE Xplore: {e}")
            return []
