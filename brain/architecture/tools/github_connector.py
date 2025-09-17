"""A connector for the GitHub API to search for code and repositories.

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Operational tooling invoked by agents/simulators when required.
"""

import requests
from typing import List, Dict

class GitHubConnector:
    """
    Handles interaction with the GitHub API for code and repository discovery.
    """
    def __init__(self, base_url: str = "https://api.github.com"):
        """Initializes the GitHub connector."""
        self.base_url = base_url
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }

    def search_repositories(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Searches for repositories on GitHub.
        Args:
            query: The search term.
            max_results: The maximum number of results to return.
        Returns:
            A list of dictionaries, each containing info about a repository.
        """
        search_url = f"{self.base_url}/search/repositories"
        params = {"q": query, "per_page": max_results, "sort": "stars", "order": "desc"}

        try:
            response = requests.get(search_url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "full_name": item["full_name"],
                    "url": item["html_url"],
                    "description": item.get("description"),
                    "stars": item.get("stargazers_count"),
                    "language": item.get("language")
                })
            return results
        except requests.exceptions.RequestException as e:
            print(f"Error searching GitHub repositories: {e}")
            return []
