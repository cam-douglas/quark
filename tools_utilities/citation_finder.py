import requests
import json
import time
from typing import List, Dict, Optional

def get_api_key(service_name: str) -> Optional[str]:
    """Reads API key from the all_api_keys.json file."""
    try:
        with open("data/credentials/all_api_keys.json", "r") as f:
            keys = json.load(f)
            if service_name == "ncbi_eutilities" and "pubmed_central" in keys.get("services", {}):
                return keys["services"]["pubmed_central"].get("api_key")
            return keys.get("services", {}).get(service_name, {}).get("api_key")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading API keys: {e}")
        return None

def search_pubmed(query: str, api_key: str, retmax: int = 3) -> List[str]:
    """Searches PubMed for a given query and returns a list of PMIDs."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "api_key": api_key,
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except requests.RequestException as e:
        print(f"Error searching PubMed for '{query}': {e}")
        return []

def fetch_citation_details(pmids: List[str], api_key: str) -> List[str]:
    """Fetches citation details (Author et al., Year) for a list of PMIDs."""
    if not pmids:
        return []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "api_key": api_key,
    }
    citations = []
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("result", {})
        
        for pmid in pmids:
            article = results.get(pmid, {})
            authors = article.get("authors", [])
            pubdate = article.get("pubdate", "")
            year = pubdate.split(" ")[0] if pubdate else "N/A"
            
            if authors:
                first_author = authors[0]["name"]
                if len(authors) > 1:
                    citation_str = f"({first_author} et al., {year})"
                else:
                    citation_str = f"({first_author}, {year})"
            else:
                citation_str = f"(PMID: {pmid}, {year})"
            citations.append(citation_str)
            
    except requests.RequestException as e:
        print(f"Error fetching citation details: {e}")
    return citations

def main():
    """Main function to find citations for a list of topics."""
    api_key = get_api_key("ncbi_eutilities")
    if not api_key:
        print("Could not find API key for NCBI E-utilities (using PubMed Central key).")
        return

    search_topics = {
        "Ventricular System Morphogenesis": "ventricular system development neural tube morphogenesis",
        "Meninges Formation": "meninges development dura mater arachnoid pia mater",
        "SHH Gradient": "sonic hedgehog gradient timing patterning",
        "BMP Gradient Dynamics": "BMP gradient signaling dynamics embryo",
        "WNT FGF Gradient Patterning": "WNT FGF gradient patterning embryo",
    }

    all_citations = {}

    for topic, query in search_topics.items():
        print(f"Searching for: {topic}...")
        pmids = search_pubmed(query, api_key, retmax=1)
        if pmids:
            print(f"Found PMIDs: {pmids}. Fetching details...")
            citations = fetch_citation_details(pmids, api_key)
            all_citations[topic] = citations
            print(f"Formatted Citations: {citations}")
        else:
            all_citations[topic] = ["No quantitative data found in top results."]
            print("No relevant PMIDs found.")
        time.sleep(1)

    print("\n--- Formatted Citations ---")
    print(json.dumps(all_citations, indent=2))

if __name__ == "__main__":
    main()